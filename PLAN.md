# TalkWalk — Steerable Recommendation Engine

> An open-source recommendation engine where users can change the algorithm with plain English, in real-time. Like Spotify's AI DJ, but generalized for any domain.

---

## Table of Contents

1. [Why TalkWalk](#1-why-talkwalk)
2. [Core Architecture](#2-core-architecture)
3. [Technical Deep Dive](#3-technical-deep-dive)
4. [Inference Flow](#4-inference-flow)
5. [Repo Structure](#5-repo-structure)
6. [Training Data](#6-training-data)
7. [Evaluation](#7-evaluation)
8. [Hardware and Dependencies](#8-hardware-and-dependencies)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [References](#10-references)

---

## 1. Why TalkWalk

Traditional recommendation systems treat language and items as separate worlds — the user speaks English, a pipeline translates that into filters and queries, and a separate model ranks results. The user has no real control over *how* the algorithm thinks.

TalkWalk unifies language and items inside a single model. When a user says "something chill," it doesn't trigger a filter — it directly shifts the probability distribution over items toward low-energy, relaxed content. The words *are* the algorithm.

| | Traditional Recommenders | TalkWalk |
|---|---|---|
| **Steering** | Filters, sliders, thumbs up/down | Plain English changes inference path in real-time |
| **Architecture** | Pipeline of separate components (retrieval, ranking, re-ranking) | Single model infers in language space; contrastive embeddings resolve to items |
| **Item understanding** | Collaborative filtering on interaction history | Contrastively trained on metadata — understands what items *are* |
| **User understanding** | Demographic buckets, sparse features | User metadata contrastively trained with language descriptions — understands who users *are* |
| **Cold start** | Needs interaction history | Metadata alone is enough — new items and new users work immediately |
| **Latency** | Multiple model calls + database queries | One forward pass + one nearest-neighbor lookup |
| **Scalability** | Item IDs grow linearly with catalog | Contrastive space is fixed-dimensional regardless of catalog size |
| **State** | Stateless or shallow session tracking | Implicit — the conversation context *is* the traversal |

### The core insight

The LLM should never know about item IDs. It should stay entirely in language space — understanding user intent, maintaining conversational context, reasoning about what to recommend. The contrastive embedding space, trained to align metadata and language, handles the translation from "what the user wants" to "which specific item that is." Inference and ID resolution are completely decoupled.

---

## 2. Core Architecture

### The Two-Space Design

```
LANGUAGE SPACE (the LLM)              CONTRASTIVE SPACE (the bridge)

User text + system text               Item metadata embeddings
       |                               User metadata embeddings
       v                               Language description embeddings
LLM reasons in natural language              |
       |                                     |
       v                                     v
Hidden state / generated description   All live in the same vector space
       |                                     |
       '----- projection ------->  nearest neighbor lookup
                                             |
                                             v
                                       Real item ID
```

The LLM stays in language. The contrastive space does ID resolution. They are decoupled.

### Training Pipeline

```
Phase 1                              Phase 2
Contrastive Pre-training         --> Projection Training
(item metadata <-> text)              (LLM hidden state -> contrastive space)
(user metadata <-> text)              lightweight linear projection
(composites   <-> text)
OpenCLIP + InfoNCE
```

No quantization. No vocabulary extension. No semantic IDs. No constrained decoding.

### Inference Pipeline

```
User input + user metadata
    |
    v
Conversation context (all prior turns + system context)
    |
    v
LLM forward pass (pure language inference)
    |
    v
Extract hidden state from final layer
    |
    v
Project into contrastive embedding space
    |
    v
Nearest neighbor search against item catalog embeddings
    |
    v
Return real item ID to client
```

### The Key Principle: Everything Reshapes the Distribution

At every point during inference, the LLM's internal representation is shaped by everything in the context:

- **User-provided English** — "something chill," "more like that," "wake me up" — each phrase shifts the LLM's hidden state, which shifts where it lands in contrastive space
- **System-provided user metadata embeddings** — the user's age, platform, time of day, listening history — injected into the prompt, these bias the model's reasoning toward regions that match the user's profile
- **System-provided contextual signals** — session context, domain rules, business constraints — further shape the LLM's output
- **Prior conversation** — every previous recommendation and user reaction is in the context window, influencing the current output

There is no distinction between "the algorithm" and "the inputs." The LLM's hidden state *is* the algorithm, and every signal — user or system, language or embedding — continuously reshapes it. The contrastive space then translates that hidden state into a concrete item. This is what makes TalkWalk fundamentally different: the recommendation algorithm is not a fixed function that takes inputs; it is a living representation that every participant (user, system, context) is constantly sculpting.

### Context Length

At each inference point, the context fed to the model is:

```
[ new user text ] + [ new system text ] + [ truncated conversation history ]
```

The conversation history is truncated to a configurable maximum length (e.g. last N turns or K tokens). This bounds memory and compute regardless of session length. The truncation is not a limitation — it mirrors how recommendation relevance naturally decays over time. Recent interactions matter more than distant ones, and the contrastive embeddings already encode long-term user preferences via the system-provided user metadata. The sliding window of recent context handles short-term trajectory; the user profile handles long-term identity.

### What Makes This Novel

1. **Inference is decoupled from IDs** — the LLM never generates item IDs; it reasons in pure language, and the contrastive space resolves to items separately
2. **Every input changes the algorithm** — user English, system metadata, prior outputs — all continuously reshape the LLM's hidden state, which determines where it lands in contrastive space
3. **No quantization artifacts** — no hierarchical commitment, no path dependence, no vocabulary extension; the contrastive space is continuous
4. **No fine-tuning needed** — the LLM already understands language; we only train a lightweight projection from its hidden state to the contrastive space
5. **Domain-agnostic** — works for music, products, movies, articles — anything with metadata

---

## 3. Technical Deep Dive

### 3.1 Phase 1: Contrastive Pre-training (Metadata <-> Text)

**Goal:** Create a shared embedding space where structured metadata and human language descriptions mean the same thing — for items, users, and their combinations.

**Architecture:** CLIP-style dual encoder with InfoNCE loss.

```python
# Encoder A: Text (pre-trained, frozen initially)
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")  # or Qwen3-0.6B

# Encoder B: Metadata (learned from scratch — shared across item and user metadata)
class MetadataEncoder(nn.Module):
    def __init__(self, metadata_dim, embed_dim):
        self.encoder = nn.Sequential(
            nn.Linear(metadata_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )

# Loss: Symmetric InfoNCE
# For batch of N (metadata, text) pairs:
# - Compute NxN cosine similarity matrix
# - Maximize diagonal (matching pairs)
# - Minimize off-diagonal (mismatched pairs)
```

#### Item Metadata Descriptions

Human language descriptions per metadata key/value — the "language bridge":

```yaml
genre:
  indie-rock: "guitar-driven independent rock music with raw emotional vocals"
  electronic: "synthesizer and beat-driven dance music"
  jazz: "improvisational music with complex harmonies and swing rhythms"
mood:
  melancholic: "sad, reflective, bittersweet feeling"
  euphoric: "intense joy and uplifting energy"
  relaxed: "calm, easygoing, peaceful atmosphere"
energy:
  0.0-0.3: "low energy, quiet, subdued"
  0.4-0.6: "moderate energy, balanced"
  0.7-1.0: "high energy, intense, driving"
```

#### User Metadata Descriptions

Every user attribute gets the same treatment — human language descriptions so the model understands *who* it's recommending to, not just *what* to recommend:

```yaml
age_range:
  13-17: "teenager, discovering tastes, trend-sensitive"
  18-24: "young adult, exploratory, open to new genres"
  25-34: "established preferences but still adventurous"
  35-49: "mature taste, values quality over novelty"
  50+: "deep knowledge of classics, selective discovery"

listening_frequency:
  casual: "listens occasionally, background music"
  regular: "daily listener, curated playlists"
  power: "music is central to life, constantly exploring"

platform:
  mobile: "on the go, shorter sessions, needs energy-aware recs"
  desktop: "focused listening sessions, deeper exploration"
  smart_speaker: "ambient, social setting, mood-driven"

time_of_day:
  morning: "waking up, needs energy ramp, fresh start"
  afternoon: "focused or working, steady rhythm"
  evening: "winding down, reflective, social"
  late_night: "introspective, deep cuts, experimental"
```

#### Combined Metadata Sets (Composite Descriptions)

The real power is in combinations. Individual metadata values are useful, but behavior emerges from intersections. We train on composite descriptions that capture the *meaning* of metadata combinations:

```yaml
age_range=18-24 + listening_frequency=power:
  "young music obsessive, constantly seeking the next underground hit,
   thrives on discovery and genre-blending"

age_range=35-49 + listening_frequency=casual:
  "knows what they like, low patience for misses,
   recommend proven quality not experiments"

age_range=18-24 + time_of_day=late_night:
  "late night young adult — experimental, introspective,
   open to ambient, lo-fi, or deep electronic"

age_range=50+ + platform=smart_speaker + time_of_day=morning:
  "older listener, morning routine, smart speaker —
   familiar comfort, jazz or classical, moderate energy"
```

#### Cross-Domain Training (User x Item)

We also train contrastively on user-item affinity descriptions — teaching the model that certain user profiles naturally align with certain item profiles:

```yaml
user={age: 18-24, freq: power, time: late_night} + item={genre: ambient, energy: 0.2}:
  "young night owl deep-listening to atmospheric ambient — high affinity"

user={age: 35-49, freq: casual} + item={genre: experimental-noise, energy: 0.9}:
  "mature casual listener vs harsh experimental — low affinity, avoid unless requested"
```

This means when the model sees a user profile at inference time, it already knows which regions of item-space are natural fits — and which require an explicit steering request to reach.

#### Training Strategy

All metadata types are trained together in the same contrastive embedding space:

```
Item metadata      <-- contrastive -->  Item descriptions
User metadata      <-- contrastive -->  User descriptions
Combined metadata  <-- contrastive -->  Composite descriptions
User x Item pairs  <-- contrastive -->  Affinity descriptions
```

This creates a unified embedding space where:
- "18-24 year old power listener" is *near* "experimental, high-discovery items"
- "50+ casual morning listener" is *near* "familiar, moderate-energy classics"
- Plain English steering ("show me something weird") moves through the same space

**Libraries:** OpenCLIP (with MetadataEncoder replacing vision tower) + info-nce-pytorch

**Output:** Dense embeddings where item metadata, user metadata, combined profiles, and natural language all live in the same vector space. Every item in the catalog has a fixed embedding vector in this space.

---

### 3.2 Phase 2: Projection Training

**Goal:** Train a lightweight projection that maps the LLM's hidden state into the contrastive embedding space — while solving three geometric problems that naive projection would create.

The LLM (Qwen3-8B or similar) processes the conversation and produces a hidden state — a rich representation of the user's intent, context, and preferences. We need to project that hidden state into the contrastive space where item embeddings live, so we can do a nearest-neighbor lookup.

```python
class ProjectionHead(nn.Module):
    """Maps LLM hidden state to contrastive embedding space."""
    def __init__(self, llm_dim, contrastive_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(llm_dim, contrastive_dim),
            nn.GELU(),
            nn.Linear(contrastive_dim, contrastive_dim),
        )

    def forward(self, hidden_state):
        return F.normalize(self.projection(hidden_state), dim=-1)

# Example: Qwen3-8B has hidden_dim=4096, contrastive space is 512-dim
proj = ProjectionHead(llm_dim=4096, contrastive_dim=512)
```

#### The Geometric Problems

A naive projection (just cosine similarity loss against target items) will fail in subtle ways. The problems are not structural — they are geometric.

**Problem 1 — Hidden State Drift.** LLM hidden states are not naturally linear preference vectors. When a user says "more energetic," the hidden state does not move along a clean "energy axis." It moves along a messy mixture of syntax, discourse state, conversational memory, politeness patterns, and reasoning traces. The projection must learn to extract *only* the preference component. Otherwise steering becomes noisy.

**Problem 2 — Projection Collapse.** Without constraints, the projection learns shortcuts: similar conversations map to the same embedding region, regardless of intent differences. You get good first recommendations but weak controllability. This is the most common failure mode in LLM-to-embedding projection systems.

**Problem 3 — Clustered Contrastive Space.** The contrastive space from Phase 1 is trained on similarity (InfoNCE), not continuity. Recommendation requires continuous preference gradients (calm -> mellow -> balanced -> upbeat -> intense). If embeddings form isolated clusters instead of smooth gradients, steering feels jumpy — small changes in language cause large jumps in recommendations.

#### The Training Objectives (5 Losses)

These are lightweight additions to the projection training, not redesigns. Together they solve all three geometric problems.

**Loss 1 — Target Similarity (baseline).** Standard cosine similarity between projected hidden state and target item embedding:

```
L_target = 1 - cosine(projection(H), target_item_embedding)
```

**Loss 2 — Directional Steering (solves Problem 1, most important).** Explicitly teaches that language edits correspond to vector directions in contrastive space.

Create paired training examples where the only difference is a steering utterance:

```
C1: "recommend something chill"           --> H1
C2: "recommend something more energetic"  --> H2
```

Compute the expected direction from metadata:
```
delta_target = embedding(energetic_items) - embedding(chill_items)
delta_model  = projection(H2) - projection(H1)
```

Loss:
```
L_direction = 1 - cosine(delta_model, delta_target)
```

This teaches: "more energetic" means move *this way* in contrastive space. Steering becomes linear and predictable. This single loss dramatically improves controllability.

**Loss 3 — Anchor Reconstruction (solves Problem 2).** Forces projected hidden states to remain semantically interpretable, preventing projection collapse.

After projection, a small decoder reconstructs the metadata description text embedding:

```
z = projection(hidden_state)
reconstructed = decoder(z)
L_reconstruct = 1 - cosine(reconstructed, text_encoder(metadata_description))
```

The projection cannot memorize conversations — it must encode semantic intent to reconstruct descriptions. This keeps the projection honest.

**Loss 4 — Conversation Delta (high ROI).** Instead of training only on final targets, also train on turn-to-turn transitions.

```
Turn 1 --> item A
Turn 2 ("more upbeat") --> item B

L_delta = margin_loss(
    projection(H2) closer to B than A,
    projection(H1) closer to A than B
)
```

This directly teaches conversational steering dynamics — the projection learns how turns change recommendations, not just what the final answer should be.

**Loss 5 — Context Dropout (quiet but critical).** Long conversations can dominate hidden states, making the projection position-dependent rather than meaning-dependent.

During projection training, randomly:
- Drop earlier turns from context
- Shuffle irrelevant history
- Vary metadata verbosity

This forces the projection to rely on semantic meaning, not prompt position. Result: robust steering even after long sessions.

#### Combined Training Loss

```
L = L_target + alpha * L_direction + beta * L_reconstruct + gamma * L_delta
```

With context dropout applied stochastically during training. The LLM is frozen throughout — only the projection head and small decoder are trained.

Suggested starting weights: alpha=1.0, beta=0.5, gamma=0.5. The directional steering loss is the most important; start there and add the others incrementally.

**Base model:** Qwen3-8B (or any open-source LLM)
- Best embedding quality at 7-8B size
- Apache 2.0 / permissive license
- No vocabulary extension needed — the model stays as-is
- No fine-tuning needed — only the projection head and decoder are trained

---

### 3.3 Contrastive Space Smoothing (Phase 1 Addition)

Phase 1's InfoNCE loss teaches similarity but not continuity. To create smooth preference gradients in the contrastive space, add synthetic interpolation training.

Take two metadata profiles at different points on an axis:
```
energy = 0.2  --> "low energy, quiet, subdued"
energy = 0.8  --> "high energy, intense, driving"
```

Generate a midpoint description:
```
energy = 0.5  --> "moderately energetic, balanced intensity"
```

Train so the midpoint embedding falls between the endpoints:
```
L_smooth = ||E(midpoint) - 0.5 * (E(low) + E(high))||^2
```

This creates continuous axes inside the contrastive space. Without this, nearest-neighbor results jump between clusters instead of gliding along gradients. Apply this along every metadata axis that has a natural ordering (energy, tempo, valence, price, etc.).

---

### 3.3 Item Index

**Goal:** Enable fast nearest-neighbor lookup in the contrastive space.

Every item in the catalog is embedded once (via the metadata encoder from Phase 1) and stored in a vector index:

```python
import faiss

# Embed all items
item_embeddings = metadata_encoder(all_item_metadata)  # shape: [num_items, contrastive_dim]

# Build index
index = faiss.IndexFlatIP(contrastive_dim)  # inner product = cosine sim on normalized vecs
index.add(item_embeddings)

# At inference: project LLM hidden state, query index
projected = projection_head(llm_hidden_state)
distances, item_indices = index.search(projected, k=10)  # top-10 nearest items
```

For large catalogs (millions of items), use approximate nearest neighbor (FAISS IVF, HNSW, or ScaNN) for sub-millisecond lookup.

New items are added by embedding their metadata and inserting into the index. No retraining needed.

---

## 4. Inference Flow

### Example Catalog

```
song_042: {genre: "indie-rock", mood: "melancholic", energy: 0.3}  --> embedding [0.12, -0.45, ...]
song_871: {genre: "electronic", mood: "euphoric", energy: 0.9}     --> embedding [0.87, 0.33, ...]
song_215: {genre: "jazz", mood: "relaxed", energy: 0.2}            --> embedding [0.08, -0.52, ...]
```

### Multi-Turn Conversation

```
TURN 1:
  System: "User profile: 25yo, prefers indie."
  User:   "Recommend something"
  LLM:    produces hidden state H1
  Project: H1 --> contrastive space --> nearest neighbor --> song_042 (indie, melancholic)
  Append to context: "Recommended: indie-rock, melancholic, low energy"

TURN 2:
  User:   "More like that"
  LLM:    sees prior context + "more like that" --> produces H2
  Project: H2 --> contrastive space --> nearest to H2 but not song_042 --> song_215 (jazz, relaxed)

  The hidden state H2 is shaped by the full context: user profile + song_042's description
  + "more like that." It naturally lands near the same region of contrastive space.

TURN 3:
  User:   "ok now something to wake me up, really intense"
  LLM:    sees full context + steering text --> produces H3
  Project: H3 --> contrastive space --> song_871 (electronic, euphoric, energy: 0.9)

  "wake me up" shifts the hidden state. The projection maps it to a completely different
  region of contrastive space. No hierarchical commitment. No path dependence.
  The jump from chill to intense is one continuous vector operation.

RETURN TO CLIENT:
  { id: "song_871" }
```

There is no quantization, no token generation for IDs, no hierarchy to commit to. The LLM reasons in language. The contrastive space resolves to items. A hard pivot ("something completely different") is just as easy as a subtle refinement ("a bit more upbeat") — both are just different directions in a continuous space.

---

## 5. Repo Structure

```
Talkwalk/
  README.md                     # Project overview, quick start, demo GIF
  PLAN.md                       # This file
  LICENSE                       # Apache 2.0

  talkwalk/
    __init__.py
    config.py                   # Hydra/YAML configuration

    # Phase 1: Contrastive pre-training
    contrastive/
      __init__.py
      metadata_encoder.py       # MetadataEncoder nn.Module
      text_encoder.py           # Text encoder wrapper
      infonce_loss.py           # Symmetric InfoNCE implementation
      train_contrastive.py      # Training script
      dataset.py                # Paired (metadata, description) dataset
      user_metadata.py          # User metadata encoding + composite profile builder

    # Phase 2: Projection
    projection/
      __init__.py
      projection_head.py        # LLM hidden state -> contrastive space
      anchor_decoder.py         # Small decoder for reconstruction loss
      losses.py                 # All 5 training losses (target, directional, reconstruct, delta, dropout)
      train_projection.py       # Projection training script
      steering_pairs.py         # Generate (base_context, steered_context) pairs for directional loss

    # Item index
    index/
      __init__.py
      catalog.py                # Embed catalog, build FAISS index
      lookup.py                 # Nearest neighbor search, real ID resolution

    # Inference
    engine/
      __init__.py
      recommender.py            # Main recommendation engine class
      prompt_builder.py         # Build prompts from user metadata + context
      session.py                # Conversation context management

    # Evaluation
    eval/
      __init__.py
      controllability.py        # Steering entropy, turns-to-satisfaction, diversity gradients
      interpretability.py       # Embedding coherence, steering traceability
      efficiency.py             # First-turn relevance, exploration velocity, cold-start
      sanity.py                 # Basic accuracy, coverage, latency
      benchmark.py              # Run full eval suite

  # Data
  data/
    descriptions/
      items/                    # Item metadata descriptions per key/value
        example_music.yaml
        example_ecommerce.yaml
      users/                    # User metadata descriptions per key/value
        demographics.yaml       # age ranges, gender, location
        behavior.yaml           # listening frequency, session patterns
        context.yaml            # time of day, platform, device
      composites/               # Combined metadata set descriptions
        user_combinations.yaml  # e.g. age + frequency + time_of_day
        user_item_affinity.yaml # cross-domain user-item affinity pairs
    scripts/
      prepare_catalog.py        # Convert raw catalog to training format
      prepare_user_profiles.py  # Convert user data to training format
      generate_descriptions.py  # LLM-augment descriptions for diversity
      generate_composites.py    # Generate combined metadata descriptions

  # Configs
  configs/
    contrastive.yaml
    projection.yaml
    index.yaml
    eval.yaml

  # Demo
  demo/
    app.py                      # Gradio/Streamlit interactive demo
    example_music.py            # Music recommendation example
    example_ecommerce.py        # E-commerce example

  tests/
    test_contrastive.py
    test_projection.py
    test_index.py
    test_engine.py

  pyproject.toml
  Makefile                      # make train-contrastive, make train-projection, etc.
```

---

## 6. Training Data

### Phase 1: Contrastive Pre-training (100K+ pairs across all metadata types)

**Item metadata pairs:**
```json
{
  "metadata": {"genre": "indie-rock", "tempo": 128, "mood": "melancholic", "energy": 0.3},
  "description": "guitar-driven independent rock with raw emotional vocals, slow melancholic energy"
}
```

**User metadata pairs:**
```json
{
  "metadata": {"age_range": "18-24", "listening_frequency": "power", "time_of_day": "late_night"},
  "description": "young music obsessive, late night deep listening, experimental and introspective"
}
```

**Combined/composite pairs:**
```json
{
  "metadata": {"user": {"age": "18-24", "freq": "power"}, "item": {"genre": "ambient", "energy": 0.2}},
  "description": "young power listener exploring atmospheric ambient — high affinity, discovery mode"
}
```

Source: Catalog metadata + user profiles + human-written key descriptions + LLM-generated composites + paraphrases for augmentation.

### Phase 2: Projection Training

(Conversation context, target item embedding) pairs:

```json
{
  "context": "User profile: 25yo, power listener. Previously recommended: indie-rock melancholic track. User says: 'something with more energy'",
  "target_item_embedding": [0.87, 0.33, ...]
}
```

Source: Historical interaction sequences reformatted as conversations, with target items embedded via the contrastive encoder from Phase 1.

### Benchmark Datasets

| Dataset | Items | Domain | Why |
|---------|-------|--------|-----|
| Amazon Reviews 2023 | 48M reviews | E-commerce | Standard benchmark |
| MovieLens 25M | 62K movies | Movies | Rich metadata for steerability testing |
| Steam | 7.8M reviews | Games | Good metadata, proven domain |

---

## 7. Evaluation

TalkWalk's advantage is not ranking accuracy — it's controllability, interpretability, and interaction efficiency. We evaluate on those axes first, with traditional metrics as a secondary sanity check.

### Primary Metrics: Controllability

| Metric | What It Measures | Protocol |
|--------|-----------------|----------|
| **Steering entropy reduction** | How much does a steering utterance narrow the output distribution? | Measure entropy of the nearest-neighbor distance distribution before and after a steering input. Higher reduction = more responsive steering. |
| **Turns-to-satisfaction** | How many turns does a user need to reach a satisfying recommendation? | Simulated and human eval: give users a target preference, count turns until the output matches. Lower is better. |
| **Controllable diversity gradient** | Can the user smoothly control how diverse recommendations are? | User says "more variety" / "stay close" — measure the resulting spread across outputs. Should scale proportionally to steering intensity. |
| **Steering precision** | When a user says "more X," does X increase without unrelated attributes changing? | Measure change in target attribute vs. change in non-target attributes. High precision = surgical steering. |
| **Steering reversibility** | Can the user undo a steer? | User steers toward X, then says "go back." Measure cosine similarity of projected hidden state to pre-steer state. |

### Primary Metrics: Interpretability

| Metric | What It Measures | Protocol |
|--------|-----------------|----------|
| **Embedding space coherence** | Do contrastive space regions correspond to human-understandable categories? | Cluster regions, label them via LLM, ask human raters if labels match contents. |
| **Steering traceability** | Can you explain *why* a recommendation changed? | Given a steer + output shift, measure whether the movement in contrastive space aligns with the steering direction. |

### Primary Metrics: Interaction Efficiency

| Metric | What It Measures | Protocol |
|--------|-----------------|----------|
| **First-turn relevance** | How good is the recommendation with zero steering? | User metadata + first prompt only. Measure user satisfaction. |
| **Exploration velocity** | How quickly can a user traverse different regions of the catalog? | Count unique embedding-space clusters reached in N turns. |
| **Cold-start quality** | How well does it work for a brand new user with only metadata? | Evaluate recommendations for users with no interaction history. |

### Secondary Metrics: Sanity Checks

| Metric | What | Baseline |
|--------|------|----------|
| Nearest-neighbor accuracy | Does the top-1 result match ground truth? | > 15% |
| Catalog coverage | % of catalog ever recommended | > 30% |
| Latency | LLM forward pass + NN lookup | < 500ms |

---

## 8. Hardware and Dependencies

### Core Dependencies

| Library | Purpose |
|---------|---------|
| PyTorch 2.x | Framework |
| Qwen3-8B | Base LLM (frozen, no fine-tuning) |
| Qwen3-0.6B | Lightweight item/user embedding |
| OpenCLIP | Contrastive training framework |
| info-nce-pytorch | InfoNCE loss |
| FAISS | Nearest-neighbor search |
| SentenceTransformers | Text encoding |
| Gradio | Demo UI |
| vLLM | Production LLM serving |

### Hardware Requirements

| Phase | Minimum | Recommended |
|-------|---------|-------------|
| Contrastive pre-train | 1x A100 40GB | 2x A100 80GB |
| Projection training | 1x GPU 16GB+ | 1x A100 40GB |
| FAISS index build | CPU is fine | GPU for large catalogs |
| Inference | 1x GPU 24GB+ (quantized LLM) | 1x A100 40GB |

### Budget option
- Quantized Qwen3-8B (4-bit) on 1x RTX 3090/4090
- Phi-3 (3.8B) instead of Qwen3-8B — half the VRAM, 2x faster

---

## 9. Implementation Roadmap

### Phase 0: Scaffolding (Week 1)
- [ ] Initialize repo structure
- [ ] Set up pyproject.toml, configs, Makefile
- [ ] Write README with architecture diagram and vision
- [ ] Pick a demo domain (music or e-commerce)
- [ ] Download benchmark dataset (Amazon Reviews or MovieLens)

### Phase 1: Contrastive Pre-training (Weeks 2-3)
- [ ] Write MetadataEncoder
- [ ] Write paired dataset loader (items, users, composites, cross-domain)
- [ ] Implement symmetric InfoNCE training loop
- [ ] Add interpolation smoothing loss for ordered metadata axes (energy, tempo, price, etc.)
- [ ] Write human language descriptions for chosen domain (items + users + combinations)
- [ ] Generate midpoint descriptions for smoothing
- [ ] Train and validate: verify that similar items/users cluster together AND gradients are smooth
- [ ] Visualize embedding space (t-SNE/UMAP) — check for continuity, not just clusters

### Phase 2: Projection + Index (Weeks 4-5)
- [ ] Embed full catalog into contrastive space, build FAISS index
- [ ] Generate steering pairs for directional loss
- [ ] Generate conversation delta pairs for turn-transition loss
- [ ] Train projection head with all 5 losses (target + directional + reconstruction + delta + context dropout)
- [ ] Validate: directional steering (does "more energetic" move the right direction?)
- [ ] Validate: given a conversation, does the projected hidden state land near the right items?

### Phase 3: Inference Engine (Week 5)
- [ ] Build prompt construction from user metadata + conversation context
- [ ] Implement LLM -> projection -> nearest neighbor pipeline
- [ ] Build session management
- [ ] End-to-end test: text in, item ID out

### Phase 4: Evaluation and Demo (Weeks 6-7)
- [ ] Run full eval suite (controllability + interpretability + efficiency)
- [ ] Build Gradio demo with interactive steering
- [ ] Record demo GIF/video for README
- [ ] Write blog post explaining the architecture

---

## 10. References

| Resource | URL |
|----------|-----|
| OpenCLIP | https://github.com/mlfoundations/open_clip |
| info-nce-pytorch | https://github.com/RElbers/info-nce-pytorch |
| FAISS | https://github.com/facebookresearch/faiss |
| REGEN dataset (Google) | https://www.kaggle.com/datasets/googleai/regen-reviews-enhanced-with-generative-narratives |
| SteerEval | https://arxiv.org/abs/2601.21105 |
| Spotify Text2Tracks | https://research.atspotify.com/2025/04/text2tracks-improving-prompt-based-music-recommendations-with-generative-retrieval |
| REGEN/LUMEN (Google) | https://arxiv.org/abs/2503.11924 |
| TIGER paper (NeurIPS 2023) | https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf |
| CLIP (OpenAI) | https://arxiv.org/abs/2103.00020 |
| LLM2CLIP | https://arxiv.org/pdf/2411.04997 |
