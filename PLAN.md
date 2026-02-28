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

TalkWalk unifies language and items inside a single embedding space. When a user says "something chill," it doesn't trigger a filter — it encodes directly into the same space where items live, landing near low-energy, relaxed content. The words *are* the coordinates.

| | Traditional Recommenders | TalkWalk |
|---|---|---|
| **Steering** | Filters, sliders, thumbs up/down | Plain English changes the query point in embedding space in real-time |
| **Architecture** | Pipeline of separate components (retrieval, ranking, re-ranking) | One shared contrastive space — all signals encode into it, nearest neighbor resolves to items |
| **Item understanding** | Collaborative filtering on interaction history | Contrastively trained on metadata — understands what items *are* |
| **User understanding** | Demographic buckets, sparse features | User metadata contrastively trained with language descriptions — understands who users *are* |
| **Cold start** | Needs interaction history | Metadata alone is enough — new items and new users work immediately |
| **Latency** | Multiple model calls + database queries | Encode + nearest-neighbor lookup |
| **Scalability** | Item IDs grow linearly with catalog | Contrastive space is fixed-dimensional regardless of catalog size |

### The core insight

We contrastively train item metadata, user metadata, combined profiles, and natural language descriptions all into the same embedding space. At inference, every signal — user text, system-provided user profile, system-provided context — encodes directly into that space. The query point for nearest-neighbor search is the combination of all these embeddings. No projection head. No LLM hidden state extraction. Everything is already in the same space because that's what Phase 1 trained for.

An LLM can optionally sit on top as a reasoning layer — processing multi-turn conversation and generating a text description of what to recommend. That description goes through the same text encoder into the same contrastive space. The LLM is a convenience for complex reasoning, not a structural requirement.

---

## 2. Core Architecture

### One Space

```
Everything encodes into the same contrastive embedding space:

  User text:        "something chill"            --> text_encoder    --> [0.08, -0.52, ...]
  User metadata:    {age: 25, freq: power}       --> metadata_encoder --> [0.15, -0.41, ...]
  System context:   {time: late_night, ...}      --> metadata_encoder --> [0.03, -0.60, ...]
  Item metadata:    {genre: jazz, energy: 0.2}   --> metadata_encoder --> [0.08, -0.52, ...]
  Item description: "relaxed jazz, low energy"   --> text_encoder    --> [0.09, -0.50, ...]

All vectors live in the same space. Similarity = relevance.
```

### Training

```
Phase 1: Contrastive Pre-training
  - Item metadata <-> item descriptions
  - User metadata <-> user descriptions
  - Composite metadata <-> composite descriptions
  - User x Item affinity pairs
  - Interpolation smoothing along ordered axes
  - All into one shared embedding space
  OpenCLIP + InfoNCE
```

That's it. One training phase. Everything else is inference.

### Inference

```
At each turn:
  1. Encode user's text with text_encoder          --> user_text_emb
  2. Encode user metadata with metadata_encoder     --> user_meta_emb
  3. Encode system context with text/meta encoder   --> system_emb
  4. Combine: query = f(user_text_emb, user_meta_emb, system_emb, prior_item_embs)
  5. Nearest neighbor search against item index     --> item ID
  6. Return item ID to client
```

No projection head. No LLM hidden state. No learned bridge between spaces. Everything is already in the same space.

### The Key Principle: Everything Reshapes the Query

At every point during inference, both the user and the system can inject signals that move the query point in contrastive space:

- **User-provided English** — "something chill," "more like that," "wake me up" — each phrase encodes into the space via the text encoder, shifting the query point
- **System-provided user metadata** — the user's age, platform, time of day, listening history — encodes via the metadata encoder, biasing the query toward regions that match the user's profile
- **System-provided contextual signals** — session context, domain rules, business constraints — encode into the same space, further shaping the query
- **Prior recommendations** — previously returned items are already embedded; they can be used to attract (more like this) or repel (something different) the query point

There is no distinction between user input and system input at the embedding level. Both are just vectors in the same space. Both reshape the query. The recommendation algorithm is not a function — it is a point in a space that every participant is constantly moving.

### Context Length

Conversation history is maintained as a list of prior embeddings (user text embeddings + returned item embeddings). At each turn, the combination function sees:

```
[ current user text emb ] + [ current system embs ] + [ last N item embs returned ]
```

Bounded by a configurable window size. No LLM context window limitations. No truncation artifacts. Just a sliding window of vectors.

### Optional: LLM as Reasoning Layer

For complex multi-turn steering where the text encoder alone may struggle (e.g., "something like that third song you recommended but more upbeat and less electronic"), an LLM can process the full conversation and generate a text description of the intent:

```
User conversation --> LLM --> "upbeat acoustic track similar to indie-folk"
                              --> text_encoder --> contrastive space --> nearest neighbor
```

The LLM's output is plain text. It goes through the same text encoder as everything else. The LLM is a preprocessor, not a structural component. It can be swapped, removed, or upgraded without changing the recommendation architecture.

---

## 3. Technical Deep Dive

### 3.1 Phase 1: Contrastive Pre-training

**Goal:** Create a single embedding space where item metadata, user metadata, combined profiles, and natural language descriptions all coexist — with smooth, continuous gradients.

**Architecture:** CLIP-style dual encoder with InfoNCE loss + interpolation smoothing.

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

# Loss: Symmetric InfoNCE + interpolation smoothing
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

Every user attribute gets the same treatment:

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

Behavior emerges from intersections. We train on composite descriptions that capture the *meaning* of metadata combinations:

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

We also train on user-item affinity descriptions:

```yaml
user={age: 18-24, freq: power, time: late_night} + item={genre: ambient, energy: 0.2}:
  "young night owl deep-listening to atmospheric ambient — high affinity"

user={age: 35-49, freq: casual} + item={genre: experimental-noise, energy: 0.9}:
  "mature casual listener vs harsh experimental — low affinity, avoid unless requested"
```

When the system encodes a user profile at inference time, the embedding already knows which regions of item-space are natural fits — and which require an explicit steering request to reach.

#### Interpolation Smoothing

InfoNCE teaches similarity but not continuity. To create smooth preference gradients, add synthetic interpolation training along every metadata axis with a natural ordering:

```
energy = 0.2  --> "low energy, quiet, subdued"
energy = 0.5  --> "moderately energetic, balanced intensity"
energy = 0.8  --> "high energy, intense, driving"

Loss: E(midpoint) should fall between E(low) and E(high)
L_smooth = ||E(mid) - 0.5 * (E(low) + E(high))||^2
```

Apply along energy, tempo, valence, price, age, etc. This creates continuous axes inside the space so that steering produces smooth glides, not abrupt jumps.

#### Training Strategy

All metadata types and smoothing are trained together:

```
Item metadata      <-- contrastive -->  Item descriptions
User metadata      <-- contrastive -->  User descriptions
Combined metadata  <-- contrastive -->  Composite descriptions
User x Item pairs  <-- contrastive -->  Affinity descriptions
Ordered axes       <-- smoothing   -->  Midpoint descriptions
```

**Libraries:** OpenCLIP (with MetadataEncoder replacing vision tower) + info-nce-pytorch

**Output:** A single embedding space where item metadata, user metadata, combined profiles, and natural language all coexist with smooth gradients. Every item in the catalog gets a fixed embedding vector. This is the only training required.

---

### 3.2 Query Combination

At inference, multiple signals encode into the contrastive space simultaneously. The query point for nearest-neighbor search is their combination.

```python
def build_query(user_text, user_metadata, system_context, prior_items, weights):
    """Combine all signals into a single query point in contrastive space."""

    # Everything encodes into the same space
    text_emb = text_encoder(user_text)                    # what the user said
    user_emb = metadata_encoder(user_metadata)            # who the user is
    ctx_emb  = text_encoder(system_context)               # system-provided context
    prior_embs = [item_index[id] for id in prior_items]   # what was already recommended

    # Weighted combination
    query = (
        weights.text * text_emb
      + weights.user * user_emb
      + weights.context * ctx_emb
    )

    # Optionally repel from prior items (diversity)
    for emb in prior_embs:
        query = query - weights.repulsion * emb

    return normalize(query)
```

The weights control how much each signal influences the recommendation:
- High `weights.text` = user's words dominate (strong steering)
- High `weights.user` = profile dominates (personalization)
- High `weights.repulsion` = avoid repeating similar items (exploration)

These can be fixed, learned, or dynamically adjusted based on steering intensity.

---

### 3.3 Item Index

Every item in the catalog is embedded once and stored in a vector index:

```python
import faiss

# Embed all items
item_embeddings = metadata_encoder(all_item_metadata)  # shape: [num_items, contrastive_dim]

# Build index
index = faiss.IndexFlatIP(contrastive_dim)  # inner product = cosine sim on normalized vecs
index.add(item_embeddings)

# At inference: query the index
query = build_query(user_text, user_meta, system_ctx, prior_items, weights)
distances, item_indices = index.search(query, k=10)  # top-10 nearest items
```

For large catalogs (millions of items), use approximate nearest neighbor (FAISS IVF, HNSW, or ScaNN) for sub-millisecond lookup.

New items are added by embedding their metadata and inserting into the index. No retraining needed.

---

## 4. Inference Flow

### Example Catalog

```
song_042: {genre: "indie-rock", mood: "melancholic", energy: 0.3}  --> emb_042
song_871: {genre: "electronic", mood: "euphoric", energy: 0.9}     --> emb_871
song_215: {genre: "jazz", mood: "relaxed", energy: 0.2}            --> emb_215
```

### Multi-Turn Conversation

```
TURN 1:
  User text:     "recommend something"
  User metadata: {age: 25, freq: power}
  System:        {time: evening}

  text_emb  = encode("recommend something")           --> generic, near center
  user_emb  = encode({age: 25, freq: power})           --> leans toward discovery
  sys_emb   = encode({time: evening})                  --> leans toward winding-down

  query = combine(text_emb, user_emb, sys_emb)        --> lands near chill/discovery region
  nearest neighbor --> song_042 (indie, melancholic)

TURN 2:
  User text:     "more like that"
  Prior items:   [song_042]

  text_emb  = encode("more like that")                --> generic "similar" signal
  user_emb  = same
  sys_emb   = same
  prior     = [emb_042]                               --> attracts query toward emb_042's region

  query = combine(text_emb, user_emb, sys_emb) + attract(emb_042)
  nearest neighbor (excluding song_042) --> song_215 (jazz, relaxed — same L1 region)

TURN 3:
  User text:     "ok now something to wake me up, really intense"
  Prior items:   [song_042, song_215]

  text_emb  = encode("something to wake me up, really intense")  --> HIGH ENERGY region
  user_emb  = same
  sys_emb   = same
  prior     = [emb_042, emb_215]                                 --> repel from chill region

  query = combine(text_emb, user_emb, sys_emb) + repel(emb_042, emb_215)
  nearest neighbor --> song_871 (electronic, euphoric, energy: 0.9)

RETURN TO CLIENT: { id: "song_871" }
```

No hierarchy. No commitment. No path dependence. "Wake me up" just moves the query point to a different region. It's one vector operation in a continuous space. A hard pivot is exactly as easy as a subtle refinement — both are just directions.

### System-Side Steering

The system can inject signals at any turn, just like the user:

```
TURN 4:
  User text:     "keep going"
  System injects: "business rule: promote new releases this week"

  sys_emb now includes encode("promote new releases this week")
  --> query shifts toward recently-added items in the catalog
  --> nearest neighbor prefers new items in the current region
```

User and system inputs are symmetric. Both are just embeddings in the same space. Both reshape the query.

---

## 5. Repo Structure

```
Talkwalk/
  README.md
  PLAN.md
  LICENSE                         # Apache 2.0

  talkwalk/
    __init__.py
    config.py                     # Hydra/YAML configuration

    # Contrastive pre-training
    contrastive/
      __init__.py
      metadata_encoder.py         # MetadataEncoder nn.Module
      text_encoder.py             # Text encoder wrapper
      infonce_loss.py             # Symmetric InfoNCE implementation
      smoothing_loss.py           # Interpolation smoothing for ordered axes
      train.py                    # Training script
      dataset.py                  # Paired (metadata, description) dataset
      user_metadata.py            # User metadata encoding + composite profile builder

    # Item index
    index/
      __init__.py
      catalog.py                  # Embed catalog, build FAISS index
      lookup.py                   # Nearest neighbor search, real ID resolution

    # Inference
    engine/
      __init__.py
      recommender.py              # Main recommendation engine class
      query_builder.py            # Combine user/system/prior embeddings into query
      session.py                  # Conversation state (prior items, user metadata)
      llm_reasoner.py             # Optional LLM for complex multi-turn reasoning

    # Evaluation
    eval/
      __init__.py
      controllability.py          # Steering entropy, turns-to-satisfaction, diversity gradients
      interpretability.py         # Embedding coherence, steering traceability
      efficiency.py               # First-turn relevance, exploration velocity, cold-start
      sanity.py                   # Basic accuracy, coverage, latency
      benchmark.py                # Run full eval suite

  # Data
  data/
    descriptions/
      items/
        example_music.yaml
        example_ecommerce.yaml
      users/
        demographics.yaml
        behavior.yaml
        context.yaml
      composites/
        user_combinations.yaml
        user_item_affinity.yaml
        interpolation_midpoints.yaml
    scripts/
      prepare_catalog.py
      prepare_user_profiles.py
      generate_descriptions.py
      generate_composites.py
      generate_midpoints.py       # Generate interpolation data for smoothing

  configs/
    contrastive.yaml
    index.yaml
    engine.yaml
    eval.yaml

  demo/
    app.py                        # Gradio interactive demo
    example_music.py
    example_ecommerce.py

  tests/
    test_contrastive.py
    test_index.py
    test_query_builder.py
    test_engine.py

  pyproject.toml
  Makefile
```

---

## 6. Training Data

### Contrastive Pre-training (100K+ pairs across all metadata types)

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

**Interpolation midpoints:**
```json
{
  "low": {"energy": 0.2, "description": "low energy, quiet, subdued"},
  "mid": {"energy": 0.5, "description": "moderately energetic, balanced intensity"},
  "high": {"energy": 0.8, "description": "high energy, intense, driving"}
}
```

Source: Catalog metadata + user profiles + human-written key descriptions + LLM-generated composites + paraphrases for augmentation.

### Benchmark Datasets

| Dataset | Items | Domain | Why |
|---------|-------|--------|-----|
| Amazon Reviews 2023 | 48M reviews | E-commerce | Standard benchmark |
| MovieLens 25M | 62K movies | Movies | Rich metadata for steerability testing |
| Steam | 7.8M reviews | Games | Good metadata, proven domain |

---

## 7. Evaluation

TalkWalk's advantage is not ranking accuracy — it's controllability, interpretability, and interaction efficiency.

### Primary Metrics: Controllability

| Metric | What It Measures | Protocol |
|--------|-----------------|----------|
| **Steering entropy reduction** | How much does a steering utterance narrow the output distribution? | Measure entropy of nearest-neighbor distances before and after a steering input. Higher reduction = more responsive. |
| **Turns-to-satisfaction** | How many turns to reach a satisfying recommendation? | Give users a target preference, count turns until output matches. Lower is better. |
| **Controllable diversity gradient** | Can the user smoothly control diversity? | User says "more variety" / "stay close" — measure spread of outputs. Should scale proportionally. |
| **Steering precision** | Does "more X" increase X without changing unrelated attributes? | Measure change in target vs. non-target attributes. |
| **Steering reversibility** | Can the user undo a steer? | Steer toward X, then "go back." Measure similarity to pre-steer query. |

### Primary Metrics: Interpretability

| Metric | What It Measures | Protocol |
|--------|-----------------|----------|
| **Embedding space coherence** | Do regions correspond to human-understandable categories? | Cluster, label via LLM, human raters verify. |
| **Steering traceability** | Can you explain *why* a recommendation changed? | Measure whether query movement aligns with steering direction. |

### Primary Metrics: Interaction Efficiency

| Metric | What It Measures | Protocol |
|--------|-----------------|----------|
| **First-turn relevance** | How good with zero steering? | User metadata + first prompt only. |
| **Exploration velocity** | How fast can a user traverse different catalog regions? | Count unique clusters reached in N turns. |
| **Cold-start quality** | How well for brand new users? | Only demographic/contextual metadata, no history. |

### Secondary: Sanity Checks

| Metric | Baseline |
|--------|----------|
| Catalog coverage (% ever recommended) | > 30% |
| Latency (encode + NN lookup) | < 100ms |

---

## 8. Hardware and Dependencies

### Core Dependencies

| Library | Purpose |
|---------|---------|
| PyTorch 2.x | Framework |
| OpenCLIP | Contrastive training framework |
| info-nce-pytorch | InfoNCE loss |
| FAISS | Nearest-neighbor search |
| SentenceTransformers | Text encoding |
| Gradio | Demo UI |
| Qwen3-0.6B (optional) | Lightweight text encoder |
| Qwen3-8B (optional) | LLM reasoning layer for complex multi-turn |
| vLLM (optional) | LLM serving if using reasoning layer |

### Hardware Requirements

| Phase | Minimum | Recommended |
|-------|---------|-------------|
| Contrastive pre-train | 1x GPU 16GB+ | 1x A100 40GB |
| FAISS index build | CPU | GPU for large catalogs |
| Inference (no LLM) | CPU | GPU for low latency |
| Inference (with LLM) | 1x GPU 24GB+ | 1x A100 40GB |

### Budget option
- Contrastive training on 1x RTX 3090/4090
- Inference without LLM runs on CPU — the encoders are small

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
- [ ] Add interpolation smoothing loss for ordered metadata axes
- [ ] Write human language descriptions for chosen domain (items + users + combinations)
- [ ] Generate midpoint descriptions for smoothing
- [ ] Train and validate: verify clustering AND gradient smoothness
- [ ] Visualize embedding space (t-SNE/UMAP)

### Phase 2: Index + Query Engine (Week 4)
- [ ] Embed full catalog into contrastive space
- [ ] Build FAISS index
- [ ] Implement query combination (user text + user meta + system context + prior items)
- [ ] Implement attract/repel for prior items
- [ ] End-to-end test: text in, item ID out

### Phase 3: Evaluation and Demo (Weeks 5-6)
- [ ] Run full eval suite (controllability + interpretability + efficiency)
- [ ] Build Gradio demo with interactive steering
- [ ] Record demo GIF/video for README
- [ ] Write blog post explaining the architecture

### Phase 4: Optional LLM Reasoning Layer (Week 7)
- [ ] Add LLM preprocessor for complex multi-turn conversations
- [ ] LLM generates text description -> text encoder -> same contrastive space
- [ ] Compare eval results with and without LLM layer
- [ ] Document when the LLM helps vs. when pure embedding is sufficient

---

## 10. References

| Resource | URL |
|----------|-----|
| OpenCLIP | https://github.com/mlfoundations/open_clip |
| info-nce-pytorch | https://github.com/RElbers/info-nce-pytorch |
| FAISS | https://github.com/facebookresearch/faiss |
| CLIP (OpenAI) | https://arxiv.org/abs/2103.00020 |
| LLM2CLIP | https://arxiv.org/pdf/2411.04997 |
| REGEN dataset (Google) | https://www.kaggle.com/datasets/googleai/regen-reviews-enhanced-with-generative-narratives |
| REGEN/LUMEN (Google) | https://arxiv.org/abs/2503.11924 |
| SteerEval | https://arxiv.org/abs/2601.21105 |
| Spotify Text2Tracks | https://research.atspotify.com/2025/04/text2tracks-improving-prompt-based-music-recommendations-with-generative-retrieval |
