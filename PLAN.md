# TalkWalk — Steerable Recommendation Engine

> An open-source recommendation engine where users can change the algorithm with plain English, in real-time. Like Spotify's AI DJ, but generalized for any domain.

---

## Table of Contents

1. [Why TalkWalk](#1-why-talkwalk)
2. [Core Architecture](#2-core-architecture)
3. [Technical Deep Dive](#3-technical-deep-dive)
4. [Inference Flow](#4-inference-flow)
5. [Constrained Decoding](#5-constrained-decoding)
6. [Repo Structure](#6-repo-structure)
7. [Training Data](#7-training-data)
8. [Evaluation](#8-evaluation)
9. [Hardware and Dependencies](#9-hardware-and-dependencies)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [References](#11-references)

---

## 1. Why TalkWalk

Traditional recommendation systems treat language and items as separate worlds — the user speaks English, a pipeline translates that into filters and queries, and a separate model ranks results. The user has no real control over *how* the algorithm thinks.

TalkWalk unifies language and items inside a single model. When a user says "something chill," it doesn't trigger a filter — it directly shifts the probability distribution over items toward low-energy, relaxed content. The words *are* the algorithm.

| | Traditional Recommenders | TalkWalk |
|---|---|---|
| **Steering** | Filters, sliders, thumbs up/down | Plain English changes inference path in real-time |
| **Architecture** | Pipeline of separate components (retrieval, ranking, re-ranking) | Single unified model — the LLM *is* the recommender |
| **Item understanding** | Collaborative filtering on interaction history | Contrastively trained on metadata — understands what items *are* |
| **User understanding** | Demographic buckets, sparse features | User metadata contrastively trained with language descriptions — understands who users *are* |
| **Cold start** | Needs interaction history | Metadata alone is enough — new items and new users work immediately |
| **Latency** | Multiple model calls + database queries | One forward pass |
| **Scalability** | Item IDs grow linearly with catalog | Semantic IDs = fixed vocabulary for millions of items |
| **State** | Stateless or shallow session tracking | Implicit — the conversation context *is* the traversal |

### The core insight

By contrastively training metadata descriptions — for both items and users — into the LLM's embedding space, and representing items as hierarchical semantic tokens in the LLM's vocabulary, we create a system where human language, user identity, and item identity all exist in the same space. Steering the algorithm is as natural as talking to it. And because the LLM already understands language sequencing, no fine-tuning is needed beyond aligning the new token embeddings.

---

## 2. Core Architecture

### The Three-Way Bridge

```
Human Language  <-- contrastive -->  Metadata  <-- quantization -->  Semantic ID Tokens
      ^                                                                     ^
      '-------------------- both live inside the same LLM -----------------'
```

### Training Pipeline

```
Phase 1                       Phase 2                    Phase 3
Contrastive Pre-training  --> Semantic ID            --> Embedding Alignment
(item metadata <-> text)      Tokenization               (new tokens into LLM vocab)
(user metadata <-> text)      (RQ-VAE via GRID)          (~1K steps, embeddings only)
(composites   <-> text)
OpenCLIP + InfoNCE                                       Qwen3-8B + Unsloth
```

### Inference Pipeline

```
User input + user metadata
    |
    v
Conversation context (already contains all prior semantic IDs + steering)
    |
    v
LLM forward pass with constrained decoding (Outlines)
    |
    v
Output: <L1_X><L2_Y><L3_Z>  -->  Lookup table  -->  real object ID
    |
    v
Return object ID to client
```

There is no separate "traversal state" to manage. The LLM's context window already contains the full history of generated semantic IDs, user steering messages, and prior recommendations. In an autoregressive model, the probability distribution over the next token is already the product of every prior token. The conversation context *is* the traversal. The distribution space *is* the state.

### The Key Principle: Everything Reshapes the Distribution

At every point during inference, the probability distribution over the next semantic ID token is shaped by **everything in the context**:

- **User-provided English** — "something chill," "more like that," "wake me up" — each phrase shifts the distribution in real-time
- **System-provided user metadata embeddings** — the user's age, platform, time of day, listening history — injected into the prompt, these bias the distribution toward regions of item-space that match the user's profile
- **System-provided contextual embeddings** — session context, domain rules, business constraints — further shape what the model considers likely
- **Prior semantic IDs in the conversation** — every previously generated recommendation is in the context window, pulling the distribution toward or away from similar items

There is no distinction between "the algorithm" and "the inputs." The distribution space *is* the algorithm, and every signal — user or system, language or embedding — continuously reshapes it. This is what makes TalkWalk fundamentally different: the recommendation algorithm is not a fixed function that takes inputs; it is a living distribution that every participant (user, system, context) is constantly sculpting.

### What Makes This Novel

1. **The LLM *is* the recommender** — no tool chain, no external ranker, no SQL queries
2. **Every input changes the algorithm** — user English, system metadata, prior outputs — all continuously reshape the distribution space over items
3. **Traversal is implicit** — the distribution space is already a product of everything the user has navigated; no separate state management needed
4. **No fine-tuning needed** — the LLM already understands language; we only align new token embeddings into its existing space
5. **Domain-agnostic** — works for music, products, movies, articles — anything with metadata

---

## 3. Technical Deep Dive

### 3.1 Phase 1: Contrastive Pre-training (Metadata <-> Text)

**Goal:** Teach the model that structured metadata and human language descriptions mean the same thing — for items, users, and their combinations.

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

**Output:** Dense embeddings where item metadata, user metadata, combined profiles, and natural language all live in the same vector space.

---

### 3.2 Phase 2: Semantic ID Tokenization

**Goal:** Convert each object into a short, fixed-length sequence of hierarchical tokens the LLM can generate.

**Method:** Residual Quantization (RQ-VAE or RK-Means) via GRID toolkit.

**Process:**

1. **Embed items** — use the contrastive pre-trained metadata encoder to produce dense vectors
2. **Train RQ-VAE** — learn 3 codebooks (256 codes each):
   - Level 1 (coarse): broad category/vibe
   - Level 2 (medium): sub-cluster within L1
   - Level 3 (fine): specific item identity
3. **Assign IDs** — each item gets a tuple like `(42, 178, 95)`

```
song_042 --> (12, 187, 45)  --> <L1_12><L2_187><L3_45>
song_871 --> (88, 302, 91)  --> <L1_88><L2_302><L3_91>
song_215 --> (12, 44, 203)  --> <L1_12><L2_44><L3_203>
                ^
      Similar items share L1 prefix
```

**Key parameters:**
- `num_hierarchies = 3` (3 levels of quantization)
- `codebook_width = 256` (256 codes per level)
- Total new tokens: 3 x 256 + 3 special = **771 tokens** added to LLM vocab
- Can address up to 256^3 = **16.7 million unique items**

**Libraries:** GRID (snap-research/GRID) or standalone RQ-VAE from semantic-ids-llm

**Output:** Lookup table mapping semantic ID tuples <-> real object IDs.

---

### 3.3 Phase 3: Embedding Alignment

**Goal:** Integrate semantic ID tokens into an LLM that already understands language sequencing.

The LLM is already trained on language. It already understands "chill," "intense," "something like that but different." We don't need to teach it language or sequencing — we only need to teach it what the new tokens mean. The contrastive pre-training (Phase 1) already aligned metadata and language in the same embedding space. Phase 3 is just connecting those embeddings to the LLM's vocabulary.

**Base model:** Qwen3-8B
- Best embedding quality at 7-8B size
- Proven with semantic IDs (Eugene Yan's semantic-ids-llm)
- Apache 2.0 / permissive license
- 151k vocab — room to add tokens without disruption
- Qwen3-0.6B available for lightweight item embedding

**Vocabulary extension:** Add ~771 new tokens:
- 3 special tokens: `<|rec|>`, `<|sid_start|>`, `<|sid_end|>`
- 768 semantic ID tokens: `<|L1_0|>` ... `<|L1_255|>`, `<|L2_0|>` ... `<|L2_255|>`, `<|L3_0|>` ... `<|L3_255|>`

**Why full fine-tuning is not needed:**

The LLM already knows how to understand natural language, sequence tokens autoregressively, and maintain context across a conversation. All we need is for the new semantic ID token embeddings to land in the right place in the model's embedding space — close to the language concepts they represent. This is an embedding alignment problem, not a behavioral fine-tuning problem.

**Embedding-only training:**
- Freeze ALL model parameters except input/output embedding layers
- Initialize new token embeddings from the contrastive pre-trained metadata encoder (not random — the embeddings already encode the right semantics)
- Train for ~1,000 steps to align new embeddings with the LLM's internal representation space
- High LR (1e-3), batch size 32, cosine annealing, 100 step warmup

The model already speaks English. Now it also speaks semantic IDs — because those IDs were contrastively trained to mean the same thing as the English descriptions.

**Optional fallback (only if embedding-only underperforms):**
- Light LoRA fine-tune on recommendation conversations
- Conservative LR (2e-5), 1-2 epochs max

---

## 4. Inference Flow

### Example Catalog

```
song_042: {genre: "indie-rock", mood: "melancholic", energy: 0.3}  --> <L1_12><L2_187><L3_45>
song_871: {genre: "electronic", mood: "euphoric", energy: 0.9}     --> <L1_88><L2_302><L3_91>
song_215: {genre: "jazz", mood: "relaxed", energy: 0.2}            --> <L1_12><L2_44><L3_203>
```

### Multi-Turn Conversation

```
TURN 1:
  System: "User profile: 25yo, prefers indie."
  User:   "Recommend something"
  Model:  <|sid_start|><|L1_12|><|L2_187|><|L3_45|><|sid_end|>  --> song_042 (indie, melancholic)

TURN 2:
  User:   "More like that"
  Model:  <|sid_start|><|L1_12|><|L2_44|><|L3_203|><|sid_end|>  --> song_215 (jazz, relaxed)

  The context already contains <L1_12><L2_187><L3_45> from turn 1.
  The distribution is already biased toward the L1_12 region (chill, low-energy).
  The prior output IS the state.

TURN 3:
  User:   "ok now something to wake me up, really intense"
  Model:  <|sid_start|><|L1_88|><|L2_302|><|L3_91|><|sid_end|>  --> song_871 (electronic, euphoric)

  "wake me up" activates HIGH ENERGY regions of embedding space.
  The model sees two prior chill recommendations in context AND the steering text.
  The distribution shifts from L1_12 (chill) to L1_88 (high-energy) in one pass.
  The entire traversal (chill -> chill -> PIVOT) is visible in the context window.

LOOKUP:
  (88, 302, 91) --> song_871

RETURN TO CLIENT:
  { id: "song_871" }
```

In an autoregressive model, the probability distribution over the next token is already the product of every prior token. The traversal history — prior semantic IDs and user steering — shapes the distribution automatically. There is nothing to "update." The context window is the state. The distribution space is the traversal.

---

## 5. Constrained Decoding

**Problem:** The LLM might generate semantic ID tokens in the wrong order or produce invalid combinations.

**Solution:** Level-separated token namespaces + grammar-constrained generation via Outlines.

Each quantization level gets its own token prefix:
```
Level 1 (coarse):  <|L1_0|> through <|L1_255|>
Level 2 (medium):  <|L2_0|> through <|L2_255|>
Level 3 (fine):    <|L3_0|> through <|L3_255|>
```

At inference, Outlines enforces the grammar:
```
Position 1: ONLY allow <|L1_*|> tokens   (256 options)
Position 2: ONLY allow <|L2_*|> tokens   (256 options)
Position 3: ONLY allow <|L3_*|> tokens   (256 options)
```

```python
import outlines

L1 = "|".join(f"<\\|L1_{i}\\|>" for i in range(256))
L2 = "|".join(f"<\\|L2_{i}\\|>" for i in range(256))
L3 = "|".join(f"<\\|L3_{i}\\|>" for i in range(256))

sid_regex = f"<\\|sid_start\\|>({L1})({L2})({L3})<\\|sid_end\\|>"
generator = outlines.generate.regex(model, sid_regex)
```

The model cannot produce tokens out of order. Invalid sequences are impossible.

For hallucinated but structurally valid IDs (a tuple that maps to no real object), constrain decoding further to only allow tokens forming valid prefixes in the codebook.

---

## 6. Repo Structure

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

    # Phase 2: Semantic ID tokenization
    tokenizer/
      __init__.py
      rqvae.py                  # RQ-VAE implementation (or GRID wrapper)
      codebook.py               # Codebook management, ID assignment
      lookup.py                 # Semantic ID <-> real ID lookup table

    # Phase 3: Embedding alignment
    model/
      __init__.py
      vocab_extension.py        # Add semantic ID tokens to LLM vocab
      train_embeddings.py       # Embedding-only alignment training

    # Inference
    engine/
      __init__.py
      recommender.py            # Main recommendation engine class
      prompt_builder.py         # Build prompts from user metadata + context
      constrained_decode.py     # Outlines grammar for semantic IDs
      session.py                # Conversation context management

    # Evaluation
    eval/
      __init__.py
      metrics.py                # Hit@K, NDCG@K, MRR, coverage, diversity
      steerability.py           # Constraint-following, preference sensitivity
      hierarchical.py           # L1/L2/L3 accuracy
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
    tokenizer.yaml
    model.yaml
    eval.yaml

  # Demo
  demo/
    app.py                      # Gradio/Streamlit interactive demo
    example_music.py            # Music recommendation example
    example_ecommerce.py        # E-commerce example

  tests/
    test_contrastive.py
    test_tokenizer.py
    test_lookup.py
    test_constrained_decode.py
    test_engine.py

  pyproject.toml
  Makefile                      # make train-contrastive, make train-tokenizer, etc.
```

---

## 7. Training Data

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

### Phase 2: Embedding Alignment (~1K steps)

Short examples pairing semantic IDs with their natural language descriptions, so the new token embeddings land in the right place in the LLM's space:

```
"indie-rock, melancholic, low energy" --> <|sid_start|><|L1_12|><|L2_187|><|L3_45|><|sid_end|>
"electronic, euphoric, high energy"   --> <|sid_start|><|L1_88|><|L2_302|><|L3_91|><|sid_end|>
"jazz, relaxed, very low energy"      --> <|sid_start|><|L1_12|><|L2_44|><|L3_203|><|sid_end|>
```

Source: Catalog items with their contrastive descriptions mapped to assigned semantic IDs.

### Benchmark Datasets

| Dataset | Items | Domain | Why |
|---------|-------|--------|-----|
| Amazon Reviews 2023 | 48M reviews | E-commerce | Standard benchmark, used by GRID and semantic-ids-llm |
| MovieLens 25M | 62K movies | Movies | Rich metadata for steerability testing |
| Steam | 7.8M reviews | Games | Good metadata, proven domain |

---

## 8. Evaluation

### Ranking Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Hit@10 | Ground-truth in top 10? | > 0.20 |
| NDCG@10 | Ranking quality weighted by position | > 0.15 |
| MRR | Avg 1/rank of first relevant item | > 0.10 |

### Hierarchical Accuracy

| Metric | Target |
|--------|--------|
| L1 Accuracy (coarse category) | > 80% |
| L2 Accuracy (sub-cluster) | > 50% |
| L3 Accuracy (exact item) | > 20% |
| Valid Format % | 100% (guaranteed by constrained decoding) |

### Steerability (the differentiator)

| Metric | Protocol |
|--------|----------|
| Constraint-following | "under $50" — does output satisfy? Target > 90% |
| Preference sensitivity | Same history + different steering text — do outputs diverge? |
| Attribute control | Ask for genre X — what fraction of outputs are genre X? |
| NL instruction quality | LLM-as-judge scores coherence 1-5 |

### Beyond-Accuracy

| Metric | What |
|--------|------|
| Catalog Coverage | % of catalog ever recommended (avoid popularity bias) |
| Diversity (ILS) | How different are items within one recommendation list |
| Novelty | Average inverse popularity of recommended items |
| Latency | Target < 500ms per recommendation |

---

## 9. Hardware and Dependencies

### Core Dependencies

| Library | Purpose |
|---------|---------|
| PyTorch 2.x | Framework |
| Qwen3-8B | Base LLM |
| Qwen3-0.6B | Lightweight item/user embedding |
| Unsloth | Efficient embedding alignment |
| GRID (snap-research) | Semantic ID tokenization (RQ-VAE / RK-Means) |
| OpenCLIP | Contrastive training framework |
| info-nce-pytorch | InfoNCE loss |
| Outlines (dottxt-ai) | Constrained decoding |
| vLLM | Production inference serving |
| SentenceTransformers | Text encoding |
| Gradio | Demo UI |

### Hardware Requirements

| Phase | Minimum | Recommended |
|-------|---------|-------------|
| Contrastive pre-train | 1x A100 40GB | 2x A100 80GB |
| RQ-VAE tokenization | CPU is fine | 1x GPU (any) |
| Embedding alignment | 1x GPU 16GB+ | 1x A100 40GB |
| Inference | 1x GPU 24GB+ (quantized) | 1x A100 40GB |

### Budget option
- Embedding alignment on 1x RTX 3090/4090 (24GB)
- Phi-3 (3.8B) instead of Qwen3-8B — half the VRAM, 2x faster, slight quality loss

---

## 10. Implementation Roadmap

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
- [ ] Write human language descriptions for chosen domain (items + users + combinations)
- [ ] Train and validate: verify that similar items/users cluster together
- [ ] Visualize embedding space (t-SNE/UMAP)

### Phase 2: Semantic ID Tokenization (Week 4)
- [ ] Integrate GRID or implement standalone RQ-VAE
- [ ] Train codebooks on contrastive embeddings
- [ ] Assign semantic IDs to full catalog
- [ ] Build and validate lookup table (SID <-> real ID)
- [ ] Verify: similar items share L1 prefixes

### Phase 3: Embedding Alignment (Week 5)
- [ ] Extend Qwen3-8B vocabulary with semantic ID tokens
- [ ] Initialize new embeddings from contrastive pre-trained encoder
- [ ] Embedding-only training (~1K steps, freeze all other params)
- [ ] Validate: L1/L2/L3 accuracy, Hit@10, NDCG@10
- [ ] If underperforms: light LoRA as fallback

### Phase 4: Inference Engine (Week 6)
- [ ] Implement constrained decoding with Outlines
- [ ] Build prompt construction from user metadata + conversation context
- [ ] Build session management
- [ ] End-to-end test: text in, object ID out

### Phase 5: Evaluation and Demo (Weeks 7-8)
- [ ] Run full eval suite (ranking + steerability + diversity)
- [ ] Build Gradio demo with interactive steering
- [ ] Record demo GIF/video for README
- [ ] Write blog post explaining the architecture
- [ ] Benchmark against baselines

---

## 11. References

| Resource | URL |
|----------|-----|
| GRID (Snap Research) | https://github.com/snap-research/GRID |
| semantic-ids-llm (Eugene Yan) | https://github.com/eugeneyan/semantic-ids-llm |
| OpenCLIP | https://github.com/mlfoundations/open_clip |
| info-nce-pytorch | https://github.com/RElbers/info-nce-pytorch |
| Outlines | https://github.com/dottxt-ai/outlines |
| REGEN dataset (Google) | https://www.kaggle.com/datasets/googleai/regen-reviews-enhanced-with-generative-narratives |
| SteerEval | https://arxiv.org/abs/2601.21105 |
| CALRec paper | https://arxiv.org/abs/2405.02429 |
| TIGER paper (NeurIPS 2023) | https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf |
| Spotify Semantic IDs | https://research.atspotify.com/2025/9/semantic-ids-for-generative-search-and-recommendation |
| Spotify Text2Tracks | https://research.atspotify.com/2025/04/text2tracks-improving-prompt-based-music-recommendations-with-generative-retrieval |
| REGEN/LUMEN (Google) | https://arxiv.org/abs/2503.11924 |
