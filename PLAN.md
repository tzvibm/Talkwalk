# TalkWalk — Steerable Recommendation Engine

> A novel open-source recommendation engine where users can change the algorithm with plain English, in real-time. Like Spotify's AI DJ, but generalized for any domain.

---

## Table of Contents

1. [Why Not RecAI](#1-why-not-recai)
2. [Core Architecture](#2-core-architecture)
3. [Technical Deep Dive](#3-technical-deep-dive)
4. [Inference Flow](#4-inference-flow)
5. [Constrained Decoding](#5-constrained-decoding)
6. [Repo Structure](#6-repo-structure)
7. [Training Data](#7-training-data)
8. [Evaluation](#8-evaluation)
9. [Hardware and Dependencies](#9-hardware-and-dependencies)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. Why Not RecAI

| Dimension | TalkWalk | RecAI |
|---|---|---|
| **Architecture** | Single unified model — the LLM *is* the recommender | 7 loosely-coupled components — LLM orchestrates external tools |
| **NL Steering** | Changes inference path directly (token probabilities shift) | LLM picks different tools / parameters |
| **Latency** | One forward pass | 2+ API calls + tool execution chain |
| **Item Understanding** | Deep — contrastively trained on metadata | Shallow — items are SQL rows the LLM never "sees" |
| **Cold Start** | Metadata alone is enough | Needs interaction history for collaborative filtering |
| **Scalability** | Semantic IDs = fixed vocab for millions of items | Candidate Bus initialized with all items (doesn't scale) |
| **Code Quality** | Fresh, modern stack | Rotting deps (LangChain 0.0.312, vLLM 0.2.7, alpha UniRec) |

**Verdict: Build from scratch. Steal ideas, not code.**

Ideas worth borrowing:
- RL reward function design (item-level + list-level compliance + KL penalty) from RecLM-gen
- Critique training data format from Google's REGEN dataset

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
Phase 1                    Phase 2                  Phase 3
Contrastive Pre-train  --> Semantic ID          --> LLM Fine-tuning
(metadata <-> text)        Tokenization             (SFT + optional RL)
OpenCLIP + InfoNCE         RQ-VAE via GRID          Qwen3-8B via Unsloth
```

### Inference Pipeline

```
User input + user metadata + traversal state
    |
    v
Build prompt (natural language + semantic ID history)
    |
    v
LLM forward pass with constrained decoding (Outlines)
    |
    v
Output: <L1_X><L2_Y><L3_Z>  -->  Lookup table  -->  real object ID
    |
    v
Update traversal state (append to session context)
    |
    v
Return object ID to client
```

### What Makes This Novel

1. **The LLM *is* the recommender** — no tool chain, no external ranker, no SQL queries
2. **Plain English changes the algorithm** — "something chill" literally shifts token probabilities toward low-energy items
3. **Stateful traversal** — accumulated navigation context feeds back into each recommendation
4. **Domain-agnostic** — works for music, products, movies, articles — anything with metadata

---

## 3. Technical Deep Dive

### 3.1 Phase 1: Contrastive Pre-training (Metadata <-> Text)

**Goal:** Teach the model that structured metadata and human language descriptions mean the same thing.

**Architecture:** CLIP-style dual encoder with InfoNCE loss.

```python
# Encoder A: Text (pre-trained, frozen initially)
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")  # or Qwen3-0.6B

# Encoder B: Metadata (learned from scratch)
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

**You provide:** Human language descriptions per metadata key/value (the "language bridge").

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

Descriptions for each object are assembled by combining per-key descriptions. Augmented with LLM-generated paraphrases for diversity.

**Libraries:** OpenCLIP (with MetadataEncoder replacing vision tower) + info-nce-pytorch

**Output:** Dense embeddings where language and metadata live in the same vector space.

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

### 3.3 Phase 3: LLM Fine-tuning

**Goal:** Teach the LLM to (a) understand semantic ID tokens, (b) generate valid recommendations, (c) follow natural language steering.

**Base model:** Qwen3-8B

Why Qwen3-8B:
- Best embedding quality at 7-8B size
- Proven with semantic IDs (Eugene Yan's semantic-ids-llm)
- Apache 2.0 / permissive license
- 151k vocab — room to add tokens without disruption
- Qwen3-0.6B available for lightweight item embedding

**Vocabulary extension:** Add ~771 new tokens:
- 3 special tokens: `<|rec|>`, `<|sid_start|>`, `<|sid_end|>`
- 768 semantic ID tokens: `<|L1_0|>` ... `<|L1_255|>`, `<|L2_0|>` ... `<|L2_255|>`, `<|L3_0|>` ... `<|L3_255|>`

**Two-stage fine-tuning:**

**Stage A — Embedding warm-up (1,000 steps):**
- Freeze ALL parameters except input/output embedding layers
- High LR (1e-3) to quickly learn new token embeddings
- New embeddings initialized to mean of existing embeddings
- Batch size 32, cosine annealing, 100 step warmup

**Stage B — Full fine-tune (3 epochs):**
- Unfreeze all parameters
- Conservative LR (2e-5), effective batch size 128
- LoRA or full fine-tune depending on hardware
- 3% warmup, weight decay 0.01, gradient clipping 1.0
- Mixed precision (bf16 on A100/H100, fp16 otherwise)

---

## 4. Inference Flow (Concrete Example)

### The Data

```
song_042: {genre: "indie-rock", mood: "melancholic", energy: 0.3}  --> <L1_12><L2_187><L3_45>
song_871: {genre: "electronic", mood: "euphoric", energy: 0.9}     --> <L1_88><L2_302><L3_91>
song_215: {genre: "jazz", mood: "relaxed", energy: 0.2}            --> <L1_12><L2_44><L3_203>
```

### Step-by-step

```
USER SESSION STATE:
  User metadata:    {age: 25, likes: [indie]}
  Traversal state:  [song_042, song_215]   (both chill, low energy)
  User says:        "ok now something to wake me up, really intense"
                           |
                           v
PROMPT CONSTRUCTED:
  "User profile: 25yo, prefers indie.
   History: <|sid_start|><|L1_12|><|L2_187|><|L3_45|><|sid_end|>,
            <|sid_start|><|L1_12|><|L2_44|><|L3_203|><|sid_end|>
   Pattern: low-energy, reflective tracks.
   User request: 'something to wake me up, really intense'
   Recommend next:"
                           |
                           v
LLM GENERATES (with constrained decoding):
  "wake me up" activates HIGH ENERGY regions of embedding space
  --> high probability on <L1_88> (high-energy cluster)

  Output: <|sid_start|><|L1_88|><|L2_302|><|L3_91|><|sid_end|>
                           |
                           v
LOOKUP:
  (88, 302, 91) --> song_871  (electronic, euphoric, energy: 0.9)
                           |
                           v
UPDATE TRAVERSAL:
  State: [song_042, song_215, song_871]
  Trajectory: chill -> chill -> HIGH ENERGY (pivot detected)
  This feeds into the NEXT recommendation
                           |
                           v
RETURN TO CLIENT:
  { id: "song_871", explanation: "High-energy electronic track to wake you up" }
```

---

## 5. Constrained Decoding

**Problem:** The LLM might generate semantic ID tokens in wrong order or produce invalid combinations.

**Solution:** Level-separated token namespaces + grammar-constrained generation via Outlines.

Each quantization level gets its own token prefix:
```
Level 1 (coarse):  <|L1_0|> through <|L1_255|>
Level 2 (medium):  <|L2_0|> through <|L2_255|>
Level 3 (fine):    <|L3_0|> through <|L3_255|>
```

At inference, Outlines enforces this grammar:
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

The model **cannot** produce tokens out of order. Invalid sequences are impossible.

For hallucinated but valid-format IDs (a tuple that maps to no real object), use constrained decoding that only allows tokens forming valid prefixes in the codebook.

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
      text_encoder.py           # Text encoder wrapper (SentenceTransformers/Qwen3-0.6B)
      infonce_loss.py           # Symmetric InfoNCE implementation
      train_contrastive.py      # Training script
      dataset.py                # Paired (metadata, description) dataset

    # Phase 2: Semantic ID tokenization
    tokenizer/
      __init__.py
      rqvae.py                  # RQ-VAE implementation (or GRID wrapper)
      codebook.py               # Codebook management, ID assignment
      lookup.py                 # Semantic ID <-> real ID lookup table

    # Phase 3: LLM fine-tuning
    model/
      __init__.py
      vocab_extension.py        # Add semantic ID tokens to LLM vocab
      train_embeddings.py       # Stage A: embedding warm-up
      train_full.py             # Stage B: full SFT fine-tune
      train_dpo.py              # Optional: DPO/RL alignment

    # Inference
    engine/
      __init__.py
      recommender.py            # Main recommendation engine class
      prompt_builder.py         # Build prompts from user/traversal state
      constrained_decode.py     # Outlines grammar for semantic IDs
      traversal.py              # Stateful traversal manager
      session.py                # User session management

    # Evaluation
    eval/
      __init__.py
      metrics.py                # Hit@K, NDCG@K, MRR, coverage, diversity
      steerability.py           # Constraint-following, preference sensitivity
      hierarchical.py           # L1/L2/L3 accuracy
      benchmark.py              # Run full eval suite

  # Data processing
  data/
    descriptions/               # Human language descriptions per metadata key
      example_music.yaml
      example_ecommerce.yaml
    scripts/
      prepare_catalog.py        # Convert raw catalog to training format
      generate_descriptions.py  # LLM-augment descriptions for diversity
      build_sft_data.py         # Generate SFT chat conversations
      build_dpo_data.py         # Generate preference pairs

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

### Phase 1: Contrastive Pre-training (100K+ pairs)

```json
{
  "metadata": {"genre": "indie-rock", "tempo": 128, "mood": "melancholic", "energy": 0.3},
  "description": "guitar-driven independent rock with raw emotional vocals, slow melancholic energy"
}
```

Source: Catalog metadata + human-written key descriptions + LLM paraphrases for augmentation.

### Phase 2: SFT Fine-tuning (50-100K conversations)

**Next-item prediction:**
```
User: History: [items with semantic IDs]. What next?
Assistant: I recommend <|sid_start|><|L1_42|><|L2_201|><|L3_67|><|sid_end|> because...
```

**Steered recommendation:**
```
User: History: [items]. Give me something with more energy.
Assistant: <|sid_start|><|L1_88|><|L2_302|><|L3_91|><|sid_end|> — high-energy electronic track...
```

**Explanation:**
```
User: Why did you recommend that?
Assistant: It shares L1 category 42 (premium audio) with your history...
```

Source: Historical interaction sequences + synthetic generation.

### Phase 3: DPO Alignment (20-50K preference pairs, optional)

```json
{
  "prompt": "History: [items]. Recommend a gift under $100.",
  "chosen": "<|sid_start|>...<|sid_end|> — matches budget, great gift.",
  "rejected": "<|sid_start|>...<|sid_end|> — repeats history, ignores constraint."
}
```

### Benchmark Datasets to Start With

| Dataset | Items | Domain | Why |
|---------|-------|--------|-----|
| Amazon Reviews 2023 | 48M reviews | E-commerce | Standard benchmark, used by GRID and semantic-ids-llm |
| MovieLens 25M | 62K movies | Movies | Rich metadata for steerability testing |
| Steam | 7.8M reviews | Games | Good metadata, proven domain |

---

## 8. Evaluation

### Ranking Metrics (offline)

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Hit@10 | Ground-truth in top 10? | > 0.20 |
| NDCG@10 | Ranking quality weighted by position | > 0.15 |
| MRR | Avg 1/rank of first relevant item | > 0.10 |

### Hierarchical Accuracy (semantic-ID specific)

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
| Qwen3-0.6B | Lightweight item embedding |
| Unsloth | Efficient fine-tuning |
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
| RQ-VAE tokenization | 1x GPU (any) | CPU is fine |
| LLM fine-tune (LoRA) | 1x A100 40GB (16GB with QLoRA) | 2x A100 80GB |
| LLM fine-tune (full) | 4x A100 80GB | 8x A100 80GB |
| Inference | 1x A100 40GB | 1x A100 80GB (or quantized on smaller) |

### Budget option
- QLoRA fine-tune on 1x RTX 4090 (24GB) or even 1x RTX 3090
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
- [ ] Write paired dataset loader
- [ ] Implement symmetric InfoNCE training loop
- [ ] Write human language descriptions for chosen domain
- [ ] Train and validate: verify that similar items cluster together in embedding space
- [ ] Visualize embedding space (t-SNE/UMAP)

### Phase 2: Semantic ID Tokenization (Week 4)
- [ ] Integrate GRID or implement standalone RQ-VAE
- [ ] Train codebooks on contrastive embeddings
- [ ] Assign semantic IDs to full catalog
- [ ] Build and validate lookup table (SID -> real ID, real ID -> SID)
- [ ] Verify: similar items share L1 prefixes

### Phase 3: LLM Fine-tuning (Weeks 5-7)
- [ ] Generate SFT training data (next-item + steered + explanation)
- [ ] Extend Qwen3-8B vocabulary with semantic ID tokens
- [ ] Stage A: embedding warm-up (1K steps)
- [ ] Stage B: full SFT fine-tune (3 epochs)
- [ ] Validate: L1/L2/L3 accuracy, Hit@10, NDCG@10
- [ ] Optional: DPO alignment on preference pairs

### Phase 4: Inference Engine (Week 8)
- [ ] Implement constrained decoding with Outlines
- [ ] Build prompt construction from user/traversal state
- [ ] Implement stateful traversal manager
- [ ] Build session management
- [ ] End-to-end test: text in -> object ID out

### Phase 5: Evaluation and Demo (Weeks 9-10)
- [ ] Run full eval suite (ranking + steerability + diversity)
- [ ] Build Gradio demo with interactive steering
- [ ] Record demo GIF/video for README
- [ ] Write blog post explaining the architecture
- [ ] Benchmark against baselines (SASRec, RecAI if possible)

---

## References

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
| Microsoft RecAI | https://github.com/microsoft/RecAI |
| REGEN/LUMEN (Google) | https://arxiv.org/abs/2503.11924 |
