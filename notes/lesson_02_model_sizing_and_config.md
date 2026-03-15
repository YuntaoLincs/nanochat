# Lesson 02 — Model Sizing & GPTConfig

**Date:** 2026-03-15
**Branch:** `learn/code`
**Status:** ✅ Complete

---

## Key Formula: Parameter Count

```
N ≈ 12 × n_layer × n_embd²       (attention + MLP only)
  + vocab_size × n_embd           (embedding table)
```

The `12` comes from: each Block has 4 weight matrices in attention (Q, K, V, out) and 2 in MLP (up, down), each of size `n_embd × n_embd` → `6 × 2 = 12`.

### Reference Table

| Model      | n_layer | n_embd | Formula result | Actual  | Gap (embeddings)       |
|------------|---------|--------|----------------|---------|------------------------|
| GPT-2 small| 12      | 768    | ~85M           | ~117M   | 50,257 × 768 ≈ 38M     |
| GPT-2 XL   | 48      | 1600   | ~1,474M        | ~1,542M | 50,257 × 1600 ≈ 80M    |
| **1B target**  | **20**  | **2048** | **~1,006M** | ~1B  | 32,768 × 2048 ≈ 67M    |

**Lesson:** For small models, the embedding table is ~30% of total params. For large models it becomes negligible. The formula gets more accurate as models scale.

---

## Solving for n_layer (1B target)

```
1,000,000,000 = 12 × n_layer × 2048²
2048²         = 4,194,304
12 × 2048²    = 50,331,648
n_layer       = 1,000,000,000 ÷ 50,331,648 ≈ 19.9 → 20
```

This matches nanochat's default `--depth=20` in `scripts/base_train.py`.

nanochat derives dimensions dynamically:
```python
model_dim = depth * aspect_ratio   # 20 * 64 = 1280 (default small)
                                   # for 1B: depth=20, n_embd=2048 (aspect_ratio=~103)
n_head    = model_dim // head_dim  # head_dim typically 128
```

---

## GPTConfig (scratch/model.py — Step 1 ✅)

```python
@dataclass
class GPTConfig:
    vocab_size:  int   = 32768
    n_layer:     int   = 20       # derived: 1B / (12 × 2048²) ≈ 20
    n_head:      int   = 16       # 2048 / 16 = 128 dims per head
    n_embd:      int   = 2048     # d_model
    block_size:  int   = 2048     # context window (max sequence length)
    dropout:     float = 0.0
```

**Important:** In a Python `@dataclass`, fields **must** have type annotations (`n_layer: int = 20`).
Without the annotation, `n_layer = 20` becomes a plain class variable — not a dataclass field — and `GPTConfig()` won't include it.

---

## Head Dimension

With `n_embd=2048` and `n_head=16`:
```
head_dim = n_embd // n_head = 2048 // 16 = 128
```

128 is the modern standard head dimension (used in LLaMA, Mistral, nanochat). It fits well with GPU Tensor Core tile sizes.

---

## Next: Step 3 — CausalSelfAttention

The hardest and most important piece. We need to implement:
1. Q, K, V projections
2. Scaled dot-product attention
3. Causal mask (no attending to future tokens)
4. Output projection
