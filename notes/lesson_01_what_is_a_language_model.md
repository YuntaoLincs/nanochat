# Lesson 01 — What is a Language Model?

## Core Question
What does a language model fundamentally do?

## Your Answers (Socratic Reconstruction)

**Q: What does an LM output?**
A: A probability distribution over the vocabulary, for each position in the sequence.

**Q: What shape is the output tensor?**
A: `(B, T, V)` = `(1, 2048, 32768)` for batch=1, seqlen=2048, vocab=32768 (= 2^15)

**Q: What is the difference between logits and probabilities?**
A: The model outputs raw **logits** (any real number). Softmax converts them to **probabilities** (positive, sum to 1).

**Q: What is the loss?**
A: **Cross-entropy loss** — measures the gap between true distribution `p` (one-hot) and predicted distribution `q`.

Formula simplification:
```
H(p, q) = -sum_i p_i * log(q_i)
         = -log(q_k)      # only the true token k survives
```

Behavior:
| Situation | Loss |
|-----------|------|
| Confident + correct (`q_k → 1`) | `-log(1) = 0` |
| Confident + wrong (`q_k → 0`) | `-log(0) = +∞` |

## Key Vocabulary
- **logits**: raw output of the final linear layer, before softmax
- **cross-entropy loss**: `-log(probability assigned to true token)`
- **greedy decoding**: argmax over logits at inference time
- **sampling**: draw from the softmax distribution (temperature controls sharpness)

## Token Embeddings

One-hot vectors are wasteful (sparse, high-dimensional) and carry **no semantic information** — `"king"` and `"queen"` are equidistant from everything.

Instead: a **learned embedding table** of shape `(V, d_model)`.

```python
# nanochat/gpt.py:172
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(padded_vocab_size, config.n_embd),  # wte = weight token embedding
    ...
})
```

- `d_model = 2048` for the 1B model
- Each token ID maps to a **dense vector** learned by gradient descent
- Semantically similar tokens end up with **geometrically close** vectors

### Padded Vocab Size (gpt.py:168)

```python
padded_vocab_size = ((vocab_size + pad_to - 1) // pad_to) * pad_to
```

GPU Tensor Cores work on tiles (16×16 blocks). Dimensions that are **multiples of 64 or 128** ensure full tile utilization. 32,768 = 2¹⁵ is already optimal.

## The Missing Piece: Position

After embedding lookup the model has shape `(B, T, d_model)` — but the embedding of `"cat"` at position 2 is **identical** to `"cat"` at position 7.

**Example:** `"I love you"` ≠ `"you love I"` — same tokens, different order, different meaning.

→ Leads directly into **Lesson 02: Positional Encoding (RoPE)**.

---

## Status: ✅ Complete
