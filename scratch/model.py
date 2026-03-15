"""
scratch/model.py
================
Build a GPT-style LLM from scratch, step by step.

Lesson map:
  Step 1  — Config dataclass
  Step 2  — Token embedding table (wte)
  Step 3  — Causal self-attention (CausalSelfAttention)
  Step 4  — MLP / Feed-forward
  Step 5  — Transformer Block (Attention + MLP)
  Step 6  — Full GPT model (stack of Blocks + lm_head)
  Step 7  — Forward pass & loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Step 1: Config
# ---------------------------------------------------------------------------
# We need a way to hold all the hyperparameters in one place.
# Fill in the fields below. Think about:
#   - How many layers?
#   - How many attention heads?
#   - What is the embedding dimension?
#   - What is the vocab size?
#   - What is the max sequence length?
#   - What dropout rate?

@dataclass
class GPTConfig:
    vocab_size:  int   = 32768
    n_layer:     int   = 20
    n_head:      int   = 16      # 2048 / 16 = 128 dims per head
    n_embd:      int   = 2048
    block_size:  int   = 2048    # context window (max sequence length)
    dropout:     float = 0.0
    

# ---------------------------------------------------------------------------
# Step 2: Token Embedding  (you already know this one!)
# ---------------------------------------------------------------------------
# Will be used inside the GPT class.
# nn.Embedding(vocab_size, n_embd)


# ---------------------------------------------------------------------------
# Step 3: Causal Self-Attention
# ---------------------------------------------------------------------------
# TODO: implement this class
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Hint: you need projections for Q, K, V and an output projection.
        # Also: you need a causal mask so token t cannot attend to t+1, t+2...

    def forward(self, x):
        # x shape: (B, T, C) where C = n_embd
        # TODO: implement
        pass


# ---------------------------------------------------------------------------
# Step 4: MLP (Feed-Forward Network)
# ---------------------------------------------------------------------------
# TODO: implement this class
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Hint: two linear layers with a non-linearity in between.
        # Classic size: hidden dim = 4 * n_embd

    def forward(self, x):
        # TODO: implement
        pass


# ---------------------------------------------------------------------------
# Step 5: Transformer Block
# ---------------------------------------------------------------------------
# TODO: implement this class
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Hint: LayerNorm → Attention → residual, then LayerNorm → MLP → residual

    def forward(self, x):
        # TODO: implement
        pass


# ---------------------------------------------------------------------------
# Step 6 & 7: GPT model
# ---------------------------------------------------------------------------
# TODO: implement this class
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Hint: you need wte, a stack of Blocks, a final LayerNorm, and lm_head

    def forward(self, idx, targets=None):
        # idx: (B, T) token ids
        # targets: (B, T) shifted token ids for computing loss
        # TODO: implement
        # Return: (logits, loss) where loss is None if targets is None
        pass
