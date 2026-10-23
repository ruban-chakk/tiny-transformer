"""A deliberately small GPT-style transformer, built from PyTorch primitives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, context: int, dropout: float):
        super().__init__()
        assert width % heads == 0
        self.heads, self.head_size = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(context, context)).bool())

    def forward(self, x):
        batch, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, time, self.heads, self.head_size)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        scores = q @ k.transpose(-2, -1) / self.head_size**0.5
        scores = scores.masked_fill(~self.mask[:time, :time], float("-inf"))
        attention = self.dropout(F.softmax(scores, dim=-1))
        joined = (attention @ v).transpose(1, 2).contiguous().view(batch, time, width)
        return self.out(joined)


class Block(nn.Module):
    def __init__(self, width, heads, context, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads, context, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, context=128, width=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, context=context, width=width,
                           heads=heads, layers=layers, dropout=dropout)
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*[Block(width, heads, context, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.output.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        time = tokens.shape[1]
        x = self.token_embedding(tokens) + self.position_embedding(torch.arange(time, device=tokens.device))
        logits = self.output(self.norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=120, temperature=0.0):
        for _ in range(max_new_tokens):
            logits, _ = self(tokens[:, -self.context :])
            next_logits = logits[:, -1]
            next_token = (next_logits.argmax(dim=-1, keepdim=True) if temperature == 0 else
                          torch.multinomial(F.softmax(next_logits / temperature, dim=-1), 1))
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens

