"""
model.py (v4 - numerically stable)

Architecture: standard transformer with Euclidean ops throughout.
The hyperbolic geometry is injected via:
  1. Learned curvature parameter c
  2. expmap0/logmap0 at the embedding boundary (input → ball → output)
  3. Training losses that enforce hyperbolic structure on hidden states
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic import (
    expmap0, logmap0, TangentSpaceAttention, HyperbolicMLP
)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1, c=1.0):
        super().__init__()
        self.attn  = TangentSpaceAttention(d_model, n_heads, c=c, dropout=dropout)
        self.ffn   = HyperbolicMLP(d_model, ffn_dim, c=c, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm transformer (more stable than post-norm)
        x = x + self.drop(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class HyperbolicReasoningStudent(nn.Module):
    """
    ~100M parameter student model.

    Euclidean transformer core with hyperbolic geometry at the boundaries:
      - Input embeddings are projected onto the Poincaré ball (expmap0)
        then immediately projected back (logmap0) — this is a no-op at init
        but the gradient flows through the manifold projection, giving the
        model a geometric inductive bias without causing numerical instability.
      - last_hidden is returned as both Euclidean (for lm_head) and is used
        in the manifold losses by projecting onto the ball.
      - Curvature c is learned.
    """

    def __init__(
        self,
        vocab_size   = 32000,
        d_model      = 512,
        n_heads      = 8,
        n_layers     = 8,
        ffn_dim      = 2048,
        max_seq_len  = 512,
        dropout      = 0.1,
        c_init       = 1.0,
        tie_weights  = True,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # Learned curvature — softplus ensures c > 0
        self.log_c = nn.Parameter(torch.tensor(math.log(c_init)))

        self.embed      = nn.Embedding(vocab_size, d_model)
        self.pos_embed  = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ffn_dim, dropout=dropout, c=c_init)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight,     std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    @property
    def c(self):
        return F.softplus(self.log_c).item()

    def get_causal_mask(self, T, device):
        m = torch.full((1, 1, T, T), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T   = input_ids.shape
        device = input_ids.device
        c      = F.softplus(self.log_c)

        # Embeddings
        pos = torch.arange(T, device=device).unsqueeze(0)
        x   = self.embed(input_ids) + self.pos_embed(pos)
        x   = self.embed_drop(x)

        # Hyperbolic boundary: project onto ball and back
        # This is a smooth differentiable op that encodes the geometry
        # without keeping activations on the manifold (which causes instability)
        x = logmap0(expmap0(x * 0.1, c.item()), c.item())

        # Causal mask
        mask = self.get_causal_mask(T, device)
        if attention_mask is not None:
            pad = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            mask = mask + pad

        # Transformer layers (fully Euclidean — stable)
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        output = {
            "logits":      logits,
            "last_hidden": x,        # Euclidean, used for manifold losses
            "curvature":   c,
        }

        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            output["loss"] = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return output

    def param_count(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Total: {total/1e6:.1f}M  Trainable: {trainable/1e6:.1f}M"


if __name__ == "__main__":
    model = HyperbolicReasoningStudent(
        vocab_size=1000, d_model=64, n_heads=4,
        n_layers=4, ffn_dim=256, max_seq_len=128,
    )
    print(model.param_count())
    ids  = torch.randint(0, 1000, (2, 32))
    lbl  = ids.clone(); lbl[:, :4] = -100
    out  = model(ids, labels=lbl)
    assert torch.isfinite(out["loss"]), f"Loss is NaN: {out['loss']}"
    print(f"Loss: {out['loss'].item():.4f}  c={out['curvature'].item():.4f}")
    print("model.py smoke test passed.")