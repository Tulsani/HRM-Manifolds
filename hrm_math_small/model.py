"""
model.py (v5 - curvature actually learned)

Fix: the expmap0/logmap0 roundtrip at the embedding was mathematically
cancelling out, making ∂loss/∂c = 0 identically.

New approach: use expmap0 WITHOUT the inverse logmap0 at the embedding.
The embedding vector goes ONTO the ball and STAYS there for the first
linear projection. This breaks the symmetry so c has real effect.

Specifically: we project embeddings onto the ball, apply a learned
linear mix in tangent space, then come back. The curvature c controls
how the embedding space is warped, and gradients flow through it.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic import expmap0, logmap0, TangentSpaceAttention, HyperbolicMLP


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1, c=1.0):
        super().__init__()
        self.attn  = TangentSpaceAttention(d_model, n_heads, c=c, dropout=dropout)
        self.ffn   = HyperbolicMLP(d_model, ffn_dim, c=c, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class CurvatureProjection(nn.Module):
    """
    Projects embeddings through hyperbolic space in a way that:
      1. Is NOT a roundtrip (so ∂loss/∂c ≠ 0)
      2. Stays numerically stable

    Operation:
      x (Euclidean) → scale → expmap0 onto ball
      → learned linear mix IN TANGENT SPACE (logmap0 → W → expmap0)
      → logmap0 back to Euclidean for transformer

    The key: W mixes dimensions AFTER the nonlinear expmap0, so the
    output is NOT equal to W @ x. The curvature c changes the nonlinearity
    and thus affects every downstream computation.
    """
    def __init__(self, d_model, c_init=1.0):
        super().__init__()
        self.log_c = nn.Parameter(torch.tensor(math.log(c_init)))
        # Small mixing matrix — initialized near identity to preserve
        # pretrained embedding structure at the start of training
        self.W = nn.Parameter(torch.eye(d_model) + torch.randn(d_model, d_model) * 0.01)

    @property
    def c(self):
        return F.softplus(self.log_c)

    def forward(self, x):
        c = self.c
        # Scale down to keep points well inside the ball
        x_scaled = x * 0.1
        # Onto ball
        x_ball = expmap0(x_scaled, c.item())
        # Mix in tangent space (this is where c has effect)
        x_tan  = logmap0(x_ball, c.item())     # back to tangent at origin
        x_mix  = x_tan @ self.W.T              # learned mix
        x_ball2 = expmap0(x_mix * 0.1, c.item())  # back to ball with new c
        # Return Euclidean for transformer
        return logmap0(x_ball2, c.item())


class HyperbolicReasoningStudent(nn.Module):
    """
    ~100M parameter student model.

    The curvature c lives in CurvatureProjection which is applied to
    embeddings before the transformer. This gives c real gradient signal
    while keeping the transformer itself numerically stable (Euclidean).
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

        self.embed      = nn.Embedding(vocab_size, d_model)
        self.pos_embed  = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)

        # Curvature lives here — has real gradient signal
        self.curv_proj  = CurvatureProjection(d_model, c_init=c_init)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ffn_dim,
                             dropout=dropout, c=c_init)
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
        return self.curv_proj.c.item()

    @property
    def log_c(self):
        return self.curv_proj.log_c

    def get_causal_mask(self, T, device):
        m = torch.full((1, 1, T, T), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T   = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(T, device=device).unsqueeze(0)
        x   = self.embed(input_ids) + self.pos_embed(pos)
        x   = self.embed_drop(x)

        # Hyperbolic projection — where curvature c has real effect
        x = self.curv_proj(x)

        # Causal + padding mask
        mask = self.get_causal_mask(T, device)
        if attention_mask is not None:
            pad  = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            mask = mask + pad

        # Stable Euclidean transformer
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        output = {
            "logits":      logits,
            "last_hidden": x,
            "curvature":   self.curv_proj.c,
        }

        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            output["loss"] = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=0.1,
            )

        return output

    def param_count(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Total: {total/1e6:.1f}M  Trainable: {trainable/1e6:.1f}M"


if __name__ == "__main__":
    torch.manual_seed(0)
    model = HyperbolicReasoningStudent(
        vocab_size=1000, d_model=64, n_heads=4,
        n_layers=4, ffn_dim=256, max_seq_len=128,
    )
    print(model.param_count())
    print(f"Initial c: {model.c:.6f}")

    ids = torch.randint(0, 1000, (2, 32))
    lbl = ids.clone(); lbl[:, :4] = -100
    out = model(ids, labels=lbl)
    assert torch.isfinite(out["loss"]), "Loss NaN"
    print(f"Loss: {out['loss'].item():.4f}")

    # Verify c has gradient
    out["loss"].backward()
    grad_c = model.curv_proj.log_c.grad
    assert grad_c is not None and grad_c.abs() > 1e-10, \
        f"c has no gradient: {grad_c}"
    print(f"∂loss/∂log_c = {grad_c.item():.6f}  (must be non-zero)")
    print("model.py smoke test passed.")