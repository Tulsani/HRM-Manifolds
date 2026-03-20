"""
hyperbolic.py (v4 - numerically stable)

Key design change: manifold ops are used ONLY for the explicit geometry
losses (step ordering, distillation). The transformer itself uses a
"tangent-space" architecture where all intermediate tensors stay Euclidean.

The hyperbolic inductive bias comes from:
  1. HypLinear projections (expmap/logmap at boundaries)
  2. The explicit order_loss and step_dist_loss at training time
  3. The curvature parameter c being learned

This avoids the boundary collapse problem entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS   = 1e-7
CLAMP = 1 - 1e-5


def clamp_to_ball(x, c=1.0):
    max_norm = CLAMP / (c ** 0.5)
    norm  = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    scale = (max_norm / norm).clamp(max=1.0)
    return x * scale


def expmap0(v, c=1.0):
    """Euclidean tangent vector → Poincaré ball point."""
    v_norm   = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    sqrt_c   = c ** 0.5
    tanh_arg = (sqrt_c * v_norm).clamp(max=15.0)
    return clamp_to_ball(torch.tanh(tanh_arg) * v / (sqrt_c * v_norm), c)


def logmap0(x, c=1.0):
    """Poincaré ball point → Euclidean tangent vector."""
    x_norm    = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    sqrt_c    = c ** 0.5
    atanh_arg = (sqrt_c * x_norm).clamp(max=1 - EPS)
    return (torch.atanh(atanh_arg) / (sqrt_c * x_norm)) * x


def mobius_add(x, y, c=1.0):
    x2    = (x * x).sum(-1, keepdim=True)
    y2    = (y * y).sum(-1, keepdim=True)
    xy    = (x * y).sum(-1, keepdim=True)
    num   = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
    denom = (1 + 2*c*xy + c**2 * x2 * y2).clamp(min=EPS)
    return clamp_to_ball(num / denom, c)


def poincare_dist(x, y, c=1.0):
    sqrt_c    = c ** 0.5
    diff      = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1).clamp(max=1 - EPS)
    return (2.0 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)


def poincare_dist_row(q, K, c=1.0):
    """Distance from one query (d,) to all keys (T, d). O(T*d) memory."""
    q_exp = q.unsqueeze(0).expand_as(K)
    diff  = mobius_add(-q_exp, K, c)
    dnorm = diff.norm(dim=-1).clamp(max=1 - EPS)
    return (2.0 / (c**0.5)) * torch.atanh((c**0.5) * dnorm)


class HypLinear(nn.Module):
    """
    Linear layer that operates on the Poincaré ball.
    Takes ball point → returns ball point.
    """
    def __init__(self, in_features, out_features, c=1.0, bias=True):
        super().__init__()
        self.c      = c
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight, gain=0.1)  # small init

    def forward(self, x):
        # x on ball → tangent → linear → back to ball
        out = expmap0(logmap0(x, self.c) @ self.weight.T, self.c)
        if self.bias is not None:
            b = expmap0(self.bias.unsqueeze(0) * 0.01, self.c)
            out = mobius_add(out, b.expand_as(out), self.c)
        return out


class HyperbolicMLP(nn.Module):
    """
    Two-layer MLP entirely in tangent space (Euclidean).
    The hyperbolic geometry lives in how these embeddings are used
    in the loss functions, not in the MLP ops themselves.
    """
    def __init__(self, input_dim, hidden_dim, c=1.0, dropout=0.1):
        super().__init__()
        self.c    = c
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, input_dim)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.lin2.weight, gain=0.1)

    def forward(self, x):
        # Pure Euclidean FFN - stable and fast
        return self.lin2(self.drop(F.gelu(self.lin1(x))))


class TangentSpaceAttention(nn.Module):
    """
    Standard multi-head attention. Hyperbolic geometry is encoded via
    the training losses, not by making attention itself curved.
    This is numerically identical to a standard transformer attention block.
    """
    def __init__(self, d_model, n_heads, c=1.0, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads
        self.c        = c
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)
        for l in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(l.weight, gain=0.1)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        h, dk   = self.n_heads, self.d_k

        Q = self.q_proj(x).view(B, T, h, dk).transpose(1, 2)
        K = self.k_proj(x).view(B, T, h, dk).transpose(1, 2)
        V = self.v_proj(x).view(B, T, h, dk).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (dk ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn = self.drop(F.softmax(scores, dim=-1))
        out  = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


# Keep HyperbolicAttention as alias for compatibility
HyperbolicAttention = TangentSpaceAttention


if __name__ == "__main__":
    print("Smoke tests...")
    c = 1.0
    v = torch.randn(4, 32) * 0.1
    b = expmap0(v, c)
    assert b.norm(dim=-1).max() < 1.0, "expmap0 outside ball"
    err = (v - logmap0(b, c)).abs().max()
    assert err < 1e-5, f"roundtrip err {err}"
    print(f"  expmap/logmap roundtrip: {err:.2e}  OK")

    d = poincare_dist(b, expmap0(torch.randn(4,32)*0.1, c), c)
    assert (d > 0).all()
    print(f"  poincare_dist: min={d.min():.4f} max={d.max():.4f}  OK")

    attn = TangentSpaceAttention(64, 4)
    x = torch.randn(2, 16, 64)
    o = attn(x)
    assert torch.isfinite(o).all()
    print(f"  Attention: {o.shape}  OK")

    mlp = HyperbolicMLP(64, 128)
    o2 = mlp(x)
    assert torch.isfinite(o2).all()
    print(f"  MLP: {o2.shape}  OK")
    print("All passed.")