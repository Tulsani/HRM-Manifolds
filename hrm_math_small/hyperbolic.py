"""
hyperbolic.py  (v2 — memory-efficient)
Fix: HyperbolicAttention no longer expands Q/K to (B,h,T,T,dk).
Uses chunked row-wise distance — O(T*dk) memory not O(T^2*dk).
Also adds TangentSpaceAttention as a faster fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS   = 1e-8
CLAMP = 1 - 1e-5


def clamp_to_ball(x, c=1.0):
    max_norm = CLAMP / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def mobius_add(x, y, c=1.0):
    x2  = (x * x).sum(dim=-1, keepdim=True)
    y2  = (y * y).sum(dim=-1, keepdim=True)
    xy  = (x * y).sum(dim=-1, keepdim=True)
    num   = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
    denom = 1 + 2*c*xy + c**2 * x2 * y2
    return clamp_to_ball(num / denom.clamp(min=EPS), c)


def expmap0(v, c=1.0):
    v_norm   = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    sqrt_c   = c ** 0.5
    tanh_arg = (sqrt_c * v_norm).clamp(max=15.0)
    return clamp_to_ball(torch.tanh(tanh_arg) * v / (sqrt_c * v_norm), c)


def logmap0(x, c=1.0):
    x_norm   = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    sqrt_c   = c ** 0.5
    atanh_arg = (sqrt_c * x_norm).clamp(max=1 - EPS)
    return (torch.atanh(atanh_arg) / (sqrt_c * x_norm)) * x


def poincare_dist(x, y, c=1.0):
    sqrt_c    = c ** 0.5
    diff      = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1).clamp(max=1 - EPS)
    return (2.0 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)


def poincare_dist_row(q_row, K, c=1.0):
    """Distance from one query (dk,) to all keys (T, dk). O(T*dk) memory."""
    q_exp  = q_row.unsqueeze(0).expand_as(K)
    diff   = mobius_add(-q_exp, K, c)
    dnorm  = diff.norm(dim=-1).clamp(max=1-EPS)
    return (2.0 / (c**0.5)) * torch.atanh((c**0.5) * dnorm)


class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c=1.0, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x_tan   = logmap0(x, self.c)
        out_tan = x_tan @ self.weight.T
        out     = expmap0(out_tan, self.c)
        if self.bias is not None:
            bias_ball = expmap0(self.bias.unsqueeze(0), self.c)
            out = mobius_add(out, bias_ball.expand_as(out), self.c)
        return out


class HyperbolicMLP(nn.Module):
    """
    FFN that takes Euclidean input, processes via HypLinear (on ball),
    and returns Euclidean output. Caller handles expmap0/logmap0 at boundaries.
    """
    def __init__(self, input_dim, hidden_dim, c=1.0, dropout=0.1):
        super().__init__()
        self.c    = c
        self.lin1 = HypLinear(input_dim, hidden_dim, c=c)
        self.lin2 = HypLinear(hidden_dim, input_dim, c=c)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: Euclidean — lift to ball, transform, return Euclidean
        h     = expmap0(x * 0.1, self.c)   # small scale before lifting
        h     = self.lin1(h)
        h_tan = logmap0(h, self.c)
        h_tan = F.relu(h_tan)
        h_tan = self.drop(h_tan)
        h     = expmap0(h_tan * 0.1, self.c)
        return logmap0(self.lin2(h), self.c)  # return Euclidean


class HyperbolicAttention(nn.Module):
    """
    Geodesic-distance attention — row-wise to avoid O(T^2*dk) memory.
    Still exact; trades T kernel launches for T^2*dk memory.
    """
    def __init__(self, d_model, n_heads, c=1.0, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads
        self.c        = c
        self.q_proj   = HypLinear(d_model, d_model, c=c)
        self.k_proj   = HypLinear(d_model, d_model, c=c)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        h, dk, c = self.n_heads, self.d_k, self.c

        x_eucl = logmap0(x, c)
        Q = self.q_proj(x).view(B, T, h, dk).permute(0, 2, 1, 3)    # (B,h,T,dk)
        K = self.k_proj(x).view(B, T, h, dk).permute(0, 2, 1, 3)
        V = self.v_proj(x_eucl).view(B, T, h, dk).permute(0, 2, 1, 3)

        scale  = dk ** 0.5
        scores = torch.zeros(B, h, T, T, device=x.device, dtype=x.dtype)

        for i in range(T):
            q_i = Q[:, :, i, :]                          # (B, h, dk)
            for b in range(B):
                for hd in range(h):
                    scores[b, hd, i] = -poincare_dist_row(
                        q_i[b, hd], K[b, hd], c
                    ) / scale

        if mask is not None:
            scores = scores + mask

        attn = self.drop(F.softmax(scores, dim=-1))
        out  = (attn @ V).permute(0, 2, 1, 3).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class TangentSpaceAttention(nn.Module):
    """
    Faster hybrid: Q/K projected through HypLinear then logmap0'd back to
    Euclidean for standard dot-product attention.  No loops, no OOM.
    Geometry still lives in the projections; only the similarity measure
    is Euclidean.  Recommended for seq_len > 128.
    """
    def __init__(self, d_model, n_heads, c=1.0, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads
        self.c        = c
        self.q_proj   = HypLinear(d_model, d_model, c=c)
        self.k_proj   = HypLinear(d_model, d_model, c=c)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        h, dk   = self.n_heads, self.d_k

        Q = logmap0(self.q_proj(x), self.c).view(B, T, h, dk).permute(0, 2, 1, 3)
        K = logmap0(self.k_proj(x), self.c).view(B, T, h, dk).permute(0, 2, 1, 3)
        V = self.v_proj(logmap0(x, self.c)).view(B, T, h, dk).permute(0, 2, 1, 3)

        scores = (Q @ K.transpose(-2, -1)) / (dk ** 0.5)
        if mask is not None:
            scores = scores + mask

        attn = self.drop(F.softmax(scores, dim=-1))
        out  = (attn @ V).permute(0, 2, 1, 3).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


if __name__ == "__main__":
    print("Smoke tests...")
    c = 1.0
    x = torch.randn(4, 8) * 0.3
    y = torch.randn(4, 8) * 0.3
    xb = expmap0(x, c)
    err = (x - logmap0(xb, c)).abs().max().item()
    print(f"  roundtrip err: {err:.2e}")
    d = poincare_dist(xb, expmap0(y, c), c)
    print(f"  dist range: {d.min():.4f} – {d.max():.4f}")

    # TangentSpaceAttention (fast path)
    a2  = TangentSpaceAttention(64, 4)
    xin = expmap0(torch.randn(2, 16, 64) * 0.1, c)
    print(f"  TangentSpaceAttention: {a2(xin).shape}  OK")

    # HyperbolicAttention (exact geodesic, small seq only)
    a1  = HyperbolicAttention(64, 4)
    print(f"  HyperbolicAttention:   {a1(xin).shape}  OK")
    print("All passed.")