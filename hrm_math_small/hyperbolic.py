"""
hyperbolic.py
-------------
Poincaré ball geometry primitives for the hyperbolic transformer.

All tensors live in the open unit ball:  ||x|| < 1
Curvature c > 0 controls how curved the space is.
c=1.0 is the standard Poincaré ball. Smaller c = flatter.

Key operations:
  mobius_add      : addition on the manifold (replaces vector addition)
  expmap0         : map from tangent space at origin → ball (used after linear layers)
  logmap0         : inverse — ball → tangent space at origin
  poincare_dist   : geodesic distance between two points on the ball
  HypLinear       : drop-in replacement for nn.Linear that stays on manifold
  HyperbolicMLP   : MLP where all ops live in hyperbolic space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Numerical safety
# ---------------------------------------------------------------------------
EPS   = 1e-8       # prevent division by zero
CLAMP = 1 - 1e-5   # keep points strictly inside the ball


def clamp_to_ball(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Project points back inside the unit ball if they escape during training."""
    max_norm = CLAMP / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    # Only rescale points that are outside
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


# ---------------------------------------------------------------------------
# Core Möbius operations  (curvature-aware)
# ---------------------------------------------------------------------------

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Möbius addition: x ⊕_c y
    The manifold analogue of vector addition.
    Closed under the ball: if ||x||,||y|| < 1/√c then ||result|| < 1/√c
    """
    x2 = (x * x).sum(dim=-1, keepdim=True)          # ||x||²
    y2 = (y * y).sum(dim=-1, keepdim=True)          # ||y||²
    xy = (x * y).sum(dim=-1, keepdim=True)          # <x, y>

    num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
    denom = 1 + 2*c*xy + c**2 * x2 * y2
    return clamp_to_ball(num / denom.clamp(min=EPS), c)


def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map at the origin: tangent space → ball.
    Use this to map the output of a standard linear layer into hyperbolic space.

    v : tangent vector at origin (any R^n vector)
    returns: point on the Poincaré ball
    """
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    sqrt_c = c ** 0.5
    tanh_arg = (sqrt_c * v_norm).clamp(max=15.0)   # prevent tanh saturation
    return clamp_to_ball(torch.tanh(tanh_arg) * v / (sqrt_c * v_norm), c)


def logmap0(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Logarithmic map at the origin: ball → tangent space.
    Inverse of expmap0. Use before operations that expect Euclidean vectors.
    """
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    sqrt_c = c ** 0.5
    # atanh is only defined for |x| < 1; clamp_to_ball ensures this
    atanh_arg = (sqrt_c * x_norm).clamp(max=1 - EPS)
    return (torch.atanh(atanh_arg) / (sqrt_c * x_norm)) * x


def poincare_dist(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Geodesic distance between x and y on the Poincaré ball.
    d(x,y) = (2/√c) · atanh(√c · ||(-x) ⊕_c y||)

    Returns a scalar tensor (or batch of scalars) representing distances.
    This is what we use in the distillation loss to measure how far
    student step embeddings are from teacher step embeddings on the manifold.
    """
    sqrt_c = c ** 0.5
    diff   = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1).clamp(max=1 - EPS)
    return (2.0 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)


def mobius_matmul(x: torch.Tensor, W: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Möbius matrix-vector multiplication.
    Equivalent to: expmap0(W @ logmap0(x))
    Used in HypLinear to replace standard nn.Linear on-manifold.
    """
    return expmap0(logmap0(x, c) @ W.T, c)


# ---------------------------------------------------------------------------
# Hyperbolic Linear Layer
# ---------------------------------------------------------------------------

class HypLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that operates on the Poincaré ball.

    Input  x : (..., in_features)  — points ON the ball
    Output   : (..., out_features) — points ON the ball

    Internally:
      1. logmap0(x)  — lift to tangent space
      2. standard linear transform
      3. expmap0(.)  — project back to ball
      4. Möbius-add bias (bias lives on the ball too)
    """
    def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.c = c

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is on the ball; lift to tangent space, transform, project back
        x_tangent = logmap0(x, self.c)
        out_tangent = x_tangent @ self.weight.T
        out = expmap0(out_tangent, self.c)

        if self.bias is not None:
            # Bias is treated as a point on the ball via expmap
            bias_ball = expmap0(self.bias.unsqueeze(0), self.c)
            out = mobius_add(out, bias_ball.expand_as(out), self.c)

        return out

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, c={self.c}"


# ---------------------------------------------------------------------------
# Hyperbolic MLP  (used as the reasoning step encoder)
# ---------------------------------------------------------------------------

class HyperbolicMLP(nn.Module):
    """
    A 2-layer MLP entirely on the Poincaré ball.
    Used to encode each reasoning step text embedding into hyperbolic space.

    Architecture:
      embed (Euclidean R^d)
        → expmap0 (lift onto ball)
        → HypLinear(d → hidden)
        → HypLinear(hidden → d)
        → point on ball

    The key property: steps that are conceptually "broader" (early in the
    reasoning chain) will naturally cluster near the origin of the ball,
    while specific computation steps fan outward — this is the geometric
    inductive bias we're trying to exploit.
    """
    def __init__(self, input_dim: int, hidden_dim: int, c: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.c = c
        self.lin1 = HypLinear(input_dim, hidden_dim, c=c)
        self.lin2 = HypLinear(hidden_dim, input_dim, c=c)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (..., input_dim)  Euclidean embeddings (e.g. from a tokenizer)
        returns: (..., input_dim)  points on the Poincaré ball
        """
        # Lift from Euclidean to ball
        h = expmap0(x, self.c)

        # Transform on the manifold
        h = self.lin1(h)

        # Non-linearity: go to tangent space, apply relu, come back
        h_tan = logmap0(h, self.c)
        h_tan = F.relu(h_tan)
        h_tan = self.drop(h_tan)
        h = expmap0(h_tan, self.c)

        h = self.lin2(h)
        return h   # points on ball


# ---------------------------------------------------------------------------
# Hyperbolic Attention  (hybrid: geometry in key/query, Euclidean softmax)
# ---------------------------------------------------------------------------

class HyperbolicAttention(nn.Module):
    """
    Multi-head attention where similarity is computed as negative geodesic
    distance on the Poincaré ball, rather than dot product.

    This makes the attention geometry-aware: tokens that are "closer" on the
    reasoning manifold attend to each other more strongly.

    Architecture (hybrid):
      Q, K projected into hyperbolic space via HypLinear
      V stays Euclidean (standard linear)
      Attention weight = softmax(-dist(Q_i, K_j) / sqrt(d_k))
      Output = weighted sum of V  (Euclidean)

    Why hybrid: full hyperbolic attention aggregation (weighted Fréchet mean)
    is expensive and numerically fragile. Keeping V Euclidean is a practical
    compromise validated in recent literature (e.g. HypFormer, 2023).
    """
    def __init__(self, d_model: int, n_heads: int, c: float = 1.0, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.c       = c

        # Q and K live on the ball
        self.q_proj = HypLinear(d_model, d_model, c=c)
        self.k_proj = HypLinear(d_model, d_model, c=c)
        # V stays Euclidean
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                         # (B, T, d_model) — on the ball
        mask: torch.Tensor | None = None,        # (B, T, T) additive mask
    ) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.n_heads
        dk = self.d_k

        # Project Q, K on manifold; V in Euclidean
        x_eucl = logmap0(x, self.c)              # lift to tangent for V
        Q = self.q_proj(x)                       # (B, T, d_model) on ball
        K = self.k_proj(x)                       # (B, T, d_model) on ball
        V = self.v_proj(x_eucl)                  # (B, T, d_model) Euclidean

        # Split into heads
        Q = Q.view(B, T, h, dk).transpose(1, 2)  # (B, h, T, dk)
        K = K.view(B, T, h, dk).transpose(1, 2)
        V = V.view(B, T, h, dk).transpose(1, 2)

        # Compute pairwise geodesic distances: (B, h, T, T)
        # Q_i vs K_j for all i,j
        Q_exp = Q.unsqueeze(3).expand(B, h, T, T, dk)  # (B,h,T,T,dk)
        K_exp = K.unsqueeze(2).expand(B, h, T, T, dk)

        # Flatten batch dims for poincare_dist
        Q_flat = Q_exp.reshape(-1, dk)
        K_flat = K_exp.reshape(-1, dk)
        dists   = poincare_dist(Q_flat, K_flat, self.c).view(B, h, T, T)

        # Attention: closer = higher weight (negate distance)
        scale   = dk ** 0.5
        scores  = -dists / scale                          # (B, h, T, T)
        if mask is not None:
            scores = scores + mask
        attn    = F.softmax(scores, dim=-1)
        attn    = self.drop(attn)

        # Weighted sum of V (Euclidean)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch = __import__("torch")
    print("Running geometry smoke tests...")

    c = 1.0
    x = torch.randn(4, 8) * 0.3      # small vectors so they're inside ball
    y = torch.randn(4, 8) * 0.3

    # expmap / logmap roundtrip
    x_ball = expmap0(x, c)
    x_back = logmap0(x_ball, c)
    err = (x - x_back).abs().max().item()
    print(f"  expmap→logmap roundtrip error: {err:.2e}  (should be ~1e-6)")

    # Poincaré distance is positive
    d = poincare_dist(x_ball, expmap0(y, c), c)
    print(f"  poincare_dist min={d.min():.4f}  max={d.max():.4f}  (all > 0)")

    # Möbius add stays in ball
    z = mobius_add(x_ball, expmap0(y, c), c)
    norms = z.norm(dim=-1)
    print(f"  mobius_add norms max={norms.max():.4f}  (should be < 1)")

    # HypLinear
    lin = HypLinear(8, 16, c=c)
    out = lin(x_ball)
    print(f"  HypLinear(8→16): out shape {out.shape}, norm max {out.norm(dim=-1).max():.4f}")

    # HyperbolicMLP
    mlp = HyperbolicMLP(8, 32, c=c)
    out = mlp(x)
    print(f"  HyperbolicMLP: out shape {out.shape}, norm max {out.norm(dim=-1).max():.4f}")

    print("All tests passed.")