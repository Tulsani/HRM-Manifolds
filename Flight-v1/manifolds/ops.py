from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-5
NORM_EPS = 1e-7   # raised from 1e-15 — prevents overflow in logmap0 scale


def _sqrt_c(c: float | Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(float(c), device=device, dtype=dtype).sqrt()


def _safe_norm(x: Tensor, keepdim: bool = True) -> Tensor:
    return x.norm(dim=-1, keepdim=keepdim).clamp_min(NORM_EPS)


def artanh(x: Tensor) -> Tensor:
    x = x.clamp(min=-1.0 + EPS, max=1.0 - EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def project_to_ball(x: Tensor, c: float, eps: float = EPS) -> Tensor:
    sqrt_c = _sqrt_c(c, x.device, x.dtype)
    max_norm = (1.0 - eps) / sqrt_c
    norm = _safe_norm(x, keepdim=True)
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def expmap0(v: Tensor, c: float) -> Tensor:
    sqrt_c = _sqrt_c(c, v.device, v.dtype)
    v_norm = _safe_norm(v, keepdim=True)
    # tanh(sqrt_c * v_norm / 2) / (sqrt_c * v_norm)
    # when v_norm -> 0, limit is 0.5, so we can safely compute
    factor = torch.tanh(sqrt_c * v_norm / 2.0) / (sqrt_c * v_norm)
    x = factor * v
    return project_to_ball(x, c)


def logmap0(x: Tensor, c: float) -> Tensor:
    x = project_to_ball(x, c)
    sqrt_c = _sqrt_c(c, x.device, x.dtype)
    x_norm = _safe_norm(x, keepdim=True)  # clamped to NORM_EPS
    # artanh(sqrt_c * x_norm) / x_norm
    # when x_norm -> 0, limit is sqrt_c, so scale -> 2.0
    # we guard by clamping x_norm to NORM_EPS before dividing
    atanh_val = artanh(sqrt_c * x_norm)
    scale = (2.0 / sqrt_c) * atanh_val / x_norm
    # final safety: clamp scale to prevent overflow
    scale = scale.clamp(max=100.0)
    return scale * x


def mobius_add(x: Tensor, y: Tensor, c: float) -> Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    c_t = torch.as_tensor(float(c), device=x.device, dtype=x.dtype)
    numerator = (1.0 + 2.0 * c_t * xy + c_t * y2) * x + (1.0 - c_t * x2) * y
    denominator = (1.0 + 2.0 * c_t * xy + (c_t ** 2) * x2 * y2).clamp_min(NORM_EPS)
    return project_to_ball(numerator / denominator, c)


def gyration(x: Tensor, y: Tensor, v: Tensor, c: float) -> Tensor:
    xy = mobius_add(x, y, c)
    yv = mobius_add(y, v, c)
    xyv = mobius_add(x, yv, c)
    return mobius_add(-xy, xyv, c)


def parallel_transport(v: Tensor, x: Tensor, y: Tensor, c: float) -> Tensor:
    return gyration(y, -x, v, c)


def poincare_distance(x: Tensor, y: Tensor, c: float) -> Tensor:
    sqrt_c = _sqrt_c(c, x.device, x.dtype)
    delta = mobius_add(-x, y, c)
    delta_norm = _safe_norm(delta, keepdim=False)
    # clamp argument to artanh to stay in (-1, 1)
    arg = (sqrt_c * delta_norm).clamp(max=1.0 - EPS)
    return (2.0 / sqrt_c) * artanh(arg)