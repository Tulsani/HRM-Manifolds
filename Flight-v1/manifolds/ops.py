from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-5
NORM_EPS = 1e-7


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
    # tanh(t)/t is stable everywhere — no 0/0 issue in forward or backward
    # because tanh(t)/t -> 1 as t -> 0 and PyTorch handles this smoothly
    t = sqrt_c * v_norm / 2.0
    # Use stable formula: tanh(t) / (sqrt_c * v_norm)
    # At v_norm -> 0: tanh(t) ≈ t, so tanh(t)/(sqrt_c*v_norm) ≈ 0.5
    factor = torch.tanh(t) / (sqrt_c * v_norm)
    x = factor * v
    return project_to_ball(x, c)


def logmap0(x: Tensor, c: float) -> Tensor:
    x = project_to_ball(x, c)
    sqrt_c = _sqrt_c(c, x.device, x.dtype)
    x_norm = _safe_norm(x, keepdim=True)

    # The problem: artanh(t) / t has NaN gradient at t -> 0 via quotient rule.
    # Fix: use torch.where to switch between Taylor expansion (small t)
    # and exact formula (large t). Both branches have stable gradients.
    #
    # Taylor series: artanh(t) / t = 1 + t²/3 + t⁴/5 + ...
    # For t < 1e-3, first two terms give error < 1e-9
    # For t >= 1e-3, use exact formula artanh(t) / t directly

    arg = (sqrt_c * x_norm).clamp(max=1.0 - EPS)  # t = sqrt_c * x_norm

    # Exact branch: artanh(t) / t — safe when t is not near 0
    # We evaluate at max(arg, 1e-3) to avoid division issues even in
    # the branch that won't be selected, preventing NaN in unused branch
    safe_arg = arg.clamp(min=1e-3)
    exact = artanh(safe_arg) / safe_arg

    # Taylor branch: 1 + t²/3 — safe near 0 with well-defined gradient
    taylor = 1.0 + (arg ** 2) / 3.0

    # Switch at threshold — torch.where has well-defined gradients
    # for both branches so backward never hits 0/0
    threshold = torch.full_like(arg, 1e-3)
    atanh_over_t = torch.where(arg < threshold, taylor, exact)

    scale = (2.0 / sqrt_c) * atanh_over_t
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
    arg = (sqrt_c * delta_norm).clamp(max=1.0 - EPS)
    return (2.0 / sqrt_c) * artanh(arg)