from __future__ import annotations

import torch

from manifolds import build_expert
from manifolds.euclidean import EuclideanExpert
from manifolds.hyperbolic import HyperbolicExpert
from manifolds.ops import expmap0, logmap0, mobius_add, poincare_distance, project_to_ball
from manifolds.product import ProductManifoldExpert
from manifolds.prototype_memory import PrototypeMemory
from training.contrastive_loss import (
    compute_pretraining_loss,
    pairwise_structure_loss,
    step_ordering_loss,
    subskill_prototype_loss,
)


def test_expmap_logmap_roundtrip() -> None:
    v = torch.randn(4, 8) * 0.05
    x = expmap0(v, c=1.0)
    restored = logmap0(x, c=1.0)
    assert torch.allclose(v, restored, atol=1e-4, rtol=1e-4)


def test_poincare_distance_symmetric() -> None:
    x = expmap0(torch.randn(3, 8) * 0.05, c=1.0)
    y = expmap0(torch.randn(3, 8) * 0.05, c=1.0)
    d_xy = poincare_distance(x, y, c=1.0)
    d_yx = poincare_distance(y, x, c=1.0)
    assert torch.allclose(d_xy, d_yx, atol=1e-5)


def test_mobius_add_handles_zero_and_noncommutative() -> None:
    zero = torch.zeros(2, 4)
    x = expmap0(torch.randn(2, 4) * 0.05, c=1.0)
    assert torch.isfinite(mobius_add(x, zero, c=1.0)).all()
    y = expmap0(torch.randn(2, 4) * 0.05, c=1.0)
    xy = mobius_add(x, y, c=1.0)
    yx = mobius_add(y, x, c=1.0)
    assert not torch.allclose(xy, yx)


def test_project_to_ball_bounds_norm() -> None:
    x = torch.randn(5, 6)
    projected = project_to_ball(x, c=1.0)
    assert torch.isfinite(projected).all()
    assert torch.all(projected.norm(dim=-1) < 1.0)


def test_ops_handle_zero_without_nan() -> None:
    zero = torch.zeros(2, 4)
    x = expmap0(zero, c=1.0)
    assert torch.isfinite(x).all()
    assert torch.isfinite(logmap0(x, c=1.0)).all()
    assert torch.isfinite(poincare_distance(x, x, c=1.0)).all()


def test_hyperbolic_expert_forward_shapes() -> None:
    expert = HyperbolicExpert(skill="multi_step_reason", input_dim=16, h_dim=8, curvature=1.0, n_prototypes=4)
    out = expert(torch.randn(3, 16))
    assert out["z"].shape == (3, 8)
    assert out["z_euclidean"].shape == (3, 8)
    assert out["distances"].shape == (3, 3)
    assert torch.isfinite(out["z_euclidean"]).all()


def test_euclidean_expert_forward_shapes() -> None:
    expert = EuclideanExpert(skill="arithmetic", input_dim=16, e_dim=8)
    out = expert(torch.randn(3, 16))
    assert out["z"].shape == (3, 8)
    assert out["z_euclidean"].shape == (3, 8)
    assert out["distances"].shape == (3, 3)
    assert torch.isfinite(out["z_euclidean"]).all()


def test_product_expert_forward_shapes() -> None:
    expert = ProductManifoldExpert(skill="algebraic", input_dim=16, h_dim=6, e_dim=6, curvature=1.0, manifold_dim=12, n_prototypes=4)
    out = expert(torch.randn(3, 16))
    assert out["z"].shape == (3, 12)
    assert out["z_euclidean"].shape == (3, 12)
    assert out["distances"].shape == (3, 3)
    assert torch.isfinite(out["z_euclidean"]).all()


def test_build_expert_returns_correct_types() -> None:
    cfg = {
        "skills": {
            "a": {"geometry": "hyperbolic", "h_dim": 8, "curvature": 1.0, "n_prototypes": 4},
            "b": {"geometry": "euclidean", "e_dim": 8, "curvature": 0.0},
            "c": {"geometry": "product", "h_dim": 4, "e_dim": 4, "curvature": 1.0, "n_prototypes": 4},
        }
    }
    assert isinstance(build_expert("a", cfg, backbone_dim=16), HyperbolicExpert)
    assert isinstance(build_expert("b", cfg, backbone_dim=16), EuclideanExpert)
    assert isinstance(build_expert("c", cfg, backbone_dim=16), ProductManifoldExpert)


def test_prototype_memory_assign_and_loss() -> None:
    memory = PrototypeMemory(n_prototypes=4, dim=8, curvature=1.0)
    z = expmap0(torch.randn(5, 8) * 0.05, c=1.0)
    assignments, distances = memory.assign(z, c=1.0)
    loss = memory.prototype_loss(z, assignments, c=1.0)
    assert assignments.min() >= 0
    assert assignments.max() < 4
    assert distances.shape == (5, 4)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_losses_are_finite_and_backward_runs() -> None:
    expert = ProductManifoldExpert(skill="algebraic", input_dim=10, h_dim=4, e_dim=4, curvature=1.0, manifold_dim=8, n_prototypes=3)
    h = torch.randn(6, 10, requires_grad=True)
    prototype_ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    step_embeddings = [torch.randn(4, 10), torch.randn(5, 10), torch.randn(3, 10)]

    z = expert.encode(h)
    l1 = subskill_prototype_loss(expert, h, prototype_ids, margin=2.0)
    l2 = pairwise_structure_loss(expert, z, h)
    l3 = step_ordering_loss(expert, step_embeddings, margin=0.5)
    total, metrics = compute_pretraining_loss(
        expert=expert,
        h=h,
        teacher_embeddings=h,
        step_embeddings=step_embeddings,
        prototype_ids=prototype_ids,
    )
    total.backward()

    assert l1.item() >= 0.0
    assert torch.isfinite(l2)
    assert l3.item() >= 0.0
    assert torch.isfinite(total)
    assert "loss_proto" in metrics and "loss_struct" in metrics and "loss_order" in metrics
