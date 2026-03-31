from __future__ import annotations

import torch
from torch import Tensor, nn

from manifolds.base import ManifoldExpert
from manifolds.euclidean import EuclideanExpert
from manifolds.hyperbolic import HyperbolicExpert


class ProductManifoldExpert(ManifoldExpert):
    def __init__(
        self,
        skill: str,
        input_dim: int,
        h_dim: int,
        e_dim: int,
        curvature: float,
        manifold_dim: int | None = None,
        n_prototypes: int = 32,
    ) -> None:
        total_dim = int(manifold_dim or (h_dim + e_dim))
        super().__init__(
            skill=skill,
            geometry="product",
            input_dim=input_dim,
            manifold_dim=total_dim,
            curvature=curvature,
        )
        self.h_dim = int(h_dim)
        self.e_dim = int(e_dim)
        self.h_branch = HyperbolicExpert(
            skill=skill,
            input_dim=input_dim,
            h_dim=h_dim,
            curvature=curvature,
            n_prototypes=n_prototypes,
        )
        self.e_branch = EuclideanExpert(skill=skill, input_dim=input_dim, e_dim=e_dim)
        self.fusion = nn.Linear(self.h_dim + self.e_dim, self.manifold_dim)

    def encode(self, h: Tensor) -> Tensor:
        z_h = self.h_branch.encode(h)
        z_e = self.e_branch.encode(h)
        z_h_flat = self.h_branch.to_euclidean(z_h)
        z_cat = torch.cat([z_h_flat, z_e], dim=-1)
        return self.fusion(z_cat)

    def distance(self, z1: Tensor, z2: Tensor) -> Tensor:
        return torch.cdist(z1, z2, p=2)

    def to_euclidean(self, z: Tensor) -> Tensor:
        return z

    def prototype_loss(self, h: Tensor, target_prototype_ids: Tensor, margin: float = 2.0) -> Tensor:
        z_h = self.h_branch.encode(h)
        return self.h_branch.prototype_memory.prototype_loss(
            z_h,
            target_prototype_ids,
            self.curvature,
            margin=margin,
        )

    def expert_config(self) -> dict[str, float | int | str]:
        config = super().expert_config()
        config.update(
            {
                "h_dim": self.h_dim,
                "e_dim": self.e_dim,
                "n_prototypes": self.h_branch.prototype_memory.n_prototypes,
            }
        )
        return config
