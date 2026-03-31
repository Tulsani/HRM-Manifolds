from __future__ import annotations

import torch
from torch import Tensor, nn

from manifolds.base import ManifoldExpert


class EuclideanExpert(ManifoldExpert):
    def __init__(self, skill: str, input_dim: int, e_dim: int) -> None:
        super().__init__(
            skill=skill,
            geometry="euclidean",
            input_dim=input_dim,
            manifold_dim=e_dim,
            curvature=0.0,
        )
        self.e_dim = int(e_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.e_dim * 2),
            nn.LayerNorm(self.e_dim * 2),
            nn.GELU(),
            nn.Linear(self.e_dim * 2, self.e_dim),
        )

    def encode(self, h: Tensor) -> Tensor:
        return self.mlp(h)

    def distance(self, z1: Tensor, z2: Tensor) -> Tensor:
        return torch.cdist(z1, z2, p=2)

    def to_euclidean(self, z: Tensor) -> Tensor:
        return z

    def expert_config(self) -> dict[str, float | int | str]:
        config = super().expert_config()
        config.update({"e_dim": self.e_dim})
        return config
