from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class ManifoldExpert(nn.Module, ABC):
    def __init__(
        self,
        skill: str,
        geometry: str,
        input_dim: int,
        manifold_dim: int,
        curvature: float = 0.0,
    ) -> None:
        super().__init__()
        self.skill = skill
        self.geometry = geometry
        self.input_dim = int(input_dim)
        self.manifold_dim = int(manifold_dim)
        self.curvature = float(curvature)

    @abstractmethod
    def encode(self, h: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def distance(self, z1: Tensor, z2: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def to_euclidean(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def prototype_loss(self, h: Tensor, target_prototype_ids: Tensor, margin: float = 2.0) -> Tensor:
        return h.new_tensor(0.0)

    def expert_config(self) -> dict[str, float | int | str]:
        return {
            "skill": self.skill,
            "geometry": self.geometry,
            "input_dim": self.input_dim,
            "manifold_dim": self.manifold_dim,
            "curvature": self.curvature,
        }

    def forward(self, h: Tensor) -> dict[str, Tensor]:
        z = self.encode(h)
        z_euclidean = self.to_euclidean(z)
        distances = self.distance(z, z)
        return {
            "z": z,
            "z_euclidean": z_euclidean,
            "distances": distances,
        }
