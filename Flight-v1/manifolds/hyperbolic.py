from __future__ import annotations

import torch
from torch import Tensor, nn

from manifolds.base import ManifoldExpert
from manifolds.ops import expmap0, logmap0, poincare_distance, project_to_ball
from manifolds.prototype_memory import PrototypeMemory


class MobiusLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        curvature: float,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.curvature = float(curvature)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        nn.init.xavier_uniform_(self.weight, gain=0.1)  # small init

    def forward(self, x: Tensor) -> Tensor:
        tangent = logmap0(x, self.curvature)
        # clamp tangent to prevent exploding linear outputs
        tangent = tangent.clamp(-10.0, 10.0)
        projected = torch.nn.functional.linear(tangent, self.weight, self.bias)
        # clamp before expmap0 to stay well inside ball
        projected = projected.clamp(-5.0, 5.0)
        return expmap0(projected, self.curvature)


class HyperbolicExpert(ManifoldExpert):
    def __init__(
        self,
        skill: str,
        input_dim: int,
        h_dim: int,
        curvature: float,
        n_prototypes: int = 32,
    ) -> None:
        super().__init__(
            skill=skill,
            geometry="hyperbolic",
            input_dim=input_dim,
            manifold_dim=h_dim,
            curvature=curvature,
        )
        self.h_dim = int(h_dim)
        self.input_proj = nn.Linear(self.input_dim, self.h_dim)
        # LayerNorm before lifting to manifold — keeps norms stable
        self.input_norm = nn.LayerNorm(self.h_dim)
        self.mobius_linear_1 = MobiusLinear(self.h_dim, self.h_dim, curvature)
        self.mobius_linear_2 = MobiusLinear(self.h_dim, self.h_dim, curvature)
        self.prototype_memory = PrototypeMemory(
            n_prototypes=n_prototypes,
            dim=self.h_dim,
            curvature=curvature,
        )

    def encode(self, h: Tensor) -> Tensor:
        h_proj = self.input_proj(h)
        # normalise before lifting — prevents large-norm inputs
        # from pushing points to the ball boundary
        h_proj = self.input_norm(h_proj)
        # scale down further so expmap0 maps to interior of ball
        h_proj = h_proj * 0.1
        z0 = expmap0(h_proj, self.curvature)
        z1 = self.mobius_linear_1(z0)
        z2 = self.mobius_linear_2(z1)
        return project_to_ball(z2, self.curvature)

    def distance(self, z1: Tensor, z2: Tensor) -> Tensor:
        return poincare_distance(
            z1.unsqueeze(1), z2.unsqueeze(0), self.curvature
        )

    def to_euclidean(self, z: Tensor) -> Tensor:
        return logmap0(z, self.curvature)

    def prototype_loss(
        self,
        h: Tensor,
        target_prototype_ids: Tensor,
        margin: float = 2.0,
    ) -> Tensor:
        z = self.encode(h)
        return self.prototype_memory.prototype_loss(
            z, target_prototype_ids, self.curvature, margin=margin
        )

    def expert_config(self) -> dict[str, float | int | str]:
        config = super().expert_config()
        config.update({
            "h_dim": self.h_dim,
            "n_prototypes": self.prototype_memory.n_prototypes,
        })
        return config