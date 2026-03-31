from __future__ import annotations

import torch
from torch import Tensor, nn

from manifolds.ops import poincare_distance, project_to_ball


class PrototypeMemory(nn.Module):
    def __init__(self, n_prototypes: int, dim: int, curvature: float) -> None:
        super().__init__()
        self.n_prototypes = max(int(n_prototypes), 1)
        self.dim = int(dim)
        self.curvature = float(curvature)
        self.prototypes = nn.Parameter(torch.empty(self.n_prototypes, self.dim))
        nn.init.uniform_(self.prototypes, -0.001, 0.001)

    def projected_prototypes(self) -> Tensor:
        return project_to_ball(self.prototypes, self.curvature)

    def assign(self, z: Tensor, c: float | None = None) -> tuple[Tensor, Tensor]:
        curvature = self.curvature if c is None else float(c)
        prototypes = project_to_ball(self.prototypes, curvature)
        distances = poincare_distance(z.unsqueeze(1), prototypes.unsqueeze(0), curvature)
        assignments = distances.argmin(dim=-1)
        return assignments, distances

    def prototype_loss(
        self,
        z: Tensor,
        target_prototype_ids: Tensor,
        c: float | None = None,
        margin: float = 2.0,
    ) -> Tensor:
        curvature = self.curvature if c is None else float(c)
        _, distances = self.assign(z, curvature)
        target_ids = target_prototype_ids.long().clamp(min=0, max=self.n_prototypes - 1)
        target_dist = distances.gather(1, target_ids.unsqueeze(1)).squeeze(1)

        if self.n_prototypes == 1:
            negative_term = torch.zeros_like(target_dist)
        else:
            negative_mask = torch.ones_like(distances, dtype=torch.bool)
            negative_mask.scatter_(1, target_ids.unsqueeze(1), False)
            negative_dist = distances.masked_select(negative_mask).view(z.size(0), self.n_prototypes - 1)
            negative_term = torch.relu(torch.as_tensor(margin, device=z.device, dtype=z.dtype) - negative_dist).mean(dim=1)

        return (target_dist.mean() + negative_term.mean()).clamp_min(0.0)
