from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from manifolds.base import ManifoldExpert


def subskill_prototype_loss(
    expert: ManifoldExpert,
    h: Tensor,
    prototype_ids: Tensor,
    margin: float = 2.0,
) -> Tensor:
    return expert.prototype_loss(h, prototype_ids, margin=margin).clamp_min(0.0)


def pairwise_structure_loss(
    expert: ManifoldExpert,
    z: Tensor,
    teacher_embeddings: Tensor,
) -> Tensor:
    teacher_embeddings = teacher_embeddings.float()
    teacher_sim = F.normalize(teacher_embeddings, dim=-1) @ F.normalize(teacher_embeddings, dim=-1).T
    teacher_sim = ((teacher_sim + 1.0) / 2.0).clamp(0.0, 1.0)

    student_dist = expert.distance(z, z).float()
    max_dist = student_dist.max().clamp_min(1e-8)
    student_sim = (1.0 - (student_dist / max_dist)).clamp(0.0, 1.0)
    return F.mse_loss(student_sim, teacher_sim)


def step_ordering_loss(
    expert: ManifoldExpert,
    step_embeddings: list[Tensor],
    margin: float = 0.5,
) -> Tensor:
    losses: list[Tensor] = []
    for step_matrix in step_embeddings:
        if step_matrix.ndim != 2 or step_matrix.size(0) < 3:
            continue
        z_steps = expert.encode(step_matrix)
        distances = expert.distance(z_steps, z_steps)
        consecutive = distances.diagonal(offset=1)
        if consecutive.numel() == 0:
            continue

        idx = torch.arange(step_matrix.size(0), device=step_matrix.device)
        non_adjacent_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() > 1
        random_distances = distances.masked_select(non_adjacent_mask)
        if random_distances.numel() == 0:
            continue

        d_consec = consecutive.mean()
        d_random = random_distances.mean()
        losses.append(torch.relu(d_consec - d_random + margin))

    if not losses:
        return next(expert.parameters()).new_tensor(0.0)
    return torch.stack(losses).mean().clamp_min(0.0)


def compute_pretraining_loss(
    expert: ManifoldExpert,
    h: Tensor,
    teacher_embeddings: Tensor,
    step_embeddings: list[Tensor],
    prototype_ids: Tensor,
    lambda_proto: float = 1.0,
    lambda_struct: float = 0.5,
    lambda_order: float = 0.5,
    margin_proto: float = 2.0,
    margin_order: float = 0.5,
) -> tuple[Tensor, dict[str, float]]:
    z = expert.encode(h)
    loss_proto = subskill_prototype_loss(expert, h, prototype_ids, margin=margin_proto)
    loss_struct = pairwise_structure_loss(expert, z, teacher_embeddings)
    loss_order = step_ordering_loss(expert, step_embeddings, margin=margin_order)
    total = lambda_proto * loss_proto + lambda_struct * loss_struct + lambda_order * loss_order
    metrics = {
        "loss": float(total.detach().item()),
        "loss_proto": float(loss_proto.detach().item()),
        "loss_struct": float(loss_struct.detach().item()),
        "loss_order": float(loss_order.detach().item()),
    }
    return total, metrics
