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
    loss = expert.prototype_loss(h, prototype_ids, margin=margin)
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
    return loss.clamp_min(0.0)


def pairwise_structure_loss(
    expert: ManifoldExpert,
    z: Tensor,
    teacher_embeddings: Tensor,
) -> Tensor:
    if z.size(0) < 2:
        return z.new_tensor(0.0).requires_grad_(True)

    teacher_embeddings = torch.nan_to_num(
        teacher_embeddings.float(), nan=0.0, posinf=1.0, neginf=-1.0
    )
    t_norm = F.normalize(teacher_embeddings, dim=-1)
    teacher_sim = (t_norm @ t_norm.T)
    # map from [-1,1] to [0,1]
    teacher_sim = ((teacher_sim + 1.0) / 2.0).clamp(0.0, 1.0)

    student_dist = expert.distance(z, z).float()
    student_dist = torch.nan_to_num(student_dist, nan=0.0, posinf=1.0, neginf=0.0)
    # guard: if all distances are zero (degenerate), return 0
    max_dist = student_dist.max()
    if max_dist < 1e-8:
        return z.new_tensor(0.0).requires_grad_(True)
    student_sim = (1.0 - (student_dist / max_dist)).clamp(0.0, 1.0)

    loss = F.mse_loss(student_sim, teacher_sim.detach())
    return torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)


def step_ordering_loss(
    expert: ManifoldExpert,
    step_embeddings: list[Tensor],
    margin: float = 0.5,
) -> Tensor:
    losses: list[Tensor] = []
    for step_matrix in step_embeddings:
        step_matrix = torch.nan_to_num(step_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        # guard: need at least 3 steps and non-empty tensor
        if step_matrix.ndim != 2 or step_matrix.size(0) < 3:
            continue

        z_steps = expert.encode(step_matrix)
        z_steps = torch.nan_to_num(z_steps, nan=0.0, posinf=1.0, neginf=-1.0)

        distances = expert.distance(z_steps, z_steps)
        distances = torch.nan_to_num(distances, nan=0.0, posinf=1.0, neginf=0.0)

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
        step_loss = torch.relu(d_consec - d_random + margin)

        losses.append(torch.nan_to_num(step_loss, nan=0.0, posinf=1.0, neginf=0.0))

    if not losses:
        return next(expert.parameters()).new_tensor(0.0).requires_grad_(True)
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
    h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
    teacher_embeddings = torch.nan_to_num(
        teacher_embeddings, nan=0.0, posinf=1.0, neginf=-1.0
    )
    step_embeddings = [
        torch.nan_to_num(step_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        for step_matrix in step_embeddings
    ]
    z = expert.encode(h)
    z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)

    loss_proto  = subskill_prototype_loss(expert, h, prototype_ids, margin=margin_proto)
    loss_struct = pairwise_structure_loss(expert, z, teacher_embeddings)
    loss_order  = step_ordering_loss(expert, step_embeddings, margin=margin_order)

    total = (
        lambda_proto  * loss_proto
      + lambda_struct * loss_struct
      + lambda_order  * loss_order
    )

    total = torch.nan_to_num(total, nan=0.0, posinf=1.0, neginf=0.0)

    metrics = {
        "loss":         float(total.detach().item()),
        "loss_proto":   float(loss_proto.detach().item()),
        "loss_struct":  float(loss_struct.detach().item()),
        "loss_order":   float(loss_order.detach().item()),
    }
    return total, metrics
