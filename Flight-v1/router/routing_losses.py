from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from router.router import RouterOutput


def skill_classification_loss(
    output: RouterOutput,
    skill_labels: Tensor,
    low_confidence: Tensor,
    incorrect: Tensor,
) -> Tensor:
    per_sample = F.cross_entropy(output.skill_logits, skill_labels, reduction="none")
    scale = torch.where(low_confidence | incorrect, 0.5, 1.0).to(per_sample.dtype)
    return (per_sample * scale).mean()


def build_subskill_target_distribution(
    subskill_multihot: Tensor,
    skill_subskill_matrix: Tensor,
) -> Tensor:
    overlap = subskill_multihot.float() @ skill_subskill_matrix.float().T
    empty_mask = overlap.sum(dim=-1, keepdim=True).eq(0)
    if empty_mask.any():
        overlap = overlap + empty_mask.float()
    return overlap / overlap.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def subskill_consistency_loss(output: RouterOutput, subskill_target_dist: Tensor) -> Tensor:
    log_probs = torch.log_softmax(output.skill_logits, dim=-1)
    return F.kl_div(log_probs, subskill_target_dist, reduction="batchmean")


def confidence_calibration_loss(output: RouterOutput, skill_labels: Tensor) -> Tensor:
    correct_mask = output.skill_idx.eq(skill_labels).float()
    return F.mse_loss(output.confidence.squeeze(-1), correct_mask)


def entropy_regularization_loss(output: RouterOutput, n_skills: int) -> Tensor:
    entropy = -(output.skill_probs * torch.log(output.skill_probs.clamp_min(1e-8))).sum(dim=-1)
    target_entropy = output.skill_probs.new_tensor(math.log(max(n_skills, 1)) * 0.5)
    return F.mse_loss(entropy.mean(), target_entropy)


def compute_router_loss(
    output: RouterOutput,
    skill_labels: Tensor,
    low_confidence: Tensor,
    incorrect: Tensor,
    subskill_target_dist: Tensor,
    n_skills: int,
    lambda_cls: float = 1.0,
    lambda_sub: float = 0.3,
    lambda_cal: float = 0.5,
    lambda_ent: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    loss_cls = skill_classification_loss(output, skill_labels, low_confidence, incorrect)
    loss_sub = subskill_consistency_loss(output, subskill_target_dist)
    loss_cal = confidence_calibration_loss(output, skill_labels)
    loss_ent = entropy_regularization_loss(output, n_skills)
    total = (
        lambda_cls * loss_cls
        + lambda_sub * loss_sub
        + lambda_cal * loss_cal
        + lambda_ent * loss_ent
    )
    metrics = {
        "loss_cls": float(loss_cls.detach().item()),
        "loss_sub": float(loss_sub.detach().item()),
        "loss_cal": float(loss_cal.detach().item()),
        "loss_ent": float(loss_ent.detach().item()),
        "loss_total": float(total.detach().item()),
    }
    return total, metrics
