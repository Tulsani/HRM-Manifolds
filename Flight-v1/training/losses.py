from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def shift_for_causal_lm(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()


def causal_lm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    shifted_logits, shifted_labels = shift_for_causal_lm(logits, labels)
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def kd_kl_divergence(student_logits: Tensor, teacher_logits: Tensor, labels: Tensor, temperature: float = 4.0) -> Tensor:
    shifted_student, shifted_labels = shift_for_causal_lm(student_logits, labels)
    shifted_teacher, _ = shift_for_causal_lm(teacher_logits, labels)
    valid_mask = shifted_labels.ne(-100)
    if not valid_mask.any():
        return shifted_student.new_tensor(0.0)

    student_log_probs = F.log_softmax(shifted_student / temperature, dim=-1)
    teacher_probs = F.softmax(shifted_teacher / temperature, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    token_kl = token_kl.masked_select(valid_mask)
    return token_kl.mean() * (temperature ** 2)


def compute_backbone_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    alpha_lm: float = 0.5,
    beta_kd: float = 0.5,
    kd_temperature: float = 4.0,
) -> tuple[Tensor, dict[str, float]]:
    lm = causal_lm_loss(student_logits, labels)
    kd = kd_kl_divergence(student_logits, teacher_logits, labels, temperature=kd_temperature)
    total = alpha_lm * lm + beta_kd * kd
    metrics = {
        "loss": float(total.detach().item()),
        "loss_lm": float(lm.detach().item()),
        "loss_kd": float(kd.detach().item()),
    }
    return total, metrics
