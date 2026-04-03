from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from manifolds.ops import expmap0


@dataclass
class LossOutput:
    total: Tensor
    l_task: Tensor
    l_kd: Tensor
    l_trace: Tensor
    l_geom: Tensor
    l_proto: Tensor
    l_diff: Tensor
    l_sep: Tensor


def lambda_ramp(step: int, ramp_steps: int) -> float:
    if ramp_steps <= 0:
        return 1.0
    return float(min(max(step, 0) / ramp_steps, 1.0))


class DistillLoss(nn.Module):
    def __init__(
        self,
        pad_token_id: int,
        kd_temperature: float = 4.0,
        lambda_1: float = 1.0,
        lambda_2: float = 0.7,
        lambda_3: float = 0.5,
        lambda_4: float = 0.3,
    ) -> None:
        super().__init__()
        self.pad_token_id = int(pad_token_id)
        self.kd_temperature = float(kd_temperature)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.lambda_3 = float(lambda_3)
        self.lambda_4 = float(lambda_4)

    def _task_loss(self, student_logits: Tensor, target_ids: Tensor) -> Tensor:
        T = min(student_logits.size(1), target_ids.size(1))
        return F.cross_entropy(
            student_logits[:, :T, :].reshape(-1, student_logits.size(-1)),
            target_ids[:, :T].reshape(-1),
            ignore_index=self.pad_token_id,
        )

    def _kd_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        target_ids: Tensor,
        sample_weights: Tensor,
    ) -> Tensor:
        T_student = student_logits.size(1)
        last_indices = (
            target_ids[:, :T_student].ne(self.pad_token_id).sum(dim=1).clamp_min(1) - 1
        ).clamp(max=T_student - 1)
        batch_indices = torch.arange(
            target_ids.size(0), device=target_ids.device
        )
        student_final = student_logits[batch_indices, last_indices]
        vocab_student = student_final.size(-1)
        teacher_trimmed = teacher_logits[:, :vocab_student]
        # Clamp logits before softmax to prevent overflow in bfloat16 backward
        teacher_trimmed = teacher_trimmed.float().clamp(-30.0, 30.0)
        student_final = student_final.float().clamp(-30.0, 30.0)

        teacher_dist = torch.softmax(
            teacher_trimmed / self.kd_temperature, dim=-1
        )
        student_log_dist = torch.log_softmax(
            student_final / self.kd_temperature, dim=-1
        )
        per_sample = (
            F.kl_div(student_log_dist, teacher_dist, reduction="none").sum(dim=-1)
            * (self.kd_temperature ** 2)
        )
        loss = (per_sample * sample_weights.float()).mean()
        return torch.nan_to_num(loss, nan=0.0)

    def _trace_loss(
        self,
        student_hidden: Tensor,
        step_embeddings: Tensor,
        step_mask: Tensor,
    ) -> Tensor:
        losses: list[Tensor] = []
        for idx in range(student_hidden.size(0)):
            valid_steps = int(step_mask[idx].sum().item())
            if valid_steps < 2:
                continue
            teacher_steps = step_embeddings[idx, :valid_steps]
            student_steps = student_hidden[idx, :valid_steps]
            teacher_sim = (
                F.normalize(teacher_steps, dim=-1)
                @ F.normalize(teacher_steps, dim=-1).T
            )
            student_sim = (
                F.normalize(student_steps, dim=-1)
                @ F.normalize(student_steps, dim=-1).T
            )
            losses.append(F.mse_loss(student_sim, teacher_sim))
        if not losses:
            return student_hidden.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _difficulty_loss(
        self, expert, z_s: Tensor, difficulty: Tensor
    ) -> Tensor:
        centroid = z_s.mean(dim=0, keepdim=True)
        d_to_centroid = (
            expert.distance(z_s, centroid).reshape(z_s.size(0), -1).mean(dim=-1)
        )
        norm_dist = d_to_centroid / d_to_centroid.max().clamp_min(1e-8)
        targets = difficulty.float() / 2.0
        return F.mse_loss(norm_dist, targets)

    def _incorrect_separation_loss(
        self,
        expert,
        z_s: Tensor,
        subskill_labels: Tensor,
        incorrect_mask: Tensor,
        margin: float = 1.0,
    ) -> Tensor:
        primary_subskill = subskill_labels[:, 0]
        losses: list[Tensor] = []
        for idx in range(z_s.size(0)):
            if not bool(incorrect_mask[idx]) or int(primary_subskill[idx].item()) < 0:
                continue
            same_mask = primary_subskill.eq(primary_subskill[idx]) & (~incorrect_mask)
            if not same_mask.any():
                continue
            distances = expert.distance(
                z_s[idx : idx + 1], z_s[same_mask]
            ).reshape(-1)
            losses.append(
                torch.relu(z_s.new_tensor(margin) - distances).mean()
            )
        if not losses:
            return z_s.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _geom_loss(
        self,
        expert,
        z_s: Tensor,
        subskill_labels: Tensor,
        difficulty: Tensor,
        incorrect_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if hasattr(expert, "prototype_memory"):
            primary = subskill_labels[:, 0].clamp_min(0)
            n_prototypes = int(expert.prototype_memory.n_prototypes)
            proto_ids = torch.remainder(primary, n_prototypes)
            l_proto = expert.prototype_memory.prototype_loss(
                z_s, proto_ids, getattr(expert, "curvature", 1.0)
            )
            if not torch.isfinite(l_proto):
                l_proto = z_s.new_tensor(0.0)
        elif hasattr(expert, "h_branch") and hasattr(
            expert.h_branch, "prototype_memory"
        ):
            primary = subskill_labels[:, 0].clamp_min(0)
            n_prototypes = int(expert.h_branch.prototype_memory.n_prototypes)
            proto_ids = torch.remainder(primary, n_prototypes)
            h_dim = int(expert.h_branch.manifold_dim)
            z_h = expmap0(
                z_s[:, :h_dim],
                getattr(expert.h_branch, "curvature", 1.0),
            )
            l_proto = expert.h_branch.prototype_memory.prototype_loss(
                z_h,
                proto_ids,
                getattr(expert.h_branch, "curvature", 1.0),
            )
            if not torch.isfinite(l_proto):
                l_proto = z_s.new_tensor(0.0)
        else:
            l_proto = z_s.new_tensor(0.0)

        l_diff = self._difficulty_loss(expert, z_s, difficulty)
        if not torch.isfinite(l_diff):
            l_diff = z_s.new_tensor(0.0)

        l_sep = self._incorrect_separation_loss(
            expert, z_s, subskill_labels, incorrect_mask
        )
        if not torch.isfinite(l_sep):
            l_sep = z_s.new_tensor(0.0)

        l_geom = l_proto + 0.5 * l_diff + 0.5 * l_sep
        if not torch.isfinite(l_geom):
            l_geom = z_s.new_tensor(0.0)

        return l_geom, l_proto, l_diff, l_sep

    def forward(
        self,
        student_logits: Tensor,
        target_ids: Tensor,
        teacher_logits: Tensor,
        student_hidden: Tensor,
        step_embeddings: Tensor,
        z_s: Tensor,
        skill_label: Tensor,
        subskill_labels: Tensor,
        difficulty: Tensor,
        incorrect_mask: Tensor,
        router_weights: Tensor,
        expert,
        sample_weights: Tensor,
        step_mask: Tensor | None = None,
        effective_lambda_3: float | None = None,
        effective_lambda_4: float | None = None,
    ) -> LossOutput:
        if step_mask is None:
            step_mask = step_embeddings.abs().sum(dim=-1).ne(0.0)

        l_task = self._task_loss(student_logits, target_ids)
        l_kd = self._kd_loss(
            student_logits, teacher_logits, target_ids, sample_weights
        )
        l_trace = self._trace_loss(student_hidden, step_embeddings, step_mask)
        l_geom, l_proto, l_diff, l_sep = self._geom_loss(
            expert, z_s, subskill_labels, difficulty, incorrect_mask
        )

        # Guard all loss terms before combining into total
        l_task  = torch.nan_to_num(l_task,  nan=0.0)
        l_kd    = torch.nan_to_num(l_kd,    nan=0.0)
        l_trace = torch.nan_to_num(l_trace, nan=0.0)
        l_geom  = torch.nan_to_num(l_geom,  nan=0.0)
        l_proto = torch.nan_to_num(l_proto, nan=0.0)
        l_diff  = torch.nan_to_num(l_diff,  nan=0.0)
        l_sep   = torch.nan_to_num(l_sep,   nan=0.0)

        route_scale = router_weights.gather(
            1, skill_label.unsqueeze(1)
        ).mean()
        lambda_3 = (
            self.lambda_3 if effective_lambda_3 is None else float(effective_lambda_3)
        )
        lambda_4 = (
            self.lambda_4 if effective_lambda_4 is None else float(effective_lambda_4)
        )
        total = route_scale * (
            self.lambda_1 * l_task
            + self.lambda_2 * l_kd
            + lambda_3 * l_trace
            + lambda_4 * l_geom
        )
        return LossOutput(
            total=total,
            l_task=l_task,
            l_kd=l_kd,
            l_trace=l_trace,
            l_geom=l_geom,
            l_proto=l_proto,
            l_diff=l_diff,
            l_sep=l_sep,
        )