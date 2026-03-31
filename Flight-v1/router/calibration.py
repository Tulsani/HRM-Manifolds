from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from router.router import Router


class RouterCalibrator:
    def __init__(self, n_bins: int = 10, temp_range: tuple[float, float] = (0.5, 5.0), temp_search_steps: int = 50) -> None:
        self.n_bins = int(n_bins)
        self.temp_range = (float(temp_range[0]), float(temp_range[1]))
        self.temp_search_steps = int(temp_search_steps)

    @torch.no_grad()
    def fit(self, router: Router, val_loader) -> float:
        device = next(router.parameters()).device
        logits_list = []
        labels_list = []
        was_training = router.training
        router.eval()
        for batch in val_loader:
            embeddings = batch["embeddings"].to(device=device, dtype=torch.float32)
            logits = router.compute_skill_logits(embeddings, apply_temperature=False)
            logits_list.append(logits)
            labels_list.append(batch["skill_labels"].to(device))
        if was_training:
            router.train()

        if not logits_list:
            router.temperature = 1.0
            return 1.0

        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        temps = torch.linspace(self.temp_range[0], self.temp_range[1], steps=self.temp_search_steps, device=device)
        best_temp = 1.0
        best_nll = float("inf")
        for temp in temps:
            nll = F.cross_entropy(logits / temp, labels).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = float(temp.item())
        router.temperature = best_temp
        return best_temp

    @torch.no_grad()
    def evaluate_calibration(self, router: Router, val_loader) -> dict[str, Any]:
        device = next(router.parameters()).device
        total = 0
        correct = 0
        confidence_list = []
        correctness_list = []
        per_skill_total = {skill: 0 for skill in router.skill_names}
        per_skill_correct = {skill: 0 for skill in router.skill_names}
        was_training = router.training
        router.eval()

        for batch in val_loader:
            embeddings = batch["embeddings"].to(device=device, dtype=torch.float32)
            labels = batch["skill_labels"].to(device)
            output = router(embeddings)
            probs = output.skill_probs
            pred_conf, preds = probs.max(dim=-1)
            matches = preds.eq(labels)

            total += labels.numel()
            correct += int(matches.sum().item())
            confidence_list.append(pred_conf.cpu())
            correctness_list.append(matches.float().cpu())
            for idx, skill_idx in enumerate(labels.tolist()):
                skill_name = router.skill_names[int(skill_idx)]
                per_skill_total[skill_name] += 1
                per_skill_correct[skill_name] += int(matches[idx].item())

        if was_training:
            router.train()

        if total == 0:
            report = {
                "temperature": float(router.temperature),
                "ece": 0.0,
                "mce": 0.0,
                "val_accuracy": 0.0,
                "per_skill_accuracy": {skill: 0.0 for skill in router.skill_names},
            }
            return report

        confidences = torch.cat(confidence_list, dim=0)
        correctness = torch.cat(correctness_list, dim=0)
        bin_edges = torch.linspace(0.0, 1.0, steps=self.n_bins + 1)
        ece = 0.0
        mce = 0.0
        for start, end in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences >= start) & (confidences < end if end < 1.0 else confidences <= end)
            if not mask.any():
                continue
            bin_acc = float(correctness[mask].mean().item())
            bin_conf = float(confidences[mask].mean().item())
            gap = abs(bin_acc - bin_conf)
            weight = float(mask.float().mean().item())
            ece += weight * gap
            mce = max(mce, gap)

        report = {
            "temperature": float(router.temperature),
            "ece": float(max(min(ece, 1.0), 0.0)),
            "mce": float(max(min(mce, 1.0), 0.0)),
            "val_accuracy": float(correct / total),
            "per_skill_accuracy": {
                skill: float(per_skill_correct[skill] / per_skill_total[skill]) if per_skill_total[skill] > 0 else 0.0
                for skill in router.skill_names
            },
        }
        return report

    def save_report(self, report: dict[str, Any], path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
