from __future__ import annotations

import json
from pathlib import Path

import torch


def compute_utility(accuracy_when_answered: float, error_rate_when_answered: float, abstain_rate: float, cost_wrong: float, cost_abstain: float) -> float:
    return float(accuracy_when_answered - cost_wrong * error_rate_when_answered - cost_abstain * abstain_rate)


class ThresholdTuner:
    def tune(
        self,
        system,
        val_loader,
        cost_wrong: float = 1.0,
        cost_abstain: float = 0.3,
        search_steps: int = 17,
        threshold_range: tuple[float, float] = (0.1, 0.9),
    ) -> dict[str, float]:
        device = next(system.parameters()).device
        thresholds = torch.linspace(threshold_range[0], threshold_range[1], steps=search_steps)
        best = {
            "threshold": float(thresholds[0].item()),
            "utility_at_threshold": float("-inf"),
            "abstain_rate_at_threshold": 0.0,
            "accuracy_when_answered": 0.0,
        }
        was_training = system.training
        system.eval()
        for threshold in thresholds.tolist():
            answered = 0
            correct = 0
            abstained = 0
            total = 0
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_ids = batch["target_ids"].to(device)
                output = system(input_ids, attention_mask=attention_mask)
                last_indices = target_ids.ne(system.tokenizer.pad_token_id).sum(dim=1).clamp_min(1) - 1
                batch_indices = torch.arange(target_ids.size(0), device=device)
                preds = output.logits[batch_indices, last_indices].argmax(dim=-1)
                gold = target_ids[batch_indices, last_indices]
                for idx in range(target_ids.size(0)):
                    total += 1
                    if float(output.router_conf) <= threshold or output.abstained:
                        abstained += 1
                    else:
                        answered += 1
                        correct += int(preds[idx].item() == gold[idx].item())
            accuracy_answered = float(correct / max(answered, 1))
            error_rate = float((answered - correct) / max(answered, 1))
            abstain_rate = float(abstained / max(total, 1))
            utility = compute_utility(accuracy_answered, error_rate, abstain_rate, cost_wrong, cost_abstain)
            if utility > best["utility_at_threshold"]:
                best = {
                    "threshold": float(threshold),
                    "utility_at_threshold": float(utility),
                    "abstain_rate_at_threshold": float(abstain_rate),
                    "accuracy_when_answered": float(accuracy_answered),
                }
        if was_training:
            system.train()
        return best

    def save(self, payload: dict[str, float], path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

