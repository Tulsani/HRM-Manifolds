from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class SystemCalibrator:
    def __init__(self, search_steps: int = 30, temp_range: tuple[float, float] = (0.5, 3.0)) -> None:
        self.search_steps = int(search_steps)
        self.temp_range = (float(temp_range[0]), float(temp_range[1]))

    @torch.no_grad()
    def fit_head_temperatures(self, system, val_loader) -> dict[str, float]:
        device = next(system.parameters()).device
        logits_by_skill: dict[str, list[torch.Tensor]] = {skill: [] for skill in system.skill_names}
        targets_by_skill: dict[str, list[torch.Tensor]] = {skill: [] for skill in system.skill_names}
        was_training = system.training
        system.eval()
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            output = system(input_ids, attention_mask=attention_mask, routing_mode="hard")
            active_skill = output.active_skill
            last_indices = target_ids.ne(system.tokenizer.pad_token_id).sum(dim=1).clamp_min(1) - 1
            batch_indices = torch.arange(target_ids.size(0), device=device)
            next_token_targets = target_ids[batch_indices, last_indices]
            logits_by_skill[active_skill].append(output.logits[batch_indices, last_indices].detach())
            targets_by_skill[active_skill].append(next_token_targets.detach())
        if was_training:
            system.train()

        temps = torch.linspace(self.temp_range[0], self.temp_range[1], steps=self.search_steps)
        head_temps: dict[str, float] = {}
        for skill in system.skill_names:
            if not logits_by_skill[skill]:
                head_temps[skill] = 1.0
                continue
            logits = torch.cat(logits_by_skill[skill], dim=0)
            targets = torch.cat(targets_by_skill[skill], dim=0)
            best_temp = 1.0
            best_nll = float("inf")
            for temp in temps:
                nll = F.cross_entropy(logits / temp, targets).item()
                if nll < best_nll:
                    best_nll = nll
                    best_temp = float(temp.item())
            head_temps[skill] = best_temp
        return head_temps

    def apply_temperatures(self, system, temperatures: dict[str, float]) -> None:
        for skill, head in system.heads.items():
            value = torch.tensor(float(temperatures.get(skill, 1.0)))
            if hasattr(head, "temperature"):
                head.temperature.copy_(value)
            else:
                head.register_buffer("temperature", value)

