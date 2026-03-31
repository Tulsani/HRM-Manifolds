from __future__ import annotations

from typing import Any

import torch


def _fit_isotonic_pairs(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    if not xs:
        return [(0.0, 0.5)]
    pairs = sorted(zip(xs, ys), key=lambda item: item[0])
    blocks = [[x, x, y, 1] for x, y in pairs]
    idx = 0
    while idx < len(blocks) - 1:
        if blocks[idx][2] <= blocks[idx + 1][2]:
            idx += 1
            continue
        left = blocks[idx]
        right = blocks[idx + 1]
        total_n = left[3] + right[3]
        avg = (left[2] * left[3] + right[2] * right[3]) / total_n
        merged = [left[0], right[1], avg, total_n]
        blocks[idx : idx + 2] = [merged]
        idx = max(idx - 1, 0)
    return [(block[1], float(block[2])) for block in blocks]


class ExpertCalibrator:
    def fit(self, system, val_loader) -> dict[str, dict]:
        device = next(system.parameters()).device
        distances: dict[str, list[float]] = {skill: [] for skill in system.skill_names}
        correctness: dict[str, list[float]] = {skill: [] for skill in system.skill_names}
        was_training = system.training
        system.eval()
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            with torch.no_grad():
                hidden_states, _ = system.backbone(input_ids, attention_mask=attention_mask)
                pooled = hidden_states.mean(dim=1)
                router_out = system.router(hidden_states)
                active_idx = router_out.skill_idx
                output = system(input_ids, attention_mask=attention_mask, routing_mode="hard")
                last_indices = target_ids.ne(system.tokenizer.pad_token_id).sum(dim=1).clamp_min(1) - 1
                batch_indices = torch.arange(target_ids.size(0), device=device)
                preds = output.logits[batch_indices, last_indices].argmax(dim=-1)
                gold = target_ids[batch_indices, last_indices]
            for skill_idx, skill in enumerate(system.skill_names):
                mask = active_idx.eq(skill_idx)
                if not mask.any():
                    continue
                expert = system.experts[skill]
                z = expert.encode(pooled[mask])
                if hasattr(expert, "prototype_memory"):
                    _, dists = expert.prototype_memory.assign(z, getattr(expert, "curvature", 1.0))
                    proto_dist = dists.min(dim=-1).values
                elif hasattr(expert, "h_branch") and hasattr(expert.h_branch, "prototype_memory"):
                    z_h = expert.h_branch.encode(pooled[mask])
                    _, dists = expert.h_branch.prototype_memory.assign(z_h, getattr(expert.h_branch, "curvature", 1.0))
                    proto_dist = dists.min(dim=-1).values
                else:
                    proto_dist = z.norm(dim=-1)
                distances[skill].extend(proto_dist.detach().cpu().tolist())
                correctness[skill].extend(preds[mask].eq(gold[mask]).float().cpu().tolist())
        if was_training:
            system.train()
        result = {}
        for skill in system.skill_names:
            mapping = _fit_isotonic_pairs(distances[skill], correctness[skill])
            result[skill] = {
                "isotonic_mapping": mapping,
                "distance_thresholds": [pair[0] for pair in mapping],
            }
        return result

