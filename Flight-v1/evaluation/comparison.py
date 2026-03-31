from __future__ import annotations

import copy
import time

import torch
from torch import nn

from evaluation.evaluator import Evaluator
from manifolds.euclidean import EuclideanExpert


class BackboneOnlySystem(nn.Module):
    def __init__(self, full_system) -> None:
        super().__init__()
        self.backbone = full_system.backbone
        self.tokenizer = full_system.tokenizer
        self.skill_names = list(full_system.skill_names)
        self.head = next(iter(full_system.heads.values()))

    def forward(self, input_ids, attention_mask=None, routing_mode="soft"):
        hidden, logits = self.backbone(input_ids, attention_mask=attention_mask)
        pooled = hidden.mean(dim=1)
        next_logits = self.head(pooled).unsqueeze(1).repeat(1, input_ids.size(1), 1)
        return type("SystemOutput", (), {"logits": next_logits, "skill_probs": None, "active_skill": self.skill_names[0], "z": pooled, "abstained": False, "router_conf": 1.0})()

    def generate(self, input_ids, attention_mask=None, max_new_tokens=64, do_sample=False, temperature=1.0):
        return self.backbone.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, eos_token_id=getattr(self.tokenizer, "eos_token_id", None))


class SingleEuclideanSystem(nn.Module):
    def __init__(self, full_system) -> None:
        super().__init__()
        self.backbone = full_system.backbone
        self.tokenizer = full_system.tokenizer
        self.skill_names = list(full_system.skill_names)
        self.expert = EuclideanExpert(skill="single", input_dim=full_system.backbone.config.d_model, e_dim=32)
        self.head = next(iter(full_system.heads.values()))

    def forward(self, input_ids, attention_mask=None, routing_mode="soft"):
        hidden, _ = self.backbone(input_ids, attention_mask=attention_mask)
        z = self.expert.encode(hidden.mean(dim=1))
        next_logits = self.head(z).unsqueeze(1).repeat(1, input_ids.size(1), 1)
        return type("SystemOutput", (), {"logits": next_logits, "skill_probs": None, "active_skill": "single", "z": z, "abstained": False, "router_conf": 1.0})()

    def generate(self, input_ids, attention_mask=None, max_new_tokens=64, do_sample=False, temperature=1.0):
        generated = input_ids
        for _ in range(max_new_tokens):
            output = self.forward(generated, attention_mask=attention_mask)
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


def compare_baselines(system, evaluator: Evaluator) -> dict[str, dict[str, float]]:
    results = {}
    for name, baseline in {
        "backbone_only": BackboneOnlySystem(system),
        "single_euclidean": SingleEuclideanSystem(system),
        "full_system": system,
    }.items():
        start = time.perf_counter()
        gsm = evaluator.evaluate_gsm8k(baseline, split="test", n_examples=50)
        math = evaluator.evaluate_math500(baseline)
        elapsed = time.perf_counter() - start
        results[name] = {
            "gsm8k": gsm.accuracy,
            "math500": math.accuracy,
            "speed_ms_per_example": float((elapsed / max(gsm.n_total + math.n_total, 1)) * 1000.0),
        }
    return results
