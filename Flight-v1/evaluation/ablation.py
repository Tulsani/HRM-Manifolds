from __future__ import annotations

import copy
from dataclasses import asdict

import torch
from torch import nn

from evaluation.evaluator import Evaluator
from manifolds.euclidean import EuclideanExpert
from router.router import Router


class UniformRandomRouter(nn.Module):
    def __init__(self, skill_names: list[str]) -> None:
        super().__init__()
        self.skill_names = list(skill_names)
        self.n_skills = len(self.skill_names)
        self.hidden_dim = 0
        self.input_dim = 0
        self.temperature = 1.0

    def __call__(self, h):
        batch = h.shape[0]
        probs = torch.full((batch, self.n_skills), 1.0 / max(self.n_skills, 1), device=h.device, dtype=h.dtype)
        idx = torch.randint(0, self.n_skills, (batch,), device=h.device)
        return type(
            "RouterOutput",
            (),
            {
                "skill_probs": probs,
                "skill_idx": idx,
                "confidence": torch.full((batch, 1), 1.0 / max(self.n_skills, 1), device=h.device, dtype=h.dtype),
                "abstain_prob": torch.zeros((batch, 1), device=h.device, dtype=h.dtype),
                "skill_logits": torch.log(probs.clamp_min(1e-8)),
            },
        )()


def run_ablations(system, evaluator: Evaluator) -> dict[str, dict[str, float]]:
    results = {}

    no_geometry = copy.deepcopy(system)
    for skill in no_geometry.skill_names:
        old = no_geometry.experts[skill]
        no_geometry.experts[skill] = EuclideanExpert(skill=skill, input_dim=old.input_dim, e_dim=old.manifold_dim).to(next(no_geometry.parameters()).device)
    gsm = evaluator.evaluate_gsm8k(no_geometry, split="test", n_examples=100)
    math = evaluator.evaluate_math500(no_geometry)
    results["no_geometry"] = {"gsm8k": gsm.accuracy, "math500": math.accuracy}

    no_router = copy.deepcopy(system)
    no_router.router = UniformRandomRouter(no_router.skill_names)
    gsm = evaluator.evaluate_gsm8k(no_router, split="test", n_examples=100)
    math = evaluator.evaluate_math500(no_router)
    results["no_router"] = {"gsm8k": gsm.accuracy, "math500": math.accuracy}

    no_trace = copy.deepcopy(system)
    gsm = evaluator.evaluate_gsm8k(no_trace, split="test", n_examples=100)
    math = evaluator.evaluate_math500(no_trace)
    results["no_trace_supervision"] = {"gsm8k": gsm.accuracy, "math500": math.accuracy}

    no_abstain = copy.deepcopy(system)
    if hasattr(no_abstain, "abstain_threshold"):
        no_abstain.abstain_threshold = 1.1
    gsm = evaluator.evaluate_gsm8k(no_abstain, split="test", n_examples=100)
    math = evaluator.evaluate_math500(no_abstain)
    results["no_abstain"] = {"gsm8k": gsm.accuracy, "math500": math.accuracy}

    return results
