from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.ablation import run_ablations
from evaluation.comparison import compare_baselines
from evaluation.evaluator import EvalResult, Evaluator
from training.distill_dataset import DistillDataset


class FinalEvaluator:
    def __init__(self, device: str = "cuda") -> None:
        self.evaluator = Evaluator(device=device)

    def evaluate_gsm8k_full(self, system) -> EvalResult:
        return self.evaluator.evaluate_gsm8k(system, split="test", n_examples=1319)

    def evaluate_math500_full(self, system) -> EvalResult:
        return self.evaluator.evaluate_math500(system)

    def evaluate_subskill_calibration(self, system) -> dict[str, dict[str, float]]:
        dataset = DistillDataset()
        routing_accuracy = self.evaluator.evaluate_subskill_calibration(system, dataset)
        answer_accuracy = {
            skill: float(value)
            for skill, value in self.evaluate_gsm8k_full(system).per_skill.items()
        }
        return {
            skill: {
                "routing_accuracy": float(routing_accuracy.get(skill, 0.0)),
                "answer_accuracy": float(answer_accuracy.get(skill, 0.0)),
                "calibration_gap": abs(float(routing_accuracy.get(skill, 0.0)) - float(answer_accuracy.get(skill, 0.0))),
            }
            for skill in system.skill_names
        }

    def evaluate_ood(self, system) -> EvalResult:
        return self.evaluator.evaluate_ood(system)

    def run_all(self, system) -> dict[str, Any]:
        gsm8k = self.evaluate_gsm8k_full(system)
        math500 = self.evaluate_math500_full(system)
        subskill = self.evaluate_subskill_calibration(system)
        ood = self.evaluate_ood(system)
        ablations = run_ablations(system, self.evaluator)
        baselines = compare_baselines(system, self.evaluator)
        return {
            "gsm8k": gsm8k.__dict__,
            "math500": math500.__dict__,
            "subskill_calibration": subskill,
            "ood": ood.__dict__,
            "ablations": ablations,
            "baselines": baselines,
        }
