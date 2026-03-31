from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset

from evaluation.metrics import extract_gsm8k_answer, extract_math_answer, loose_match


@dataclass
class EvalResult:
    accuracy: float
    n_correct: int
    n_total: int
    per_skill: dict[str, float]
    abstain_rate: float
    avg_router_conf: float


class Evaluator:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"

    def _run_generation(self, system, prompt: str, max_new_tokens: int = 128) -> tuple[str, str, bool, float]:
        encoded = system.tokenizer.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        encoded = {key: value.to(next(system.parameters()).device) for key, value in encoded.items()}
        output = system(encoded["input_ids"], encoded.get("attention_mask"))
        generated = system.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text = system.tokenizer.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text, output.active_skill, output.abstained, output.router_conf

    def evaluate_gsm8k(self, system, split: str = "test", n_examples: int = 500) -> EvalResult:
        dataset = load_dataset("gsm8k", "main", split=split)
        n_total = min(len(dataset), n_examples)
        n_correct = 0
        abstains = 0
        confs = []
        per_skill_total = {skill: 0 for skill in system.skill_names}
        per_skill_correct = {skill: 0 for skill in system.skill_names}
        for idx in range(n_total):
            example = dataset[idx]
            text, skill, abstained, conf = self._run_generation(system, example["question"])
            pred = extract_gsm8k_answer(text)
            gold = extract_gsm8k_answer(example["answer"])
            is_correct = pred is not None and gold is not None and loose_match(pred, gold)
            n_correct += int(is_correct)
            abstains += int(abstained)
            confs.append(conf)
            per_skill_total[skill] += 1
            per_skill_correct[skill] += int(is_correct)
        return EvalResult(
            accuracy=float(n_correct / max(n_total, 1)),
            n_correct=n_correct,
            n_total=n_total,
            per_skill={
                skill: float(per_skill_correct[skill] / per_skill_total[skill]) if per_skill_total[skill] > 0 else 0.0
                for skill in system.skill_names
            },
            abstain_rate=float(abstains / max(n_total, 1)),
            avg_router_conf=float(sum(confs) / max(len(confs), 1)),
        )

    def evaluate_math500(self, system) -> EvalResult:
        dataset = load_dataset("hendrycks/competition_math", split="test")
        n_total = min(len(dataset), 500)
        n_correct = 0
        abstains = 0
        confs = []
        per_skill_total = {skill: 0 for skill in system.skill_names}
        per_skill_correct = {skill: 0 for skill in system.skill_names}
        for idx in range(n_total):
            example = dataset[idx]
            text, skill, abstained, conf = self._run_generation(system, example["problem"])
            pred = extract_math_answer(text)
            gold = extract_math_answer(example["solution"])
            is_correct = loose_match(pred, gold)
            n_correct += int(is_correct)
            abstains += int(abstained)
            confs.append(conf)
            per_skill_total[skill] += 1
            per_skill_correct[skill] += int(is_correct)
        return EvalResult(
            accuracy=float(n_correct / max(n_total, 1)),
            n_correct=n_correct,
            n_total=n_total,
            per_skill={
                skill: float(per_skill_correct[skill] / per_skill_total[skill]) if per_skill_total[skill] > 0 else 0.0
                for skill in system.skill_names
            },
            abstain_rate=float(abstains / max(n_total, 1)),
            avg_router_conf=float(sum(confs) / max(len(confs), 1)),
        )

    def evaluate_subskill_calibration(self, system, dataset) -> dict[str, Any]:
        total = {skill: 0 for skill in system.skill_names}
        routed = {skill: 0 for skill in system.skill_names}
        for item in dataset:
            prompt = item["problem_text"]
            encoded = system.tokenizer.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            encoded = {key: value.to(next(system.parameters()).device) for key, value in encoded.items()}
            output = system(encoded["input_ids"], encoded.get("attention_mask"))
            gold_skill = dataset.skill_names[int(item["skill_label"])]
            total[gold_skill] += 1
            routed[gold_skill] += int(output.active_skill == gold_skill)
        return {
            skill: float(routed[skill] / total[skill]) if total[skill] > 0 else 0.0
            for skill in system.skill_names
        }

    def evaluate_ood(self, system) -> EvalResult:
        return EvalResult(
            accuracy=0.0,
            n_correct=0,
            n_total=0,
            per_skill={skill: 0.0 for skill in system.skill_names},
            abstain_rate=0.0,
            avg_router_conf=0.0,
        )
