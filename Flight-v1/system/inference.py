from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch

from evaluation.metrics import extract_gsm8k_answer, extract_math_answer
from system.student_system import StudentSystem


@dataclass
class InferenceResult:
    problem: str
    generation: str
    answer: str | None
    skill_used: str
    router_conf: float
    abstained: bool
    routing_dist: dict[str, float]


class SkillDistillInference:
    def __init__(self, checkpoint_path: str, backbone_config_path: str = "configs/backbone_small.yaml", device: str = "cuda") -> None:
        actual_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.device = torch.device(actual_device)
        self.system = StudentSystem.from_full_checkpoint(
            checkpoint_path=checkpoint_path,
            backbone_config_path=backbone_config_path,
            device=self.device,
        )
        self.system.eval()

    def _tokenize_problem(self, problem: str) -> dict[str, torch.Tensor]:
        encoded = self.system.tokenizer.tokenizer(problem, return_tensors="pt", add_special_tokens=True)
        return {key: value.to(self.device) for key, value in encoded.items()}

    @torch.no_grad()
    def solve(self, problem: str, max_tokens: int = 256, show_routing: bool = False) -> InferenceResult:
        encoded = self._tokenize_problem(problem)
        generated = self.system.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        output = self.system(encoded["input_ids"], encoded.get("attention_mask"))
        generation = self.system.tokenizer.tokenizer.decode(generated[0], skip_special_tokens=True)
        answer = extract_math_answer(generation) or (
            str(extract_gsm8k_answer(generation)) if extract_gsm8k_answer(generation) is not None else None
        )
        routing_dist = {
            skill: float(prob.item())
            for skill, prob in zip(self.system.skill_names, output.skill_probs[0])
        } if show_routing else {}
        return InferenceResult(
            problem=problem,
            generation=generation,
            answer=answer,
            skill_used=output.active_skill,
            router_conf=output.router_conf,
            abstained=output.abstained,
            routing_dist=routing_dist,
        )

    @torch.no_grad()
    def batch_solve(self, problems: list[str], batch_size: int = 8) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        for start in range(0, len(problems), batch_size):
            for problem in problems[start : start + batch_size]:
                results.append(self.solve(problem))
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained Skill Distillation system.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to full system checkpoint.")
    parser.add_argument("--problem", type=str, required=True, help="Math reasoning problem to solve.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference = SkillDistillInference(checkpoint_path=args.checkpoint, device=args.device)
    result = inference.solve(args.problem, show_routing=True)
    print(f"Generation: {result.generation}")
    print(f"Answer: {result.answer}")
    print(f"Skill used: {result.skill_used}")
    print(f"Router confidence: {result.router_conf:.4f}")
    print(f"Abstained: {result.abstained}")
    print(f"Routing distribution: {result.routing_dist}")


if __name__ == "__main__":
    main()
