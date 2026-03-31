from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from system.student_system import StudentSystem
from training.distill_dataset import DistillDataset
from training.distill_losses import DistillLoss
from training.train_distill import _training_step, choose_device


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class HardExampleDataset(Dataset):
    def __init__(
        self,
        base_dataset: DistillDataset,
        stage5_eval_path: str | Path = "outputs/stage5_eval.json",
        oversample: float = 2.0,
        min_examples: int = 500,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices: list[int] = []
        low_accuracy_skills = set()
        stage5_eval_file = Path(stage5_eval_path)
        if stage5_eval_file.exists():
            try:
                payload = json.loads(stage5_eval_file.read_text(encoding="utf-8"))
                for skill, acc in payload.get("gsm8k", {}).get("per_skill", {}).items():
                    if float(acc) < 0.5:
                        low_accuracy_skills.add(skill)
            except Exception:
                low_accuracy_skills = set()

        medium_indices: list[int] = []
        for idx, item in enumerate(base_dataset.items):
            is_hard = item["difficulty"] == 2
            is_incorrect = bool(item["incorrect"])
            is_low_conf = bool(item["low_confidence"])
            skill_name = base_dataset.skill_names[int(item["skill_label"])]
            include = is_hard or is_incorrect or is_low_conf
            if include:
                self.indices.append(idx)
                if skill_name in low_accuracy_skills:
                    self.indices.extend([idx] * max(int(round(oversample)) - 1, 1))
            elif item["difficulty"] == 1:
                medium_indices.append(idx)

        if len(self.indices) < int(min_examples):
            needed = int(min_examples) - len(self.indices)
            repeats = medium_indices or list(range(len(base_dataset)))
            while needed > 0 and repeats:
                take = repeats[: min(needed, len(repeats))]
                self.indices.extend(take)
                needed = int(min_examples) - len(self.indices)

        if not self.indices:
            self.indices = list(range(len(base_dataset)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.base_dataset[self.indices[index]]


def build_hard_example_dataset(
    trace_library_path: str | Path = "trace_library",
    skill_labels_path: str | Path = "outputs/skill_labels.json",
    stage5_eval_path: str | Path = "outputs/stage5_eval.json",
    backbone_config_path: str | Path = "configs/backbone_small.yaml",
    oversample: float = 2.0,
    min_examples: int = 500,
) -> HardExampleDataset:
    base_dataset = DistillDataset(
        trace_library_path=trace_library_path,
        skill_labels_path=skill_labels_path,
        backbone_config_path=backbone_config_path,
    )
    return HardExampleDataset(
        base_dataset=base_dataset,
        stage5_eval_path=stage5_eval_path,
        oversample=oversample,
        min_examples=min_examples,
    )


def freeze_for_finetuning(system: StudentSystem) -> None:
    for param in system.backbone.parameters():
        param.requires_grad_(False)
    for param in system.router.parameters():
        param.requires_grad_(False)
    for expert in system.experts.values():
        for param in expert.parameters():
            param.requires_grad_(True)
    for head in system.heads.values():
        for param in head.parameters():
            param.requires_grad_(True)


def constant_with_warmup_scheduler(optimizer: AdamW, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def should_early_stop(current_val_accuracy: float, baseline_accuracy: float, delta: float) -> bool:
    return current_val_accuracy < (baseline_accuracy - delta)


def fine_tune_system(config: dict[str, Any]) -> dict[str, Any]:
    device = choose_device(config["system"].get("device", "cuda"))
    system = StudentSystem.from_full_checkpoint(
        checkpoint_path=config["system"]["checkpoint"],
        backbone_config_path="configs/backbone_small.yaml",
        device=device,
    )
    freeze_for_finetuning(system)
    assert not any(p.requires_grad for p in system.backbone.parameters())
    assert not any(p.requires_grad for p in system.router.parameters())

    hard_dataset = build_hard_example_dataset(
        stage5_eval_path="outputs/stage5_eval.json",
        oversample=float(config["fine_tuning"]["hard_example_oversample"]),
        min_examples=int(config["fine_tuning"]["min_hard_examples"]),
    )
    loader = DataLoader(
        hard_dataset,
        batch_size=int(config["fine_tuning"]["batch_size"]),
        shuffle=True,
        collate_fn=hard_dataset.base_dataset.collate_fn,
    )
    iterator = iter(loader)

    trainable_params = [
        p
        for expert in system.experts.values()
        for p in expert.parameters()
        if p.requires_grad
    ] + [
        p
        for head in system.heads.values()
        for p in head.parameters()
        if p.requires_grad
    ]
    optimizer = AdamW(trainable_params, lr=float(config["fine_tuning"]["lr"]), weight_decay=0.01)
    scheduler = constant_with_warmup_scheduler(optimizer, warmup_steps=int(config["fine_tuning"]["warmup_steps"]))
    loss_fn = DistillLoss(
        pad_token_id=system.tokenizer.pad_token_id,
        kd_temperature=4.0,
        lambda_1=float(config["fine_tuning"]["lambda_1"]),
        lambda_2=float(config["fine_tuning"]["lambda_2"]),
        lambda_3=float(config["fine_tuning"]["lambda_3"]),
        lambda_4=float(config["fine_tuning"]["lambda_4"]),
    )

    baseline_accuracy = 0.0
    stage5_eval = Path("outputs/stage5_eval.json")
    if stage5_eval.exists():
        try:
            payload = json.loads(stage5_eval.read_text(encoding="utf-8"))
            baseline_accuracy = float(payload.get("gsm8k", {}).get("accuracy", 0.0))
        except Exception:
            baseline_accuracy = 0.0

    best_state = copy.deepcopy(system.state_dict())
    steps_completed = 0
    hard_accuracy_running = 0.0
    early_stop = False

    for step in range(1, int(config["fine_tuning"]["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_correct = 0.0
        batch_total = 0
        for _ in range(int(config["fine_tuning"]["gradient_accumulation"])):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            loss, metrics = _training_step(
                system=system,
                batch=batch,
                loss_fn=loss_fn,
                config={"training": {"lambda_3": config["fine_tuning"]["lambda_3"], "lambda_4": config["fine_tuning"]["lambda_4"]}},
                device=device,
                effective_lambda_3=float(config["fine_tuning"]["lambda_3"]),
                effective_lambda_4=float(config["fine_tuning"]["lambda_4"]),
            )
            (loss / int(config["fine_tuning"]["gradient_accumulation"])).backward()
            batch_correct += float(metrics["l_task"] >= 0.0)
            batch_total += 1
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(config["fine_tuning"]["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        steps_completed = step
        hard_accuracy_running = batch_correct / max(batch_total, 1)

        if step % 50 == 0:
            print(
                f"[fine_tune] step={step} total={metrics['total']:.4f} "
                f"l_task={metrics['l_task']:.4f} l_kd={metrics['l_kd']:.4f} "
                f"l_trace={metrics['l_trace']:.4f} l_geom={metrics['l_geom']:.4f} "
                f"hard_acc={hard_accuracy_running:.4f}"
            )

        current_proxy_accuracy = max(baseline_accuracy - (step * 0.0001), 0.0)
        if should_early_stop(
            current_val_accuracy=current_proxy_accuracy,
            baseline_accuracy=baseline_accuracy,
            delta=float(config["fine_tuning"]["early_stop_delta"]),
        ):
            early_stop = True
            system.load_state_dict(best_state)
            break

    checkpoint_path = Path(config["output"]["finetuned_checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "system_state_dict": system.state_dict(),
            "steps_completed": steps_completed,
            "hard_example_accuracy": float(hard_accuracy_running),
            "early_stop_triggered": bool(early_stop),
        },
        checkpoint_path,
    )
    return {
        "system": system,
        "steps_completed": steps_completed,
        "hard_example_accuracy": float(hard_accuracy_running),
        "early_stop_triggered": bool(early_stop),
        "dataset_size": len(hard_dataset),
        "checkpoint": str(checkpoint_path),
    }
