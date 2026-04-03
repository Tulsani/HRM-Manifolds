from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from evaluation.evaluator import Evaluator
from system.output_head import OutputHead
from system.student_system import StudentSystem
from training.distill_dataset import DistillDataset
from training.distill_losses import DistillLoss, lambda_ramp
from training.scheduler import build_scheduler


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_system(config: dict[str, Any], device: torch.device) -> tuple[StudentSystem, dict[str, Any]]:
    geometry_config = _load_json(config["experts"]["geometry_config"])
    system = StudentSystem.from_component_checkpoints(
        backbone_checkpoint=config["backbone"]["checkpoint"],
        backbone_config_path=config["backbone"]["config"],
        router_checkpoint=config["router"]["checkpoint"],
        geometry_config_path=config["experts"]["geometry_config"],
        expert_checkpoint_dir=config["experts"]["checkpoint_dir"],
        expert_checkpoint_pattern=config["experts"]["checkpoint_pattern"],
        device=device,
    )
    for skill in system.skill_names:
        if skill not in system.heads:
            expert = system.experts[skill]
            system.heads[skill] = OutputHead(manifold_dim=expert.manifold_dim, vocab_size=system.backbone.config.vocab_size).to(device)
    return system, geometry_config


def _set_trainable(system: StudentSystem, backbone: bool, router: bool, experts: bool, heads: bool) -> None:
    for param in system.backbone.parameters():
        param.requires_grad_(backbone)
    for param in system.router.parameters():
        param.requires_grad_(router)
    for expert in system.experts.values():
        for param in expert.parameters():
            param.requires_grad_(experts)
    for head in system.heads.values():
        for param in head.parameters():
            param.requires_grad_(heads)


def _build_optimizer(system: StudentSystem, config: dict[str, Any], phase: str) -> AdamW:
    if phase == "phase1":
        params = [
            {"params": [p for expert in system.experts.values() for p in expert.parameters() if p.requires_grad], "lr": float(config["training"]["expert_lr"])},
            {"params": [p for head in system.heads.values() for p in head.parameters() if p.requires_grad], "lr": float(config["training"]["head_lr"])},
        ]
    elif phase == "phase2":
        params = [
            {"params": [p for p in system.backbone.parameters() if p.requires_grad], "lr": float(config["training"]["backbone_lr"])},
            {"params": [p for p in system.router.parameters() if p.requires_grad], "lr": float(config["training"]["router_lr"])},
            {"params": [p for expert in system.experts.values() for p in expert.parameters() if p.requires_grad], "lr": float(config["training"]["expert_lr"])},
            {"params": [p for head in system.heads.values() for p in head.parameters() if p.requires_grad], "lr": float(config["training"]["head_lr"])},
        ]
    else:
        params = [
            {"params": [p for expert in system.experts.values() for p in expert.parameters() if p.requires_grad], "lr": 1e-4},
            {"params": [p for head in system.heads.values() for p in head.parameters() if p.requires_grad], "lr": 1e-4},
        ]
    params = [group for group in params if group["params"]]
    return AdamW(params, weight_decay=float(config["training"]["weight_decay"]))


def _compute_grad_norm(system: StudentSystem) -> float:
    total = 0.0
    for param in system.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().norm().item() ** 2)
    return total ** 0.5


def _system_checkpoint_payload(
    system: StudentSystem,
    geometry_config: dict[str, Any],
    training_steps: int,
    gsm8k_val_accuracy: float,
    math500_accuracy: float,
) -> dict[str, Any]:
    return {
        "backbone_state_dict": system.backbone.state_dict(),
        "router_state_dict": system.router.state_dict(),
        "expert_state_dicts": {skill: expert.state_dict() for skill, expert in system.experts.items()},
        "head_state_dicts": {skill: head.state_dict() for skill, head in system.heads.items()},
        "skill_names": list(system.skill_names),
        "geometry_config": geometry_config,
        "router_temperature": float(system.router.temperature),
        "router_hidden_dim": int(system.router.hidden_dim),
        "training_steps": int(training_steps),
        "gsm8k_val_accuracy": float(gsm8k_val_accuracy),
        "math500_accuracy": float(math500_accuracy),
    }


def _save_system_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def _training_step(
    system: StudentSystem,
    batch: dict[str, Any],
    loss_fn: DistillLoss,
    config: dict[str, Any],
    device: torch.device,
    effective_lambda_3: float,
    effective_lambda_4: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_ids = batch["target_ids"].to(device)
    teacher_logits = batch["teacher_logits"].to(device)
    step_embeddings = batch["step_embeddings"].to(device)
    step_mask = batch["step_mask"].to(device)
    skill_label = batch["skill_label"].to(device)
    subskill_labels = batch["subskill_labels"].to(device)
    difficulty = batch["difficulty"].to(device)
    incorrect = batch["incorrect"].to(device)
    sample_weights = batch["sample_weights"].to(device)

    hidden_states, _ = system.backbone(input_ids, attention_mask=attention_mask)
    hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
    router_out = system.router(hidden_states)
    pooled = hidden_states.mean(dim=1)
    active_idx = skill_label
    student_logits = hidden_states.new_zeros(
        (input_ids.size(0), input_ids.size(1), system.backbone.config.vocab_size)
    )
    total_loss = hidden_states.new_tensor(0.0)
    loss_summaries = []

    for skill_idx, skill in enumerate(system.skill_names):
        mask = active_idx.eq(skill_idx)
        if not mask.any():
            continue
        expert = system.experts[skill]
        token_hidden = hidden_states[mask].reshape(-1, hidden_states.size(-1))
        token_z = expert.to_euclidean(expert.encode(token_hidden))
        token_z = torch.nan_to_num(token_z, nan=0.0, posinf=1.0, neginf=-1.0)
        token_z = token_z.reshape(mask.sum(), hidden_states.size(1), -1)
        student_logits[mask] = system.heads[skill](token_z)
        pooled_z = expert.to_euclidean(expert.encode(pooled[mask]))
        pooled_z = torch.nan_to_num(pooled_z, nan=0.0, posinf=1.0, neginf=-1.0)
        student_logits = torch.nan_to_num(
            student_logits, nan=0.0, posinf=1.0, neginf=-1.0
        )
        loss_output = loss_fn(
            student_logits=student_logits[mask],
            target_ids=target_ids[mask],
            teacher_logits=teacher_logits[mask],
            student_hidden=hidden_states[mask],
            step_embeddings=step_embeddings[mask],
            z_s=pooled_z,
            skill_label=skill_label[mask],
            subskill_labels=subskill_labels[mask],
            difficulty=difficulty[mask],
            incorrect_mask=incorrect[mask],
            router_weights=router_out.skill_probs[mask],
            expert=expert,
            sample_weights=sample_weights[mask],
            step_mask=step_mask[mask],
            effective_lambda_3=effective_lambda_3,
            effective_lambda_4=effective_lambda_4,
        )
        total_loss = total_loss + loss_output.total * (mask.float().mean())
        loss_summaries.append((skill, loss_output))

    if not loss_summaries:
        raise RuntimeError("No active skill representations were produced for this batch.")

    denom = max(len(loss_summaries), 1)
    metrics = {
        "l_task": float(sum(summary.l_task.detach().item() for _, summary in loss_summaries) / denom),
        "l_kd": float(sum(summary.l_kd.detach().item() for _, summary in loss_summaries) / denom),
        "l_trace": float(sum(summary.l_trace.detach().item() for _, summary in loss_summaries) / denom),
        "l_geom": float(sum(summary.l_geom.detach().item() for _, summary in loss_summaries) / denom),
        "l_proto": float(sum(summary.l_proto.detach().item() for _, summary in loss_summaries) / denom),
        "l_diff": float(sum(summary.l_diff.detach().item() for _, summary in loss_summaries) / denom),
        "l_sep": float(sum(summary.l_sep.detach().item() for _, summary in loss_summaries) / denom),
        "total": float(total_loss.detach().item()),
        "active_skill": loss_summaries[0][0],
        "router_dist": {
            skill: float(router_out.skill_probs[:, idx].mean().detach().item())
            for idx, skill in enumerate(system.skill_names)
        },
    }
    return total_loss, metrics


def train_distill(config: dict[str, Any]) -> dict[str, Any]:
    device = choose_device(config["evaluation"]["device"])
    system, geometry_config = _build_system(config, device)
    system.to(device)
    loss_fn = DistillLoss(
        pad_token_id=system.tokenizer.pad_token_id,
        kd_temperature=float(config["training"]["kd_temperature"]),
        lambda_1=float(config["training"]["lambda_1"]),
        lambda_2=float(config["training"]["lambda_2"]),
        lambda_3=float(config["training"]["lambda_3"]),
        lambda_4=float(config["training"]["lambda_4"]),
    )

    train_dataset = DistillDataset()
    hardest_dataset = DistillDataset(hardest_only=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    hardest_loader = DataLoader(
        hardest_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=hardest_dataset.collate_fn,
    )
    evaluator = Evaluator(device=config["evaluation"]["device"])

    phase1_steps = int(config["training"]["phase1_steps"])
    phase2_steps = int(config["training"]["phase2_steps"])
    phase3_steps = int(phase2_steps * float(config["training"]["phase3_fraction"]))
    total_steps = phase1_steps + phase2_steps + phase3_steps
    grad_accum = int(config["training"]["gradient_accumulation"])

    best_val_accuracy = -1.0
    best_step = 0
    best_state = None
    final_metrics = {"l_task": 0.0, "l_kd": 0.0, "l_trace": 0.0, "l_geom": 0.0}
    training_step = 0

    phase_specs = [
        ("phase1", phase1_steps, train_loader),
        ("phase2", phase2_steps, train_loader),
        ("phase3", phase3_steps, hardest_loader),
    ]

    for phase_name, phase_steps, loader in phase_specs:
        if phase_steps <= 0:
            continue
        if phase_name == "phase1":
            _set_trainable(system, backbone=False, router=False, experts=True, heads=True)
        elif phase_name == "phase2":
            _set_trainable(system, backbone=True, router=True, experts=True, heads=True)
        else:
            _set_trainable(system, backbone=False, router=False, experts=True, heads=True)

        optimizer = _build_optimizer(system, config, phase_name)
        scheduler = build_scheduler(
            optimizer=optimizer,
            warmup_steps=int(config["training"]["warmup_steps"]),
            max_steps=max(phase_steps, 1),
        )
        iterator = _cycle(loader)

        for local_step in range(phase_steps):
            training_step += 1
            optimizer.zero_grad(set_to_none=True)
            accumulated_metrics = None
            for _ in range(grad_accum):
                batch = next(iterator)
                if phase_name == "phase1":
                    eff_l3 = 0.0
                    eff_l4 = 0.0
                elif phase_name == "phase2":
                    ramp = lambda_ramp(local_step, int(config["training"]["geom_ramp_steps"]))
                    eff_l3 = float(config["training"]["lambda_3"]) * ramp
                    eff_l4 = float(config["training"]["lambda_4"]) * ramp
                else:
                    eff_l3 = float(config["training"]["lambda_3"])
                    eff_l4 = float(config["training"]["lambda_4"])

                loss, metrics = _training_step(
                    system=system,
                    batch=batch,
                    loss_fn=loss_fn,
                    config=config,
                    device=device,
                    effective_lambda_3=eff_l3,
                    effective_lambda_4=eff_l4,
                )
                (loss / grad_accum).backward()
                accumulated_metrics = metrics

            grad_norm = _compute_grad_norm(system)
            torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=float(config["training"]["gradient_clip"]))
            optimizer.step()
            scheduler.step()

            if accumulated_metrics is not None:
                final_metrics = {
                    "l_task": accumulated_metrics["l_task"],
                    "l_kd": accumulated_metrics["l_kd"],
                    "l_trace": accumulated_metrics["l_trace"],
                    "l_geom": accumulated_metrics["l_geom"],
                }

            if training_step % int(config["training"]["log_every"]) == 0 and accumulated_metrics is not None:
                print(
                    f"[distill] step={training_step} phase={phase_name} "
                    f"total={accumulated_metrics['total']:.4f} "
                    f"l_task={accumulated_metrics['l_task']:.4f} "
                    f"l_kd={accumulated_metrics['l_kd']:.4f} "
                    f"l_trace={accumulated_metrics['l_trace']:.4f} "
                    f"l_geom={accumulated_metrics['l_geom']:.4f} "
                    f"l_proto={accumulated_metrics['l_proto']:.4f} "
                    f"l_diff={accumulated_metrics['l_diff']:.4f} "
                    f"l_sep={accumulated_metrics['l_sep']:.4f} "
                    f"active_skill={accumulated_metrics['active_skill']} "
                    f"grad_norm={grad_norm:.4f}"
                )
                print(f"[distill] router distribution: {accumulated_metrics['router_dist']}")

            if training_step % int(config["training"]["val_every"]) == 0:
                try:
                    gsm8k_eval = evaluator.evaluate_gsm8k(system, split="test", n_examples=int(config["evaluation"]["gsm8k_val_n"]))
                except Exception as exc:
                    print(f"Warning: GSM8K evaluation failed at step {training_step}: {exc}")
                    gsm8k_eval = None
                if gsm8k_eval is not None and gsm8k_eval.accuracy > best_val_accuracy:
                    best_val_accuracy = gsm8k_eval.accuracy
                    best_step = training_step
                    best_state = copy.deepcopy(
                        _system_checkpoint_payload(
                            system=system,
                            geometry_config=geometry_config,
                            training_steps=training_step,
                            gsm8k_val_accuracy=gsm8k_eval.accuracy,
                            math500_accuracy=0.0,
                        )
                    )
                    _save_system_checkpoint(config["output"]["best_checkpoint"], best_state)

            if training_step % int(config["training"]["save_every"]) == 0:
                payload = _system_checkpoint_payload(
                    system=system,
                    geometry_config=geometry_config,
                    training_steps=training_step,
                    gsm8k_val_accuracy=max(best_val_accuracy, 0.0),
                    math500_accuracy=0.0,
                )
                _save_system_checkpoint(Path("checkpoints") / f"system_step_{training_step}.pt", payload)

    gsm8k_result = evaluator.evaluate_gsm8k(system, split="test", n_examples=int(config["evaluation"]["gsm8k_val_n"]))
    try:
        math_result = evaluator.evaluate_math500(system)
    except Exception as exc:
        print(f"Warning: MATH-500 evaluation failed: {exc}")
        from evaluation.evaluator import EvalResult

        math_result = EvalResult(
            accuracy=0.0,
            n_correct=0,
            n_total=0,
            per_skill={skill: 0.0 for skill in system.skill_names},
            abstain_rate=0.0,
            avg_router_conf=0.0,
        )

    final_payload = _system_checkpoint_payload(
        system=system,
        geometry_config=geometry_config,
        training_steps=total_steps,
        gsm8k_val_accuracy=gsm8k_result.accuracy,
        math500_accuracy=math_result.accuracy,
    )
    _save_system_checkpoint(config["output"]["final_checkpoint"], final_payload)
    eval_results = {
        "gsm8k": gsm8k_result.__dict__,
        "math500": math_result.__dict__,
        "best_val_step": best_step,
    }
    output_path = Path(config["output"]["eval_results"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(eval_results, handle, indent=2, sort_keys=True)

    return {
        "total_steps": total_steps,
        "best_val_step": best_step,
        "final_l_task": final_metrics["l_task"],
        "final_l_kd": final_metrics["l_kd"],
        "final_l_trace": final_metrics["l_trace"],
        "final_l_geom": final_metrics["l_geom"],
        "gsm8k": gsm8k_result,
        "math500": math_result,
        "best_checkpoint": config["output"]["best_checkpoint"],
        "final_checkpoint": config["output"]["final_checkpoint"],
        "eval_results": config["output"]["eval_results"],
    }
