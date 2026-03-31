from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam, AdamW

from router.calibration import RouterCalibrator
from router.router import Router
from router.routing_dataset import build_routing_dataloaders
from router.routing_losses import build_subskill_target_distribution, compute_router_loss
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


def _build_skill_subskill_matrix(dataset) -> torch.Tensor:
    matrix = torch.zeros((dataset.skill_names.__len__(), dataset.n_subskills), dtype=torch.float32)
    for skill, indices in dataset.skill_to_subskills.items():
        skill_idx = dataset.skill_to_idx[skill]
        if indices:
            matrix[skill_idx, list(indices)] = 1.0
    return matrix


@torch.no_grad()
def evaluate_router(router: Router, val_loader) -> dict[str, Any]:
    device = next(router.parameters()).device
    total = 0
    correct = 0
    per_skill_total = {skill: 0 for skill in router.skill_names}
    per_skill_correct = {skill: 0 for skill in router.skill_names}
    was_training = router.training
    router.eval()
    for batch in val_loader:
        embeddings = batch["embeddings"].to(device=device, dtype=torch.float32)
        labels = batch["skill_labels"].to(device)
        output = router(embeddings)
        preds = output.skill_idx
        matches = preds.eq(labels)
        total += labels.numel()
        correct += int(matches.sum().item())
        for idx, skill_idx in enumerate(labels.tolist()):
            skill_name = router.skill_names[int(skill_idx)]
            per_skill_total[skill_name] += 1
            per_skill_correct[skill_name] += int(matches[idx].item())
    if was_training:
        router.train()
    return {
        "val_accuracy": float(correct / total) if total > 0 else 0.0,
        "per_skill_accuracy": {
            skill: float(per_skill_correct[skill] / per_skill_total[skill]) if per_skill_total[skill] > 0 else 0.0
            for skill in router.skill_names
        },
    }


def _save_router_checkpoint(router: Router, path: str | Path, val_accuracy: float, ece: float) -> None:
    payload = {
        "model_state_dict": router.state_dict(),
        "skill_names": list(router.skill_names),
        "n_skills": int(router.n_skills),
        "hidden_dim": int(router.hidden_dim),
        "input_dim": int(router.input_dim),
        "temperature": float(router.temperature),
        "val_accuracy": float(val_accuracy),
        "ece": float(ece),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def _abstain_labels(batch: dict[str, Any], router: Router, per_skill_accuracy: dict[str, float]) -> torch.Tensor:
    labels = []
    for skill_idx, difficulty, low_conf, incorrect in zip(
        batch["skill_labels"].tolist(),
        batch["difficulty"].tolist(),
        batch["low_confidence"].tolist(),
        batch["incorrect"].tolist(),
    ):
        skill_name = router.skill_names[int(skill_idx)]
        hard_and_unreliable = int(difficulty) == 2 and float(per_skill_accuracy.get(skill_name, 0.0)) < 0.6
        label = bool(low_conf) or bool(incorrect) or hard_and_unreliable
        labels.append(float(label))
    return torch.tensor(labels, dtype=torch.float32)


def train_router(config: dict[str, Any]) -> dict[str, Any]:
    train_dataset, val_dataset, train_loader, val_loader = build_routing_dataloaders(
        embeddings_path=config["data"]["embeddings"],
        skill_labels_path=config["data"]["skill_labels"],
        train_split=float(config["data"]["train_split"]),
        batch_size=int(config["training"]["batch_size"]),
    )

    input_dim = int(train_dataset[0]["embedding"].shape[-1])
    router = Router(
        input_dim=input_dim,
        skill_names=train_dataset.skill_names,
        hidden_dim=int(config["router"]["hidden_dim"]),
        dropout=float(config["router"]["dropout"]),
    )
    device = choose_device(config["backbone"].get("device", "cuda"))
    router.to(device)

    skill_subskill_matrix = _build_skill_subskill_matrix(train_dataset).to(device)
    optimizer = AdamW(
        router.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        warmup_steps=int(config["training"]["warmup_steps"]),
        max_steps=int(config["training"]["phase1_steps"]),
    )

    best_val_accuracy = -1.0
    best_state = copy.deepcopy(router.state_dict())
    best_metrics = {"val_accuracy": 0.0, "per_skill_accuracy": {skill: 0.0 for skill in router.skill_names}}
    iterator = _cycle(train_loader)
    phase1_steps = int(config["training"]["phase1_steps"])

    for step in range(1, phase1_steps + 1):
        batch = next(iterator)
        embeddings = batch["embeddings"].to(device=device, dtype=torch.float32)
        skill_labels = batch["skill_labels"].to(device)
        low_confidence = batch["low_confidence"].to(device)
        incorrect = batch["incorrect"].to(device)
        subskill_multihot = batch["subskill_multihot"].to(device)

        optimizer.zero_grad(set_to_none=True)
        output = router(embeddings)
        subskill_target_dist = build_subskill_target_distribution(subskill_multihot, skill_subskill_matrix)
        loss, metrics = compute_router_loss(
            output=output,
            skill_labels=skill_labels,
            low_confidence=low_confidence,
            incorrect=incorrect,
            subskill_target_dist=subskill_target_dist,
            n_skills=router.n_skills,
            lambda_cls=float(config["losses"]["lambda_cls"]),
            lambda_sub=float(config["losses"]["lambda_sub"]),
            lambda_cal=float(config["losses"]["lambda_cal"]),
            lambda_ent=float(config["losses"]["lambda_ent"]),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=float(config["training"]["gradient_clip"]))
        optimizer.step()
        scheduler.step()

        if step % int(config["training"]["log_every"]) == 0:
            val_metrics = evaluate_router(router, val_loader)
            print(
                f"[router] step={step} "
                f"L_cls={metrics['loss_cls']:.4f} "
                f"L_sub={metrics['loss_sub']:.4f} "
                f"L_cal={metrics['loss_cal']:.4f} "
                f"L_ent={metrics['loss_ent']:.4f} "
                f"L_total={metrics['loss_total']:.4f} "
                f"val_acc={val_metrics['val_accuracy']:.4f}"
            )
            print(f"[router] per-skill val accuracy: {val_metrics['per_skill_accuracy']}")
            if val_metrics["val_accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["val_accuracy"]
                best_state = copy.deepcopy(router.state_dict())
                best_metrics = val_metrics
                _save_router_checkpoint(
                    router,
                    config["output"]["best_checkpoint"],
                    val_accuracy=best_val_accuracy,
                    ece=0.0,
                )

    final_phase1_metrics = evaluate_router(router, val_loader)
    if final_phase1_metrics["val_accuracy"] > best_val_accuracy:
        best_val_accuracy = final_phase1_metrics["val_accuracy"]
        best_state = copy.deepcopy(router.state_dict())
        best_metrics = final_phase1_metrics
        _save_router_checkpoint(
            router,
            config["output"]["best_checkpoint"],
            val_accuracy=best_val_accuracy,
            ece=0.0,
        )

    router.load_state_dict(best_state)

    for param in router.parameters():
        param.requires_grad_(False)
    for param in router.abstain_head.parameters():
        param.requires_grad_(True)

    phase2_optimizer = Adam(router.abstain_head.parameters(), lr=1e-4)
    phase2_steps = int(config["training"]["phase2_steps"])
    abstain_threshold = float(config["router"]["abstain_threshold"])
    phase2_precision = 0.0
    phase2_recall = 0.0
    iterator = _cycle(train_loader)
    router.eval()
    router.abstain_head.train()

    for step in range(1, phase2_steps + 1):
        batch = next(iterator)
        embeddings = batch["embeddings"].to(device=device, dtype=torch.float32)
        targets = _abstain_labels(batch, router, best_metrics["per_skill_accuracy"]).to(device)

        phase2_optimizer.zero_grad(set_to_none=True)
        logits = router.compute_abstain_logits(embeddings).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        phase2_optimizer.step()

        if step % 50 == 0:
            preds = (torch.sigmoid(logits) > abstain_threshold).float()
            tp = ((preds == 1.0) & (targets == 1.0)).sum().item()
            fp = ((preds == 1.0) & (targets == 0.0)).sum().item()
            fn = ((preds == 0.0) & (targets == 1.0)).sum().item()
            phase2_precision = float(tp / max(tp + fp, 1))
            phase2_recall = float(tp / max(tp + fn, 1))
            print(
                f"[abstain] step={step} loss={loss.item():.4f} "
                f"precision={phase2_precision:.4f} recall={phase2_recall:.4f}"
            )

    with torch.no_grad():
        preds = (torch.sigmoid(logits) > abstain_threshold).float()
        tp = ((preds == 1.0) & (targets == 1.0)).sum().item()
        fp = ((preds == 1.0) & (targets == 0.0)).sum().item()
        fn = ((preds == 0.0) & (targets == 1.0)).sum().item()
        phase2_precision = float(tp / max(tp + fp, 1))
        phase2_recall = float(tp / max(tp + fn, 1))

    calibrator = RouterCalibrator(
        n_bins=int(config["calibration"]["n_bins"]),
        temp_range=tuple(config["calibration"]["temp_range"]),
        temp_search_steps=int(config["calibration"]["temp_search_steps"]),
    )
    temperature = calibrator.fit(router, val_loader)
    calibration_report = calibrator.evaluate_calibration(router, val_loader)
    calibrator.save_report(calibration_report, config["output"]["calibration_report"])
    _save_router_checkpoint(
        router,
        config["output"]["final_checkpoint"],
        val_accuracy=calibration_report["val_accuracy"],
        ece=calibration_report["ece"],
    )

    return {
        "best_val_accuracy": best_metrics["val_accuracy"],
        "per_skill_accuracy": best_metrics["per_skill_accuracy"],
        "abstain_precision": phase2_precision,
        "abstain_recall": phase2_recall,
        "temperature": temperature,
        "ece": calibration_report["ece"],
        "mce": calibration_report["mce"],
        "final_val_accuracy": calibration_report["val_accuracy"],
        "best_checkpoint": config["output"]["best_checkpoint"],
        "final_checkpoint": config["output"]["final_checkpoint"],
        "calibration_report": config["output"]["calibration_report"],
    }
