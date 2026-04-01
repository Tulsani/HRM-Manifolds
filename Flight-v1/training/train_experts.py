from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import Adam

from manifolds import build_expert
from training.contrastive_loss import compute_pretraining_loss
from training.expert_dataset import build_expert_dataloader


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


def train_all_experts(config: dict[str, Any]) -> list[dict[str, Any]]:
    geometry_config = _load_json(config["manifold"]["geometry_config"])
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pretrain_cfg = config["pretraining"]
    device = choose_device(config["backbone"].get("device", "cuda"))
    backbone_dim = int(geometry_config["backbone_dim"])
    results: list[dict[str, Any]] = []

    for skill, skill_cfg in geometry_config["skills"].items():
        dataset, loader = build_expert_dataloader(
            skill=skill,
            batch_size=int(pretrain_cfg["batch_size"]),
        )
        if len(dataset) == 0:
            print(f"Warning: skill '{skill}' has no examples in the expert dataset; skipping training.")
            continue

        local_skill_cfg = {
            **skill_cfg,
            "n_prototypes": int(config["manifold"].get("n_prototypes", 32)),
        }
        if local_skill_cfg.get("geometry") == "product" and "manifold_dim" not in local_skill_cfg:
            local_skill_cfg["manifold_dim"] = int(config["manifold"]["default_manifold_dim"])

        local_geometry_config = {"skills": {skill: local_skill_cfg}}
        expert = build_expert(skill, local_geometry_config, backbone_dim=backbone_dim).to(device)
        expert.train()
        optimizer = Adam(expert.parameters(), lr=float(pretrain_cfg["lr"]))
        iterator = _cycle(loader)
        final_metrics = {"loss": 0.0, "loss_proto": 0.0, "loss_struct": 0.0, "loss_order": 0.0}
        max_steps = int(pretrain_cfg["max_steps"])

        diagnostic_batch = next(iterator)
        diagnostic_h = diagnostic_batch["teacher_embeddings"].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            diagnostic_z = expert.encode(diagnostic_h)
            encode_finite = bool(torch.isfinite(diagnostic_z).all().item())
            has_nan = bool(torch.isnan(diagnostic_z).any().item())
            print(
                f"[{skill}] startup encode diagnostic: "
                f"finite={encode_finite} nan={has_nan} "
                f"input_finite={bool(torch.isfinite(diagnostic_h).all().item())}"
            )

        for step in range(1, max_steps + 1):
            batch = next(iterator)
            teacher_embeddings = batch["teacher_embeddings"].to(device=device, dtype=torch.float32)
            step_embeddings = [tensor.to(device=device, dtype=torch.float32) for tensor in batch["step_embeddings"]]
            prototype_ids = batch["prototype_ids"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_pretraining_loss(
                expert=expert,
                h=teacher_embeddings,
                teacher_embeddings=teacher_embeddings,
                step_embeddings=step_embeddings,
                prototype_ids=prototype_ids,
                lambda_proto=float(pretrain_cfg["lambda_proto"]),
                lambda_struct=float(pretrain_cfg["lambda_struct"]),
                lambda_order=float(pretrain_cfg["lambda_order"]),
                margin_proto=float(pretrain_cfg["margin_proto"]),
                margin_order=float(pretrain_cfg["margin_order"]),
            )
            if loss.requires_grad and float(loss.item()) != 0.0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    expert.parameters(),
                    max_norm=float(pretrain_cfg["gradient_clip"])
                )
                optimizer.step()
            else:
                # Loss is zero or detached — skip backward, still update metrics
                optimizer.zero_grad(set_to_none=True)
            final_metrics = metrics

            if step % int(pretrain_cfg["log_every"]) == 0:
                print(
                    f"[{skill}] step={step} "
                    f"L1={metrics['loss_proto']:.4f} "
                    f"L2={metrics['loss_struct']:.4f} "
                    f"L3={metrics['loss_order']:.4f} "
                    f"L={metrics['loss']:.4f}"
                )

        checkpoint_path = checkpoint_dir / config["output"]["checkpoint_prefix"].format(skill=skill)
        checkpoint_payload = {
            "model_state_dict": expert.state_dict(),
            "skill": skill,
            "geometry": expert.geometry,
            "config": expert.expert_config(),
            "pretrain_loss_final": float(final_metrics["loss"]),
        }
        torch.save(checkpoint_payload, checkpoint_path)
        results.append(
            {
                "skill": skill,
                "geometry": expert.geometry,
                "steps": max_steps,
                "loss_final": float(final_metrics["loss"]),
                "checkpoint_path": str(checkpoint_path),
            }
        )

    return results
