from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from router.calibration import RouterCalibrator
from router.router import Router
from router.routing_dataset import RoutingDataset, build_routing_dataloaders, collate_routing_batch
from router.routing_losses import (
    build_subskill_target_distribution,
    compute_router_loss,
    confidence_calibration_loss,
    entropy_regularization_loss,
    skill_classification_loss,
    subskill_consistency_loss,
)


def _write_mock_artifacts(tmp_path) -> tuple[str, str]:
    embeddings = {}
    labels = {"labels": {}, "summary": {"skill_counts": {}, "low_confidence_count": 0, "incorrect_count": 0, "total": 0}}
    skill_specs = {
        "algebraic": 4,
        "arithmetic": 4,
        "fallback": 4,
    }
    counter = 0
    for skill, count in skill_specs.items():
        labels["summary"]["skill_counts"][skill] = count
        for idx in range(count):
            example_id = f"ex_{counter:03d}"
            embeddings[example_id] = np.random.randn(8).astype(np.float32)
            labels["labels"][example_id] = {
                "skill": skill,
                "subskills": ["equation_setup"] if skill == "algebraic" else (["multiplication"] if skill == "arithmetic" else ["ratio"]),
                "n_steps": 3,
                "low_confidence": idx == 0 and skill == "fallback",
                "incorrect": idx == 1 and skill == "fallback",
                "difficulty": "hard" if idx == 0 else "easy",
                "source": "math",
            }
            counter += 1
    labels["summary"]["low_confidence_count"] = 1
    labels["summary"]["incorrect_count"] = 1
    labels["summary"]["total"] = counter

    embeddings_path = tmp_path / "problem_embeddings.npz"
    labels_path = tmp_path / "skill_labels.json"
    np.savez(embeddings_path, **embeddings)
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return str(embeddings_path), str(labels_path)


def test_router_forward_and_helpers() -> None:
    router = Router(input_dim=8, skill_names=["a", "b", "c"], hidden_dim=16, dropout=0.0)
    output = router(torch.randn(4, 8))
    assert output.skill_probs.shape == (4, 3)
    assert output.skill_logits.shape == (4, 3)
    assert output.confidence.shape == (4, 1)
    assert output.abstain_prob.shape == (4, 1)
    assert torch.allclose(output.skill_probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.all((output.confidence > 0.0) & (output.confidence < 1.0))
    assert torch.all((output.abstain_prob > 0.0) & (output.abstain_prob < 1.0))

    single = torch.randn(1, 8)
    skill_name, confidence = router.hard_route(single)
    soft = router.soft_route(single, threshold=0.1)
    abstain = router.should_abstain(single, threshold=0.5)
    assert skill_name in {"a", "b", "c"}
    assert isinstance(confidence, float)
    assert all(weight > 0.1 for weight in soft.values())
    assert isinstance(abstain, bool)


def test_routing_dataset_and_split(tmp_path) -> None:
    embeddings_path, labels_path = _write_mock_artifacts(tmp_path)
    train_dataset, val_dataset, _, _ = build_routing_dataloaders(
        embeddings_path=embeddings_path,
        skill_labels_path=labels_path,
        train_split=0.85,
        batch_size=4,
    )
    item = train_dataset[0]
    assert item["embedding"].shape == (8,)
    assert isinstance(item["skill_label"], int)
    assert isinstance(item["subskill_labels"], list)
    assert isinstance(item["low_confidence"], bool)
    assert isinstance(item["incorrect"], bool)

    val_skills = {val_dataset.labels[example_id]["skill"] for example_id in val_dataset.example_ids}
    assert {"algebraic", "arithmetic", "fallback"}.issubset(val_skills)

    weights = train_dataset.get_skill_weights()
    assert torch.all(weights > 0)
    assert torch.isclose(weights.sum(), torch.tensor(float(len(train_dataset.skill_names))), atol=1e-5)

    flagged = [train_dataset[idx] for idx in range(len(train_dataset)) if train_dataset[idx]["low_confidence"] or train_dataset[idx]["incorrect"]]
    assert any(item["low_confidence"] for item in flagged) or any(item["incorrect"] for item in flagged)


def test_router_losses_backward() -> None:
    router = Router(input_dim=8, skill_names=["algebraic", "arithmetic", "fallback"], hidden_dim=16, dropout=0.0)
    embeddings = torch.randn(6, 8, requires_grad=True)
    output = router(embeddings)
    skill_labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    low_confidence = torch.tensor([False, True, False, False, False, False])
    incorrect = torch.tensor([False, False, True, False, False, False])
    subskill_multihot = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    skill_subskill_matrix = torch.eye(3)
    target_dist = build_subskill_target_distribution(subskill_multihot, skill_subskill_matrix)

    l_cls = skill_classification_loss(output, skill_labels, low_confidence, incorrect)
    l_sub = subskill_consistency_loss(output, target_dist)
    l_cal = confidence_calibration_loss(output, skill_labels)
    l_ent = entropy_regularization_loss(output, n_skills=3)
    total, metrics = compute_router_loss(
        output=output,
        skill_labels=skill_labels,
        low_confidence=low_confidence,
        incorrect=incorrect,
        subskill_target_dist=target_dist,
        n_skills=3,
    )
    total.backward()

    assert l_cls.item() >= 0.0 and torch.isfinite(l_cls)
    assert l_sub.item() >= 0.0 and torch.isfinite(l_sub)
    assert l_cal.item() >= 0.0 and torch.isfinite(l_cal)
    assert l_ent.item() >= 0.0 and torch.isfinite(l_ent)
    assert torch.isfinite(total)
    assert "loss_total" in metrics


def test_calibration_pipeline(tmp_path) -> None:
    embeddings_path, labels_path = _write_mock_artifacts(tmp_path)
    _, val_dataset, _, val_loader = build_routing_dataloaders(
        embeddings_path=embeddings_path,
        skill_labels_path=labels_path,
        train_split=0.85,
        batch_size=4,
    )
    router = Router(input_dim=8, skill_names=val_dataset.skill_names, hidden_dim=16, dropout=0.0)
    calibrator = RouterCalibrator(n_bins=10, temp_range=(0.5, 5.0), temp_search_steps=20)
    temperature = calibrator.fit(router, val_loader)
    report = calibrator.evaluate_calibration(router, val_loader)
    report_path = tmp_path / "router_calibration.json"
    calibrator.save_report(report, report_path)
    loaded = json.loads(report_path.read_text(encoding="utf-8"))

    assert 0.5 <= temperature <= 5.0
    assert 0.0 <= report["ece"] <= 1.0
    assert 0.0 <= report["mce"] <= 1.0
    assert "temperature" in loaded
    assert "per_skill_accuracy" in loaded
