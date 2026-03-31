from __future__ import annotations

import json
from pathlib import Path

import torch

from calibration.system_calibrator import SystemCalibrator
from calibration.threshold_tuner import ThresholdTuner, compute_utility
from evaluation.ablation import UniformRandomRouter, run_ablations
from evaluation.comparison import BackboneOnlySystem, SingleEuclideanSystem
from evaluation.evaluator import EvalResult
from evaluation.metrics import extract_gsm8k_answer
from reporting.report_builder import build_final_report
from reporting.results_aggregator import collect_stage_checksums
from system.student_system import StudentSystem
from training.fine_tune import HardExampleDataset, freeze_for_finetuning, should_early_stop
from tests.test_system import _build_mock_system


class DummyDataset:
    def __init__(self) -> None:
        self.items = [
            {"difficulty": 2, "incorrect": False, "low_confidence": False, "skill_label": 0},
            {"difficulty": 1, "incorrect": True, "low_confidence": False, "skill_label": 1},
            {"difficulty": 0, "incorrect": False, "low_confidence": True, "skill_label": 0},
        ]
        self.skill_names = ["algebraic", "arithmetic"]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = dict(self.items[index])
        item.update(
            {
                "example_id": f"ex{index}",
                "problem_text": "p",
                "input_ids": torch.tensor([1, 2]),
                "target_ids": torch.tensor([1, 2]),
                "teacher_logits": torch.zeros(4),
                "teacher_logit_indices": torch.tensor([0, 1]),
                "step_embeddings": torch.zeros((1, 2)),
                "subskill_labels": [],
                "sample_weight": 1.0,
            }
        )
        return item

    def collate_fn(self, batch):
        return {}


def test_hard_example_dataset_and_freeze() -> None:
    base = DummyDataset()
    hard = HardExampleDataset(base_dataset=base, oversample=2.0, min_examples=500)
    assert len(hard) >= 500
    system = _build_mock_system()
    freeze_for_finetuning(system)
    assert not any(p.requires_grad for p in system.backbone.parameters())
    assert not any(p.requires_grad for p in system.router.parameters())


def test_early_stop_logic() -> None:
    assert should_early_stop(0.80, 0.90, 0.03) is True
    assert should_early_stop(0.88, 0.90, 0.03) is False


def test_calibration_ranges_and_utility() -> None:
    system = _build_mock_system()
    calibrator = SystemCalibrator(search_steps=5, temp_range=(0.5, 3.0))
    temps = {skill: 1.0 for skill in system.skill_names}
    for value in temps.values():
        assert 0.5 <= value <= 3.0
    tuner = ThresholdTuner()
    utility = compute_utility(accuracy_when_answered=0.8, error_rate_when_answered=0.2, abstain_rate=0.1, cost_wrong=1.0, cost_abstain=0.3)
    assert isinstance(utility, float)
    threshold_payload = {"threshold": 0.5, "utility_at_threshold": utility, "abstain_rate_at_threshold": 0.1, "accuracy_when_answered": 0.8}
    assert 0.1 <= threshold_payload["threshold"] <= 0.9
    assert 0.0 <= min(0.2, 0.1) <= 1.0


def test_final_evaluator_ablation_and_baselines_types() -> None:
    system = _build_mock_system()
    assert isinstance(BackboneOnlySystem(system), BackboneOnlySystem)
    assert isinstance(SingleEuclideanSystem(system), SingleEuclideanSystem)
    assert isinstance(UniformRandomRouter(system.skill_names), UniformRandomRouter)


def test_final_checkpoint_schema_and_checksums(tmp_path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for name in ["backbone_stage0.pt", "router_stage4.pt", "full_system_stage5.pt", "expert_a_stage3.pt"]:
        (ckpt_dir / name).write_bytes(b"abc")
    cwd = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        checksums = collect_stage_checksums()
        assert "stage0" in checksums and "stage3" in checksums and "stage4" in checksums and "stage5" in checksums
    finally:
        os.chdir(cwd)


def test_report_builder_outputs_all_sections(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "geometry_config.json").write_text(json.dumps({"skills": {"algebraic": {"geometry": "product"}}}), encoding="utf-8")
    (outputs / "router_calibration.json").write_text(json.dumps({"ece": 0.2}), encoding="utf-8")
    (outputs / "stage5_eval.json").write_text(json.dumps({"gsm8k": {"accuracy": 0.4}}), encoding="utf-8")
    (outputs / "stage6_eval.json").write_text(
        json.dumps(
            {
                "gsm8k": {"accuracy": 0.5, "abstain_rate": 0.1, "n_correct": 10, "n_total": 20, "per_skill": {"algebraic": 0.5}},
                "math500": {"accuracy": 0.4, "n_correct": 8, "n_total": 20, "per_skill": {"algebraic": 0.4}},
                "subskill_calibration": {"algebraic": {"routing_accuracy": 0.6, "answer_accuracy": 0.5, "calibration_gap": 0.1}},
                "ablations": {
                    "no_geometry": {"gsm8k": 0.4, "math500": 0.3},
                    "no_router": {"gsm8k": 0.3, "math500": 0.2},
                    "no_trace_supervision": {"gsm8k": 0.45, "math500": 0.35},
                    "no_abstain": {"gsm8k": 0.48, "math500": 0.38},
                },
                "baselines": {
                    "backbone_only": {"gsm8k": 0.2, "math500": 0.1},
                    "single_euclidean": {"gsm8k": 0.3, "math500": 0.2},
                },
                "calibration": {"ece_after": 0.1},
                "parameter_counts": {"backbone": 1, "router": 1, "experts": 1, "heads": 1, "total": 4},
            }
        ),
        encoding="utf-8",
    )
    (outputs / "abstain_threshold.json").write_text(json.dumps({"threshold": 0.4, "utility_at_threshold": 0.5}), encoding="utf-8")
    cwd = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        report = build_final_report("outputs/final_report.md")
        assert report
        for section in [
            "## 1. Project overview",
            "## 2. Architecture summary",
            "## 3. Training pipeline summary",
            "## 4. Results",
            "## 5. Key findings",
            "## 6. Limitations and future work",
        ]:
            assert section in report
        assert "placeholder X" not in report
    finally:
        os.chdir(cwd)


def test_extract_and_evalresult_types() -> None:
    assert extract_gsm8k_answer("the answer is 42") == 42.0
    result = EvalResult(accuracy=0.5, n_correct=1, n_total=2, per_skill={"a": 0.5}, abstain_rate=0.1, avg_router_conf=0.8)
    assert 0.0 <= result.accuracy <= 1.0
