from __future__ import annotations

import json

import numpy as np

from analysis.delta_analyzer import compute_step_delta_consistency
from analysis.geometry_probe import decide_geometry, estimate_gromov_delta
from analysis.report_generator import write_stage2_report
from analysis.skill_refiner import build_skill_label_payload
from data.trace_schema import SparseLogits, TracePackage


def _make_trace(
    example_id: str,
    skill: str = "fallback",
    source: str = "math",
    subject: str = "",
    confidence: float = 0.9,
    answer_correct: bool = True,
    n_steps: int = 2,
    subskills: list[str] | None = None,
) -> TracePackage:
    return TracePackage(
        example_id=example_id,
        source=source,
        difficulty="medium",
        problem="Mock problem",
        final_answer="42",
        answer_correct=answer_correct,
        soft_logits=SparseLogits(indices=np.array([1], dtype=np.int32), values=np.array([0.5], dtype=np.float16)),
        confidence=confidence,
        rationale="First set an equation. Then substitute. Finally simplify.",
        step_texts=["First set an equation.", "Then substitute.", "Finally simplify."][:n_steps],
        n_steps=n_steps,
        skill=skill,
        subskills=subskills if subskills is not None else ["equation_setup", "substitution"],
        math_subject=subject,
        teacher_model="teacher",
        timestamp="2026-03-30T00:00:00+00:00",
    )


def test_skill_refiner_subject_alignment_rules() -> None:
    traces = [
        _make_trace("ex_alg", subject="Algebra"),
        _make_trace("ex_num", subject="Number Theory"),
        _make_trace("ex_geo", subject="Geometry"),
        _make_trace("ex_cnt", subject="Counting/Prob"),
    ]
    payload = build_skill_label_payload(traces)
    labels = payload["labels"]

    assert labels["ex_alg"]["skill"] == "algebraic"
    assert labels["ex_num"]["skill"] == "number_theory"
    assert labels["ex_geo"]["skill"] == "geometric"
    assert labels["ex_cnt"]["skill"] == "combinatorics"


def test_skill_refiner_flags_low_confidence_and_incorrect() -> None:
    payload = build_skill_label_payload(
        [
            _make_trace("ex1", confidence=0.2, answer_correct=False, source="gsm8k"),
            _make_trace("ex2", confidence=0.8, answer_correct=True, source="gsm8k"),
        ]
    )
    assert payload["labels"]["ex1"]["low_confidence"] is True
    assert payload["labels"]["ex1"]["incorrect"] is True
    assert payload["summary"]["low_confidence_count"] == 1
    assert payload["summary"]["incorrect_count"] == 1


def test_gromov_delta_estimator_in_range() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(16, 8)).astype(np.float32)
    stats = estimate_gromov_delta(embeddings, n_triples=50)
    assert stats["delta_norm"] is not None
    assert 0.0 <= stats["delta_norm"] <= 1.0


def test_step_delta_analyzer_in_range() -> None:
    matrix = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    score = compute_step_delta_consistency(matrix)
    assert -1.0 <= score <= 1.0


def test_geometry_decision_logic_returns_valid_geometry() -> None:
    cases = [
        (0.05, 3.5, 0.2, 0.2),
        (0.2, 2.0, 0.6, 0.2),
        (0.4, 1.5, 0.6, 0.8),
        (0.4, 1.0, 0.1, 0.1),
        (None, 0.0, 0.0, 0.0),
    ]
    for delta_norm, depth, consistency, purity in cases:
        decision = decide_geometry(delta_norm, depth, consistency, purity)
        assert decision["geometry"] in {"hyperbolic", "euclidean", "product"}


def test_stage2_json_schemas_loadable(tmp_path) -> None:
    skill_payload = {
        "labels": {
            "ex1": {
                "skill": "algebraic",
                "subskills": ["equation_setup"],
                "n_steps": 2,
                "low_confidence": False,
                "incorrect": False,
                "difficulty": "medium",
                "source": "math",
            }
        },
        "summary": {
            "skill_counts": {"algebraic": 1},
            "low_confidence_count": 0,
            "incorrect_count": 0,
            "total": 1,
        },
    }
    geometry_payload = {
        "skills": {
            "algebraic": {
                "geometry": "product",
                "curvature": -1.0,
                "h_dim": 16,
                "e_dim": 16,
                "n_examples": 1,
                "test_results": {
                    "delta_norm": None,
                    "avg_depth": 0.0,
                    "branching_factor": 0.0,
                    "step_consistency": 0.0,
                    "nn_purity": 0.0,
                    "degenerate": False,
                },
            }
        },
        "default_geometry": "product",
        "backbone_dim": 512,
        "summary": {
            "analyzed_skill_groups": 0,
            "weak_signal_skills": ["algebraic"],
            "ambiguous_skills": [],
            "total_examples": 1,
        },
    }

    skill_path = tmp_path / "skill_labels.json"
    geometry_path = tmp_path / "geometry_config.json"
    skill_path.write_text(json.dumps(skill_payload, indent=2), encoding="utf-8")
    geometry_path.write_text(json.dumps(geometry_payload, indent=2), encoding="utf-8")

    loaded_skill = json.loads(skill_path.read_text(encoding="utf-8"))
    loaded_geometry = json.loads(geometry_path.read_text(encoding="utf-8"))

    assert "labels" in loaded_skill and "summary" in loaded_skill
    assert "skills" in loaded_geometry and "default_geometry" in loaded_geometry


def test_report_generator_runs_on_mock_data(tmp_path) -> None:
    skill_payload = {
        "labels": {},
        "summary": {
            "skill_counts": {"algebraic": 10},
            "low_confidence_count": 1,
            "incorrect_count": 2,
            "total": 10,
        },
    }
    geometry_payload = {
        "skills": {
            "algebraic": {
                "geometry": "product",
                "curvature": -1.0,
                "h_dim": 16,
                "e_dim": 16,
                "n_examples": 10,
                "test_results": {
                    "delta_norm": 0.2,
                    "avg_depth": 2.5,
                    "branching_factor": 1.8,
                    "step_consistency": 0.5,
                    "nn_purity": 0.4,
                    "degenerate": False,
                },
            }
        },
        "default_geometry": "product",
        "backbone_dim": 512,
        "summary": {
            "analyzed_skill_groups": 1,
            "weak_signal_skills": ["algebraic"],
            "ambiguous_skills": [],
            "total_examples": 10,
        },
    }

    report_path = tmp_path / "stage2_report.md"
    report_text = write_stage2_report(skill_payload, geometry_payload, report_path)

    assert report_path.exists()
    assert "Stage 2 Geometry Analysis Report" in report_text
