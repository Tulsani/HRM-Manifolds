from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from data.trace_schema import TracePackage, load_sparse_logits

SUBJECT_OVERRIDES = {
    "algebra": "algebraic",
    "number theory": "number_theory",
    "geometry": "geometric",
    "counting & probability": "combinatorics",
    "counting/prob": "combinatorics",
    "counting/probability": "combinatorics",
    "precalculus": "algebraic",
    "intermediate algebra": "algebraic",
    "intermediate alg": "algebraic",
}


def _load_traces_for_source(trace_dir: str | Path, source: str) -> list[TracePackage]:
    trace_dir = Path(trace_dir)
    jsonl_path = trace_dir / f"{source}_traces.jsonl"
    logits_path = trace_dir / "logits" / f"{source}_logits.npz"
    logits = load_sparse_logits(logits_path)
    traces: list[TracePackage] = []
    if not jsonl_path.exists():
        return traces
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            traces.append(TracePackage.from_jsonl(line, sparse_logits=logits.get(payload["example_id"])))
    return traces


def load_all_traces(trace_dir: str | Path) -> list[TracePackage]:
    traces: list[TracePackage] = []
    for source in ("gsm8k", "math"):
        traces.extend(_load_traces_for_source(trace_dir, source))
    traces.sort(key=lambda trace: trace.example_id)
    return traces


def subject_override(math_subject: str) -> str | None:
    subject = math_subject.strip().lower()
    return SUBJECT_OVERRIDES.get(subject)


def refine_skill(trace: TracePackage) -> dict[str, Any]:
    skill = trace.skill
    if trace.source == "math":
        overridden = subject_override(trace.math_subject)
        if overridden is not None:
            skill = overridden

    distinct_subskills = len(set(trace.subskills))
    if trace.n_steps >= 4 and distinct_subskills >= 3:
        skill = "multi_step_reason"

    return {
        "skill": skill,
        "subskills": list(trace.subskills),
        "n_steps": int(trace.n_steps),
        "low_confidence": bool(trace.confidence < 0.3),
        "incorrect": bool(not trace.answer_correct),
        "difficulty": trace.difficulty,
        "source": trace.source,
    }


def build_skill_label_payload(traces: list[TracePackage]) -> dict[str, Any]:
    labels: dict[str, dict[str, Any]] = {}
    skill_counts: Counter[str] = Counter()
    low_confidence_count = 0
    incorrect_count = 0

    for trace in traces:
        refined = refine_skill(trace)
        labels[trace.example_id] = refined
        skill_counts[refined["skill"]] += 1
        low_confidence_count += int(refined["low_confidence"])
        incorrect_count += int(refined["incorrect"])

    return {
        "labels": labels,
        "summary": {
            "skill_counts": dict(sorted(skill_counts.items())),
            "low_confidence_count": low_confidence_count,
            "incorrect_count": incorrect_count,
            "total": len(labels),
        },
    }


def refine_skill_labels(trace_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    traces = load_all_traces(trace_dir)
    payload = build_skill_label_payload(traces)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload
