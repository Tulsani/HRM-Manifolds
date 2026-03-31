from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SparseLogits:
    indices: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int32)
        self.values = np.asarray(self.values, dtype=np.float16)
        if self.indices.ndim != 1 or self.values.ndim != 1:
            raise ValueError("Sparse logits indices and values must be 1D arrays.")
        if len(self.indices) != len(self.values):
            raise ValueError("Sparse logits indices and values must have the same length.")

    def to_json(self) -> dict[str, list[Any]]:
        return {
            "indices": self.indices.astype(np.int32).tolist(),
            "values": self.values.astype(np.float16).tolist(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SparseLogits":
        return cls(indices=np.asarray(payload["indices"]), values=np.asarray(payload["values"]))


@dataclass
class TracePackage:
    example_id: str
    source: str
    difficulty: str
    problem: str
    final_answer: str
    answer_correct: bool
    soft_logits: SparseLogits
    confidence: float
    rationale: str
    step_texts: list[str]
    n_steps: int
    skill: str
    subskills: list[str]
    math_subject: str
    teacher_model: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_logits: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload["soft_logits"] = self.soft_logits.to_json() if include_logits else {
            "nnz": int(len(self.soft_logits.indices))
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TracePackage":
        soft_logits = payload.get("soft_logits", {"indices": [], "values": []})
        return cls(
            example_id=payload["example_id"],
            source=payload["source"],
            difficulty=payload["difficulty"],
            problem=payload["problem"],
            final_answer=payload["final_answer"],
            answer_correct=bool(payload["answer_correct"]),
            soft_logits=SparseLogits.from_json(soft_logits),
            confidence=float(payload["confidence"]),
            rationale=payload["rationale"],
            step_texts=list(payload["step_texts"]),
            n_steps=int(payload["n_steps"]),
            skill=payload["skill"],
            subskills=list(payload["subskills"]),
            math_subject=payload["math_subject"],
            teacher_model=payload["teacher_model"],
            timestamp=payload["timestamp"],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(include_logits=False), ensure_ascii=True)

    @classmethod
    def from_jsonl(cls, line: str, sparse_logits: SparseLogits | None = None) -> "TracePackage":
        payload = json.loads(line)
        if sparse_logits is not None:
            payload["soft_logits"] = sparse_logits.to_json()
        return cls.from_dict(payload)


def save_sparse_logits(path: str | Path, logits_by_example: dict[str, SparseLogits]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for example_id, sparse in logits_by_example.items():
        arrays[f"{example_id}__indices"] = sparse.indices.astype(np.int32)
        arrays[f"{example_id}__values"] = sparse.values.astype(np.float16)
    np.savez(output_path, **arrays)


def load_sparse_logits(path: str | Path) -> dict[str, SparseLogits]:
    input_path = Path(path)
    if not input_path.exists():
        return {}

    loaded = np.load(input_path, allow_pickle=False)
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for key in loaded.files:
        example_id, suffix = key.rsplit("__", maxsplit=1)
        grouped.setdefault(example_id, {})[suffix] = loaded[key]

    return {
        example_id: SparseLogits(indices=parts.get("indices", np.array([], dtype=np.int32)), values=parts.get("values", np.array([], dtype=np.float16)))
        for example_id, parts in grouped.items()
    }
