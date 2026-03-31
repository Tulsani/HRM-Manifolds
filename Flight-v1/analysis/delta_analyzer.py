from __future__ import annotations

from typing import Iterable

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def compute_step_delta_consistency(step_matrix: np.ndarray) -> float:
    if step_matrix.ndim != 2 or step_matrix.shape[0] < 3:
        return 0.0
    deltas = step_matrix[1:] - step_matrix[:-1]
    if deltas.shape[0] < 2:
        return 0.0

    similarities = [
        _cosine_similarity(deltas[idx], deltas[idx + 1])
        for idx in range(deltas.shape[0] - 1)
    ]
    if not similarities:
        return 0.0
    return float(np.clip(np.mean(similarities), -1.0, 1.0))


def summarize_step_delta_consistency(step_matrices: Iterable[np.ndarray]) -> float:
    scores = [compute_step_delta_consistency(matrix) for matrix in step_matrices]
    if not scores:
        return 0.0
    return float(np.clip(np.mean(scores), -1.0, 1.0))
