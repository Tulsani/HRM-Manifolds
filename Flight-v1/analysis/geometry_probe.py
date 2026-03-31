from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analysis.delta_analyzer import summarize_step_delta_consistency


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def cosine_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    normalized = _normalize_rows(matrix.astype(np.float32))
    similarities = normalized @ normalized.T
    distances = 1.0 - np.clip(similarities, -1.0, 1.0)
    np.fill_diagonal(distances, 0.0)
    return distances


def estimate_gromov_delta(
    embeddings: np.ndarray,
    n_triples: int = 500,
    rng_seed: int = 42,
) -> dict[str, Any]:
    n_examples = embeddings.shape[0]
    if n_examples < 3:
        return {"delta_norm": None, "degenerate": False, "diameter": 0.0}
    if np.allclose(embeddings, embeddings[0]):
        return {"delta_norm": 0.0, "degenerate": True, "diameter": 0.0}

    distances = cosine_distance_matrix(embeddings)
    diameter = float(np.max(distances))
    if diameter <= 1e-8:
        return {"delta_norm": 0.0, "degenerate": True, "diameter": diameter}

    rng = random.Random(rng_seed)
    slack_values: list[float] = []
    indices = list(range(n_examples))
    for _ in range(n_triples):
        if n_examples >= 4:
            a, b, c, d = rng.sample(indices, 4)
        else:
            sample = rng.sample(indices, 3)
            a, b, c = sample
            d = sample[0]

        s1 = distances[a, b] + distances[c, d]
        s2 = distances[a, c] + distances[b, d]
        s3 = distances[a, d] + distances[b, c]
        ordered = sorted([s1, s2, s3], reverse=True)
        slack_values.append(max(ordered[0] - ordered[1], 0.0) / 2.0)

    delta = float(max(slack_values) if slack_values else 0.0)
    return {
        "delta_norm": float(np.clip(delta / diameter, 0.0, 1.0)),
        "degenerate": False,
        "diameter": diameter,
    }


def build_subskill_trie_stats(subskill_chains: list[list[str]]) -> dict[str, float]:
    if not subskill_chains:
        return {"avg_depth": 0.0, "max_depth": 0.0, "branching_factor": 0.0}

    branching_values: list[int] = []
    total_depth = 0
    max_depth = 0
    root: dict[str, Any] = {}

    for chain in subskill_chains:
        node = root
        total_depth += len(chain)
        max_depth = max(max_depth, len(chain))
        for step in chain:
            node = node.setdefault(step, {})

    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            branching_values.append(len(node))
            stack.extend(node.values())

    return {
        "avg_depth": float(total_depth / max(len(subskill_chains), 1)),
        "max_depth": float(max_depth),
        "branching_factor": float(np.mean(branching_values) if branching_values else 0.0),
    }


def compute_nn_purity(
    embeddings: np.ndarray,
    primary_subskills: list[str],
    nn_k: int = 5,
) -> float:
    n_examples = len(primary_subskills)
    if n_examples <= 1:
        return 0.0

    distances = cosine_distance_matrix(embeddings)
    purity_scores: list[float] = []
    for idx in range(n_examples):
        order = np.argsort(distances[idx])
        neighbors = [candidate for candidate in order if candidate != idx][:nn_k]
        if not neighbors:
            continue
        same = sum(primary_subskills[candidate] == primary_subskills[idx] for candidate in neighbors)
        purity_scores.append(same / len(neighbors))
    if not purity_scores:
        return 0.0
    return float(np.mean(purity_scores))


def decide_geometry(
    delta_norm: float | None,
    avg_depth: float,
    consistency: float,
    purity: float,
) -> dict[str, Any]:
    if delta_norm is not None and delta_norm < 0.1 and avg_depth >= 3.0:
        return {"geometry": "hyperbolic", "curvature": -1.0, "h_dim": 32, "e_dim": 0}
    if delta_norm is not None and delta_norm < 0.3 and consistency > 0.4:
        return {"geometry": "product", "curvature": -1.0, "h_dim": 16, "e_dim": 16}
    if consistency > 0.4 and purity > 0.5:
        return {"geometry": "euclidean", "curvature": 0.0, "h_dim": 0, "e_dim": 32}
    return {"geometry": "product", "curvature": -1.0, "h_dim": 16, "e_dim": 16}


def _decision_is_ambiguous(results: dict[str, float | None]) -> bool:
    delta_norm = results.get("delta_norm")
    checks = [
        delta_norm is not None and abs(float(delta_norm) - 0.1) < 0.03,
        delta_norm is not None and abs(float(delta_norm) - 0.3) < 0.03,
        abs(float(results.get("step_consistency", 0.0)) - 0.4) < 0.05,
        abs(float(results.get("nn_purity", 0.0)) - 0.5) < 0.05,
    ]
    return any(checks)


def run_geometry_analysis(config: dict[str, Any]) -> dict[str, Any]:
    probe_cfg = config["geometry_probe"]
    output_cfg = config["output"]
    labels_payload = _load_json(output_cfg["skill_labels"])
    labels = labels_payload["labels"]
    summary = labels_payload["summary"]
    embeddings_dir = Path(output_cfg["embeddings_dir"])
    problem_embeddings = _load_npz(embeddings_dir / "problem_embeddings.npz")
    step_embeddings = _load_npz(embeddings_dir / "step_embeddings.npz")

    grouped_example_ids: dict[str, list[str]] = defaultdict(list)
    grouped_subskills: dict[str, list[list[str]]] = defaultdict(list)
    grouped_primary_subskills: dict[str, list[str]] = defaultdict(list)
    for example_id, label_info in labels.items():
        skill = label_info["skill"]
        grouped_example_ids[skill].append(example_id)
        subskills = list(label_info.get("subskills", []))
        grouped_subskills[skill].append(subskills)
        grouped_primary_subskills[skill].append(subskills[0] if subskills else "none")

    all_skills = sorted(grouped_example_ids)
    skills_output: dict[str, Any] = {}
    weak_signal_skills: list[str] = []
    ambiguous_skills: list[str] = []
    analyzed_count = 0
    backbone_dim = 0

    for skill in all_skills:
        example_ids = [example_id for example_id in grouped_example_ids[skill] if example_id in problem_embeddings]
        primary_subskill_by_example = {
            example_id: grouped_primary_subskills[skill][idx]
            for idx, example_id in enumerate(grouped_example_ids[skill])
            if example_id in problem_embeddings
        }
        if example_ids:
            backbone_dim = int(problem_embeddings[example_ids[0]].shape[-1])
        n_examples = len(example_ids)

        if n_examples < int(probe_cfg["min_examples_per_skill"]):
            weak_signal_skills.append(skill)
            print(f"Warning: skill '{skill}' has only {n_examples} examples; using default product geometry.")
            test_results = {
                "delta_norm": None,
                "avg_depth": 0.0,
                "branching_factor": 0.0,
                "step_consistency": 0.0,
                "nn_purity": 0.0,
                "degenerate": False,
            }
            decision = decide_geometry(None, 0.0, 0.0, 0.0)
        else:
            analyzed_count += 1
            sample_size = min(int(probe_cfg["sample_size"]), n_examples)
            rng = random.Random(42)
            sampled_ids = rng.sample(example_ids, sample_size) if sample_size < n_examples else list(example_ids)
            sampled_embeddings = np.stack([problem_embeddings[example_id] for example_id in sampled_ids], axis=0)

            delta_stats = estimate_gromov_delta(
                sampled_embeddings,
                n_triples=int(probe_cfg["n_triples"]),
            )
            trie_stats = build_subskill_trie_stats(grouped_subskills[skill])
            consistency = summarize_step_delta_consistency(
                [step_embeddings.get(example_id, np.zeros((0, backbone_dim), dtype=np.float32)) for example_id in example_ids]
            )
            purity = compute_nn_purity(
                sampled_embeddings,
                [primary_subskill_by_example.get(example_id, "none") for example_id in sampled_ids],
                nn_k=int(probe_cfg["nn_k"]),
            )

            test_results = {
                "delta_norm": delta_stats["delta_norm"],
                "avg_depth": trie_stats["avg_depth"],
                "branching_factor": trie_stats["branching_factor"],
                "step_consistency": consistency,
                "nn_purity": purity,
                "degenerate": bool(delta_stats["degenerate"]),
            }
            decision = decide_geometry(
                delta_norm=delta_stats["delta_norm"],
                avg_depth=trie_stats["avg_depth"],
                consistency=consistency,
                purity=purity,
            )
            if _decision_is_ambiguous(test_results):
                ambiguous_skills.append(skill)

        skills_output[skill] = {
            "geometry": decision["geometry"],
            "curvature": decision["curvature"],
            "h_dim": decision["h_dim"],
            "e_dim": decision["e_dim"],
            "n_examples": n_examples,
            "test_results": test_results,
        }

    geometry_payload = {
        "skills": skills_output,
        "default_geometry": "product",
        "backbone_dim": backbone_dim,
        "summary": {
            "analyzed_skill_groups": analyzed_count,
            "weak_signal_skills": weak_signal_skills,
            "ambiguous_skills": ambiguous_skills,
            "total_examples": summary["total"],
        },
    }

    output_path = Path(output_cfg["geometry_config"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(geometry_payload, handle, indent=2, sort_keys=True)
    return geometry_payload
