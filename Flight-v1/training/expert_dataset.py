from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_SKILL_LABELS = Path("outputs/skill_labels.json")
DEFAULT_PROBLEM_EMBEDDINGS = Path("trace_library/embeddings/problem_embeddings.npz")
DEFAULT_STEP_EMBEDDINGS = Path("trace_library/embeddings/step_embeddings.npz")


class SkillExpertDataset(Dataset):
    def __init__(
        self,
        skill: str,
        skill_labels_path: str | Path = DEFAULT_SKILL_LABELS,
        problem_embeddings_path: str | Path = DEFAULT_PROBLEM_EMBEDDINGS,
        step_embeddings_path: str | Path = DEFAULT_STEP_EMBEDDINGS,
    ) -> None:
        self.skill = skill
        with Path(skill_labels_path).open("r", encoding="utf-8") as handle:
            labels_payload = json.load(handle)
        self.labels = labels_payload["labels"]

        problem_loaded = np.load(problem_embeddings_path, allow_pickle=False)
        step_loaded = np.load(step_embeddings_path, allow_pickle=False)
        self.problem_embeddings = {key: problem_loaded[key].astype(np.float32) for key in problem_loaded.files}
        self.step_embeddings = {key: step_loaded[key].astype(np.float32) for key in step_loaded.files}

        self.example_ids = [
            example_id
            for example_id, info in sorted(self.labels.items())
            if info["skill"] == skill and example_id in self.problem_embeddings
        ]

        primary_subskills = sorted(
            {
                (self.labels[example_id]["subskills"][0] if self.labels[example_id]["subskills"] else "none")
                for example_id in self.example_ids
            }
        )
        self.prototype_to_id = {name: idx for idx, name in enumerate(primary_subskills)}
        self.id_to_prototype = {idx: name for name, idx in self.prototype_to_id.items()}

    def __len__(self) -> int:
        return len(self.example_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example_id = self.example_ids[index]
        label_info = self.labels[example_id]
        primary_subskill = label_info["subskills"][0] if label_info["subskills"] else "none"
        return {
            "example_id": example_id,
            "teacher_embedding": torch.from_numpy(self.problem_embeddings[example_id]),
            "step_embeddings": torch.from_numpy(
                self.step_embeddings.get(
                    example_id,
                    np.zeros((0, self.problem_embeddings[example_id].shape[-1]), dtype=np.float32),
                )
            ),
            "prototype_id": int(self.prototype_to_id[primary_subskill]),
            "primary_subskill": primary_subskill,
        }


def collate_expert_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "example_ids": [item["example_id"] for item in batch],
        "teacher_embeddings": torch.stack([item["teacher_embedding"] for item in batch], dim=0),
        "step_embeddings": [item["step_embeddings"] for item in batch],
        "prototype_ids": torch.tensor([item["prototype_id"] for item in batch], dtype=torch.long),
        "primary_subskills": [item["primary_subskill"] for item in batch],
    }


def build_expert_dataloader(
    skill: str,
    batch_size: int,
    skill_labels_path: str | Path = DEFAULT_SKILL_LABELS,
    problem_embeddings_path: str | Path = DEFAULT_PROBLEM_EMBEDDINGS,
    step_embeddings_path: str | Path = DEFAULT_STEP_EMBEDDINGS,
) -> tuple[SkillExpertDataset, DataLoader]:
    dataset = SkillExpertDataset(
        skill=skill,
        skill_labels_path=skill_labels_path,
        problem_embeddings_path=problem_embeddings_path,
        step_embeddings_path=step_embeddings_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_expert_batch,
    )
    return dataset, loader
