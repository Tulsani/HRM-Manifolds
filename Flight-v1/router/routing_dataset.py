from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


DIFFICULTY_TO_ID = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
}


class RoutingDataset(Dataset):
    def __init__(
        self,
        embeddings_path: str | Path,
        skill_labels_path: str | Path,
        split: str = "train",
        train_split: float = 0.85,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        embeddings_npz = np.load(embeddings_path, allow_pickle=False)
        self.embeddings = {key: embeddings_npz[key].astype(np.float32) for key in embeddings_npz.files}
        with Path(skill_labels_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.labels = payload["labels"]
        self.summary = payload.get("summary", {})

        self.skill_names = sorted({info["skill"] for info in self.labels.values()})
        self.skill_to_idx = {skill: idx for idx, skill in enumerate(self.skill_names)}

        subskills = sorted(
            {
                subskill
                for info in self.labels.values()
                for subskill in info.get("subskills", [])
            }
        )
        self.subskill_names = subskills
        self.subskill_to_idx = {subskill: idx for idx, subskill in enumerate(self.subskill_names)}
        self.n_subskills = len(self.subskill_names)

        grouped_ids: dict[str, list[str]] = defaultdict(list)
        for example_id, info in self.labels.items():
            if example_id in self.embeddings:
                grouped_ids[info["skill"]].append(example_id)

        rng = random.Random(seed)
        split_ids: list[str] = []
        for skill in self.skill_names:
            ids = sorted(grouped_ids.get(skill, []))
            rng.shuffle(ids)
            if not ids:
                continue
            if len(ids) == 1:
                selected = ids
            else:
                val_count = min(max(1, int(round(len(ids) * (1.0 - train_split)))), len(ids) - 1)
                train_count = len(ids) - val_count
                train_ids = ids[:train_count]
                val_ids = ids[train_count:]
                selected = train_ids if split == "train" else val_ids
            split_ids.extend(selected)

        self.example_ids = sorted(split_ids)
        self.skill_counts = Counter(self.labels[example_id]["skill"] for example_id in self.example_ids)
        self.skill_to_subskills = self._build_skill_to_subskills()

    def _build_skill_to_subskills(self) -> dict[str, set[int]]:
        mapping: dict[str, set[int]] = {skill: set() for skill in self.skill_names}
        for example_id in self.example_ids:
            info = self.labels[example_id]
            indices = [self.subskill_to_idx[sub] for sub in info.get("subskills", []) if sub in self.subskill_to_idx]
            mapping[info["skill"]].update(indices)
        return mapping

    def __len__(self) -> int:
        return len(self.example_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example_id = self.example_ids[index]
        info = self.labels[example_id]
        subskill_labels = [self.subskill_to_idx[sub] for sub in info.get("subskills", []) if sub in self.subskill_to_idx]
        return {
            "example_id": example_id,
            "embedding": torch.from_numpy(self.embeddings[example_id]),
            "skill_label": int(self.skill_to_idx[info["skill"]]),
            "subskill_labels": subskill_labels,
            "difficulty": int(DIFFICULTY_TO_ID.get(info["difficulty"], 1)),
            "low_confidence": bool(info.get("low_confidence", False)),
            "incorrect": bool(info.get("incorrect", False)),
        }

    def get_skill_weights(self) -> torch.Tensor:
        counts = torch.tensor(
            [float(self.skill_counts.get(skill, 0)) for skill in self.skill_names],
            dtype=torch.float32,
        ).clamp_min(1.0)
        weights = 1.0 / counts
        weights = weights * (len(self.skill_names) / weights.sum().clamp_min(1e-8))
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        skill_weights = self.get_skill_weights()
        sample_weights = [
            float(skill_weights[self.skill_to_idx[self.labels[example_id]["skill"]]].item())
            for example_id in self.example_ids
        ]
        return torch.tensor(sample_weights, dtype=torch.float32)


def collate_routing_batch(batch: list[dict[str, Any]], n_subskills: int) -> dict[str, Any]:
    embeddings = torch.stack([item["embedding"] for item in batch], dim=0)
    subskill_multihot = torch.zeros((len(batch), n_subskills), dtype=torch.float32)
    for row, item in enumerate(batch):
        if item["subskill_labels"]:
            subskill_multihot[row, item["subskill_labels"]] = 1.0
    return {
        "example_ids": [item["example_id"] for item in batch],
        "embeddings": embeddings,
        "skill_labels": torch.tensor([item["skill_label"] for item in batch], dtype=torch.long),
        "subskill_labels": [item["subskill_labels"] for item in batch],
        "subskill_multihot": subskill_multihot,
        "difficulty": torch.tensor([item["difficulty"] for item in batch], dtype=torch.long),
        "low_confidence": torch.tensor([item["low_confidence"] for item in batch], dtype=torch.bool),
        "incorrect": torch.tensor([item["incorrect"] for item in batch], dtype=torch.bool),
    }


def build_routing_dataloaders(
    embeddings_path: str | Path,
    skill_labels_path: str | Path,
    train_split: float,
    batch_size: int,
    seed: int = 42,
) -> tuple[RoutingDataset, RoutingDataset, DataLoader, DataLoader]:
    train_dataset = RoutingDataset(
        embeddings_path=embeddings_path,
        skill_labels_path=skill_labels_path,
        split="train",
        train_split=train_split,
        seed=seed,
    )
    val_dataset = RoutingDataset(
        embeddings_path=embeddings_path,
        skill_labels_path=skill_labels_path,
        split="val",
        train_split=train_split,
        seed=seed,
    )

    sampler = WeightedRandomSampler(
        weights=train_dataset.get_sample_weights(),
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda batch: collate_routing_batch(batch, train_dataset.n_subskills),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_routing_batch(batch, val_dataset.n_subskills),
    )
    return train_dataset, val_dataset, train_loader, val_loader
