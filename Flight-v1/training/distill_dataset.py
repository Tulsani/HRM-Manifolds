from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from backbone.tokenizer import TokenizerWrapper
from data.trace_schema import TracePackage, load_sparse_logits


class DistillDataset(Dataset):
    def __init__(
        self,
        trace_library_path: str | Path = "trace_library",
        skill_labels_path: str | Path = "outputs/skill_labels.json",
        problem_embeddings_path: str | Path = "trace_library/embeddings/problem_embeddings.npz",
        step_embeddings_path: str | Path = "trace_library/embeddings/step_embeddings.npz",
        backbone_config_path: str | Path = "configs/backbone_small.yaml",
        tokenizer_override: TokenizerWrapper | Any | None = None,
        hardest_only: bool = False,
    ) -> None:
        self.trace_library_path = Path(trace_library_path)
        with Path(skill_labels_path).open("r", encoding="utf-8") as handle:
            labels_payload = json.load(handle)
        self.skill_labels = labels_payload["labels"]
        self.skill_names = sorted({info["skill"] for info in self.skill_labels.values()})
        self.skill_to_idx = {skill: idx for idx, skill in enumerate(self.skill_names)}
        self.subskill_names = sorted(
            {
                subskill
                for info in self.skill_labels.values()
                for subskill in info.get("subskills", [])
            }
        )
        self.subskill_to_idx = {name: idx for idx, name in enumerate(self.subskill_names)}
        self.step_embeddings = np.load(step_embeddings_path, allow_pickle=False)
        self.problem_embeddings = np.load(problem_embeddings_path, allow_pickle=False)

        if tokenizer_override is not None:
            self.tokenizer = tokenizer_override
        else:
            with open(backbone_config_path, "r", encoding="utf-8") as handle:
                backbone_cfg = yaml.safe_load(handle)
            tokenizer_name = backbone_cfg.get("tokenizer", {}).get("model_name") or backbone_cfg.get("teacher", {}).get("model_name")
            self.tokenizer = TokenizerWrapper(model_name=tokenizer_name, max_length=backbone_cfg["model"]["max_seq_len"])

        self.vocab_size = int(self.tokenizer.vocab_size)
        self.items: list[dict[str, Any]] = []
        for source in ("gsm8k", "math"):
            jsonl_path = self.trace_library_path / f"{source}_traces.jsonl"
            logits_path = self.trace_library_path / "logits" / f"{source}_logits.npz"
            logits_map = load_sparse_logits(logits_path)
            if not jsonl_path.exists():
                continue
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    example_id = payload["example_id"]
                    if example_id not in self.skill_labels:
                        continue
                    trace = TracePackage.from_jsonl(line, sparse_logits=logits_map.get(example_id))
                    label_info = self.skill_labels[example_id]
                    if hardest_only and not (label_info["difficulty"] == "hard" or label_info["incorrect"]):
                        continue
                    self.items.append(self._build_item(trace, label_info))

    def _sample_weight(self, low_confidence: bool, incorrect: bool) -> float:
        weight = 1.0
        if low_confidence:
            weight = min(weight, 0.7)
        if incorrect:
            weight = min(weight, 0.5)
        return weight

    def _build_item(self, trace: TracePackage, label_info: dict[str, Any]) -> dict[str, Any]:
        input_ids = self.tokenizer.tokenizer(trace.problem, add_special_tokens=True)["input_ids"]
        target_text = trace.rationale if trace.rationale else trace.final_answer
        target_ids = self.tokenizer.tokenizer(target_text, add_special_tokens=True)["input_ids"]
        dense_teacher_logits = torch.zeros(self.vocab_size, dtype=torch.float32)
        if trace.soft_logits.indices.size > 0:
            dense_teacher_logits[torch.from_numpy(trace.soft_logits.indices).long()] = torch.from_numpy(trace.soft_logits.values.astype(np.float32))
        subskill_labels = [self.subskill_to_idx[sub] for sub in label_info.get("subskills", []) if sub in self.subskill_to_idx]
        step_embed = self.step_embeddings[trace.example_id].astype(np.float32) if trace.example_id in self.step_embeddings.files else np.zeros((0, self.problem_embeddings[trace.example_id].shape[-1]), dtype=np.float32)
        return {
            "example_id": trace.example_id,
            "problem_text": trace.problem,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "teacher_logits": dense_teacher_logits,
            "teacher_logit_indices": torch.from_numpy(trace.soft_logits.indices.astype(np.int64)),
            "step_embeddings": torch.from_numpy(step_embed),
            "skill_label": int(self.skill_to_idx[label_info["skill"]]),
            "subskill_labels": subskill_labels,
            "difficulty": {"easy": 0, "medium": 1, "hard": 2}.get(label_info["difficulty"], 1),
            "low_confidence": bool(label_info["low_confidence"]),
            "incorrect": bool(label_info["incorrect"]),
            "sample_weight": self._sample_weight(bool(label_info["low_confidence"]), bool(label_info["incorrect"])),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        pad_id = int(self.tokenizer.pad_token_id)
        max_input = max(item["input_ids"].numel() for item in batch)
        max_target = max(item["target_ids"].numel() for item in batch)
        max_steps = max(item["step_embeddings"].shape[0] for item in batch)
        step_dim = batch[0]["step_embeddings"].shape[-1] if max_steps > 0 else (batch[0]["teacher_logits"].numel() * 0 + self.problem_embeddings[batch[0]["example_id"]].shape[-1])
        max_subskills = max(len(item["subskill_labels"]) for item in batch)

        input_ids = torch.full((len(batch), max_input), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_input), dtype=torch.long)
        target_ids = torch.full((len(batch), max_target), pad_id, dtype=torch.long)
        step_embeddings = torch.zeros((len(batch), max_steps, step_dim), dtype=torch.float32)
        step_mask = torch.zeros((len(batch), max_steps), dtype=torch.bool)
        subskill_labels = torch.full((len(batch), max_subskills), -1, dtype=torch.long)

        for idx, item in enumerate(batch):
            input_len = item["input_ids"].numel()
            target_len = item["target_ids"].numel()
            step_len = item["step_embeddings"].shape[0]
            input_ids[idx, :input_len] = item["input_ids"]
            attention_mask[idx, :input_len] = 1
            target_ids[idx, :target_len] = item["target_ids"]
            if step_len > 0:
                step_embeddings[idx, :step_len] = item["step_embeddings"]
                step_mask[idx, :step_len] = True
            if item["subskill_labels"]:
                subskill_labels[idx, : len(item["subskill_labels"])] = torch.tensor(item["subskill_labels"], dtype=torch.long)

        return {
            "example_ids": [item["example_id"] for item in batch],
            "problem_text": [item["problem_text"] for item in batch],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "teacher_logits": torch.stack([item["teacher_logits"] for item in batch], dim=0),
            "teacher_logit_indices": [item["teacher_logit_indices"] for item in batch],
            "step_embeddings": step_embeddings,
            "step_mask": step_mask,
            "skill_label": torch.tensor([item["skill_label"] for item in batch], dtype=torch.long),
            "subskill_labels": subskill_labels,
            "difficulty": torch.tensor([item["difficulty"] for item in batch], dtype=torch.long),
            "low_confidence": torch.tensor([item["low_confidence"] for item in batch], dtype=torch.bool),
            "incorrect": torch.tensor([item["incorrect"] for item in batch], dtype=torch.bool),
            "sample_weights": torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float32),
        }
