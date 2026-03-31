from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from data.trace_schema import TracePackage, load_sparse_logits


class TraceLibraryDataset(Dataset):
    def __init__(self, trace_jsonl: str | Path, logits_npz: str | Path) -> None:
        self.trace_jsonl = Path(trace_jsonl)
        self.logits = load_sparse_logits(logits_npz)
        with self.trace_jsonl.open("r", encoding="utf-8") as handle:
            self.lines = [line.rstrip("\n") for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index: int) -> TracePackage:
        payload = json.loads(self.lines[index])
        example_id = payload["example_id"]
        sparse_logits = self.logits.get(example_id)
        return TracePackage.from_jsonl(self.lines[index], sparse_logits=sparse_logits)
