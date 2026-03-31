from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

from backbone.tokenizer import TokenizerWrapper


def format_example(example: dict[str, Any]) -> str:
    if "question" in example and "answer" in example:
        return f"Question: {example['question'].strip()}\nAnswer: {example['answer'].strip()}"
    instruction = example.get("instruction") or example.get("inputs") or example.get("input") or ""
    output = example.get("output") or example.get("targets") or example.get("response") or ""
    parts = []
    if instruction:
        parts.append(f"Instruction: {instruction.strip()}")
    if output:
        parts.append(f"Response: {output.strip()}")
    text = "\n".join(parts).strip()
    return text or str(example)


def _sample_dataset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None or len(dataset) <= limit:
        return dataset
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(limit))


def _load_instruction_dataset(name: str, split: str, subset_limit: int, seed: int) -> Dataset:
    if name.lower() == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split=split)
    elif name.lower() == "flan":
        dataset = load_dataset("Muennighoff/flan", "flan2021", split=split)
    else:
        dataset = load_dataset(name, split=split)
    return _sample_dataset(dataset, subset_limit, seed)


def build_training_datasets(
    instruction_dataset: str = "alpaca",
    instruction_limit: int = 50_000,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    gsm8k = load_dataset("gsm8k", "main")
    gsm_train = gsm8k["train"].map(lambda ex: {"text": format_example(ex)}, remove_columns=gsm8k["train"].column_names)
    gsm_eval = gsm_train.select(range(max(len(gsm_train) - 500, 0), len(gsm_train)))
    gsm_train = gsm_train.select(range(max(len(gsm_train) - 500, 0)))
    instruction_train = _load_instruction_dataset(instruction_dataset, "train", instruction_limit, seed)
    instruction_train = instruction_train.map(
        lambda ex: {"text": format_example(ex)},
        remove_columns=instruction_train.column_names,
    )

    mixed_train = concatenate_datasets([gsm_train, instruction_train]).shuffle(seed=seed)
    return mixed_train, gsm_eval


@dataclass
class DataCollatorForPackedCausalLM:
    tokenizer: TokenizerWrapper
    max_length: int

    def _pack(self, tokenized_texts: Iterable[list[int]]) -> list[list[int]]:
        packed: list[list[int]] = []
        current: list[int] = []
        eos = [self.tokenizer.eos_token_id]

        for ids in tokenized_texts:
            ids = ids[: self.max_length - 1] + eos
            if len(current) + len(ids) > self.max_length and current:
                packed.append(current[: self.max_length])
                current = []
            current.extend(ids)

            while len(current) >= self.max_length:
                packed.append(current[: self.max_length])
                current = current[self.max_length :]

        if current:
            packed.append(current)
        return packed

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [sample["text"] if "text" in sample else format_example(sample) for sample in batch]
        tokenized = self.tokenizer.encode_batch(texts)["input_ids"]
        packed = self._pack(tokenized)
        if not packed:
            packed = [[self.tokenizer.eos_token_id]]

        batch_size = len(packed)
        input_ids = torch.full(
            (batch_size, self.max_length),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, self.max_length), dtype=torch.long)
        labels = torch.full((batch_size, self.max_length), fill_value=-100, dtype=torch.long)

        for idx, seq in enumerate(packed):
            seq_len = min(len(seq), self.max_length)
            input_ids[idx, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)
            attention_mask[idx, :seq_len] = 1
            labels[idx, :seq_len] = input_ids[idx, :seq_len]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_dataloaders(
    tokenizer: TokenizerWrapper,
    batch_size: int,
    max_length: int,
    instruction_dataset: str = "alpaca",
    instruction_limit: int = 50_000,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, eval_dataset = build_training_datasets(
        instruction_dataset=instruction_dataset,
        instruction_limit=instruction_limit,
        seed=seed,
    )
    collator = DataCollatorForPackedCausalLM(tokenizer=tokenizer, max_length=max_length)
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )
    return train_loader, eval_loader
