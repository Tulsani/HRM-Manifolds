from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from transformers import AutoTokenizer


@dataclass
class TokenizerWrapper:
    model_name: str
    max_length: int = 1024

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = self.max_length

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def encode_batch(self, texts: Iterable[str]) -> dict:
        return self.tokenizer(
            list(texts),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
        )

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
