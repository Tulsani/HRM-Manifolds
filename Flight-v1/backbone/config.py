from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BackboneConfig:
    vocab_size: int = 32000
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    ffn_multiplier: int = 4
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    gradient_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        return self.d_model * self.ffn_multiplier

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "BackboneConfig":
        return cls(**data)
