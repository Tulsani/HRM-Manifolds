from __future__ import annotations

from torch import Tensor, nn


class OutputHead(nn.Module):
    def __init__(self, manifold_dim: int, vocab_size: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.manifold_dim = int(manifold_dim)
        self.hidden_dim = int(hidden_dim or (self.manifold_dim * 2))
        self.vocab_size = int(vocab_size)
        self.norm = nn.LayerNorm(self.manifold_dim)
        self.proj1 = nn.Linear(self.manifold_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, z: Tensor) -> Tensor:
        if z.ndim == 2:
            return self.proj2(self.act(self.proj1(self.norm(z))))
        if z.ndim == 3:
            return self.proj2(self.act(self.proj1(self.norm(z))))
        raise ValueError(f"OutputHead expected [B, D] or [B, T, D], got shape {tuple(z.shape)}")
