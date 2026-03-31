from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RouterOutput:
    skill_probs: Tensor
    skill_idx: Tensor
    confidence: Tensor
    abstain_prob: Tensor
    skill_logits: Tensor


class Router(nn.Module):
    def __init__(
        self,
        input_dim: int,
        skill_names: list[str],
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.skill_names = list(skill_names)
        self.n_skills = len(self.skill_names)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.temperature = 1.0

        self.proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_p)
        self.skill_head = nn.Linear(self.hidden_dim, self.n_skills)
        self.conf_head = nn.Linear(self.hidden_dim, 1)
        self.abstain_head = nn.Linear(self.hidden_dim, 1)

    def _pool(self, h: Tensor) -> Tensor:
        if h.ndim == 3:
            return h.mean(dim=1)
        if h.ndim == 2:
            return h
        raise ValueError(f"Router expected [B, D] or [B, T, D], got shape {tuple(h.shape)}")

    def encode_features(self, h: Tensor) -> Tensor:
        pooled = self._pool(h)
        return self.dropout(self.act(self.norm(self.proj(pooled))))

    def compute_skill_logits(self, h: Tensor, apply_temperature: bool = True) -> Tensor:
        features = self.encode_features(h)
        logits = self.skill_head(features)
        if apply_temperature:
            logits = logits / max(float(self.temperature), 1e-6)
        return logits

    def compute_abstain_logits(self, h: Tensor) -> Tensor:
        features = self.encode_features(h)
        return self.abstain_head(features)

    def forward(self, h: Tensor, return_all: bool = False) -> RouterOutput:
        feat = self.encode_features(h)
        skill_logits = self.skill_head(feat) / max(float(self.temperature), 1e-6)
        skill_probs = torch.softmax(skill_logits, dim=-1)
        confidence = torch.sigmoid(self.conf_head(feat))
        abstain_prob = torch.sigmoid(self.abstain_head(feat))
        output = RouterOutput(
            skill_probs=skill_probs,
            skill_idx=skill_probs.argmax(dim=-1),
            confidence=confidence,
            abstain_prob=abstain_prob,
            skill_logits=skill_logits,
        )
        return output

    @torch.no_grad()
    def hard_route(self, h: Tensor) -> tuple[str, float]:
        output = self.forward(h)
        idx = int(output.skill_idx.view(-1)[0].item())
        conf = float(output.confidence.view(-1)[0].item())
        return self.skill_names[idx], conf

    @torch.no_grad()
    def soft_route(self, h: Tensor, threshold: float = 0.1) -> dict[str, float]:
        output = self.forward(h)
        probs = output.skill_probs.view(-1)
        return {
            skill: float(prob.item())
            for skill, prob in zip(self.skill_names, probs)
            if float(prob.item()) > threshold
        }

    @torch.no_grad()
    def should_abstain(self, h: Tensor, threshold: float = 0.5) -> bool:
        output = self.forward(h)
        return bool(float(output.abstain_prob.view(-1)[0].item()) > threshold)
