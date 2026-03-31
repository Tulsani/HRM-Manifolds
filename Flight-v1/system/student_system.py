from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from backbone.config import BackboneConfig
from backbone.model import CausalTransformerBackbone
from backbone.tokenizer import TokenizerWrapper
from manifolds import build_expert
from router.router import Router
from system.output_head import OutputHead


@dataclass
class SystemOutput:
    logits: Tensor
    skill_probs: Tensor
    active_skill: str
    z: Tensor
    abstained: bool
    router_conf: float


class StudentSystem(nn.Module):
    def __init__(
        self,
        backbone: CausalTransformerBackbone,
        router: Router,
        experts: nn.ModuleDict,
        heads: nn.ModuleDict,
        tokenizer: TokenizerWrapper | Any,
        routing_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.router = router
        self.experts = experts
        self.heads = heads
        self.tokenizer = tokenizer
        self.skill_names = list(router.skill_names)
        self.routing_threshold = float(routing_threshold)

    def _combine_token_representations(
        self,
        hidden_states: Tensor,
        skill_probs: Tensor,
        routing_mode: str,
    ) -> tuple[Tensor, list[str]]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch_size * seq_len, hidden_dim)
        per_skill_token_z: dict[str, Tensor] = {}
        for skill in self.skill_names:
            expert = self.experts[skill]
            z_skill = expert.encode(flat_hidden)
            z_skill_euclid = expert.to_euclidean(z_skill).reshape(batch_size, seq_len, -1)
            per_skill_token_z[skill] = z_skill_euclid

        active_indices = skill_probs.argmax(dim=-1)
        active_skill_names = [self.skill_names[int(idx)] for idx in active_indices.tolist()]

        if routing_mode == "hard":
            combined = torch.zeros_like(per_skill_token_z[self.skill_names[0]])
            for skill_idx, skill in enumerate(self.skill_names):
                mask = active_indices.eq(skill_idx)
                if mask.any():
                    combined[mask] = per_skill_token_z[skill][mask]
            return combined, active_skill_names

        if routing_mode != "soft":
            raise ValueError("routing_mode must be 'soft' or 'hard'")

        combined = torch.zeros_like(per_skill_token_z[self.skill_names[0]])
        for skill_idx, skill in enumerate(self.skill_names):
            weights = skill_probs[:, skill_idx].view(batch_size, 1, 1)
            weights = torch.where(weights > self.routing_threshold, weights, torch.zeros_like(weights))
            combined = combined + weights * per_skill_token_z[skill]
        return combined, active_skill_names

    def _decode_logits(self, z_tokens: Tensor, active_skill_names: list[str]) -> Tensor:
        batch_size = z_tokens.size(0)
        vocab_size = next(iter(self.heads.values())).vocab_size
        logits = z_tokens.new_zeros((batch_size, z_tokens.size(1), vocab_size))
        for skill in self.skill_names:
            indices = [idx for idx, name in enumerate(active_skill_names) if name == skill]
            if not indices:
                continue
            skill_tensor = torch.tensor(indices, device=z_tokens.device, dtype=torch.long)
            logits.index_copy_(0, skill_tensor, self.heads[skill](z_tokens.index_select(0, skill_tensor)))
        return logits

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        routing_mode: str = "soft",
    ) -> SystemOutput:
        hidden_states, _ = self.backbone(input_ids, attention_mask=attention_mask)
        router_out = self.router(hidden_states)
        z_tokens, active_skill_names = self._combine_token_representations(
            hidden_states=hidden_states,
            skill_probs=router_out.skill_probs,
            routing_mode=routing_mode,
        )
        logits = self._decode_logits(z_tokens, active_skill_names)
        pooled_z = z_tokens.mean(dim=1)
        abstained = bool(router_out.abstain_prob.view(-1)[0].item() > 0.5)
        return SystemOutput(
            logits=logits,
            skill_probs=router_out.skill_probs,
            active_skill=active_skill_names[0] if active_skill_names else self.skill_names[0],
            z=pooled_z,
            abstained=abstained,
            router_conf=float(router_out.confidence.view(-1)[0].item()),
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = False,
        routing_mode: str = "soft",
    ) -> Tensor:
        generated = input_ids
        current_attention = attention_mask
        for step in range(max_new_tokens):
            output = self.forward(generated, attention_mask=current_attention, routing_mode=routing_mode)
            if step == 0 and output.abstained:
                return generated
            next_token_logits = output.logits[:, -1, :] / max(float(temperature), 1e-5)
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if current_attention is not None:
                current_attention = torch.cat(
                    [current_attention, torch.ones_like(next_token, device=current_attention.device)],
                    dim=1,
                )
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return generated

    @classmethod
    def from_component_checkpoints(
        cls,
        backbone_checkpoint: str | Path,
        backbone_config_path: str | Path,
        router_checkpoint: str | Path,
        geometry_config_path: str | Path,
        expert_checkpoint_dir: str | Path,
        expert_checkpoint_pattern: str = "expert_{skill}_stage3.pt",
        tokenizer_override: TokenizerWrapper | Any | None = None,
        device: str | torch.device = "cpu",
    ) -> "StudentSystem":
        device = torch.device(device)
        with open(backbone_config_path, "r", encoding="utf-8") as handle:
            backbone_cfg_raw = yaml.safe_load(handle)
        tokenizer = tokenizer_override
        if tokenizer is None:
            tokenizer_name = backbone_cfg_raw.get("tokenizer", {}).get("model_name") or backbone_cfg_raw.get("teacher", {}).get("model_name")
            tokenizer = TokenizerWrapper(model_name=tokenizer_name, max_length=backbone_cfg_raw["model"]["max_seq_len"])

        backbone_cfg = BackboneConfig.from_dict({**backbone_cfg_raw["model"], "vocab_size": getattr(tokenizer, "vocab_size", backbone_cfg_raw["model"]["vocab_size"])})
        backbone = CausalTransformerBackbone(backbone_cfg)
        backbone_ckpt = Path(backbone_checkpoint)
        if backbone_ckpt.exists():
            payload = torch.load(backbone_ckpt, map_location=device)
            state_dict = payload.get("model_state", payload)
            backbone.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: missing backbone checkpoint at {backbone_ckpt}; using fresh backbone.")
        backbone.to(device)

        with open(geometry_config_path, "r", encoding="utf-8") as handle:
            geometry_config = json.load(handle)

        router_ckpt = Path(router_checkpoint)
        if router_ckpt.exists():
            router_payload = torch.load(router_ckpt, map_location=device)
            skill_names = list(router_payload["skill_names"])
            router = Router(
                input_dim=int(router_payload["input_dim"]),
                skill_names=skill_names,
                hidden_dim=int(router_payload["hidden_dim"]),
                dropout=0.0,
            )
            router.load_state_dict(router_payload["model_state_dict"], strict=False)
            router.temperature = float(router_payload.get("temperature", 1.0))
        else:
            skill_names = sorted(geometry_config["skills"])
            router = Router(input_dim=backbone.config.d_model, skill_names=skill_names)
            print(f"Warning: missing router checkpoint at {router_ckpt}; using fresh router.")
        router.to(device)

        experts = nn.ModuleDict()
        heads = nn.ModuleDict()
        checkpoint_dir = Path(expert_checkpoint_dir)
        for skill in skill_names:
            expert = build_expert(skill, geometry_config, backbone_dim=backbone.config.d_model)
            expert_ckpt = checkpoint_dir / expert_checkpoint_pattern.format(skill=skill)
            if expert_ckpt.exists():
                expert_payload = torch.load(expert_ckpt, map_location=device)
                expert.load_state_dict(expert_payload.get("model_state_dict", expert_payload.get("model_state", {})), strict=False)
            else:
                print(f"Warning: missing expert checkpoint for skill '{skill}' at {expert_ckpt}; using fresh expert.")
            experts[skill] = expert.to(device)
            heads[skill] = OutputHead(manifold_dim=expert.manifold_dim, vocab_size=backbone.config.vocab_size).to(device)

        system = cls(backbone=backbone, router=router, experts=experts, heads=heads, tokenizer=tokenizer)
        system.to(device)
        return system

    @classmethod
    def from_full_checkpoint(
        cls,
        checkpoint_path: str | Path,
        backbone_config_path: str | Path,
        tokenizer_override: TokenizerWrapper | Any | None = None,
        device: str | torch.device = "cpu",
    ) -> "StudentSystem":
        device = torch.device(device)
        payload = torch.load(checkpoint_path, map_location=device)
        geometry_config = payload["geometry_config"]
        with open(backbone_config_path, "r", encoding="utf-8") as handle:
            backbone_cfg_raw = yaml.safe_load(handle)
        tokenizer = tokenizer_override
        if tokenizer is None:
            tokenizer_name = backbone_cfg_raw.get("tokenizer", {}).get("model_name") or backbone_cfg_raw.get("teacher", {}).get("model_name")
            tokenizer = TokenizerWrapper(model_name=tokenizer_name, max_length=backbone_cfg_raw["model"]["max_seq_len"])
        backbone_cfg = BackboneConfig.from_dict({**backbone_cfg_raw["model"], "vocab_size": getattr(tokenizer, "vocab_size", backbone_cfg_raw["model"]["vocab_size"])})
        backbone = CausalTransformerBackbone(backbone_cfg)
        backbone.load_state_dict(payload["backbone_state_dict"], strict=False)
        backbone.to(device)

        router = Router(
            input_dim=int(backbone.config.d_model),
            skill_names=list(payload["skill_names"]),
            hidden_dim=int(payload.get("router_hidden_dim", 256)),
            dropout=0.0,
        )
        router.load_state_dict(payload["router_state_dict"], strict=False)
        router.temperature = float(payload.get("router_temperature", 1.0))
        router.to(device)

        experts = nn.ModuleDict()
        heads = nn.ModuleDict()
        for skill in payload["skill_names"]:
            expert = build_expert(skill, geometry_config, backbone_dim=backbone.config.d_model)
            expert.load_state_dict(payload["expert_state_dicts"][skill], strict=False)
            experts[skill] = expert.to(device)
            head = OutputHead(manifold_dim=expert.manifold_dim, vocab_size=backbone.config.vocab_size)
            head.load_state_dict(payload["head_state_dicts"][skill], strict=False)
            heads[skill] = head.to(device)

        system = cls(backbone=backbone, router=router, experts=experts, heads=heads, tokenizer=tokenizer)
        system.to(device)
        return system
