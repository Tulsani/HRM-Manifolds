from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from backbone.config import BackboneConfig
from backbone.model import CausalTransformerBackbone
from backbone.tokenizer import TokenizerWrapper
from data.trace_schema import TracePackage, load_sparse_logits


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _choose_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = mapping.get(name, torch.float16)
    if device.type != "cuda" and dtype != torch.float32:
        return torch.float32
    return dtype


def _load_traces(trace_dir: str | Path) -> list[TracePackage]:
    trace_dir = Path(trace_dir)
    traces: list[TracePackage] = []
    for source in ("gsm8k", "math"):
        jsonl_path = trace_dir / f"{source}_traces.jsonl"
        logits_path = trace_dir / "logits" / f"{source}_logits.npz"
        logits = load_sparse_logits(logits_path)
        if not jsonl_path.exists():
            continue
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                traces.append(TracePackage.from_jsonl(line, sparse_logits=logits.get(payload["example_id"])))
    traces.sort(key=lambda trace: trace.example_id)
    return traces


def _load_backbone(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[CausalTransformerBackbone, TokenizerWrapper]:
    cfg = _load_yaml(config_path)
    checkpoint_file = Path(checkpoint_path)
    checkpoint_payload: dict[str, Any] | None = None
    if checkpoint_file.exists():
        checkpoint_payload = torch.load(checkpoint_file, map_location="cpu")

    config_dict = cfg["model"]
    if checkpoint_payload is not None and checkpoint_payload.get("config"):
        config_dict = checkpoint_payload["config"]

    model_cfg = BackboneConfig.from_dict(config_dict)
    tokenizer_name = cfg.get("tokenizer", {}).get("model_name") or cfg.get("teacher", {}).get("model_name")
    if tokenizer_name is None:
        raise ValueError("Backbone config must include tokenizer.model_name or teacher.model_name.")
    tokenizer = TokenizerWrapper(model_name=tokenizer_name, max_length=model_cfg.max_seq_len)

    if model_cfg.vocab_size != tokenizer.vocab_size:
        model_cfg = BackboneConfig.from_dict({**model_cfg.to_dict(), "vocab_size": tokenizer.vocab_size})

    model = CausalTransformerBackbone(model_cfg)
    if checkpoint_file.exists():
        state_dict = checkpoint_payload.get("model_state", checkpoint_payload) if checkpoint_payload is not None else {}
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: backbone checkpoint not found at {checkpoint_file}. Using a freshly initialized backbone.")

    model.eval()
    model.to(device=device)
    if device.type == "cuda":
        model = model.to(dtype=dtype)
    return model, tokenizer


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom


@torch.no_grad()
def _encode_texts(
    model: CausalTransformerBackbone,
    tokenizer: TokenizerWrapper,
    texts: list[str],
    device: torch.device,
) -> np.ndarray:
    encoded = tokenizer.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.max_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    hidden_states, _ = model(input_ids, attention_mask=attention_mask)
    pooled = _mean_pool(hidden_states, attention_mask)
    return pooled.float().cpu().numpy()


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def extract_and_save_embeddings(config: dict[str, Any]) -> dict[str, Any]:
    backbone_cfg = config["backbone"]
    trace_dir = Path(config["trace_library"]["path"])
    embeddings_dir = Path(config["output"]["embeddings_dir"])
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    device = _choose_device(backbone_cfg.get("device", "cuda"))
    dtype = _resolve_dtype(backbone_cfg.get("dtype", "float16"), device)
    model, tokenizer = _load_backbone(
        checkpoint_path=backbone_cfg["checkpoint"],
        config_path=backbone_cfg["config"],
        device=device,
        dtype=dtype,
    )

    traces = _load_traces(trace_dir)
    batch_size = 32

    problem_embeddings: dict[str, np.ndarray] = {}
    for batch in _batched(traces, batch_size):
        vectors = _encode_texts(model, tokenizer, [trace.problem for trace in batch], device)
        for trace, vector in zip(batch, vectors):
            problem_embeddings[trace.example_id] = vector.astype(np.float32)

    step_embeddings: dict[str, np.ndarray] = {}
    for trace in traces:
        if not trace.step_texts:
            step_embeddings[trace.example_id] = np.zeros((0, model.config.d_model), dtype=np.float32)
            continue
        matrices: list[np.ndarray] = []
        for batch in _batched(trace.step_texts, batch_size):
            matrices.append(_encode_texts(model, tokenizer, batch, device))
        step_embeddings[trace.example_id] = np.concatenate(matrices, axis=0).astype(np.float32)

    np.savez(embeddings_dir / "problem_embeddings.npz", **problem_embeddings)
    np.savez(embeddings_dir / "step_embeddings.npz", **step_embeddings)

    return {
        "count": len(traces),
        "backbone_dim": model.config.d_model,
        "problem_embeddings_path": str(embeddings_dir / "problem_embeddings.npz"),
        "step_embeddings_path": str(embeddings_dir / "step_embeddings.npz"),
    }
