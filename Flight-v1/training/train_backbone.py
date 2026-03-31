from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from backbone.config import BackboneConfig
from backbone.model import CausalTransformerBackbone
from backbone.tokenizer import TokenizerWrapper
from data.loader import build_dataloaders
from training.losses import causal_lm_loss, compute_backbone_loss
from training.scheduler import build_scheduler
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logging import MetricsLogger, format_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 0 Euclidean backbone.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def maybe_compile(model: nn.Module) -> nn.Module:
    if hasattr(torch, "compile") and os.environ.get("DISABLE_TORCH_COMPILE", "0") != "1":
        return torch.compile(model)
    return model


def build_student(config: dict[str, Any], vocab_size: int) -> CausalTransformerBackbone:
    model_cfg = BackboneConfig.from_dict({**config["model"], "vocab_size": vocab_size})
    student = CausalTransformerBackbone(model_cfg)
    if model_cfg.gradient_checkpointing:
        student.enable_gradient_checkpointing()
    return student


def load_teacher(model_name: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    teacher = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    teacher.eval()
    teacher.to(device)
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


@torch.no_grad()
def evaluate_perplexity(
    model: CausalTransformerBackbone,
    eval_loader,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_tokens = 0
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            _, logits = model(input_ids, attention_mask=attention_mask)
        loss = causal_lm_loss(logits.float(), labels)
        valid_tokens = labels[:, 1:].ne(-100).sum().item()
        total_loss += loss.item() * max(valid_tokens, 1)
        total_tokens += max(valid_tokens, 1)
        total_batches += 1
    mean_nll = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(mean_nll, 20))
    model.train()
    return {"eval_lm_loss": mean_nll, "eval_ppl": ppl, "eval_batches": total_batches}


@torch.no_grad()
def sample_completions(
    model: CausalTransformerBackbone,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    device: torch.device,
) -> list[str]:
    samples = []
    model.eval()
    for prompt in prompts:
        encoded = tokenizer.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            temperature=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
        samples.append(tokenizer.decode(generated[0].tolist()))
    model.train()
    return samples


def train() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    training_cfg = cfg["training"]
    teacher_cfg = cfg["teacher"]
    logging_cfg = cfg["logging"]
    tokenizer_cfg = cfg.get("tokenizer", {})

    device = choose_device(teacher_cfg.get("device", "cuda"))
    train_dtype = torch.bfloat16 if training_cfg.get("bf16", True) and device.type == "cuda" else torch.float32
    teacher_dtype = resolve_dtype(teacher_cfg.get("dtype", "float16"))

    tokenizer_name = tokenizer_cfg.get("model_name", teacher_cfg["model_name"])
    tokenizer = TokenizerWrapper(model_name=tokenizer_name, max_length=cfg["model"]["max_seq_len"])
    student = build_student(cfg, vocab_size=tokenizer.vocab_size).to(device)
    teacher = load_teacher(teacher_cfg["model_name"], device, teacher_dtype)

    if device.type == "cuda":
        student = maybe_compile(student)

    train_loader, eval_loader = build_dataloaders(
        tokenizer=tokenizer,
        batch_size=training_cfg["batch_size"],
        max_length=cfg["model"]["max_seq_len"],
        instruction_dataset=cfg.get("data", {}).get("instruction_dataset", "alpaca"),
        instruction_limit=cfg.get("data", {}).get("instruction_limit", 50_000),
        seed=seed,
    )

    optimizer = AdamW(
        student.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        betas=(0.9, 0.95),
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=int(training_cfg["warmup_steps"]),
        max_steps=int(training_cfg["max_steps"]),
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, student, optimizer=optimizer, scheduler=scheduler, map_location=device)

    metrics_logger = MetricsLogger(log_every=int(logging_cfg["log_every"]))
    scaler = None
    grad_accum = int(training_cfg["gradient_accumulation_steps"])
    max_steps = int(training_cfg["max_steps"])
    alpha_lm = float(training_cfg["alpha_lm"])
    beta_kd = float(training_cfg["beta_kd"])
    kd_temperature = float(training_cfg["kd_temperature"])
    autocast_enabled = device.type == "cuda" and train_dtype in {torch.float16, torch.bfloat16}
    prompts = cfg.get(
        "eval_prompts",
        [
            "Question: If a train leaves at 3 PM and arrives two hours later, what time is it?",
            "Instruction: Summarize why checking your work matters.\nResponse:",
            "Question: What is 17 plus 26?\nAnswer:",
            "Instruction: Write one sentence about geometric distillation.\nResponse:",
            "Question: A box has 4 rows of 3 apples. How many apples are there?\nAnswer:",
        ],
    )

    student.train()
    running_metrics: dict[str, float] = {}
    progress = tqdm(total=max_steps, initial=start_step, desc="Training backbone")
    train_iter = iter(train_loader)

    for step in range(start_step, max_steps):
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=autocast_enabled):
                _, student_logits = student(input_ids=input_ids, attention_mask=attention_mask)
                loss, metrics = compute_backbone_loss(
                    student_logits=student_logits.float(),
                    teacher_logits=teacher_logits.float(),
                    labels=labels,
                    alpha_lm=alpha_lm,
                    beta_kd=beta_kd,
                    kd_temperature=kd_temperature,
                )
                loss = loss / grad_accum

            loss.backward()
            running_metrics = metrics

        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float(training_cfg["gradient_clip"]))
        optimizer.step()
        scheduler.step()

        running_metrics["grad_norm"] = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
        running_metrics["lr"] = float(scheduler.get_last_lr()[0])
        metrics_logger.update(running_metrics)
        global_step = step + 1
        progress.update(1)

        if global_step % int(logging_cfg["log_every"]) == 0:
            progress.write(format_metrics(global_step, metrics_logger.mean()))
            metrics_logger.reset()

        if global_step % int(logging_cfg["save_every"]) == 0:
            save_checkpoint(
                path="checkpoints/backbone_stage0.pt",
                model=student,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                extra={"config": cfg},
            )

    progress.close()

    eval_metrics = evaluate_perplexity(
        model=student,
        eval_loader=eval_loader,
        device=device,
        autocast_enabled=autocast_enabled,
        autocast_dtype=train_dtype,
    )
    completions = sample_completions(student, tokenizer, prompts=prompts[:5], device=device)
    final_loss = running_metrics.get("loss", float("nan"))
    final_losses = {
        "final_lm_loss": running_metrics.get("loss_lm", float("nan")),
        "final_kd_loss": running_metrics.get("loss_kd", float("nan")),
    }

    save_checkpoint(
        path="checkpoints/backbone_stage0.pt",
        model=student,
        optimizer=optimizer,
        scheduler=scheduler,
        step=max_steps,
        extra={"config": cfg, "eval": eval_metrics, "losses": final_losses},
    )

    for idx, text in enumerate(completions, start=1):
        print(f"[sample_{idx}] {text}\n")
    print(format_metrics(max_steps, {**eval_metrics, **final_losses}))
    print(f"Backbone parameters: {student._orig_mod.num_parameters() / 1e6 if hasattr(student, '_orig_mod') else student.num_parameters() / 1e6:.2f} M")
    print(f"Training complete. Final loss: {final_loss:.4f}")
    print("Checkpoint saved: checkpoints/backbone_stage0.pt")
    print("Ready for Stage 1.")


if __name__ == "__main__":
    train()
