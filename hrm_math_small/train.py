"""
train.py
--------
Distillation training loop for HyperbolicReasoningStudent.

Loss function (three components):
  1. LM loss      : standard cross-entropy on next-token prediction
                    ensures the student can generate text at all

  2. Step loss    : for each reasoning step i, the student's hyperbolic
                    embedding of step i should be close (in geodesic distance)
                    to the teacher's embedding of step i.
                    L_step = mean over steps of poincare_dist(student_i, teacher_i)

  3. Order loss   : earlier steps should be closer to the manifold origin
                    than later steps (encodes the broad→specific hierarchy).
                    L_order = mean of max(0, dist(origin, step_i+1) - dist(origin, step_i) + margin)
                    (a soft constraint: step i+1 should be farther from origin than step i)

Total loss = λ_lm * L_lm + λ_step * L_step + λ_order * L_order

Usage:
  python train.py \
    --traces parsed_traces.jsonl \
    --teacher_model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --output_dir ./checkpoints \
    --epochs 3 \
    --batch_size 8 \
    --lr 3e-4
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from hyperbolic import poincare_dist, expmap0, logmap0
from model import HyperbolicReasoningStudent


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ReasoningTraceDataset(Dataset):
    """
    Loads parsed_traces.jsonl and prepares:
      - input_ids    : tokenized "Question: {q}\nReasoning:\n{steps joined}"
      - labels       : same as input_ids (LM objective — predict next token)
      - step_spans   : list of (start, end) token indices for each step
                       used to extract per-step embeddings for the manifold loss
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_seq_len: int = 512,
        min_steps:   int = 2,      # drop traces with fewer steps
    ):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.records     = []

        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                steps = row.get("think_steps", [])
                if len(steps) < min_steps:
                    continue
                if not row.get("question", "").strip():
                    continue
                self.records.append({
                    "question":   row["question"],
                    "steps":      steps,
                    "gt_answer":  row.get("gt_answer", ""),
                })

        print(f"Loaded {len(self.records)} traces (min_steps={min_steps})")

    def __len__(self):
        return len(self.records)

    def _build_text_and_spans(self, record: dict):
        """
        Build the full text and identify byte-level spans for each step.
        Returns (text, step_texts) where step_texts[i] is the text of step i.
        """
        q     = record["question"].strip()
        steps = record["steps"]
        header = f"Question: {q}\nReasoning:\n"
        step_texts = [f"Step {i+1}: {s.strip()}" for i, s in enumerate(steps)]
        full_text  = header + "\n".join(step_texts) + f"\nAnswer: {record['gt_answer']}"
        return full_text, header, step_texts

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        full_text, header, step_texts = self._build_text_and_spans(record)

        # Tokenize full sequence
        enc = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Labels: same as input_ids, but -100 for padding
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Find token-level spans for each step
        # We tokenize the header to find where steps begin
        header_ids = self.tokenizer(
            header, add_special_tokens=False
        )["input_ids"]
        header_len = len(header_ids)

        step_spans = []
        cursor = header_len
        for step_text in step_texts:
            step_ids = self.tokenizer(
                step_text + "\n", add_special_tokens=False
            )["input_ids"]
            start = min(cursor, self.max_seq_len - 1)
            end   = min(cursor + len(step_ids), self.max_seq_len)
            step_spans.append((start, end))
            cursor = end
            if cursor >= self.max_seq_len:
                break

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "step_spans":     step_spans,   # list of (int, int) — variable length
            "n_steps":        len(step_spans),
        }


def collate_fn(batch):
    """Custom collate: stack tensors, keep step_spans as list of lists."""
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "step_spans":     [b["step_spans"]                 for b in batch],
        "n_steps":        [b["n_steps"]                    for b in batch],
    }


# ---------------------------------------------------------------------------
# Teacher embedding extractor
# ---------------------------------------------------------------------------

class TeacherEmbedder(nn.Module):
    """
    Wraps a HuggingFace model to extract per-step hidden states.
    We use the last hidden state averaged over each step's token span.
    The teacher is frozen — no gradients flow through it.
    """
    def __init__(self, model_id: str, device: torch.device):
        super().__init__()
        print(f"Loading teacher embedder: {model_id}")
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def get_step_embeddings(
        self,
        input_ids:      torch.Tensor,   # (B, T)
        attention_mask: torch.Tensor,   # (B, T)
        step_spans:     list,           # list of list of (start, end)
    ) -> list[list[torch.Tensor]]:
        """
        Returns: batch_embeddings[b][i] = mean hidden state for step i in batch item b
                 shape of each: (d_teacher,)
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use second-to-last layer for richer representations
        hidden = out.hidden_states[-2].float()   # (B, T, d_teacher)

        batch_embeddings = []
        for b, spans in enumerate(step_spans):
            step_embs = []
            for (start, end) in spans:
                if end <= start:
                    continue
                # Mean pool over the step's token range
                step_hidden = hidden[b, start:end, :]     # (step_len, d_teacher)
                step_emb    = step_hidden.mean(dim=0)     # (d_teacher,)
                step_embs.append(step_emb)
            batch_embeddings.append(step_embs)

        return batch_embeddings


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def step_distillation_loss(
    student_hidden: torch.Tensor,    # (B, T, d_student) — student's last hidden (Euclidean)
    teacher_step_embs: list,         # batch_embeddings from TeacherEmbedder
    step_spans: list,                # batch list of (start, end)
    proj: nn.Linear,                 # projects teacher d → student d
    c: float,                        # curvature
    device: torch.device,
) -> torch.Tensor:
    """
    For each (batch item, step), compute poincaré_dist between:
      - student's mean embedding of that step (lifted to ball)
      - teacher's mean embedding of that step (projected + lifted to ball)
    Returns mean distance across all steps in batch.
    """
    total_dist = torch.tensor(0.0, device=device, requires_grad=True)
    n_pairs    = 0

    for b, spans in enumerate(step_spans):
        teacher_embs = teacher_step_embs[b]
        for i, (start, end) in enumerate(spans):
            if i >= len(teacher_embs) or end <= start:
                continue

            # Student: mean pool step tokens, lift to ball
            s_vec = student_hidden[b, start:end, :].mean(dim=0)   # (d_student,)
            s_ball = expmap0(s_vec.unsqueeze(0), c)                # (1, d_student)

            # Teacher: project to student dim, lift to ball
            t_vec  = teacher_embs[i].to(device)                   # (d_teacher,)
            t_proj = proj(t_vec.unsqueeze(0))                      # (1, d_student)
            t_ball = expmap0(t_proj, c)                            # (1, d_student)

            dist = poincare_dist(s_ball, t_ball, c)                # scalar
            total_dist = total_dist + dist.mean()
            n_pairs += 1

    return total_dist / max(n_pairs, 1)


def step_order_loss(
    student_hidden: torch.Tensor,   # (B, T, d_student)
    step_spans: list,               # batch list of (start, end)
    c: float,
    device: torch.device,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    Hierarchy loss: step i+1 should be farther from the origin than step i.
    Encodes the broad→specific structure in hyperbolic space.

    L_order = mean of relu(dist(origin, step_i) - dist(origin, step_{i+1}) + margin)

    If step i is already closer to origin than step i+1, loss=0.
    If step i is farther (wrong order), we pay a penalty.
    """
    origin = torch.zeros(1, student_hidden.shape[-1], device=device)
    total  = torch.tensor(0.0, device=device, requires_grad=True)
    n_pairs = 0

    for b, spans in enumerate(step_spans):
        step_balls = []
        for (start, end) in spans:
            if end <= start:
                continue
            s_vec  = student_hidden[b, start:end, :].mean(dim=0)
            s_ball = expmap0(s_vec.unsqueeze(0), c)
            step_balls.append(s_ball)

        for i in range(len(step_balls) - 1):
            d_i   = poincare_dist(origin, step_balls[i],   c).mean()
            d_i1  = poincare_dist(origin, step_balls[i+1], c).mean()
            # step i should be CLOSER to origin (broader concept)
            # step i+1 should be FARTHER (more specific)
            penalty = F.relu(d_i - d_i1 + margin)
            total   = total + penalty
            n_pairs += 1

    return total / max(n_pairs, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer — use teacher's tokenizer so vocab aligns
    print(f"Loading tokenizer: {args.teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # Dataset
    dataset = ReasoningTraceDataset(
        args.traces, tokenizer, max_seq_len=args.max_seq_len
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Student model
    student = HyperbolicReasoningStudent(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)
    print(f"Student: {student.param_count()}")
    print(f"Initial curvature: {student.c:.4f}")

    # Teacher embedder (optional — skip if you want LM-only training first)
    teacher = None
    proj    = None
    if args.use_teacher and args.teacher_model:
        teacher = TeacherEmbedder(args.teacher_model, device)
        # Projection: teacher hidden dim → student d_model
        # We probe the teacher to find its hidden size
        dummy = torch.zeros(1, 4, dtype=torch.long).to(device)
        dummy_mask = torch.ones(1, 4, dtype=torch.long).to(device)
        with torch.no_grad():
            t_out = teacher.model(dummy, attention_mask=dummy_mask, output_hidden_states=True)
        teacher_dim = t_out.hidden_states[-1].shape[-1]
        proj = nn.Linear(teacher_dim, args.d_model, bias=False).to(device)
        print(f"Teacher hidden dim: {teacher_dim} → projected to {args.d_model}")

    # Optimizer: separate LR for curvature (it needs to move slowly)
    curvature_params = [student.log_c]
    other_params     = [p for n, p in student.named_parameters() if n != "log_c"]
    optimizer = torch.optim.AdamW([
        {"params": other_params,     "lr": args.lr,       "weight_decay": 0.01},
        {"params": curvature_params, "lr": args.lr * 0.1, "weight_decay": 0.0},
    ])

    # Cosine LR schedule with warmup
    total_steps   = len(loader) * args.epochs
    warmup_steps  = total_steps // 10
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_lm_loss    = 0.0
        epoch_step_loss  = 0.0
        epoch_order_loss = 0.0
        epoch_total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            step_spans     = batch["step_spans"]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                # Forward pass
                out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                lm_loss = out["loss"]
                c       = out["curvature"].item()

                # Get student hidden states for manifold losses
                # Re-run without labels to get hidden states cleanly
                # (In production you'd return hidden_states from forward)
                with torch.no_grad():
                    # Use logits as proxy: project back from manifold
                    # For now we extract from the model internals
                    pass

                # Manifold losses (compute if teacher available)
                step_loss  = torch.tensor(0.0, device=device)
                order_loss = torch.tensor(0.0, device=device)

                if teacher is not None and proj is not None:
                    teacher_embs = teacher.get_step_embeddings(
                        input_ids, attention_mask, step_spans
                    )
                    # We need student hidden states — temporarily re-run
                    # TODO in next iteration: return hidden from student.forward
                    # For now skip manifold loss until we wire up hidden states
                    pass

                total_loss = (
                    args.lambda_lm    * lm_loss +
                    args.lambda_step  * step_loss +
                    args.lambda_order * order_loss
                )

            # Backward
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_lm_loss    += lm_loss.item()
            epoch_step_loss  += step_loss.item()
            epoch_order_loss += order_loss.item()
            epoch_total_loss += total_loss.item()
            global_step += 1

            pbar.set_postfix({
                "lm":    f"{lm_loss.item():.3f}",
                "c":     f"{c:.3f}",
                "lr":    f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # Checkpoint every N steps
            if global_step % args.save_every == 0:
                ckpt_path = Path(args.output_dir) / f"step_{global_step}.pt"
                torch.save({
                    "step":         global_step,
                    "epoch":        epoch,
                    "model_state":  student.state_dict(),
                    "optim_state":  optimizer.state_dict(),
                    "curvature":    student.c,
                    "lm_loss":      lm_loss.item(),
                }, ckpt_path)
                print(f"\nCheckpoint saved: {ckpt_path}")

        n = len(loader)
        print(f"\nEpoch {epoch} — "
              f"LM: {epoch_lm_loss/n:.4f}  "
              f"Step: {epoch_step_loss/n:.4f}  "
              f"Order: {epoch_order_loss/n:.4f}  "
              f"Total: {epoch_total_loss/n:.4f}  "
              f"c={student.c:.4f}")

    # Final save
    final_path = Path(args.output_dir) / "final_model.pt"
    torch.save({
        "model_state": student.state_dict(),
        "curvature":   student.c,
        "config": {
            "vocab_size": vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "ffn_dim": args.ffn_dim,
        }
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    print(f"Final curvature learned: {student.c:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces",        default="parsed_traces.jsonl")
    parser.add_argument("--teacher_model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--output_dir",    default="./checkpoints")
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--max_seq_len",   type=int,   default=512)
    parser.add_argument("--d_model",       type=int,   default=512)
    parser.add_argument("--n_heads",       type=int,   default=8)
    parser.add_argument("--n_layers",      type=int,   default=8)
    parser.add_argument("--ffn_dim",       type=int,   default=2048)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--save_every",    type=int,   default=200)
    parser.add_argument("--use_teacher",   action="store_true",
                        help="Enable teacher embedding loss (needs teacher model loaded)")
    # Loss weights
    parser.add_argument("--lambda_lm",    type=float, default=1.0)
    parser.add_argument("--lambda_step",  type=float, default=0.5)
    parser.add_argument("--lambda_order", type=float, default=0.3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()