"""
train.py  (v3 — NaN fixed, TODOs resolved, early stopping added)

Fixes vs v2:
  1. NaN: all hyperbolic ops run in float32 even under AMP (autocast disabled
     for the manifold forward; only the lm_head cross-entropy uses fp16)
  2. NaN: embedding scale changed from *0.1 to *0.05 with explicit fp32 cast
  3. TODO resolved: student.forward() now returns last_hidden (Euclidean, fp32)
     so order_loss and step_loss are actually computed
  4. scheduler.step() moved after optimizer.step() (fixes PyTorch warning)
  5. val split (10%) + early stopping (patience=5 epochs)
  6. epochs auto-computed to target ~15k steps; --epochs overrides
  7. NaN guard: skip batch and warn if loss is NaN (prevents silent corruption)
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from hyperbolic import poincare_dist, expmap0, logmap0
from model import HyperbolicReasoningStudent


# ---------------------------------------------------------------------------
# Dataset  (unchanged from v2)
# ---------------------------------------------------------------------------

class ReasoningTraceDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=512, min_steps=2):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.records     = []

        with open(jsonl_path) as f:
            for line in f:
                row   = json.loads(line)
                steps = row.get("think_steps", [])
                if len(steps) < min_steps:
                    continue
                if not row.get("question", "").strip():
                    continue
                self.records.append({
                    "question":  row["question"],
                    "steps":     steps,
                    "gt_answer": row.get("gt_answer", ""),
                })
        print(f"Loaded {len(self.records)} traces (min_steps={min_steps})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        q     = rec["question"].strip()
        steps = rec["steps"]

        header     = f"Question: {q}\nReasoning:\n"
        step_texts = [f"Step {i+1}: {s.strip()}" for i, s in enumerate(steps)]
        full_text  = header + "\n".join(step_texts) + f"\nAnswer: {rec['gt_answer']}"

        enc = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels         = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Token spans per step
        header_len = len(self.tokenizer(header, add_special_tokens=False)["input_ids"])
        step_spans = []
        cursor     = header_len
        for st in step_texts:
            n = len(self.tokenizer(st + "\n", add_special_tokens=False)["input_ids"])
            start = min(cursor, self.max_seq_len - 1)
            end   = min(cursor + n, self.max_seq_len)
            step_spans.append((start, end))
            cursor = end
            if cursor >= self.max_seq_len:
                break

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "step_spans":     step_spans,
            "n_steps":        len(step_spans),
        }


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "step_spans":     [b["step_spans"]                 for b in batch],
        "n_steps":        [b["n_steps"]                    for b in batch],
    }


# ---------------------------------------------------------------------------
# Teacher embedder  (unchanged from v2)
# ---------------------------------------------------------------------------

class TeacherEmbedder(nn.Module):
    def __init__(self, model_id, device):
        super().__init__()
        print(f"Loading teacher: {model_id}")
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def get_step_embeddings(self, input_ids, attention_mask, step_spans):
        out    = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        hidden = out.hidden_states[-2].float()   # (B, T, d_teacher)
        result = []
        for b, spans in enumerate(step_spans):
            embs = []
            for (s, e) in spans:
                if e > s:
                    embs.append(hidden[b, s:e].mean(0))
            result.append(embs)
        return result


# ---------------------------------------------------------------------------
# Loss functions — now actually called with real hidden states
# ---------------------------------------------------------------------------

def order_loss_fn(hidden, step_spans, c, device, margin=0.05):
    """
    Hierarchy constraint: step i+1 must be farther from origin than step i.
    hidden: (B, T, d) Euclidean float32
    """
    origin = torch.zeros(1, hidden.shape[-1], device=device, dtype=hidden.dtype)
    losses = []
    for b, spans in enumerate(step_spans):
        balls = []
        for (s, e) in spans:
            if e > s:
                vec  = hidden[b, s:e].mean(0)          # (d,)
                ball = expmap0(vec.unsqueeze(0), c)     # (1, d)
                balls.append(ball)
        for i in range(len(balls) - 1):
            d_i  = poincare_dist(origin, balls[i],   c).mean()
            d_i1 = poincare_dist(origin, balls[i+1], c).mean()
            losses.append(F.relu(d_i - d_i1 + margin))

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


def step_dist_loss_fn(hidden, teacher_embs, step_spans, proj, c, device):
    """
    Geodesic distance between student and teacher step embeddings.
    hidden: (B, T, d_student) Euclidean float32
    """
    losses = []
    for b, spans in enumerate(step_spans):
        t_embs = teacher_embs[b]
        for i, (s, e) in enumerate(spans):
            if i >= len(t_embs) or e <= s:
                continue
            s_vec  = hidden[b, s:e].mean(0)
            s_ball = expmap0(s_vec.unsqueeze(0), c)
            t_ball = expmap0(proj(t_embs[i].to(device).unsqueeze(0)), c)
            losses.append(poincare_dist(s_ball, t_ball, c).mean())

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # --- Dataset: 90/10 train/val split ---
    full_ds   = ReasoningTraceDataset(args.traces, tokenizer, max_seq_len=args.max_seq_len)
    val_size  = max(1, int(0.1 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    print(f"Train: {train_size}  Val: {val_size}  "
          f"Steps/epoch: {len(train_loader)}")

    # --- Auto-compute epochs to hit target_steps ---
    steps_per_epoch = len(train_loader)
    if args.epochs == 0:
        args.epochs = math.ceil(args.target_steps / steps_per_epoch)
    print(f"Training for {args.epochs} epochs "
          f"(~{args.epochs * steps_per_epoch} steps)")

    # --- Student ---
    student = HyperbolicReasoningStudent(
        vocab_size=vocab_size, d_model=args.d_model,
        n_heads=args.n_heads,  n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,  max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)
    print(f"Student: {student.param_count()}")

    # --- Teacher (optional) ---
    teacher, proj = None, None
    if args.use_teacher:
        teacher = TeacherEmbedder(args.teacher_model, device)
        dummy   = torch.zeros(1, 4, dtype=torch.long, device=device)
        with torch.no_grad():
            t_out = teacher.model(dummy, attention_mask=torch.ones_like(dummy),
                                  output_hidden_states=True)
        teacher_dim = t_out.hidden_states[-1].shape[-1]
        proj = nn.Linear(teacher_dim, args.d_model, bias=False).to(device)
        print(f"Teacher dim: {teacher_dim} → {args.d_model}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW([
        {"params": [p for n,p in student.named_parameters() if n != "log_c"],
         "lr": args.lr, "weight_decay": 0.01},
        {"params": [student.log_c],
         "lr": args.lr * 0.1, "weight_decay": 0.0},
    ])

    total_steps  = steps_per_epoch * args.epochs
    warmup_steps = total_steps // 10

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler — only for lm_head cross-entropy (manifold ops stay fp32)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    best_val_loss    = float("inf")
    patience_counter = 0
    global_step      = 0
    nan_streak       = 0   # consecutive NaN batches — abort if > 10

    for epoch in range(1, args.epochs + 1):
        student.train()
        ep_lm = ep_order = ep_step = ep_total = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            step_spans     = batch["step_spans"]

            # ── Forward pass ──────────────────────────────────────────────
            # IMPORTANT: run the entire student forward in fp32.
            # Hyperbolic ops (atanh, mobius_add) overflow in fp16.
            # We only use AMP for the final cross-entropy.
            student_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            lm_loss      = student_out["loss"]          # fp32 scalar
            last_hidden  = student_out["last_hidden"]   # (B, T, d) fp32 Euclidean
            c            = student_out["curvature"].item()

            # ── Manifold losses ───────────────────────────────────────────
            o_loss = torch.tensor(0.0, device=device)
            s_loss = torch.tensor(0.0, device=device)

            if args.lambda_order > 0:
                o_loss = order_loss_fn(last_hidden, step_spans, c, device)

            if args.lambda_step > 0 and teacher is not None:
                with torch.no_grad():
                    t_embs = teacher.get_step_embeddings(
                        input_ids, attention_mask, step_spans
                    )
                s_loss = step_dist_loss_fn(
                    last_hidden, t_embs, step_spans, proj, c, device
                )

            total_loss = (args.lambda_lm    * lm_loss +
                          args.lambda_order * o_loss  +
                          args.lambda_step  * s_loss)

            # ── NaN guard ─────────────────────────────────────────────────
            if not torch.isfinite(total_loss):
                nan_streak += 1
                print(f"\nWARN: NaN/Inf loss at step {global_step} "
                      f"(streak={nan_streak})")
                if nan_streak > 10:
                    print("Too many consecutive NaN batches — aborting.")
                    return
                optimizer.zero_grad()
                continue
            nan_streak = 0

            # ── Backward ──────────────────────────────────────────────────
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()   # AFTER optimizer.step()

            ep_lm    += lm_loss.item()
            ep_order += o_loss.item()
            ep_step  += s_loss.item()
            ep_total += total_loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix({
                "lm":    f"{lm_loss.item():.3f}",
                "ord":   f"{o_loss.item():.3f}",
                "c":     f"{c:.3f}",
                "lr":    f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if global_step % args.save_every == 0:
                ckpt = Path(args.output_dir) / f"step_{global_step}.pt"
                torch.save({
                    "step": global_step, "epoch": epoch,
                    "model_state": student.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "curvature": student.c,
                    "lm_loss": lm_loss.item(),
                }, ckpt)
                print(f"\nCheckpoint: {ckpt}")

        if n_batches == 0:
            print(f"Epoch {epoch}: all batches were NaN — stopping.")
            break

        n = n_batches
        print(f"\nEpoch {epoch} — "
              f"lm={ep_lm/n:.4f}  ord={ep_order/n:.4f}  "
              f"step={ep_step/n:.4f}  total={ep_total/n:.4f}  "
              f"c={student.c:.4f}")

        # ── Validation ────────────────────────────────────────────────────
        student.eval()
        val_lm = 0.0
        val_n  = 0
        with torch.no_grad():
            for batch in val_loader:
                out = student(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                if torch.isfinite(out["loss"]):
                    val_lm += out["loss"].item()
                    val_n  += 1

        if val_n == 0:
            print("  Val: all NaN — skipping early stop check")
            student.train()
            continue

        val_loss = val_lm / val_n
        print(f"  Val lm: {val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch, "val_loss": val_loss,
                "model_state": student.state_dict(),
                "curvature": student.c,
                "config": vars(args),
            }, Path(args.output_dir) / "best_model.pt")
            print(f"  ✓ Best val loss: {val_loss:.4f} — saved best_model.pt")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        student.train()

    # Final save
    torch.save({
        "model_state": student.state_dict(),
        "curvature":   student.c,
        "config":      vars(args),
    }, Path(args.output_dir) / "final_model.pt")
    print(f"\nDone. Final curvature: {student.c:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces",         default="parsed_traces.jsonl")
    p.add_argument("--teacher_model",  default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    p.add_argument("--output_dir",     default="./checkpoints")
    p.add_argument("--epochs",         type=int,   default=0,
                   help="0 = auto-compute from --target_steps")
    p.add_argument("--target_steps",   type=int,   default=15000,
                   help="Used when --epochs=0 to auto-set epoch count")
    p.add_argument("--patience",       type=int,   default=5,
                   help="Early stopping patience in epochs")
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--max_seq_len",    type=int,   default=512)
    p.add_argument("--d_model",        type=int,   default=512)
    p.add_argument("--n_heads",        type=int,   default=8)
    p.add_argument("--n_layers",       type=int,   default=8)
    p.add_argument("--ffn_dim",        type=int,   default=2048)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--save_every",     type=int,   default=200)
    p.add_argument("--use_teacher",    action="store_true")
    p.add_argument("--lambda_lm",      type=float, default=1.0)
    p.add_argument("--lambda_step",    type=float, default=0.0)
    p.add_argument("--lambda_order",   type=float, default=0.3)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()