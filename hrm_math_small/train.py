"""
train.py (v5)

Fixes vs v4:
  1. Curvature frozen: log_c grad was being swamped — fix by using
     a much larger LR multiplier (10x not 0.1x) and logging c every step
  2. Overfitting: add weight decay 0.1, dropout 0.15, label smoothing 0.1
  3. Order loss flat: the loss was computing on last_hidden (Euclidean)
     but poincare_dist needs ball points — fix by explicitly expmap0-ing
     before computing distances, and increase lambda_order to 1.0
  4. Val gap: reduce LR to 1e-4 (3e-4 was too high for this dataset size)
"""

import argparse, json, math
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
# Dataset
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
                if len(steps) < min_steps: continue
                if not row.get("question", "").strip(): continue
                self.records.append({
                    "question":  row["question"],
                    "steps":     steps,
                    "gt_answer": row.get("gt_answer", ""),
                })
        print(f"Loaded {len(self.records)} traces")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        steps = rec["steps"]
        header     = f"Question: {rec['question'].strip()}\nReasoning:\n"
        step_texts = [f"Step {i+1}: {s.strip()}" for i,s in enumerate(steps)]
        full_text  = header + "\n".join(step_texts) + f"\nAnswer: {rec['gt_answer']}"

        enc = self.tokenizer(full_text, max_length=self.max_seq_len,
                             truncation=True, padding="max_length",
                             return_tensors="pt")
        ids  = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        lbl  = ids.clone(); lbl[mask == 0] = -100

        header_len = len(self.tokenizer(
            header, add_special_tokens=False)["input_ids"])
        spans, cursor = [], header_len
        for st in step_texts:
            n     = len(self.tokenizer(st+"\n", add_special_tokens=False)["input_ids"])
            start = min(cursor, self.max_seq_len-1)
            end   = min(cursor+n, self.max_seq_len)
            spans.append((start, end))
            cursor = end
            if cursor >= self.max_seq_len: break

        return {"input_ids": ids, "attention_mask": mask,
                "labels": lbl, "step_spans": spans, "n_steps": len(spans)}


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "step_spans":     [b["step_spans"]  for b in batch],
        "n_steps":        [b["n_steps"]     for b in batch],
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def order_loss_fn(hidden, step_spans, c, device, margin=0.1):
    """
    Hierarchy loss: each step should be farther from origin than the previous.
    Projects Euclidean hidden states onto the ball first.
    """
    origin = torch.zeros(1, hidden.shape[-1], device=device, dtype=hidden.dtype)
    losses = []
    for b, spans in enumerate(step_spans):
        balls = []
        for (s, e) in spans:
            if e > s:
                vec  = hidden[b, s:e].mean(0)
                # Scale to reasonable ball magnitude before expmap
                vec_scaled = vec / (vec.norm() + 1e-8) * 0.3
                balls.append(expmap0(vec_scaled.unsqueeze(0), c))

        for i in range(len(balls) - 1):
            d_i  = poincare_dist(origin, balls[i],   c).mean()
            d_i1 = poincare_dist(origin, balls[i+1], c).mean()
            losses.append(F.relu(d_i - d_i1 + margin))

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


def step_dist_loss_fn(hidden, teacher_embs, step_spans, proj, c, device):
    losses = []
    for b, spans in enumerate(step_spans):
        t_embs = teacher_embs[b]
        for i, (s, e) in enumerate(spans):
            if i >= len(t_embs) or e <= s: continue
            s_vec  = hidden[b, s:e].mean(0)
            s_scaled = s_vec / (s_vec.norm() + 1e-8) * 0.3
            s_ball = expmap0(s_scaled.unsqueeze(0), c)
            t_vec  = teacher_embs[i].to(device) if isinstance(t_embs[i], torch.Tensor) else t_embs[i].to(device)
            t_proj = proj(t_vec.unsqueeze(0))
            t_scaled = t_proj / (t_proj.norm() + 1e-8) * 0.3
            t_ball = expmap0(t_scaled, c)
            losses.append(poincare_dist(s_ball, t_ball, c).mean())
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


class TeacherEmbedder(nn.Module):
    def __init__(self, model_id, device):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def get_step_embeddings(self, input_ids, attention_mask, step_spans):
        out    = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        hidden = out.hidden_states[-2].float()
        result = []
        for b, spans in enumerate(step_spans):
            embs = []
            for (s, e) in spans:
                if e > s: embs.append(hidden[b, s:e].mean(0))
            result.append(embs)
        return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    full_ds    = ReasoningTraceDataset(
        args.traces, tokenizer, max_seq_len=args.max_seq_len)
    val_size   = max(1, int(0.1 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)

    steps_per_epoch = len(train_loader)
    if args.epochs == 0:
        args.epochs = math.ceil(args.target_steps / steps_per_epoch)
    print(f"Train: {train_size}  Val: {val_size}  "
          f"Steps/epoch: {steps_per_epoch}  Epochs: {args.epochs}")

    student = HyperbolicReasoningStudent(
        vocab_size=vocab_size, d_model=args.d_model,
        n_heads=args.n_heads,  n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,  max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)
    print(f"Student: {student.param_count()}")

    teacher, proj = None, None
    if args.use_teacher:
        teacher = TeacherEmbedder(args.teacher_model, device)
        dummy   = torch.zeros(1, 4, dtype=torch.long, device=device)
        with torch.no_grad():
            t_out = teacher.model(dummy, attention_mask=torch.ones_like(dummy),
                                  output_hidden_states=True)
        teacher_dim = t_out.hidden_states[-1].shape[-1]
        proj = nn.Linear(teacher_dim, args.d_model, bias=False).to(device)
        print(f"Teacher dim {teacher_dim} → {args.d_model}")

    # Three param groups:
    #   1. curv_proj (log_c + W): high LR, no weight decay — geometry params
    #   2. embeddings: standard LR, no weight decay
    #   3. everything else: standard LR + weight decay
    curv_params  = list(student.curv_proj.parameters())
    embed_params = list(student.embed.parameters()) + list(student.pos_embed.parameters())
    curv_ids     = {id(p) for p in curv_params}
    embed_ids    = {id(p) for p in embed_params}
    other_params = [p for p in student.parameters()
                    if id(p) not in curv_ids and id(p) not in embed_ids]
    optimizer = torch.optim.AdamW([
        {"params": curv_params,  "lr": args.lr * 5,  "weight_decay": 0.0},
        {"params": embed_params, "lr": args.lr,       "weight_decay": 0.0},
        {"params": other_params, "lr": args.lr,       "weight_decay": args.weight_decay},
    ])

    total_steps  = steps_per_epoch * args.epochs
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    best_val   = float("inf")
    patience   = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        ep = {"lm": 0., "ord": 0., "step": 0., "total": 0.}
        n_ok = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbl   = batch["labels"].to(device)
            spans = batch["step_spans"]

            out     = student(input_ids=ids, attention_mask=mask, labels=lbl)
            lm_loss = out["loss"]
            hidden  = out["last_hidden"]   # (B, T, d) Euclidean fp32
            c       = out["curvature"].item()

            o_loss = torch.tensor(0., device=device)
            s_loss = torch.tensor(0., device=device)

            if args.lambda_order > 0:
                o_loss = order_loss_fn(hidden, spans, c, device)
            if args.lambda_step > 0 and teacher is not None:
                with torch.no_grad():
                    t_embs = teacher.get_step_embeddings(ids, mask, spans)
                s_loss = step_dist_loss_fn(hidden, t_embs, spans, proj, c, device)

            # FIX 2: label smoothing built into cross_entropy in model,
            # here we just use weighted sum
            total = (args.lambda_lm * lm_loss +
                     args.lambda_order * o_loss +
                     args.lambda_step  * s_loss)

            if not torch.isfinite(total):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep["lm"]    += lm_loss.item()
            ep["ord"]   += o_loss.item()
            ep["step"]  += s_loss.item()
            ep["total"] += total.item()
            n_ok        += 1
            global_step += 1

            pbar.set_postfix({
                "lm":  f"{lm_loss.item():.3f}",
                "ord": f"{o_loss.item():.3f}",
                "c":   f"{c:.4f}",        # 4 decimal places to see movement
                "lr":  f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if global_step % args.save_every == 0:
                ckpt = Path(args.output_dir) / f"step_{global_step}.pt"
                torch.save({
                    "step": global_step, "epoch": epoch,
                    "model_state": student.state_dict(),
                    "curvature": student.c, "lm_loss": lm_loss.item(),
                }, ckpt)

        if n_ok == 0: break
        n = n_ok
        print(f"\nEpoch {epoch} — lm={ep['lm']/n:.4f}  ord={ep['ord']/n:.4f}  "
              f"total={ep['total']/n:.4f}  c={student.c:.6f}")  # 6dp for c

        # Validation
        student.eval()
        val_lm, val_n = 0., 0
        with torch.no_grad():
            for batch in val_loader:
                out = student(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device))
                if torch.isfinite(out["loss"]):
                    val_lm += out["loss"].item(); val_n += 1

        if val_n == 0: student.train(); continue
        val_loss = val_lm / val_n
        print(f"  Val lm: {val_loss:.4f}  c={student.c:.6f}")

        if val_loss < best_val - 1e-4:
            best_val  = val_loss
            patience  = 0
            torch.save({
                "epoch": epoch, "val_loss": val_loss,
                "model_state": student.state_dict(),
                "curvature": student.c, "config": vars(args),
            }, Path(args.output_dir) / "best_model.pt")
            print(f"  ✓ Best: {val_loss:.4f}")
        else:
            patience += 1
            print(f"  No improvement ({patience}/{args.patience})")
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        student.train()

    torch.save({"model_state": student.state_dict(), "curvature": student.c,
                "config": vars(args)},
               Path(args.output_dir) / "final_model.pt")
    print(f"\nDone.  c={student.c:.6f}  best_val={best_val:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces",         default="parsed_traces.jsonl")
    p.add_argument("--teacher_model",  default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    p.add_argument("--output_dir",     default="./checkpoints")
    p.add_argument("--epochs",         type=int,   default=0)
    p.add_argument("--target_steps",   type=int,   default=15000)
    p.add_argument("--patience",       type=int,   default=7)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=1e-4)   # reduced from 3e-4
    p.add_argument("--weight_decay",   type=float, default=0.1)    # increased from 0.01
    p.add_argument("--max_seq_len",    type=int,   default=512)
    p.add_argument("--d_model",        type=int,   default=512)
    p.add_argument("--n_heads",        type=int,   default=8)
    p.add_argument("--n_layers",       type=int,   default=8)
    p.add_argument("--ffn_dim",        type=int,   default=2048)
    p.add_argument("--dropout",        type=float, default=0.15)   # increased from 0.1
    p.add_argument("--save_every",     type=int,   default=200)
    p.add_argument("--use_teacher",    action="store_true")
    p.add_argument("--lambda_lm",      type=float, default=1.0)
    p.add_argument("--lambda_step",    type=float, default=0.0)
    p.add_argument("--lambda_order",   type=float, default=1.0)    # increased from 0.3
    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()