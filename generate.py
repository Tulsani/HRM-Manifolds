"""
02_generate.py
--------------
Load a local reasoning model and generate chain-of-thought traces
over GSM8K problems. Saves raw outputs to raw_traces.jsonl.

Usage:
  python 02_generate.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --split train \
    --n 2000 \
    --batch_size 4 \
    --out raw_traces.jsonl

Model options (pick based on your VRAM from 01_setup.py):
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     (~14GB bf16)
  deepseek-ai/DeepSeek-R1-Distill-Qwen-14B    (~28GB bf16)
  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B    (~65GB bf16 / ~18GB 4bit)
  Qwen/QwQ-32B-Preview                         (~65GB bf16 / ~18GB 4bit)
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
import jsonlines
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# We explicitly ask for <think>...</think> wrapping so the parser in 03_parse.py
# can reliably extract the reasoning chain vs the final answer.
SYSTEM_PROMPT = """You are a careful mathematical reasoner.
For every problem:
1. Reason step by step inside <think>...</think> tags.
2. Each reasoning step should be on its own line, starting with "Step N:".
3. After </think>, write only the final numeric answer on a single line prefixed with "Answer:".

Example format:
<think>
Step 1: Identify what is being asked.
Step 2: ...
Step N: Final calculation gives X.
</think>
Answer: X"""

def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    # apply_chat_template handles model-specific formatting
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_id: str, load_in_4bit: bool = False):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batch generation

    print(f"Loading model: {model_id}  (4bit={load_in_4bit})")
    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. VRAM used: {vram:.1f} GB\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 1024,
) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(
        outputs[:, prompt_len:],
        skip_special_tokens=True,
    )
    return decoded


# ---------------------------------------------------------------------------
# Ground truth answer extraction
# ---------------------------------------------------------------------------
def extract_gsm8k_answer(solution: str) -> str | None:
    """GSM8K solutions end with #### <number>"""
    match = re.search(r"####\s*([\d,.\-]+)", solution)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--split",      default="train", choices=["train", "test"])
    parser.add_argument("--n",          type=int, default=2000, help="Number of problems to process")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--load_4bit",  action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--out",        default="raw_traces.jsonl")
    parser.add_argument("--resume",     action="store_true", help="Skip already-processed problems")
    args = parser.parse_args()

    # Load dataset
    print("Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    ds = ds.select(range(min(args.n, len(ds))))
    print(f"Using {len(ds)} problems from GSM8K {args.split}\n")

    # Resume: track already-done question hashes
    done_questions = set()
    if args.resume and Path(args.out).exists():
        with jsonlines.open(args.out) as reader:
            for row in reader:
                done_questions.add(row["question"])
        print(f"Resuming — {len(done_questions)} already done\n")

    model, tokenizer = load_model(args.model, load_in_4bit=args.load_4bit)

    out_path = Path(args.out)
    total_written = len(done_questions)

    # Batch loop
    problems = [p for p in ds if p["question"] not in done_questions]
    batches  = [problems[i:i+args.batch_size] for i in range(0, len(problems), args.batch_size)]

    with jsonlines.open(out_path, mode="a") as writer:
        for batch in tqdm(batches, desc="Generating traces"):
            questions  = [p["question"]  for p in batch]
            solutions  = [p["answer"]    for p in batch]
            gt_answers = [extract_gsm8k_answer(s) for s in solutions]
            prompts    = [build_prompt(tokenizer, q) for q in questions]

            try:
                responses = generate_batch(
                    model, tokenizer, prompts,
                    max_new_tokens=args.max_new_tokens,
                )
            except RuntimeError as e:
                print(f"\nOOM or generation error: {e} — skipping batch")
                torch.cuda.empty_cache()
                continue

            for question, solution, gt_ans, response in zip(
                questions, solutions, gt_answers, responses
            ):
                writer.write({
                    "question":    question,
                    "gt_solution": solution,
                    "gt_answer":   gt_ans,
                    "raw_output":  response,
                    "model":       args.model,
                    "timestamp":   time.time(),
                })
                total_written += 1

    print(f"\nDone. Wrote {total_written} raw traces to {out_path}")


if __name__ == "__main__":
    main()