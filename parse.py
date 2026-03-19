"""
03_parse.py
-----------
Parse raw model outputs into structured reasoning traces.
Each output trace becomes:
  {
    "question":    str,
    "gt_answer":   str,
    "think_steps": ["Step 1: ...", "Step 2: ...", ...],  # hierarchical chain
    "model_answer": str,
    "correct":     bool,
    "n_steps":     int,
    "raw_output":  str,   # kept for debugging
  }

Usage:
  python 03_parse.py --inp raw_traces.jsonl --out parsed_traces.jsonl
"""

import argparse
import re
from pathlib import Path

import jsonlines
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_think_block(text: str) -> str | None:
    """Extract content between <think>...</think> tags."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: if no tags, treat everything before "Answer:" as thinking
    parts = re.split(r"\bAnswer\s*:", text, flags=re.IGNORECASE, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip()
    return None


def extract_model_answer(text: str) -> str | None:
    """Extract the final answer after 'Answer:' tag."""
    match = re.search(r"\bAnswer\s*:\s*([\d,.\-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last standalone number in the text
    numbers = re.findall(r"(?<!\w)([\d,.\-]+)(?!\w)", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def parse_steps(think_block: str) -> list[str]:
    """
    Extract individual reasoning steps from the think block.
    Handles multiple formats the model might produce:
      - "Step N: ..."  (our prompted format)
      - Numbered lines "1. ..."
      - Plain newline-separated paragraphs (fallback)
    Returns a list of step strings, cleaned.
    """
    # Try "Step N: ..." format first
    steps = re.findall(r"Step\s*\d+\s*:(.+?)(?=Step\s*\d+\s*:|$)", think_block, re.DOTALL)
    if len(steps) >= 2:
        return [s.strip() for s in steps if s.strip()]

    # Try "N. ..." numbered list
    steps = re.findall(r"^\d+\.\s*(.+)", think_block, re.MULTILINE)
    if len(steps) >= 2:
        return [s.strip() for s in steps if s.strip()]

    # Fallback: split on double newlines (paragraphs)
    steps = [p.strip() for p in re.split(r"\n\n+", think_block) if p.strip()]
    if steps:
        return steps

    # Last resort: split on single newlines
    steps = [l.strip() for l in think_block.splitlines() if l.strip()]
    return steps


def answers_match(model_ans: str | None, gt_ans: str | None) -> bool:
    """Numeric comparison, tolerating minor float differences."""
    if model_ans is None or gt_ans is None:
        return False
    try:
        return abs(float(model_ans) - float(gt_ans)) < 1e-3
    except ValueError:
        return model_ans.strip() == gt_ans.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default="raw_traces.jsonl")
    parser.add_argument("--out", default="parsed_traces.jsonl")
    args = parser.parse_args()

    inp_path = Path(args.inp)
    out_path = Path(args.out)

    if not inp_path.exists():
        raise FileNotFoundError(f"Input file not found: {inp_path}")

    stats = {"total": 0, "parsed": 0, "correct": 0, "no_think": 0, "no_answer": 0}

    with jsonlines.open(inp_path) as reader, \
         jsonlines.open(out_path, mode="w") as writer:

        rows = list(reader)
        for row in tqdm(rows, desc="Parsing traces"):
            stats["total"] += 1
            raw = row.get("raw_output", "")

            think_block  = extract_think_block(raw)
            model_answer = extract_model_answer(raw)

            if think_block is None:
                stats["no_think"] += 1
            if model_answer is None:
                stats["no_answer"] += 1

            steps   = parse_steps(think_block) if think_block else []
            correct = answers_match(model_answer, row.get("gt_answer"))

            if correct:
                stats["correct"] += 1
            stats["parsed"] += 1

            writer.write({
                "question":     row["question"],
                "gt_answer":    row.get("gt_answer"),
                "think_steps":  steps,
                "model_answer": model_answer,
                "correct":      correct,
                "n_steps":      len(steps),
                "model":        row.get("model"),
                "raw_output":   raw,
            })

    # Summary
    t = stats["total"]
    print(f"\nParsing complete")
    print(f"  Total rows    : {t}")
    print(f"  Parsed        : {stats['parsed']}")
    print(f"  Correct       : {stats['correct']}  ({100*stats['correct']/max(t,1):.1f}%)")
    print(f"  No <think>    : {stats['no_think']}")
    print(f"  No answer     : {stats['no_answer']}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()