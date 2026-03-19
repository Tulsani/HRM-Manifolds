"""
04_filter.py
------------
Quality-filter parsed traces and produce the final traces.jsonl
that will be used for distillation training.

Filtering criteria:
  1. Answer must be correct (model_answer == gt_answer)
  2. Must have >= MIN_STEPS reasoning steps
  3. Must have <= MAX_STEPS (remove runaway/looping traces)
  4. Each step must be >= MIN_STEP_CHARS characters (remove trivial steps)
  5. No degenerate repetition (a single step repeated > REPEAT_THRESHOLD times)

Usage:
  python 04_filter.py --inp parsed_traces.jsonl --out traces.jsonl
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import jsonlines
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Filter thresholds (tune these based on your dataset statistics)
# ---------------------------------------------------------------------------
MIN_STEPS        = 3      # too few steps = not enough reasoning signal
MAX_STEPS        = 30     # too many = likely a looping/degenerate trace
MIN_STEP_CHARS   = 20     # steps shorter than this are noise
REPEAT_THRESHOLD = 0.4    # if >40% of steps are near-identical, reject


# ---------------------------------------------------------------------------
# Filter functions
# ---------------------------------------------------------------------------

def is_repetitive(steps: list[str], threshold: float = REPEAT_THRESHOLD) -> bool:
    """Detect traces where the model is looping / repeating itself."""
    if len(steps) < 4:
        return False
    # Normalize: lowercase, strip whitespace, collapse spaces
    normalized = [re.sub(r"\s+", " ", s.lower().strip()) for s in steps]
    counts = Counter(normalized)
    most_common_count = counts.most_common(1)[0][1]
    return (most_common_count / len(steps)) > threshold


def passes_filters(row: dict) -> tuple[bool, str]:
    """
    Returns (keep: bool, reason: str).
    reason is populated only when keep=False.
    """
    if not row.get("correct", False):
        return False, "wrong_answer"

    steps = row.get("think_steps", [])
    n     = len(steps)

    if n < MIN_STEPS:
        return False, f"too_few_steps ({n})"

    if n > MAX_STEPS:
        return False, f"too_many_steps ({n})"

    # Check step quality
    short_steps = [s for s in steps if len(s) < MIN_STEP_CHARS]
    if len(short_steps) > n * 0.5:
        return False, "too_many_short_steps"

    if is_repetitive(steps):
        return False, "repetitive_trace"

    if not row.get("question", "").strip():
        return False, "empty_question"

    return True, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp",           default="parsed_traces.jsonl")
    parser.add_argument("--out",           default="traces.jsonl")
    parser.add_argument("--min_steps",     type=int,   default=MIN_STEPS)
    parser.add_argument("--max_steps",     type=int,   default=MAX_STEPS)
    parser.add_argument("--min_step_chars",type=int,   default=MIN_STEP_CHARS)
    args = parser.parse_args()

    # Allow CLI overrides
    global MIN_STEPS, MAX_STEPS, MIN_STEP_CHARS
    MIN_STEPS      = args.min_steps
    MAX_STEPS      = args.max_steps
    MIN_STEP_CHARS = args.min_step_chars

    inp_path = Path(args.inp)
    out_path = Path(args.out)

    if not inp_path.exists():
        raise FileNotFoundError(f"Input not found: {inp_path}")

    rejection_reasons = Counter()
    kept = 0
    total = 0

    with jsonlines.open(inp_path) as reader, \
         jsonlines.open(out_path, mode="w") as writer:

        rows = list(reader)
        for row in tqdm(rows, desc="Filtering"):
            total += 1
            keep, reason = passes_filters(row)

            if not keep:
                rejection_reasons[reason] += 1
                continue

            # Write final clean record — drop raw_output to save space
            writer.write({
                "question":     row["question"],
                "gt_answer":    row["gt_answer"],
                "think_steps":  row["think_steps"],
                "model_answer": row["model_answer"],
                "n_steps":      row["n_steps"],
                "model":        row.get("model", ""),
            })
            kept += 1

    # Summary
    print(f"\nFiltering complete")
    print(f"  Input  : {total}")
    print(f"  Kept   : {kept}  ({100*kept/max(total,1):.1f}%)")
    print(f"  Dropped: {total - kept}")
    print()
    print("Rejection breakdown:")
    for reason, count in rejection_reasons.most_common():
        print(f"  {reason:<30} {count:>5}  ({100*count/total:.1f}%)")
    print(f"\n  -> {out_path}")

    # Step count distribution for the kept traces
    if kept > 0:
        with jsonlines.open(out_path) as reader:
            step_counts = [row["n_steps"] for row in reader]
        avg = sum(step_counts) / len(step_counts)
        mn  = min(step_counts)
        mx  = max(step_counts)
        print(f"\nStep count stats (kept traces):")
        print(f"  min={mn}  avg={avg:.1f}  max={mx}")


if __name__ == "__main__":
    main()