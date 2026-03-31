from __future__ import annotations

import argparse

from data.trace_generator import load_config, run_trace_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 1 teacher traces.")
    parser.add_argument("--config", type=str, required=True, help="Path to the trace generation YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = run_trace_generation(config)

    gsm8k = summary["per_source"]["gsm8k"]
    math = summary["per_source"]["math"]
    skill_counts = summary["skills"]

    print(f"GSM8K traces collected: {gsm8k['count']} / {gsm8k['target']}")
    print(f"MATH traces collected:  {math['count']} / {math['target']}")
    print(f"Teacher accuracy GSM8K: {gsm8k['accuracy']:.2f}%")
    print(f"Teacher accuracy MATH:  {math['accuracy']:.2f}%")
    print("Skill distribution:")
    for skill in [
        "arithmetic",
        "algebraic",
        "multi_step_reason",
        "geometric",
        "number_theory",
        "combinatorics",
        "fallback",
    ]:
        print(f"  {skill:<18} {skill_counts.get(skill, 0)}")
    print(f"Trace library saved to: {summary['trace_dir']}")
    print("Ready for Stage 2.")


if __name__ == "__main__":
    main()
