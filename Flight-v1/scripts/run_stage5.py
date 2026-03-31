from __future__ import annotations

import argparse

from training.train_distill import load_config, train_distill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 5 full-system distillation.")
    parser.add_argument("--config", type=str, required=True, help="Path to Stage 5 YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    results = train_distill(config)

    print("Stage 5 distillation complete.\n")
    print("Training summary:")
    print(f"  Total steps:           {results['total_steps']}")
    print(f"  Best val step:         {results['best_val_step']}")
    print(f"  Final L_task:          {results['final_l_task']:.4f}")
    print(f"  Final L_KD:            {results['final_l_kd']:.4f}")
    print(f"  Final L_trace:         {results['final_l_trace']:.4f}")
    print(f"  Final L_geom:          {results['final_l_geom']:.4f}\n")

    print("Evaluation results:")
    print(f"  GSM8K val accuracy:    {results['gsm8k'].accuracy * 100:.2f}%  ({results['gsm8k'].n_correct}/{results['gsm8k'].n_total} correct)")
    print(f"  MATH-500 accuracy:     {results['math500'].accuracy * 100:.2f}%  ({results['math500'].n_correct}/{results['math500'].n_total} correct)")
    print(f"  Abstain rate:          {results['gsm8k'].abstain_rate * 100:.2f}%")
    print(f"  Avg router confidence: {results['gsm8k'].avg_router_conf:.4f}\n")

    print("Per-skill accuracy (GSM8K):")
    for skill, value in sorted(results["gsm8k"].per_skill.items()):
        print(f"  {skill:<22} {value * 100:.2f}%")
    print("\nPer-skill accuracy (MATH-500):")
    for skill, value in sorted(results["math500"].per_skill.items()):
        print(f"  {skill:<22} {value * 100:.2f}%")

    print("\nCheckpoints saved:")
    print(f"  {results['best_checkpoint']}")
    print(f"  {results['final_checkpoint']}")
    print(f"Eval results: {results['eval_results']}")
    print("Ready for Stage 6.")


if __name__ == "__main__":
    main()
