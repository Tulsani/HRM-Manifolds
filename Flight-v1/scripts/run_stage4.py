from __future__ import annotations

import argparse

from training.train_router import load_config, train_router


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 router training.")
    parser.add_argument("--config", type=str, required=True, help="Path to Stage 4 YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    results = train_router(config)

    print("Router training complete.\n")
    print("Phase 1 — skill routing:")
    print(f"  Best val accuracy:       {results['best_val_accuracy'] * 100:.2f}%")
    print("  Per-skill accuracy:")
    for skill, value in sorted(results["per_skill_accuracy"].items()):
        print(f"    {skill:<22} {value * 100:.2f}%")
    print("\nPhase 2 — abstain head:")
    print(f"  Abstain precision:       {results['abstain_precision'] * 100:.2f}%")
    print(f"  Abstain recall:          {results['abstain_recall'] * 100:.2f}%")
    print("\nPhase 3 — calibration:")
    print(f"  Temperature:             {results['temperature']:.4f}")
    print(f"  ECE:                     {results['ece']:.4f}")
    print(f"  MCE:                     {results['mce']:.4f}")
    print(f"  Final val accuracy:      {results['final_val_accuracy'] * 100:.2f}%")
    print("\nCheckpoints saved:")
    print(f"  {results['best_checkpoint']}")
    print(f"  {results['final_checkpoint']}")
    print(f"Calibration report: {results['calibration_report']}")
    print("Ready for Stage 5.")


if __name__ == "__main__":
    main()
