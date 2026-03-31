from __future__ import annotations

import argparse

from training.train_experts import load_config, train_all_experts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 manifold expert pretraining.")
    parser.add_argument("--config", type=str, required=True, help="Path to Stage 3 YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    results = train_all_experts(config)

    print("Expert pretraining complete.")
    print("Skills trained:")
    for result in results:
        print(
            f"  {result['skill']:<18} ({result['geometry']})"
            f"    steps={result['steps']}  L_final={result['loss_final']:.4f}"
        )
    print("Checkpoints saved:")
    for result in results:
        print(f"  {result['checkpoint_path']}")
    print("Ready for Stage 4.")


if __name__ == "__main__":
    main()
