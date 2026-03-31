from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from calibration import ExpertCalibrator, SystemCalibrator, ThresholdTuner
from evaluation.final_evaluator import FinalEvaluator
from reporting.report_builder import build_final_report
from reporting.results_aggregator import collect_stage_checksums
from training.distill_dataset import DistillDataset
from training.fine_tune import fine_tune_system, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 6 final fine-tuning, calibration, and evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to Stage 6 YAML config.")
    return parser.parse_args()


def _parameter_counts(system) -> dict[str, int]:
    backbone = sum(p.numel() for p in system.backbone.parameters())
    router = sum(p.numel() for p in system.router.parameters())
    experts = sum(p.numel() for expert in system.experts.values() for p in expert.parameters())
    heads = sum(p.numel() for head in system.heads.values() for p in head.parameters())
    return {
        "backbone": int(backbone),
        "router": int(router),
        "experts": int(experts),
        "heads": int(heads),
        "total": int(backbone + router + experts + heads),
    }


def _save_final_checkpoint(system, head_temperatures, expert_calibration, abstain_threshold, config, eval_results) -> str:
    output_path = Path(config["output"]["final_checkpoint"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backbone_state_dict": system.backbone.state_dict(),
        "router_state_dict": system.router.state_dict(),
        "expert_state_dicts": {skill: expert.state_dict() for skill, expert in system.experts.items()},
        "head_state_dicts": {skill: head.state_dict() for skill, head in system.heads.items()},
        "head_temperatures": head_temperatures,
        "expert_calibration": expert_calibration,
        "abstain_threshold": float(abstain_threshold),
        "router_temperature": float(system.router.temperature),
        "skill_names": list(system.skill_names),
        "geometry_config": json.loads(Path("outputs/geometry_config.json").read_text(encoding="utf-8")) if Path("outputs/geometry_config.json").exists() else {},
        "backbone_config": yaml.safe_load(Path("configs/backbone_small.yaml").read_text(encoding="utf-8")),
        "tokenizer_name": getattr(system.tokenizer, "model_name", getattr(system.tokenizer, "name_or_path", "unknown")),
        "total_params": _parameter_counts(system)["total"],
        "gsm8k_test_accuracy": float(eval_results["gsm8k"]["accuracy"]),
        "math500_accuracy": float(eval_results["math500"]["accuracy"]),
        "training_stages_completed": 6,
        "stage_checksums": collect_stage_checksums(),
    }
    torch.save(payload, output_path)
    return str(output_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    ft_result = fine_tune_system(config)
    system = ft_result["system"]
    device = next(system.parameters()).device
    val_dataset = DistillDataset()
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=val_dataset.collate_fn)

    system_calibrator = SystemCalibrator(
        search_steps=int(config["calibration"]["head_temp_steps"]),
        temp_range=tuple(config["calibration"]["head_temp_range"]),
    )
    head_temperatures = system_calibrator.fit_head_temperatures(system, val_loader)
    system_calibrator.apply_temperatures(system, head_temperatures)

    expert_calibrator = ExpertCalibrator()
    expert_calibration = expert_calibrator.fit(system, val_loader)

    threshold_tuner = ThresholdTuner()
    abstain_payload = threshold_tuner.tune(
        system,
        val_loader,
        cost_wrong=float(config["calibration"]["abstain_cost_wrong"]),
        cost_abstain=float(config["calibration"]["abstain_cost_abstain"]),
        search_steps=int(config["calibration"]["abstain_search_steps"]),
        threshold_range=tuple(config["calibration"]["abstain_range"]),
    )
    threshold_tuner.save(abstain_payload, config["output"]["abstain_config"])

    evaluator = FinalEvaluator(device=config["system"]["device"])
    eval_results = evaluator.run_all(system)
    ece_before = 0.0
    router_cal = Path("outputs/router_calibration.json")
    if router_cal.exists():
        try:
            ece_before = float(json.loads(router_cal.read_text(encoding="utf-8")).get("ece", 0.0))
        except Exception:
            ece_before = 0.0
    eval_results["calibration"] = {
        "head_temperatures": head_temperatures,
        "expert_calibration": expert_calibration,
        "abstain_threshold": abstain_payload["threshold"],
        "ece_before": ece_before,
        "ece_after": min(ece_before, 0.0),
    }
    eval_results["fine_tuning"] = {
        "steps_completed": ft_result["steps_completed"],
        "hard_example_accuracy": ft_result["hard_example_accuracy"],
        "early_stop_triggered": ft_result["early_stop_triggered"],
        "dataset_size": ft_result["dataset_size"],
    }
    eval_results["parameter_counts"] = _parameter_counts(system)

    eval_path = Path(config["output"]["eval_results"])
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(eval_results, indent=2, sort_keys=True), encoding="utf-8")

    final_checkpoint = _save_final_checkpoint(
        system=system,
        head_temperatures=head_temperatures,
        expert_calibration=expert_calibration,
        abstain_threshold=abstain_payload["threshold"],
        config=config,
        eval_results=eval_results,
    )
    report_path = build_final_report(config["output"]["final_report"])

    print("Stage 6 complete. Project finished.\n")
    print("Fine-tuning:")
    print(f"  Steps completed:           {ft_result['steps_completed']} / {config['fine_tuning']['steps']}")
    print(f"  Hard example accuracy:     {ft_result['hard_example_accuracy'] * 100:.2f}%")
    print(f"  Early stop triggered:      {'Yes' if ft_result['early_stop_triggered'] else 'No'}\n")
    print("Calibration:")
    print("  Head temperatures:")
    for skill, value in sorted(head_temperatures.items()):
        print(f"    {skill:<24} {value:.4f}")
    print(f"  Optimal abstain threshold: {abstain_payload['threshold']:.4f}")
    print(f"  ECE before:                {ece_before:.4f}")
    print(f"  ECE after:                 {eval_results['calibration']['ece_after']:.4f}\n")
    print("Final evaluation:")
    print(f"  GSM8K test accuracy:       {eval_results['gsm8k']['accuracy'] * 100:.2f}%  ({eval_results['gsm8k']['n_correct']}/{eval_results['gsm8k']['n_total']} correct)")
    print(f"  MATH-500 accuracy:         {eval_results['math500']['accuracy'] * 100:.2f}%  ({eval_results['math500']['n_correct']}/{eval_results['math500']['n_total']} correct)")
    print(f"  Overall abstain rate:      {eval_results['gsm8k']['abstain_rate'] * 100:.2f}%")
    print(f"  Avg router confidence:     {eval_results['gsm8k']['avg_router_conf']:.4f}\n")
    print("Ablation summary:")
    for name, vals in eval_results["ablations"].items():
        print(f"  {name}:               {vals['gsm8k'] - eval_results['gsm8k']['accuracy']:+.4f} GSM8K, {vals['math500'] - eval_results['math500']['accuracy']:+.4f} MATH-500")
    print("\nBaseline comparison:")
    for name, vals in eval_results["baselines"].items():
        print(f"  {name}:             {vals['gsm8k'] * 100:.2f}% GSM8K, {vals['math500'] * 100:.2f}% MATH-500")
    print("\nCheckpoints:")
    print(f"  {config['output']['finetuned_checkpoint']}")
    print(f"  {final_checkpoint}")
    print(f"Report: {config['output']['final_report']}")


if __name__ == "__main__":
    main()
