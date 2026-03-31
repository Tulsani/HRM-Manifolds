from __future__ import annotations

import argparse

import yaml

from analysis.embedding_extractor import extract_and_save_embeddings
from analysis.geometry_probe import run_geometry_analysis
from analysis.report_generator import write_stage2_report
from analysis.skill_refiner import refine_skill_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 skill refinement and geometry analysis.")
    parser.add_argument("--config", type=str, required=True, help="Path to Stage 2 YAML config.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    skill_labels_payload = refine_skill_labels(
        trace_dir=config["trace_library"]["path"],
        output_path=config["output"]["skill_labels"],
    )
    embedding_summary = extract_and_save_embeddings(config)
    geometry_payload = run_geometry_analysis(config)
    write_stage2_report(
        skill_labels_payload=skill_labels_payload,
        geometry_payload=geometry_payload,
        output_path=config["output"]["report"],
    )

    print(f"Skill labels written:    {config['output']['skill_labels']}")
    print(f"Embeddings extracted:    {embedding_summary['count']} examples")
    print(f"Geometry analysis done:  {geometry_payload['summary']['analyzed_skill_groups']} skill groups analyzed")
    print("Geometry decisions:")
    for skill, info in sorted(geometry_payload["skills"].items()):
        delta = info["test_results"]["delta_norm"]
        delta_text = "n/a" if delta is None else f"{delta:.2f}"
        depth_text = f"{info['test_results']['avg_depth']:.2f}"
        print(f"  {skill:<18} {info['geometry']:<10} (delta={delta_text}, depth={depth_text})")
    print(f"Report written:          {config['output']['report']}")
    print(f"geometry_config.json written: {config['output']['geometry_config']}")
    print("Ready for Stage 3.")


if __name__ == "__main__":
    main()
