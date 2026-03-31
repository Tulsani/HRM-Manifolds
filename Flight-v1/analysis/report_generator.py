from __future__ import annotations

from pathlib import Path
from typing import Any


def _format_skill_distribution(skill_counts: dict[str, int]) -> list[str]:
    lines = ["| Skill | Count |", "| --- | ---: |"]
    for skill, count in sorted(skill_counts.items()):
        lines.append(f"| {skill} | {count} |")
    return lines


def _build_justification(skill: str, info: dict[str, Any]) -> str:
    results = info["test_results"]
    geometry = info["geometry"]
    if info["n_examples"] < 50:
        return "This skill group is small, so Stage 2 kept the safe default product manifold until more evidence is available."
    if geometry == "hyperbolic":
        return "The skill shows tree-like and hierarchical structure, which makes negative curvature a good fit for organizing expert states."
    if geometry == "euclidean":
        return "The skill shows stable directional step transitions and clean local neighborhoods, so a flat manifold should model it well."
    return "The evidence is mixed across hierarchy and sequential structure, so a hybrid product manifold is the most robust choice."


def write_stage2_report(
    skill_labels_payload: dict[str, Any],
    geometry_payload: dict[str, Any],
    output_path: str | Path,
) -> str:
    summary = skill_labels_payload["summary"]
    lines = [
        "# Stage 2 Geometry Analysis Report",
        "",
        "## Skill label summary",
        f"- Total examples: {summary['total']}",
        f"- Low-confidence examples: {summary['low_confidence_count']}",
        f"- Incorrect examples: {summary['incorrect_count']}",
        "",
    ]
    lines.extend(_format_skill_distribution(summary["skill_counts"]))
    lines.extend(["", "## Geometry decisions", ""])

    for skill, info in sorted(geometry_payload["skills"].items()):
        results = info["test_results"]
        delta_text = "n/a" if results["delta_norm"] is None else f"{results['delta_norm']:.2f}"
        lines.extend(
            [
                f"### {skill}",
                f"- Examples: {info['n_examples']}",
                f"- Geometry chosen: {info['geometry']}",
                (
                    "- Key evidence: "
                    f"delta_norm={delta_text}, "
                    f"depth={results['avg_depth']:.2f}, "
                    f"consistency={results['step_consistency']:.2f}, "
                    f"purity={results['nn_purity']:.2f}"
                ),
                f"- Justification: {_build_justification(skill, info)}",
                "",
            ]
        )

    lines.append("## Recommendations for Stage 3")
    weak_signal = geometry_payload["summary"]["weak_signal_skills"]
    ambiguous = geometry_payload["summary"]["ambiguous_skills"]
    lines.append(
        "- Weak signal skills: "
        + (", ".join(weak_signal) if weak_signal else "none")
    )
    lines.append(
        "- Ambiguous geometry decisions: "
        + (", ".join(ambiguous) if ambiguous else "none")
    )
    if weak_signal:
        lines.append("- Suggestion: consider merging the smallest skills into a broader expert until each group has at least 50 examples.")
    else:
        lines.append("- Suggestion: current skill groups are large enough to keep separate experts in Stage 3.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines) + "\n"
    output_path.write_text(report_text, encoding="utf-8")
    return report_text
