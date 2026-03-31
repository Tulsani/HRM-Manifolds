from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_stage_checksums() -> dict[str, str | dict[str, str]]:
    checksums: dict[str, Any] = {}
    stage0 = Path("checkpoints/backbone_stage0.pt")
    checksums["stage0"] = file_sha256(stage0) if stage0.exists() else ""
    stage3 = {}
    for path in sorted(Path("checkpoints").glob("expert_*_stage3.pt")):
        stage3[path.name] = file_sha256(path)
    checksums["stage3"] = stage3
    stage4 = Path("checkpoints/router_stage4.pt")
    checksums["stage4"] = file_sha256(stage4) if stage4.exists() else ""
    stage5 = Path("checkpoints/full_system_stage5.pt")
    checksums["stage5"] = file_sha256(stage5) if stage5.exists() else ""
    return checksums


def load_json_if_exists(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def aggregate_results() -> dict[str, Any]:
    return {
        "stage2": load_json_if_exists("outputs/geometry_config.json"),
        "stage4": load_json_if_exists("outputs/router_calibration.json"),
        "stage5": load_json_if_exists("outputs/stage5_eval.json"),
        "stage6": load_json_if_exists("outputs/stage6_eval.json"),
        "abstain": load_json_if_exists("outputs/abstain_threshold.json"),
        "checksums": collect_stage_checksums(),
    }
