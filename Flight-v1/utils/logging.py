from __future__ import annotations

from collections import defaultdict


class MetricsLogger:
    def __init__(self, log_every: int = 50) -> None:
        self.log_every = log_every
        self.storage: dict[str, list[float]] = defaultdict(list)

    def update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.storage[key].append(float(value))

    def mean(self) -> dict[str, float]:
        return {key: sum(values) / len(values) for key, values in self.storage.items() if values}

    def reset(self) -> None:
        self.storage.clear()


def format_metrics(step: int, metrics: dict[str, float]) -> str:
    ordered = " | ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
    return f"step={step} | {ordered}"
