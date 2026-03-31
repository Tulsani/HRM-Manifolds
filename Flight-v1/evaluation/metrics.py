from __future__ import annotations

import math
import re
from fractions import Fraction


NUMBER_PATTERN = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def extract_gsm8k_answer(text: str) -> float | None:
    if "####" in text:
        candidate = text.split("####")[-1]
        numbers = NUMBER_PATTERN.findall(candidate)
        if numbers:
            return float(numbers[-1].replace(",", ""))
    eq_match = re.findall(r"=\s*(-?\d+(?:\.\d+)?)", text)
    if eq_match:
        return float(eq_match[-1])
    numbers = NUMBER_PATTERN.findall(text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    return None


def extract_math_answer(text: str) -> str | None:
    match = BOXED_PATTERN.findall(text)
    if match:
        return match[-1].strip()
    return None


def _normalize_fraction(text: str) -> str:
    stripped = "".join(text.split())
    if "/" in stripped:
        try:
            return str(Fraction(stripped))
        except (ValueError, ZeroDivisionError):
            return stripped
    return stripped


def loose_match(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    try:
        pred_f = float(pred)
        gold_f = float(gold)
        return math.isclose(pred_f, gold_f, rel_tol=1e-3, abs_tol=1e-3)
    except (TypeError, ValueError):
        pred_s = _normalize_fraction(str(pred))
        gold_s = _normalize_fraction(str(gold))
        return pred_s == gold_s
