from __future__ import annotations

import re
from collections import Counter


VALID_SKILLS = {
    "arithmetic",
    "algebraic",
    "multi_step_reason",
    "geometric",
    "number_theory",
    "combinatorics",
    "fallback",
}

_SUBSKILL_PATTERNS = [
    (re.compile(r"simplif", re.IGNORECASE), "simplification"),
    (re.compile(r"substitut", re.IGNORECASE), "substitution"),
    (re.compile(r"equation", re.IGNORECASE), "equation_setup"),
    (re.compile(r"multiply|product", re.IGNORECASE), "multiplication"),
    (re.compile(r"divide|quotient", re.IGNORECASE), "division"),
    (re.compile(r"factor", re.IGNORECASE), "factoring"),
    (re.compile(r"expand", re.IGNORECASE), "expansion"),
    (re.compile(r"percent", re.IGNORECASE), "percentage"),
    (re.compile(r"ratio", re.IGNORECASE), "ratio"),
]

_PRIMARY_SKILL_PATTERNS = {
    "geometric": re.compile(r"angle|perimeter|area|volume|triangle|circle|radius|diameter|polygon", re.IGNORECASE),
    "number_theory": re.compile(r"prime|divisib|mod|remainder|factorial|gcd|lcm|integer", re.IGNORECASE),
    "combinatorics": re.compile(r"probabil|count|arrange|combination|permutation|choose|ways", re.IGNORECASE),
    "algebraic": re.compile(r"equation|solve for|variable|substitut|system of equations|quadratic", re.IGNORECASE),
    "arithmetic": re.compile(r"sum|total|difference|average|add|subtract|multiply|divide|cost|price", re.IGNORECASE),
}


def extract_subskills(step_texts: list[str]) -> list[str]:
    subskills: list[str] = []
    for step in step_texts:
        for pattern, label in _SUBSKILL_PATTERNS:
            if pattern.search(step):
                subskills.append(label)
    return subskills


def infer_primary_skill(problem: str, rationale: str, step_texts: list[str], source: str, math_subject: str = "") -> str:
    subject_blob = f"{problem}\n{rationale}\n{math_subject}".lower()
    for label in ("geometric", "number_theory", "combinatorics", "algebraic"):
        if _PRIMARY_SKILL_PATTERNS[label].search(subject_blob):
            return label

    if len(step_texts) >= 3:
        return "multi_step_reason"
    if source == "gsm8k" and _PRIMARY_SKILL_PATTERNS["arithmetic"].search(subject_blob):
        return "arithmetic"
    if source == "gsm8k":
        return "arithmetic"
    if _PRIMARY_SKILL_PATTERNS["arithmetic"].search(subject_blob):
        return "arithmetic"
    return "fallback"


def tag_trace(problem: str, rationale: str, step_texts: list[str], source: str, math_subject: str = "") -> tuple[str, list[str]]:
    skill = infer_primary_skill(problem=problem, rationale=rationale, step_texts=step_texts, source=source, math_subject=math_subject)
    subskills = extract_subskills(step_texts)
    if skill not in VALID_SKILLS:
        raise ValueError(f"Invalid skill label generated: {skill}")
    return skill, subskills


def build_skill_index(example_to_skill: dict[str, str]) -> dict[str, list[str]]:
    grouped = {skill: [] for skill in VALID_SKILLS}
    for example_id, skill in example_to_skill.items():
        grouped.setdefault(skill, []).append(example_id)
    for ids in grouped.values():
        ids.sort()
    return grouped


def count_skills(skills: list[str]) -> Counter[str]:
    return Counter(skills)
