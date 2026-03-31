from __future__ import annotations

import numpy as np

from data.skill_tagger import VALID_SKILLS, tag_trace
from data.trace_generator import count_processed_examples, get_resume_state, get_output_paths, split_rationale_into_steps
from data.trace_schema import SparseLogits, TracePackage, load_sparse_logits, save_sparse_logits


def test_trace_package_serializes_round_trip() -> None:
    package = TracePackage(
        example_id="gsm8k_train_0042",
        source="gsm8k",
        difficulty="easy",
        problem="What is 2 + 2?",
        final_answer="4",
        answer_correct=True,
        soft_logits=SparseLogits(indices=np.array([1, 5, 9]), values=np.array([1.5, 0.5, -0.25])),
        confidence=0.87,
        rationale="First add 2 and 2.\nTherefore the answer is 4.",
        step_texts=["First add 2 and 2.", "Therefore the answer is 4."],
        n_steps=2,
        skill="arithmetic",
        subskills=["simplification"],
        math_subject="",
        teacher_model="Qwen/Qwen2.5-7B-Instruct",
        timestamp="2026-03-30T00:00:00+00:00",
    )

    line = package.to_jsonl()
    restored = TracePackage.from_jsonl(line, sparse_logits=package.soft_logits)

    assert restored.example_id == package.example_id
    assert restored.final_answer == package.final_answer
    assert restored.skill == package.skill
    assert restored.soft_logits.indices.tolist() == [1, 5, 9]
    assert np.allclose(restored.soft_logits.values.astype(np.float32), np.array([1.5, 0.5, -0.25], dtype=np.float32))


def test_skill_tagger_returns_valid_labels() -> None:
    cases = [
        ("Add 12 and 7 to find the total cost.", "First we add the numbers.", ["First we add the numbers."], "gsm8k", "", "arithmetic"),
        ("Solve the equation 2x + 3 = 7.", "Step 1: Set up the equation and substitute.", ["Step 1: Set up the equation and substitute."], "math", "algebra", "algebraic"),
        ("Find the area of a circle with radius 3.", "First compute the area.", ["First compute the area."], "math", "geometry", "geometric"),
        ("How many ways can 3 students sit in 3 chairs?", "Then count the permutations.", ["Then count the permutations."], "math", "counting", "combinatorics"),
        ("A store changes prices in several stages before tax.", "First compute the discount. Then divide by 2. Finally simplify.", ["First compute the discount.", "Then divide by 2.", "Finally simplify."], "gsm8k", "", "multi_step_reason"),
    ]

    for problem, rationale, step_texts, source, subject, expected in cases:
        skill, subskills = tag_trace(problem, rationale, step_texts, source, math_subject=subject)
        assert skill in VALID_SKILLS
        assert skill == expected
        assert isinstance(subskills, list)


def test_step_splitter_handles_edge_cases() -> None:
    assert split_rationale_into_steps("") == []
    assert split_rationale_into_steps("This is a single sufficiently long reasoning step.") == [
        "This is a single sufficiently long reasoning step."
    ]
    assert split_rationale_into_steps("First compute x. Then simplify.\nFinally answer.") == [
        "First compute x.",
        "Then simplify.",
        "Finally answer.",
    ]


def test_sparse_logits_round_trip(tmp_path) -> None:
    path = tmp_path / "logits.npz"
    original = {
        "ex1": SparseLogits(indices=np.array([2, 4], dtype=np.int32), values=np.array([0.5, -1.25], dtype=np.float16)),
        "ex2": SparseLogits(indices=np.array([1], dtype=np.int32), values=np.array([3.0], dtype=np.float16)),
    }
    save_sparse_logits(path, original)
    restored = load_sparse_logits(path)

    assert restored.keys() == original.keys()
    assert restored["ex1"].indices.tolist() == [2, 4]
    assert np.allclose(restored["ex1"].values.astype(np.float32), np.array([0.5, -1.25], dtype=np.float32))
    assert restored["ex2"].indices.tolist() == [1]


def test_resume_logic_skips_processed_examples(tmp_path) -> None:
    trace_dir = tmp_path / "trace_library"
    paths = get_output_paths(trace_dir, "gsm8k")
    paths.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    paths.logits_path.parent.mkdir(parents=True, exist_ok=True)

    examples = [
        TracePackage(
            example_id=f"gsm8k_train_{idx:04d}",
            source="gsm8k",
            difficulty="easy",
            problem=f"Problem {idx}",
            final_answer=str(idx),
            answer_correct=True,
            soft_logits=SparseLogits(indices=np.array([idx], dtype=np.int32), values=np.array([0.1], dtype=np.float16)),
            confidence=0.5,
            rationale="First reason. Then answer.",
            step_texts=["First reason.", "Then answer."],
            n_steps=2,
            skill="arithmetic",
            subskills=[],
            math_subject="",
            teacher_model="teacher",
            timestamp="2026-03-30T00:00:00+00:00",
        )
        for idx in range(2)
    ]

    with paths.jsonl_path.open("w", encoding="utf-8") as handle:
        for trace in examples:
            handle.write(trace.to_jsonl())
            handle.write("\n")
    save_sparse_logits(paths.logits_path, {trace.example_id: trace.soft_logits for trace in examples})

    processed_count, logits_map = get_resume_state(trace_dir, "gsm8k")

    assert count_processed_examples(paths.jsonl_path) == 2
    assert processed_count == 2
    assert sorted(logits_map) == ["gsm8k_train_0000", "gsm8k_train_0001"]
