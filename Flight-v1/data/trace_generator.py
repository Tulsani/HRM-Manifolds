from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.skill_tagger import build_skill_index, count_skills, tag_trace
from data.trace_schema import SparseLogits, TracePackage, load_sparse_logits, save_sparse_logits

PROMPT_TEMPLATE = "Solve this step by step.\n\nProblem: {problem}\n\nSolution:"
STEP_PATTERN = re.compile(
    r"(?=(?:^|\s)(?:Step\s*\d+[:.)-]?|First\b|Then\b|Next\b|Finally\b|So\b|Therefore\b|\d+\.\s))",
    re.IGNORECASE,
)
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
NUMBER_PATTERN = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

HENDRYCKS_MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


@dataclass
class GenerationPaths:
    jsonl_path: Path
    logits_path: Path


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def split_rationale_into_steps(rationale: str, max_steps: int = 12) -> list[str]:
    text = rationale.strip()
    if not text:
        return []
    pieces: list[str] = []
    for segment in text.splitlines():
        segment = segment.strip()
        if not segment:
            continue
        split_segments = [
            part.strip(" -\t")
            for part in STEP_PATTERN.split(segment)
            if part.strip()
        ]
        if split_segments:
            pieces.extend(split_segments)
        else:
            pieces.append(segment)
    steps = [piece for piece in pieces if len(piece) >= 10]
    if not steps and len(text) >= 10:
        return [text[:]]
    return steps[:max_steps]


def extract_final_answer(generation: str) -> str:
    stripped = generation.strip()
    if not stripped:
        return ""
    answer_markers = [
        "final answer:",
        "answer:",
        "therefore,",
        "therefore",
        "thus,",
        "thus",
        "so,",
    ]
    lowered = stripped.lower()
    for marker in answer_markers:
        idx = lowered.rfind(marker)
        if idx != -1:
            candidate = stripped[idx + len(marker):].strip()
            if candidate:
                return candidate.splitlines()[-1].strip()
    return stripped.splitlines()[-1].strip()


def normalize_number(text: str) -> str:
    numbers = NUMBER_PATTERN.findall(text)
    if not numbers:
        return ""
    return numbers[-1].replace(",", "")


def extract_boxed_expression(text: str) -> str:
    boxed = BOXED_PATTERN.findall(text)
    if boxed:
        return boxed[-1].strip()
    return text.strip().splitlines()[-1].strip()


def verify_gsm8k_answer(ground_truth: str, generation: str) -> bool:
    gold = ""
    if "####" in ground_truth:
        gold = normalize_number(ground_truth.split("####", maxsplit=1)[-1])
    pred = normalize_number(generation)
    return bool(gold) and bool(pred) and gold == pred


def verify_math_answer(ground_truth: str, generation: str) -> bool:
    gold = re.sub(r"\s+", "", extract_boxed_expression(ground_truth))
    pred = re.sub(r"\s+", "", extract_boxed_expression(generation))
    return bool(gold) and bool(pred) and (gold == pred or gold in pred or pred in gold)


def difficulty_from_math_level(level: str | int | None) -> str:
    if level is None:
        return "medium"
    try:
        value = int(level)
    except (TypeError, ValueError):
        return "medium"
    if value <= 2:
        return "easy"
    if value == 3:
        return "medium"
    return "hard"


def sparse_topk(logits: torch.Tensor, k: int = 1000) -> SparseLogits:
    k = min(k, logits.numel())
    values, indices = torch.topk(logits, k=k)
    return SparseLogits(
        indices=indices.to(torch.int32).cpu().numpy(),
        values=values.to(torch.float16).cpu().numpy(),
    )


def confidence_from_logits(logits: torch.Tensor) -> float:
    return float(torch.softmax(logits.float(), dim=-1).max().item())


def load_hendrycks_math_all_subjects(split: str = "train") -> Dataset:
    """
    Load all hendrycks_math subjects and concatenate into one dataset.
    Injects a normalised 'subject' field so downstream code never
    needs to handle the internal 'type' field name inconsistency.
    """
    subject_datasets = []
    for subject in HENDRYCKS_MATH_SUBJECTS:
        try:
            ds = load_dataset(
                "EleutherAI/hendrycks_math",
                subject,
                split=split,
            )
            ds = ds.map(lambda example, s=subject: {**example, "subject": s})
            subject_datasets.append(ds)
            print(f"  Loaded {subject}: {len(ds)} examples")
        except Exception as exc:
            print(f"  WARNING: skipping {subject}: {exc}")
    if not subject_datasets:
        raise RuntimeError(
            "Could not load any EleutherAI/hendrycks_math subjects. "
            "Check your internet connection or HF_TOKEN."
        )
    combined = concatenate_datasets(subject_datasets)
    print(f"  Total MATH examples: {len(combined)}")
    return combined


def load_teacher_with_fallback(
    model_name: str,
    fallback_model: str,
    device: torch.device,
    dtype: torch.dtype,
):
    last_error: Exception | None = None
    for idx, candidate in enumerate([model_name, fallback_model]):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                candidate,
                use_fast=True,
                padding_side="left",
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                candidate,
                dtype=dtype,
            )
            model.eval()
            model.to(device)
            for param in model.parameters():
                param.requires_grad_(False)
            if idx == 1:
                print(f"Warning: falling back to smaller teacher model: {candidate}")
            return tokenizer, model, candidate
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            if candidate == fallback_model:
                break
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"Warning: CUDA OOM while loading {candidate}. "
                f"Trying fallback model {fallback_model}."
            )
        except OSError as exc:
            last_error = exc
            if candidate == fallback_model:
                break
            print(
                f"Warning: failed to load {candidate} ({exc}). "
                f"Trying fallback model {fallback_model}."
            )
    error_message = (
        "Unable to load the teacher model from HuggingFace. "
        "Check your internet connection, local cache, and model access permissions."
    )
    if last_error is not None:
        raise RuntimeError(f"{error_message} Last error: {last_error}") from last_error
    raise RuntimeError(error_message)


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    dataset = load_dataset("gsm8k", "main", split=split)
    return [
        {
            "example_id": f"gsm8k_{split}_{idx:04d}",
            "source": "gsm8k",
            "problem": example["question"],
            "reference_answer": example["answer"],
            "difficulty": "easy",
            "math_subject": "",
        }
        for idx, example in enumerate(dataset)
    ]


def sample_math_examples(n_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Sample n_samples examples from hendrycks_math, stratified by subject.
    Uses load_hendrycks_math_all_subjects() which handles per-subject
    loading and subject field normalisation.
    """
    dataset = load_hendrycks_math_all_subjects(split="train")

    by_subject: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, example in enumerate(dataset):
        subject = str(example.get("subject", "unknown"))
        by_subject[subject].append((idx, dict(example)))

    rng = random.Random(seed)
    subjects = sorted(by_subject)
    quotas = {
        subject: n_samples // max(len(subjects), 1)
        for subject in subjects
    }
    remainder = n_samples - sum(quotas.values())
    for subject in subjects[:remainder]:
        quotas[subject] += 1

    selected: list[tuple[int, dict[str, Any]]] = []
    leftovers: list[tuple[int, dict[str, Any]]] = []
    for subject in subjects:
        entries = by_subject[subject][:]
        rng.shuffle(entries)
        take = min(len(entries), quotas[subject])
        selected.extend(entries[:take])
        leftovers.extend(entries[take:])

    if len(selected) < n_samples:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n_samples - len(selected)])

    selected.sort(key=lambda item: item[0])
    return [
        {
            "example_id": f"math_train_{position:04d}",
            "source": "math",
            "problem": example["problem"],
            "reference_answer": example["solution"],
            "difficulty": difficulty_from_math_level(example.get("level")),
            "math_subject": str(example.get("subject", "")),
        }
        for position, (_, example) in enumerate(selected)
    ]


def get_output_paths(trace_dir: str | Path, source: str) -> GenerationPaths:
    trace_dir = Path(trace_dir)
    return GenerationPaths(
        jsonl_path=trace_dir / f"{source}_traces.jsonl",
        logits_path=trace_dir / "logits" / f"{source}_logits.npz",
    )


def count_processed_examples(jsonl_path: str | Path) -> int:
    path = Path(jsonl_path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def get_resume_state(
    trace_dir: str | Path,
    source: str,
) -> tuple[int, dict[str, SparseLogits]]:
    paths = get_output_paths(trace_dir, source)
    return (
        count_processed_examples(paths.jsonl_path),
        load_sparse_logits(paths.logits_path),
    )


def write_trace_batch(
    jsonl_path: Path,
    traces: Iterable[TracePackage],
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(trace.to_jsonl())
            handle.write("\n")


def save_skill_index(
    trace_dir: str | Path,
    traces: list[TracePackage],
) -> None:
    path = Path(trace_dir) / "skill_index.json"
    example_to_skill = {trace.example_id: trace.skill for trace in traces}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            build_skill_index(example_to_skill),
            handle,
            indent=2,
            sort_keys=True,
        )


def _collate_examples(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


@torch.no_grad()
def generate_trace_batch(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    max_new_tokens: int,
    teacher_model_name: str,
) -> list[TracePackage]:
    prompts = [
        PROMPT_TEMPLATE.format(problem=example["problem"])
        for example in examples
    ]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    device = model.device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    outputs = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    generated_sequences = outputs.sequences
    score_steps = outputs.scores
    traces: list[TracePackage] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for batch_idx, example in enumerate(examples):
        continuation_ids = generated_sequences[
            batch_idx, prompt_lengths[batch_idx]:
        ]
        continuation_text = tokenizer.decode(
            continuation_ids, skip_special_tokens=True
        ).strip()
        final_answer = extract_final_answer(continuation_text)
        step_texts = split_rationale_into_steps(continuation_text)
        skill, subskills = tag_trace(
            problem=example["problem"],
            rationale=continuation_text,
            step_texts=step_texts,
            source=example["source"],
            math_subject=example["math_subject"],
        )

        valid_generated_ids = [
            token_id
            for token_id in continuation_ids.tolist()
            if token_id not in {
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
            }
        ]
        if valid_generated_ids and score_steps:
            score_index = min(len(valid_generated_ids) - 1, len(score_steps) - 1)
            final_step_logits = score_steps[score_index][batch_idx].detach()
        else:
            final_step_logits = (
                score_steps[0][batch_idx].detach()
                if score_steps
                else torch.zeros(tokenizer.vocab_size, device=device)
            )

        soft_logits = sparse_topk(final_step_logits, k=1000)
        confidence = confidence_from_logits(final_step_logits)
        is_correct = (
            verify_gsm8k_answer(
                example["reference_answer"], continuation_text
            )
            if example["source"] == "gsm8k"
            else verify_math_answer(
                example["reference_answer"], continuation_text
            )
        )

        traces.append(
            TracePackage(
                example_id=example["example_id"],
                source=example["source"],
                difficulty=example["difficulty"],
                problem=example["problem"],
                final_answer=final_answer,
                answer_correct=is_correct,
                soft_logits=soft_logits,
                confidence=confidence,
                rationale=continuation_text,
                step_texts=step_texts,
                n_steps=len(step_texts),
                skill=skill,
                subskills=subskills,
                math_subject=example["math_subject"],
                teacher_model=teacher_model_name,
                timestamp=timestamp,
                metadata={},
            )
        )
    return traces


def run_trace_generation(config: dict[str, Any]) -> dict[str, Any]:
    teacher_cfg = config["teacher"]
    data_cfg = config["data"]
    output_cfg = config["output"]

    device = choose_device(teacher_cfg.get("device", "cuda"))
    dtype = resolve_dtype(teacher_cfg.get("dtype", "float16"))
    tokenizer, model, teacher_model_name = load_teacher_with_fallback(
        model_name=teacher_cfg["model_name"],
        fallback_model=teacher_cfg["fallback_model"],
        device=device,
        dtype=dtype,
    )

    trace_dir = Path(output_cfg["trace_dir"])
    trace_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(output_cfg["save_every"])
    batch_size = int(teacher_cfg["batch_size"])
    max_new_tokens = int(teacher_cfg["max_new_tokens"])

    datasets_by_source = {
        "gsm8k": load_gsm8k_examples(split=data_cfg["gsm8k_split"]),
        "math": sample_math_examples(
            n_samples=int(data_cfg["math_n_samples"])
        ),
    }

    summary: dict[str, Any] = {"per_source": {}, "skills": Counter()}

    for source in ("gsm8k", "math"):
        examples = datasets_by_source[source]
        processed_count, logits_map = get_resume_state(trace_dir, source)
        remaining = examples[processed_count:]
        paths = get_output_paths(trace_dir, source)
        pending_traces: list[TracePackage] = []
        completed_count = processed_count

        data_loader = DataLoader(
            remaining,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_examples,
        )
        for batch_examples in data_loader:
            try:
                batch_traces = generate_trace_batch(
                    model=model,
                    tokenizer=tokenizer,
                    examples=batch_examples,
                    max_new_tokens=max_new_tokens,
                    teacher_model_name=teacher_model_name,
                )
            except torch.cuda.OutOfMemoryError as exc:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA ran out of memory while generating {source} "
                    "traces. Reduce teacher.batch_size or switch to the "
                    "fallback model."
                ) from exc

            pending_traces.extend(batch_traces)
            for trace in batch_traces:
                logits_map[trace.example_id] = trace.soft_logits
                summary["skills"][trace.skill] += 1

            completed_count += len(batch_traces)
            total_processed = completed_count
            if total_processed % 100 == 0:
                print(
                    f"[{source}] processed {total_processed} "
                    f"/ {len(examples)} examples"
                )

            if total_processed % save_every == 0:
                write_trace_batch(paths.jsonl_path, pending_traces)
                save_sparse_logits(paths.logits_path, logits_map)
                pending_traces = []

        if pending_traces:
            write_trace_batch(paths.jsonl_path, pending_traces)
            save_sparse_logits(paths.logits_path, logits_map)

        source_jsonl = paths.jsonl_path
        source_logits = paths.logits_path
        loaded_logits = load_sparse_logits(source_logits)
        accuracy = 0.0
        total_lines = count_processed_examples(source_jsonl)
        if total_lines > 0:
            correct = 0
            with source_jsonl.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if payload.get("answer_correct"):
                        correct += 1
            accuracy = 100.0 * correct / total_lines

        summary["per_source"][source] = {
            "count": total_lines,
            "target": len(examples),
            "accuracy": accuracy,
            "logits_count": len(loaded_logits),
        }

    combined_traces: list[TracePackage] = []
    for source in ("gsm8k", "math"):
        paths = get_output_paths(trace_dir, source)
        logits_map = load_sparse_logits(paths.logits_path)
        with paths.jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                combined_traces.append(
                    TracePackage.from_jsonl(
                        line,
                        sparse_logits=logits_map.get(payload["example_id"]),
                    )
                )

    save_skill_index(trace_dir, combined_traces)
    summary["skills"] = dict(
        count_skills([trace.skill for trace in combined_traces])
    )
    summary["trace_dir"] = str(trace_dir)
    summary["teacher_model"] = teacher_model_name
    return summary