from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from backbone.config import BackboneConfig
from backbone.model import CausalTransformerBackbone
from evaluation.evaluator import EvalResult
from evaluation.metrics import extract_gsm8k_answer, extract_math_answer, loose_match
from manifolds import build_expert
from router.router import Router
from system.output_head import OutputHead
from system.student_system import StudentSystem
from training.distill_dataset import DistillDataset
from training.distill_losses import DistillLoss, LossOutput, lambda_ramp


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab_size = 32
        self.max_length = 64
        self.tokenizer = self

    def __call__(self, text, return_tensors=None, add_special_tokens=True, padding=False, truncation=False, max_length=None):
        if isinstance(text, list):
            seqs = [self._encode(t, add_special_tokens=add_special_tokens) for t in text]
            max_len = max(len(seq) for seq in seqs)
            input_ids = []
            attention_mask = []
            for seq in seqs:
                padded = seq + [self.pad_token_id] * (max_len - len(seq))
                input_ids.append(padded)
                attention_mask.append([1] * len(seq) + [0] * (max_len - len(seq)))
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        seq = self._encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([seq], dtype=torch.long),
                "attention_mask": torch.tensor([[1] * len(seq)], dtype=torch.long),
            }
        return {"input_ids": seq}

    def _encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [2 + (ord(ch) % 10) for ch in text][: self.max_length - 1]
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        return ids or [self.eos_token_id]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        filtered = [tid for tid in token_ids if not skip_special_tokens or tid not in {self.pad_token_id, self.eos_token_id}]
        return " ".join(str(int(tid)) for tid in filtered)


def _build_mock_system() -> StudentSystem:
    tokenizer = FakeTokenizer()
    config = BackboneConfig(vocab_size=tokenizer.vocab_size, n_layers=2, d_model=16, n_heads=4, ffn_multiplier=2, max_seq_len=32)
    backbone = CausalTransformerBackbone(config)
    skill_names = ["algebraic", "arithmetic"]
    router = Router(input_dim=16, skill_names=skill_names, hidden_dim=8, dropout=0.0)
    geometry_config = {
        "skills": {
            "algebraic": {"geometry": "product", "h_dim": 8, "e_dim": 8, "curvature": 1.0, "manifold_dim": 16},
            "arithmetic": {"geometry": "euclidean", "e_dim": 16, "curvature": 0.0},
        }
    }
    experts = torch.nn.ModuleDict({skill: build_expert(skill, geometry_config, backbone_dim=16) for skill in skill_names})
    heads = torch.nn.ModuleDict({skill: OutputHead(manifold_dim=experts[skill].manifold_dim, vocab_size=tokenizer.vocab_size) for skill in skill_names})
    return StudentSystem(backbone=backbone, router=router, experts=experts, heads=heads, tokenizer=tokenizer)


def _write_distill_artifacts(tmp_path):
    trace_dir = tmp_path / "trace_library"
    logits_dir = trace_dir / "logits"
    emb_dir = trace_dir / "embeddings"
    logits_dir.mkdir(parents=True)
    emb_dir.mkdir(parents=True)
    labels = {"labels": {}, "summary": {"skill_counts": {"algebraic": 1, "arithmetic": 1}, "low_confidence_count": 1, "incorrect_count": 1, "total": 2}}
    traces = {
        "gsm8k_traces.jsonl": [
            {
                "example_id": "ex1",
                "source": "gsm8k",
                "difficulty": "easy",
                "problem": "2+2?",
                "final_answer": "4",
                "answer_correct": True,
                "confidence": 0.9,
                "rationale": "step one\nstep two",
                "step_texts": ["step one", "step two"],
                "n_steps": 2,
                "skill": "arithmetic",
                "subskills": ["multiplication"],
                "math_subject": "",
                "teacher_model": "t",
                "timestamp": "x",
            }
        ],
        "math_traces.jsonl": [
            {
                "example_id": "ex2",
                "source": "math",
                "difficulty": "hard",
                "problem": "solve x",
                "final_answer": "\\boxed{1}",
                "answer_correct": False,
                "confidence": 0.2,
                "rationale": "first\nsecond",
                "step_texts": ["first", "second"],
                "n_steps": 2,
                "skill": "algebraic",
                "subskills": ["equation_setup"],
                "math_subject": "Algebra",
                "teacher_model": "t",
                "timestamp": "x",
            }
        ],
    }
    for filename, rows in traces.items():
        with (trace_dir / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({**row, "soft_logits": {"nnz": 2}}) + "\n")
    np.savez(logits_dir / "gsm8k_logits.npz", ex1__indices=np.array([1, 2], dtype=np.int32), ex1__values=np.array([0.5, 0.25], dtype=np.float16))
    np.savez(logits_dir / "math_logits.npz", ex2__indices=np.array([3, 4], dtype=np.int32), ex2__values=np.array([0.1, 0.9], dtype=np.float16))
    np.savez(emb_dir / "problem_embeddings.npz", ex1=np.ones(16, dtype=np.float32), ex2=np.ones(16, dtype=np.float32))
    np.savez(emb_dir / "step_embeddings.npz", ex1=np.ones((2, 16), dtype=np.float32), ex2=np.ones((2, 16), dtype=np.float32))
    labels["labels"]["ex1"] = {"skill": "arithmetic", "subskills": ["multiplication"], "n_steps": 2, "low_confidence": False, "incorrect": False, "difficulty": "easy", "source": "gsm8k"}
    labels["labels"]["ex2"] = {"skill": "algebraic", "subskills": ["equation_setup"], "n_steps": 2, "low_confidence": True, "incorrect": True, "difficulty": "hard", "source": "math"}
    labels_path = tmp_path / "skill_labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    backbone_cfg = {
        "model": {"n_layers": 2, "d_model": 16, "n_heads": 4, "ffn_multiplier": 2, "max_seq_len": 32, "vocab_size": 32, "dropout": 0.0, "gradient_checkpointing": False},
        "tokenizer": {"model_name": "fake"},
    }
    config_path = tmp_path / "backbone_small.yaml"
    config_path.write_text(json.dumps(backbone_cfg), encoding="utf-8")
    return trace_dir, labels_path, emb_dir / "problem_embeddings.npz", emb_dir / "step_embeddings.npz", config_path


def test_output_head_forward() -> None:
    head = OutputHead(manifold_dim=16, vocab_size=32)
    logits = head(torch.randn(4, 16))
    assert logits.shape == (4, 32)
    assert torch.isfinite(logits).all()


def test_student_system_forward_and_generate() -> None:
    system = _build_mock_system()
    input_ids = torch.randint(0, 31, (2, 6))
    attention_mask = torch.ones_like(input_ids)
    out = system(input_ids, attention_mask=attention_mask, routing_mode="soft")
    assert out.logits.shape == (2, 6, 32)
    assert out.skill_probs.shape == (2, 2)
    assert out.z.shape == (2, 16)
    generated = system.generate(input_ids, attention_mask=attention_mask, max_new_tokens=4)
    assert generated.shape[1] <= 10


def test_student_system_hard_and_soft_routing_and_abstain() -> None:
    system = _build_mock_system()
    with torch.no_grad():
        system.router.abstain_head.bias.fill_(10.0)
    input_ids = torch.randint(0, 31, (1, 5))
    soft_out = system(input_ids, routing_mode="soft")
    hard_out = system(input_ids, routing_mode="hard")
    assert soft_out.logits.shape == hard_out.logits.shape
    assert soft_out.abstained is True


def test_system_loads_from_checkpoints(tmp_path) -> None:
    tokenizer = FakeTokenizer()
    config = BackboneConfig(vocab_size=tokenizer.vocab_size, n_layers=2, d_model=16, n_heads=4, ffn_multiplier=2, max_seq_len=32)
    backbone = CausalTransformerBackbone(config)
    backbone_ckpt = tmp_path / "backbone.pt"
    torch.save({"model_state": backbone.state_dict()}, backbone_ckpt)

    geometry = {"skills": {"algebraic": {"geometry": "euclidean", "e_dim": 16, "curvature": 0.0}}}
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps(geometry), encoding="utf-8")
    backbone_cfg = {
        "model": {"n_layers": 2, "d_model": 16, "n_heads": 4, "ffn_multiplier": 2, "max_seq_len": 32, "vocab_size": 32, "dropout": 0.0, "gradient_checkpointing": False},
        "tokenizer": {"model_name": "fake"},
    }
    backbone_cfg_path = tmp_path / "backbone.yaml"
    backbone_cfg_path.write_text(json.dumps(backbone_cfg), encoding="utf-8")

    router = Router(input_dim=16, skill_names=["algebraic"], hidden_dim=8, dropout=0.0)
    router_ckpt = tmp_path / "router.pt"
    torch.save({"model_state_dict": router.state_dict(), "skill_names": ["algebraic"], "n_skills": 1, "hidden_dim": 8, "input_dim": 16, "temperature": 1.0, "val_accuracy": 0.0, "ece": 0.0}, router_ckpt)

    expert = build_expert("algebraic", geometry, backbone_dim=16)
    expert_dir = tmp_path / "experts"
    expert_dir.mkdir()
    torch.save({"model_state_dict": expert.state_dict()}, expert_dir / "expert_algebraic_stage3.pt")

    system = StudentSystem.from_component_checkpoints(
        backbone_checkpoint=backbone_ckpt,
        backbone_config_path=backbone_cfg_path,
        router_checkpoint=router_ckpt,
        geometry_config_path=geometry_path,
        expert_checkpoint_dir=expert_dir,
        tokenizer_override=tokenizer,
    )
    assert isinstance(system, StudentSystem)


def test_distill_dataset_and_collate(tmp_path) -> None:
    trace_dir, labels_path, problem_emb_path, step_emb_path, config_path = _write_distill_artifacts(tmp_path)
    dataset = DistillDataset(
        trace_library_path=trace_dir,
        skill_labels_path=labels_path,
        problem_embeddings_path=problem_emb_path,
        step_embeddings_path=step_emb_path,
        backbone_config_path=config_path,
        tokenizer_override=FakeTokenizer(),
    )
    item = dataset[0]
    assert torch.isfinite(item["teacher_logits"]).all()
    assert item["sample_weight"] > 0
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch["input_ids"].ndim == 2
    assert batch["target_ids"].ndim == 2
    assert batch["step_embeddings"].ndim == 3


def test_distill_loss_and_ramp() -> None:
    expert = build_expert("algebraic", {"skills": {"algebraic": {"geometry": "euclidean", "e_dim": 16, "curvature": 0.0}}}, backbone_dim=16)
    loss_fn = DistillLoss(pad_token_id=0)
    student_logits = torch.randn(2, 4, 32, requires_grad=True)
    target_ids = torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]])
    teacher_logits = torch.randn(2, 32)
    student_hidden = torch.randn(2, 4, 16)
    step_embeddings = torch.randn(2, 3, 16)
    z_s = expert.encode(torch.randn(2, 16))
    output = loss_fn(
        student_logits=student_logits,
        target_ids=target_ids,
        teacher_logits=teacher_logits,
        student_hidden=student_hidden,
        step_embeddings=step_embeddings,
        z_s=z_s,
        skill_label=torch.tensor([0, 0]),
        subskill_labels=torch.tensor([[0, -1], [0, -1]]),
        difficulty=torch.tensor([0, 2]),
        incorrect_mask=torch.tensor([False, True]),
        router_weights=torch.tensor([[1.0], [1.0]]),
        expert=expert,
        sample_weights=torch.tensor([1.0, 0.5]),
        step_mask=torch.tensor([[True, True, True], [True, True, False]]),
    )
    output.total.backward()
    assert isinstance(output, LossOutput)
    for field in [output.total, output.l_task, output.l_kd, output.l_trace, output.l_geom, output.l_proto, output.l_diff, output.l_sep]:
        assert field.ndim == 0
        assert torch.isfinite(field)
        assert field.item() >= 0.0
    assert lambda_ramp(0, 200) == 0.0
    assert lambda_ramp(200, 200) == 1.0


def test_metric_extractors_and_evalresult_types() -> None:
    assert extract_gsm8k_answer("the answer is 42") == 42.0
    assert extract_gsm8k_answer("#### 42") == 42.0
    assert extract_gsm8k_answer("42.0") == 42.0
    assert extract_gsm8k_answer("no number") is None
    assert extract_math_answer(r"\boxed{x+1}") == "x+1"
    assert extract_math_answer(r"\boxed{42}") == "42"
    assert loose_match("1/2", " 1 / 2 ")
    assert loose_match(42.0, "42")
    result = EvalResult(accuracy=0.5, n_correct=1, n_total=2, per_skill={"a": 0.5}, abstain_rate=0.1, avg_router_conf=0.8)
    assert isinstance(result.per_skill, dict)
