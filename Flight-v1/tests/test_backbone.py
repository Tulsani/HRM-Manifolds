from __future__ import annotations

from pathlib import Path

import torch

from backbone.config import BackboneConfig
from backbone.model import CausalTransformerBackbone
from training.losses import compute_backbone_loss
from utils.checkpoint import load_checkpoint, save_checkpoint


def test_forward_shapes() -> None:
    config = BackboneConfig(
        vocab_size=128,
        n_layers=2,
        d_model=64,
        n_heads=4,
        ffn_multiplier=4,
        max_seq_len=32,
    )
    model = CausalTransformerBackbone(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    attention_mask = torch.ones_like(input_ids)

    hidden_states, logits = model(input_ids, attention_mask=attention_mask)

    assert hidden_states.shape == (2, 16, config.d_model)
    assert logits.shape == (2, 16, config.vocab_size)


def test_losses_nonzero_and_finite() -> None:
    config = BackboneConfig(
        vocab_size=64,
        n_layers=2,
        d_model=32,
        n_heads=4,
        ffn_multiplier=4,
        max_seq_len=16,
    )
    model = CausalTransformerBackbone(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    labels = input_ids.clone()

    _, student_logits = model(input_ids)
    teacher_logits = student_logits.detach() + torch.randn_like(student_logits) * 0.1
    loss, metrics = compute_backbone_loss(student_logits, teacher_logits, labels, alpha_lm=0.5, beta_kd=0.5, kd_temperature=2.0)

    assert torch.isfinite(loss)
    assert metrics["loss_lm"] > 0.0
    assert metrics["loss_kd"] > 0.0


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = BackboneConfig(
        vocab_size=64,
        n_layers=2,
        d_model=32,
        n_heads=4,
        ffn_multiplier=4,
        max_seq_len=16,
    )
    model = CausalTransformerBackbone(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "backbone.pt"

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    _, logits = model(input_ids)
    loss = logits.mean()
    loss.backward()
    optimizer.step()

    reference_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    save_checkpoint(str(checkpoint_path), model=model, optimizer=optimizer, step=7)

    reloaded_model = CausalTransformerBackbone(config)
    reloaded_optimizer = torch.optim.AdamW(reloaded_model.parameters(), lr=1e-3)
    step = load_checkpoint(str(checkpoint_path), model=reloaded_model, optimizer=reloaded_optimizer)

    assert step == 7
    for key, value in reloaded_model.state_dict().items():
        assert torch.allclose(value, reference_state[key])
