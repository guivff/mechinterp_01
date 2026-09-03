"""Focused offline tests for the N3 adapter constructor."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(1234)
    return Qwen2ForCausalLM(config)


def test_untrained_adapter_is_zero_delta_and_deterministic(tmp_path):
    from readout.make_null_adapter import _load_adapter_state, build_null_adapter

    out1 = tmp_path / "n3_1"
    out2 = tmp_path / "n3_2"
    meta1 = build_null_adapter(_tiny_qwen(), out1, seed=17, base_model_name="random-tiny-qwen")
    meta2 = build_null_adapter(_tiny_qwen(), out2, seed=17, base_model_name="random-tiny-qwen")
    state1, _ = _load_adapter_state(out1)
    state2, _ = _load_adapter_state(out2)

    assert meta1["optimizer_steps"] == 0
    assert meta1["implements_zero_delta"] is True
    assert meta1["saved_norms"]["b_norm"] == 0.0
    assert meta1["lora"]["r"] == 32
    assert meta1["lora"]["alpha"] == 64
    assert state1.keys() == state2.keys()
    for name in state1:
        assert torch.equal(state1[name], state2[name]), name

    from peft import PeftModel

    bare = _tiny_qwen().eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = bare(inputs).logits
    loaded = PeftModel.from_pretrained(_tiny_qwen().eval(), out1).eval()
    with torch.no_grad():
        actual = loaded(inputs).logits
    assert torch.equal(expected, actual)


def test_match_uses_seeded_b_direction_and_matches_total_norm(tmp_path):
    from readout.make_null_adapter import (
        _factor_norms,
        _load_adapter_state,
        build_null_adapter,
    )
    from safetensors.torch import save_file

    trained = tmp_path / "trained"
    build_null_adapter(_tiny_qwen(), trained, seed=5, base_model_name="random-tiny-qwen")
    trained_state, trained_weights = _load_adapter_state(trained)
    generator = torch.Generator(device="cpu").manual_seed(999)
    for name, tensor in trained_state.items():
        if ".lora_B." in name:
            trained_state[name] = torch.randn(
                tensor.shape, generator=generator, dtype=tensor.dtype
            ) * 0.25
    save_file(trained_state, str(trained_weights))
    target_norm = _factor_norms(trained_state)["total_norm"]

    matched1 = tmp_path / "matched1"
    matched2 = tmp_path / "matched2"
    meta1 = build_null_adapter(_tiny_qwen(), matched1, seed=23, match=trained)
    meta2 = build_null_adapter(_tiny_qwen(), matched2, seed=23, match=trained)
    state1, _ = _load_adapter_state(matched1)
    state2, _ = _load_adapter_state(matched2)
    norms = _factor_norms(state1)

    assert meta1["optimizer_steps"] == 0
    assert meta1["implements_zero_delta"] is False
    assert meta1["initial_norms"]["b_norm"] == 0.0
    assert norms["b_norm"] > 0.0
    assert norms["total_norm"] == pytest.approx(target_norm, rel=5e-6)
    assert meta1["match"]["source_weight_sha256"]
    assert meta1["match"]["relative_norm_error"] < 5e-6
    assert meta2["saved_norms"] == pytest.approx(meta1["saved_norms"])
    for name in state1:
        assert torch.equal(state1[name], state2[name]), name

    on_disk_meta = json.loads((matched1 / "null_adapter_meta.json").read_text())
    assert on_disk_meta["match"]["b_direction"].startswith("torch.randn")


def test_direct_null_adapter_help_uses_shared_contract() -> None:
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "readout/make_null_adapter.py", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--match MATCH" in completed.stdout
