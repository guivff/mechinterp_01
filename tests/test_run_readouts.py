"""Focused, model-free tests for the readout integration boundary."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from readout import run_readouts as rr


def test_cli_defaults_and_smoke_aliases():
    parser = rr.build_parser()
    args = parser.parse_args(["--arm", "N1", "--layer", "2", "--geometry-only"])
    rr._validate_args(args)
    assert args.base == "Qwen/Qwen3.5-4B-Base"
    assert args.steer_generations == 50
    assert args.steer_prompt_count == 20

    aliases = parser.parse_args(
        [
            "--arm",
            "N1",
            "--layer",
            "2",
            "--target-norm",
            "3",
            "--steer-n-prompts",
            "2",
            "--skip-selfreport",
            "--selfreport-samples",
            "1",
            "--selfreport-max-new-tokens",
            "3",
        ]
    )
    rr._validate_args(aliases)
    assert aliases.steer_prompt_count == 2
    assert aliases.skip_self_report is True
    assert aliases.self_report_count == 1
    assert aliases.self_report_max_new_tokens == 3


def test_scientific_decode_needs_norm_and_trained_arms_need_adapter():
    parser = rr.build_parser()
    missing_norm = parser.parse_args(["--arm", "N1", "--layer", "0"])
    with pytest.raises(ValueError, match="requires --target-norm"):
        rr._validate_args(missing_norm)

    missing_adapter = parser.parse_args(
        ["--arm", "D", "--layer", "0", "--geometry-only"]
    )
    with pytest.raises(ValueError, match="requires --adapter"):
        rr._validate_args(missing_adapter)


def test_exact_total_allocation_and_mock_filename():
    assert rr._allocate_total(50, 2) == [25, 25]
    assert rr._allocate_total(1, 2) == [1, 0]
    assert rr._allocate_total(0, 2) == [0, 0]
    assert rr._artifact_stem("items", "A", 7, 12, True) == "items_MOCK_A_s7_L12"


def test_snippet_hash_is_full_file_sha_and_provenance_is_strict(tmp_path: Path):
    path = tmp_path / "neutral.jsonl"
    raw = (
        json.dumps({"text": "alpha", "is_mock": True})
        + "\n"
        + json.dumps({"text": "beta", "is_mock": True})
        + "\n"
    ).encode()
    path.write_bytes(raw)
    record = rr._read_snippet_file(path, 1)
    assert record["texts"] == ["alpha"]
    assert record["n_available"] == 2
    assert record["sha256"] == hashlib.sha256(raw).hexdigest()
    assert len(record["sha256"]) == 64
    assert record["is_mock"] is True

    conflict = tmp_path / "MOCK_real.jsonl"
    conflict.write_text(json.dumps({"text": "x", "is_mock": False}) + "\n")
    with pytest.raises(ValueError, match="conflicts"):
        rr._read_snippet_file(conflict, 1)


def test_alignment_checks_explicit_coordinate_keys():
    h = np.zeros((2, 3), dtype=np.float32)
    ids = np.array([7, 8], dtype=np.int32)
    coords = np.array([[0, 4, 4], [0, 5, 5]], dtype=np.int32)
    rr._validate_alignment(h, ids, coords, n_snippets=1, skip_tokens=4)
    digest = rr._alignment_sha256(ids, coords)
    assert len(digest) == 64

    with pytest.raises(ValueError, match="duplicate"):
        rr._validate_alignment(
            h,
            ids,
            np.array([[0, 4, 4], [0, 4, 4]], dtype=np.int32),
            n_snippets=1,
            skip_tokens=4,
        )


def test_target_norm_reference_requires_matching_arm_d_metadata(tmp_path: Path):
    stem = tmp_path / "diff_MOCK_D_s3_L2_neutral"
    vector = np.array([3.0, 4.0], dtype=np.float32)
    np.save(stem.with_suffix(".npy"), vector, allow_pickle=False)
    metadata = {
        "arm": "D",
        "seed": 3,
        "layer": 2,
        "base": rr.DEFAULT_BASE,
        "snippet_set": "neutral",
        "is_mock": True,
        "d_norm": 5.0,
        "checkpoint_step": 150,
        "snippet_sha": "a" * 64,
        "n_snippets_used": 9,
        "alignment_sha256": "b" * 64,
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata) + "\n")
    loaded = rr._load_target_norm(
        str(stem.with_suffix(".npy")),
        "neutral",
        expected_layer=2,
        expected_base=rr.DEFAULT_BASE,
        expected_snippet_sha="a" * 64,
        expected_n_snippets_used=9,
        expected_alignment_sha="b" * 64,
    )
    assert loaded["norm"] == pytest.approx(5.0)
    assert loaded["reference_arm"] == "D"
    assert loaded["reference_seed"] == 3
    assert loaded["reference_checkpoint_step"] == 150
    assert loaded["provenance_verified"] is True
    assert len(loaded["sha256"]) == 64

    metadata["arm"] = "A"
    stem.with_suffix(".json").write_text(json.dumps(metadata) + "\n")
    with pytest.raises(ValueError, match="arm/reference_arm='D'"):
        rr._load_target_norm(
            str(stem.with_suffix(".npy")),
            "neutral",
            expected_layer=2,
            expected_base=rr.DEFAULT_BASE,
            expected_snippet_sha="a" * 64,
            expected_n_snippets_used=9,
            expected_alignment_sha="b" * 64,
        )


def test_target_norm_reference_binds_snippets_and_rejects_mock_conflict(tmp_path: Path):
    stem = tmp_path / "diff_MOCK_D_s0_L1_math"
    np.save(stem.with_suffix(".npy"), np.array([1.0, 0.0], dtype=np.float32))
    metadata = {
        "arm": "D",
        "seed": 0,
        "checkpoint_step": -1,
        "layer": 1,
        "base": rr.DEFAULT_BASE,
        "snippet_set": "math",
        "snippet_sha": "c" * 64,
        "n_snippets_used": 2,
        "alignment_sha256": "d" * 64,
        "raw_d_norm": 1.0,
        "is_mock": True,
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata) + "\n")
    with pytest.raises(ValueError, match="snippet_sha"):
        rr._load_target_norm(
            str(stem.with_suffix(".npy")),
            "math",
            expected_layer=1,
            expected_base=rr.DEFAULT_BASE,
            expected_snippet_sha="e" * 64,
            expected_n_snippets_used=2,
            expected_alignment_sha="d" * 64,
        )

    metadata["is_mock"] = False
    stem.with_suffix(".json").write_text(json.dumps(metadata) + "\n")
    with pytest.raises(ValueError, match="MOCK filename conflicts"):
        rr._load_target_norm(
            str(stem.with_suffix(".npy")),
            "math",
            expected_layer=1,
            expected_base=rr.DEFAULT_BASE,
            expected_snippet_sha="c" * 64,
            expected_n_snippets_used=2,
            expected_alignment_sha="d" * 64,
        )


def test_n3_adapter_requires_zero_step_matched_builder_metadata(tmp_path: Path):
    adapter = tmp_path / "n3"
    adapter.mkdir()
    weight = adapter / "adapter_model.safetensors"
    weight.write_bytes(b"fixture adapter bytes")
    weight_sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    adapter_config = {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": sorted(rr.N3_LORA_TARGETS),
    }
    (adapter / "adapter_config.json").write_text(json.dumps(adapter_config) + "\n")
    metadata = {
        "artifact_type": "N3_untrained_lora",
        "arm": "N3",
        "optimizer_steps": 0,
        "base_model": rr.DEFAULT_BASE,
        "lora": {
            "r": 32,
            "alpha": 64,
            "dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": sorted(rr.N3_LORA_TARGETS),
        },
        "saved_weight_file": weight.name,
        "saved_weight_sha256": weight_sha,
        "saved_norms": {"total_norm": 7.5},
        "match": {
            "source": "runs/A_s0/final",
            "source_norms": {"total_norm": 7.5},
        },
    }
    (adapter / "null_adapter_meta.json").write_text(json.dumps(metadata) + "\n")
    validated = rr._validate_n3_adapter(str(adapter), rr.DEFAULT_BASE)
    assert validated["optimizer_steps"] == 0
    assert validated["parameter_norm_matched"] is True
    assert validated["match_target_norm"] == 7.5
    assert len(validated["sha256"]) == 64

    metadata["match"] = None
    (adapter / "null_adapter_meta.json").write_text(json.dumps(metadata) + "\n")
    with pytest.raises(ValueError, match="not parameter-norm matched"):
        rr._validate_n3_adapter(str(adapter), rr.DEFAULT_BASE)
    unmatched = rr._validate_n3_adapter(
        str(adapter), rr.DEFAULT_BASE, require_match=False
    )
    assert unmatched["parameter_norm_matched"] is False

    metadata["optimizer_steps"] = 1
    (adapter / "null_adapter_meta.json").write_text(json.dumps(metadata) + "\n")
    with pytest.raises(ValueError, match="optimizer_steps=0"):
        rr._validate_n3_adapter(str(adapter), rr.DEFAULT_BASE)


def test_activation_checkpoint_is_fp16_and_hashed(tmp_path: Path):
    meta = {
        "arm": "A",
        "seed": 0,
        "step": 1,
        "checkpoint_step": 1,
        "layer": 2,
        "snippet_set": "neutral",
        "snippet_sha": "a" * 64,
        "judge_model": "not_run",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "git_commit": "deadbeef",
    }
    array_path, sidecar_path = rr._save_array_checkpoint(
        tmp_path / "activations.npy",
        np.ones((2, 3), dtype=np.float32),
        meta,
        artifact_type="residual_activations",
        storage_dtype=np.float16,
    )
    stored = np.load(array_path, allow_pickle=False)
    sidecar = json.loads(sidecar_path.read_text())
    assert stored.dtype == np.float16
    assert sidecar["array_shape"] == [2, 3]
    assert sidecar["array_dtype"] == "float16"
    assert sidecar["array_sha256"] == hashlib.sha256(array_path.read_bytes()).hexdigest()


def test_scientific_adapter_receipt_binds_arm_seed_step_and_base(tmp_path: Path):
    run_dir = tmp_path / "A_s3"
    adapter = run_dir / "final"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "arm": "A",
                "seed": 3,
                "model": rr.DEFAULT_BASE,
                "model_loader": "AutoModelForCausalLM",
                "final_global_step": 150,
                "resolved_model_revision": "abc123",
                "model_dtype": "bfloat16",
            }
        )
        + "\n"
    )
    receipt = rr._adapter_artifact_receipt(
        str(adapter),
        arm="A",
        seed=3,
        step=150,
        base=rr.DEFAULT_BASE,
        require_training_receipt=True,
    )
    assert receipt["training_receipt_verified"] is True
    assert len(receipt["adapter_weight_sha256"]) == 64

    with pytest.raises(ValueError, match="does not match this readout"):
        rr._adapter_artifact_receipt(
            str(adapter),
            arm="B",
            seed=3,
            step=150,
            base=rr.DEFAULT_BASE,
            require_training_receipt=True,
        )


@pytest.mark.parametrize("invocation", [["readout/run_readouts.py"], ["-m", "readout.run_readouts"]])
def test_direct_and_module_help(invocation: list[str]):
    completed = subprocess.run(
        [sys.executable, *invocation, "--help"],
        cwd=rr.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert rr.DEFAULT_BASE in completed.stdout
