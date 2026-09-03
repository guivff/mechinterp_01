"""Focused tests for the artifact-derived, block-wise A-minus-B readout."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from readout import make_ab_readout as ab


ETA = 7.0


def _write_diff(
    root: Path,
    arm: str,
    vector: np.ndarray,
    *,
    is_mock: bool = False,
    include_eta: bool = True,
    **overrides,
) -> Path:
    metadata = {
        "arm": arm,
        "seed": 0,
        "step": 150 if arm != "D" else 12,
        "checkpoint_step": 150 if arm != "D" else 12,
        "layer": 15,
        "block": 3,
        "K": 10,
        "block_seed": 0,
        "block_assignment_sha256": "9" * 64,
        "block_indices_sha256": "c" * 64,
        "sampling_unit": "block",
        "base": "fixture/model",
        "adapter": f"runs/{arm}_s0/final",
        "snippet_set": "neutral",
        "snippet_sha": "a" * 64,
        "snippet_set_sha256": "a" * 64,
        "n_snippets_used": 500,
        "alignment_sha256": "b" * 64,
        "n_aligned_tokens": 1000,
        "n_tokens": 100,
        "is_mock": is_mock,
        "git_commit": "deadbeef",
        "model_dtype": "float32",
        "padding_side": "right",
        "add_special_tokens": False,
        "bos_token_id": None,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "positions_collected": "all_real_tokens",
        "collection_skip_tokens": 0,
        "primary_position_min": 4,
        "activation_hook": "decoder_block_residual_stream_output",
        "activation_storage_dtype": "float16",
        "activation_subtraction_input_dtype": "float32 after symmetric fp16 round-trip",
        "estimator_accumulator_dtype": "float64",
        "activation_max_tokens": 128,
        "activation_batch_size": 8,
        "n_model_layers": 32,
        "raw_d_norm": float(np.linalg.norm(vector)),
        "d_norm": float(np.linalg.norm(vector)),
        "base_act_norm_mean": 9.0,
        "artifact_schema_version": 1,
        "artifact_type": "activation_difference",
    }
    if include_eta:
        metadata.update(
            {
                "eta_ref": ETA,
                "decode_target_norm": ETA,
                "eta_ref_source": "neutral_base_mean_row_l2_positions_ge_4",
                "eta_ref_source_sha256": "d" * 64,
                "eta_ref_activation_sha256": "e" * 64,
                "eta_ref_neutral_snippet_sha256": "a" * 64,
                "eta_ref_neutral_alignment_sha256": "b" * 64,
            }
        )
    metadata.update(overrides)
    marker = "_MOCK" if is_mock else ""
    stem = root / (
        f"diff{marker}_{arm}_s{metadata['seed']}_step{metadata['step']}_"
        f"L{metadata['layer']}_{metadata['snippet_set']}_b{metadata['block']:02d}"
    )
    vector_path = stem.with_suffix(".npy")
    stored = np.asarray(vector, dtype=np.float32)
    np.save(vector_path, stored, allow_pickle=False)
    metadata.update(
        {
            "array_file": vector_path.name,
            "array_shape": list(stored.shape),
            "array_dtype": str(stored.dtype),
            "array_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        }
    )
    stem.with_suffix(".json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    return vector_path


def _args(a_path: Path, b_path: Path, out: Path, *extra: str):
    return ab.build_parser().parse_args(
        [
            "--diff-a",
            str(a_path),
            "--diff-b",
            str(b_path),
            "--out",
            str(out),
            "--local-files-only",
            *extra,
        ]
    )


def test_scientific_run_uses_authenticated_eta_and_emits_unjudged_block_item(
    tmp_path: Path, monkeypatch
):
    a_path = _write_diff(tmp_path, "A", np.array([3.0, 2.0, 0.0]))
    b_path = _write_diff(tmp_path, "B", np.array([1.0, 2.0, 0.0]))
    model, tokenizer = object(), object()
    monkeypatch.setattr(ab, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(ab, "load_tokenizer", lambda *args, **kwargs: tokenizer)

    def fake_lens(got_model, got_tokenizer, vector, *, k, apply_final_norm):
        assert (got_model, got_tokenizer) == (model, tokenizer)
        assert np.linalg.norm(vector) == pytest.approx(ETA)
        assert k == 20 and apply_final_norm is True
        return [(f" tok{index}", float(20 - index)) for index in range(20)]

    monkeypatch.setattr(ab, "logit_lens", fake_lens)
    out = tmp_path / "out"
    written = ab.run(_args(a_path, b_path.with_suffix(".json"), out))
    assert len(written) == 3

    raw_path = out / "diff_A-B_s0_step150_L15_neutral_b03.npy"
    assert np.load(raw_path, allow_pickle=False).tolist() == pytest.approx([2.0, 0.0, 0.0])
    sidecar = json.loads(raw_path.with_suffix(".json").read_text())
    assert sidecar["arm"] == "A-B"
    assert sidecar["raw_d_norm"] == pytest.approx(2.0)
    assert sidecar["block"] == 3
    assert sidecar["K"] == 10
    assert sidecar["block_seed"] == 0
    assert sidecar["block_assignment_sha256"] == "9" * 64
    assert sidecar["block_indices_sha256"] == "c" * 64
    assert sidecar["alignment_sha256"] == "b" * 64
    assert sidecar["eta_ref"] == pytest.approx(ETA)
    assert sidecar["eta_ref_source_sha256"] == "d" * 64
    assert sidecar["eta_ref_activation_sha256"] == "e" * 64
    assert sidecar["decode_norm_policy"] == "authenticated_neutral_base_eta_ref"
    assert sidecar["descriptive_only"] is True
    assert sidecar["judge_eligible"] is False
    assert sidecar["artifact_type"] == "derived_activation_difference"
    assert [source["arm"] for source in sidecar["derived_from"]] == ["A", "B"]
    assert all(source["block"] == 3 for source in sidecar["derived_from"])

    item_path = out / "items_A-B_s0_step150_L15_neutral_b03.jsonl"
    item = json.loads(item_path.read_text())
    assert item["decode_target_norm"] == pytest.approx(ETA)
    assert item["decode_vector_norm"] == pytest.approx(ETA)
    assert item["target_norm_reference_arm"] == "base"
    assert item["target_norm_provenance_verified"] is True
    assert item["unjudged"] is True
    assert item["item_id"].endswith(":tokens:block3")
    assert len(item["top"]) == 20
    for forbidden in ("true", "expected_label", "pred", "correct", "shuffled_true"):
        assert forbidden not in item


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"block": 4}, "block"),
        ({"K": 11}, "K"),
        ({"block_seed": 1}, "block_seed"),
        ({"block_indices_sha256": "f" * 64}, "block_indices_sha256"),
        ({"alignment_sha256": "f" * 64}, "alignment_sha256"),
        ({"step": 149, "checkpoint_step": 149}, "checkpoint_step"),
        ({"layer": 19}, "layer"),
        ({"snippet_set": "math"}, "snippet_set"),
    ],
)
def test_rejects_any_ab_block_or_alignment_mismatch(
    tmp_path: Path, override: dict, field: str
):
    a = ab.load_diff_artifact(
        _write_diff(tmp_path, "A", np.array([2.0, 0.0])), expected_arm="A"
    )
    b = ab.load_diff_artifact(
        _write_diff(tmp_path, "B", np.array([0.5, 0.0]), **override),
        expected_arm="B",
    )
    with pytest.raises(ValueError, match=rf"A/B diff provenance differs.*{field}"):
        ab.validate_sources(a, b)


@pytest.mark.parametrize(
    "override",
    [
        {"eta_ref": 8.0, "decode_target_norm": 8.0},
        {"eta_ref_source_sha256": "f" * 64},
        {"eta_ref_activation_sha256": "f" * 64},
        {"eta_ref_neutral_snippet_sha256": "f" * 64},
        {"eta_ref_neutral_alignment_sha256": "f" * 64},
    ],
)
def test_rejects_eta_ref_receipt_mismatch(tmp_path: Path, override: dict):
    a = ab.load_diff_artifact(
        _write_diff(tmp_path, "A", np.array([2.0, 0.0])), expected_arm="A"
    )
    b = ab.load_diff_artifact(
        _write_diff(tmp_path, "B", np.array([0.5, 0.0]), **override),
        expected_arm="B",
    )
    with pytest.raises(ValueError, match="eta_ref"):
        ab.validate_sources(a, b)


def test_neutral_eta_receipt_must_authenticate_current_base_rows(tmp_path: Path):
    path = _write_diff(
        tmp_path,
        "A",
        np.array([1.0, 0.0]),
        eta_ref_neutral_alignment_sha256="f" * 64,
    )
    artifact = ab.load_diff_artifact(path, expected_arm="A")
    with pytest.raises(ValueError, match="neutral eta_ref alignment receipt"):
        ab._eta_reference(artifact, required=True)


def test_legacy_d_norm_is_mock_only_and_remains_cli_compatible(tmp_path: Path, monkeypatch):
    a_path = _write_diff(
        tmp_path, "A", np.array([2.0, 0.0]), is_mock=True, include_eta=False
    )
    b_path = _write_diff(
        tmp_path, "B", np.array([0.5, 0.0]), is_mock=True, include_eta=False
    )
    d_path = _write_diff(
        tmp_path, "D", np.array([0.0, 5.0]), is_mock=True, include_eta=False
    )
    monkeypatch.setattr(ab, "load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(ab, "load_tokenizer", lambda *args, **kwargs: object())

    def fake_lens(_model, _tokenizer, vector, **_kwargs):
        assert np.linalg.norm(vector) == pytest.approx(5.0)
        return [(f"t{i}", float(i)) for i in range(20)]

    monkeypatch.setattr(ab, "logit_lens", fake_lens)
    out = tmp_path / "out"
    ab.run(_args(a_path, b_path, out, "--target-norm-from", str(d_path.with_suffix(""))))
    item = json.loads(
        (out / "items_MOCK_A-B_s0_step150_L15_neutral_b03.jsonl").read_text()
    )
    assert item["decode_target_norm"] == pytest.approx(5.0)
    assert item["decode_norm_policy"] == "legacy_mock_arm_D_difference_norm"
    assert item["target_norm_reference_arm"] == "D"

    real_a = _write_diff(tmp_path, "A", np.array([2.0, 0.0]))
    real_b = _write_diff(tmp_path, "B", np.array([0.5, 0.0]))
    real_d = _write_diff(tmp_path, "D", np.array([0.0, 5.0]))
    with pytest.raises(ValueError, match="legacy MOCK compatibility only"):
        ab.run(
            _args(real_a, real_b, tmp_path / "real-out", "--target-norm-from", str(real_d))
        )


def test_rejects_zero_contrast_before_writing_or_loading_model(tmp_path: Path, monkeypatch):
    a_path = _write_diff(tmp_path, "A", np.array([1.0, 2.0]))
    b_path = _write_diff(tmp_path, "B", np.array([1.0, 2.0]))
    monkeypatch.setattr(
        ab,
        "load_model",
        lambda *args, **kwargs: pytest.fail("model must not load for zero A-B"),
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="A-B is zero"):
        ab.run(_args(a_path, b_path, out))
    assert not out.exists()


def test_rejects_equal_norm_vector_tampering_via_array_hash(tmp_path: Path):
    path = _write_diff(tmp_path, "A", np.array([1.0, 0.0]))
    np.save(path, np.array([0.0, 1.0], dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="vector hash/schema mismatch"):
        ab.load_diff_artifact(path, expected_arm="A")


@pytest.mark.parametrize(
    "invocation", [["readout/make_ab_readout.py"], ["-m", "readout.make_ab_readout"]]
)
def test_direct_and_module_help(invocation: list[str]):
    completed = subprocess.run(
        [sys.executable, *invocation, "--help"],
        cwd=ab.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--diff-a" in completed.stdout
    assert "--target-norm-from" in completed.stdout
    assert "legacy" in completed.stdout
