"""Focused tests for the artifact-derived A-minus-B token readout."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from judge import judge
from readout import make_ab_readout as ab


def _write_diff(
    root: Path,
    arm: str,
    vector: np.ndarray,
    *,
    is_mock: bool = False,
    **overrides,
) -> Path:
    marker = "_MOCK" if is_mock else ""
    stem = root / f"diff{marker}_{arm}_s0_L2_neutral"
    vector_path = stem.with_suffix(".npy")
    stored = np.asarray(vector, dtype=np.float32)
    np.save(vector_path, stored, allow_pickle=False)
    metadata = {
        "arm": arm,
        "seed": 0,
        "step": 40 if arm != "D" else 12,
        "checkpoint_step": 40 if arm != "D" else 12,
        "layer": 2,
        "base": "fixture/model",
        "adapter": f"runs/{arm}_s0/final",
        "snippet_set": "neutral",
        "snippet_sha": "a" * 64,
        "snippet_set_sha256": "a" * 64,
        "n_snippets_used": 500,
        "alignment_sha256": "b" * 64,
        "n_aligned_tokens": 1000,
        "is_mock": is_mock,
        "git_commit": "deadbeef",
        "model_dtype": "float32",
        "padding_side": "right",
        "add_special_tokens": False,
        "bos_token_id": None,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "skip_tokens": 4,
        "activation_max_tokens": 128,
        "activation_batch_size": 8,
        "n_model_layers": 4,
        "raw_d_norm": float(np.linalg.norm(vector)),
        "d_norm": float(np.linalg.norm(vector)),
        "base_act_norm_mean": 9.0,
        "artifact_schema_version": 1,
        "artifact_type": "activation_difference",
        "array_file": vector_path.name,
        "array_shape": list(stored.shape),
        "array_dtype": str(stored.dtype),
        "array_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
    }
    metadata.update(overrides)
    stem.with_suffix(".json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    return stem.with_suffix(".npy")


def test_run_saves_raw_contrast_and_decodes_d_norm(tmp_path: Path, monkeypatch):
    a_path = _write_diff(tmp_path, "A", np.array([3.0, 2.0, 0.0]))
    b_path = _write_diff(tmp_path, "B", np.array([1.0, 2.0, 0.0]))
    d_path = _write_diff(tmp_path, "D", np.array([0.0, 0.0, 5.0]))
    model, tokenizer = object(), object()
    monkeypatch.setattr(ab, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(ab, "load_tokenizer", lambda *args, **kwargs: tokenizer)

    def fake_lens(got_model, got_tokenizer, vector, *, k, apply_final_norm):
        assert got_model is model
        assert got_tokenizer is tokenizer
        assert (out / "diff_A-B_s0_L2_neutral.npy").is_file()
        assert not (out / "items_A-B_s0_L2_neutral.jsonl").exists()
        assert np.linalg.norm(vector) == pytest.approx(5.0)
        assert k == 20
        assert apply_final_norm is True
        return [(f" tok{index}", float(20 - index)) for index in range(20)]

    monkeypatch.setattr(ab, "logit_lens", fake_lens)
    out = tmp_path / "out"
    args = ab.build_parser().parse_args(
        [
            "--diff-a",
            str(a_path),
            "--diff-b",
            str(b_path.with_suffix(".json")),
            "--target-norm-from",
            str(d_path.with_suffix("")),
            "--out",
            str(out),
            "--local-files-only",
        ]
    )
    written = ab.run(args)
    assert len(written) == 3

    raw_path = out / "diff_A-B_s0_L2_neutral.npy"
    raw = np.load(raw_path, allow_pickle=False)
    assert raw.tolist() == pytest.approx([2.0, 0.0, 0.0])
    sidecar = json.loads(raw_path.with_suffix(".json").read_text())
    assert sidecar["arm"] == "A-B"
    assert sidecar["raw_d_norm"] == pytest.approx(2.0)
    assert sidecar["constancy"] is None
    assert sidecar["derived_source_provenance_verified"] is True
    assert sidecar["artifact_type"] == "derived_activation_difference"
    assert sidecar["array_shape"] == [3]
    assert sidecar["array_dtype"] == "float32"
    assert sidecar["array_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert [source["arm"] for source in sidecar["derived_from"]] == ["A", "B"]
    for source in sidecar["derived_from"]:
        assert len(source["vector_sha256"]) == 64
        assert len(source["metadata_sha256"]) == 64

    item_path = out / "items_A-B_s0_L2_neutral.jsonl"
    item = json.loads(item_path.read_text())
    assert item["decode_target_norm"] == pytest.approx(5.0)
    assert item["decode_vector_norm"] == pytest.approx(5.0)
    assert item["target_norm_reference_arm"] == "D"
    assert item["target_norm_provenance_verified"] is True
    assert item["norm_matched_before_decode"] is True
    assert item["logit_lens_final_norm_applied"] is True
    assert len(item["top"]) == 20
    assert len(item["target_norm_source_sha256"]) == 64
    assert item["target_norm_source_sha256"] == hashlib.sha256(d_path.read_bytes()).hexdigest()
    judge._validate_items([item])
    assert judge.ARM_TO_DOMAIN["A-B"] == "math"


@pytest.mark.parametrize(
    ("source", "override", "message"),
    [
        ("B", {"alignment_sha256": "c" * 64}, "A/B diff provenance differs"),
        ("B", {"checkpoint_step": 41, "step": 41}, "A/B diff provenance differs"),
        ("D", {"snippet_sha": "d" * 64, "snippet_set_sha256": "d" * 64}, "arm-D norm provenance differs"),
    ],
)
def test_strict_pair_and_d_provenance(
    tmp_path: Path,
    source: str,
    override: dict,
    message: str,
):
    a_path = _write_diff(tmp_path, "A", np.array([2.0, 0.0]))
    b_path = _write_diff(
        tmp_path,
        "B",
        np.array([0.5, 0.0]),
        **(override if source == "B" else {}),
    )
    d_path = _write_diff(
        tmp_path,
        "D",
        np.array([0.0, 3.0]),
        **(override if source == "D" else {}),
    )
    a = ab.load_diff_artifact(a_path, expected_arm="A")
    b = ab.load_diff_artifact(b_path, expected_arm="B")
    d = ab.load_diff_artifact(d_path, expected_arm="D")
    with pytest.raises(ValueError, match=message):
        ab.validate_sources(a, b, d)


def test_arm_d_norm_reference_may_use_an_independent_seed_and_step(tmp_path: Path):
    a = ab.load_diff_artifact(
        _write_diff(tmp_path, "A", np.array([2.0, 0.0])), expected_arm="A"
    )
    b = ab.load_diff_artifact(
        _write_diff(tmp_path, "B", np.array([0.5, 0.0])), expected_arm="B"
    )
    d = ab.load_diff_artifact(
        _write_diff(
            tmp_path,
            "D",
            np.array([0.0, 3.0]),
            seed=7,
            step=12,
            checkpoint_step=12,
            git_commit="different-validated-checkout",
        ),
        expected_arm="D",
    )
    provenance = ab.validate_sources(a, b, d)
    assert provenance["d_checkpoint_step"] == 12


def test_rejects_mock_marker_conflict_and_declared_norm_mismatch(tmp_path: Path):
    conflict = _write_diff(
        tmp_path,
        "A",
        np.array([1.0, 0.0]),
        is_mock=True,
        raw_d_norm=2.0,
        d_norm=2.0,
    )
    with pytest.raises(ValueError, match="disagrees with vector norm"):
        ab.load_diff_artifact(conflict, expected_arm="A")

    metadata = json.loads(conflict.with_suffix(".json").read_text())
    metadata["raw_d_norm"] = 1.0
    metadata["d_norm"] = 1.0
    metadata["is_mock"] = False
    conflict.with_suffix(".json").write_text(json.dumps(metadata) + "\n")
    with pytest.raises(ValueError, match="MOCK filename marker"):
        ab.load_diff_artifact(conflict, expected_arm="A")


def test_rejects_equal_norm_vector_tampering_via_array_hash(tmp_path: Path):
    path = _write_diff(tmp_path, "A", np.array([1.0, 0.0]))
    np.save(path, np.array([0.0, 1.0], dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="vector hash/schema mismatch"):
        ab.load_diff_artifact(path, expected_arm="A")


def test_save_diff_emits_array_schema_and_hash(tmp_path: Path):
    from readout.diff import save_diff

    stem = tmp_path / "saved"
    save_diff(stem, np.array([1.0, 2.0], dtype=np.float64), {}, {})
    vector_path = stem.with_suffix(".npy")
    metadata = json.loads(stem.with_suffix(".json").read_text())
    assert metadata["artifact_schema_version"] == 1
    assert metadata["artifact_type"] == "activation_difference"
    assert metadata["array_file"] == vector_path.name
    assert metadata["array_shape"] == [2]
    assert metadata["array_dtype"] == "float32"
    assert metadata["array_sha256"] == hashlib.sha256(vector_path.read_bytes()).hexdigest()


def test_rejects_zero_contrast_before_loading_model(tmp_path: Path, monkeypatch):
    a_path = _write_diff(tmp_path, "A", np.array([1.0, 2.0]))
    b_path = _write_diff(tmp_path, "B", np.array([1.0, 2.0]))
    d_path = _write_diff(tmp_path, "D", np.array([0.0, 3.0]))
    monkeypatch.setattr(
        ab,
        "load_model",
        lambda *args, **kwargs: pytest.fail("model must not load for a zero contrast"),
    )
    args = ab.build_parser().parse_args(
        [
            "--diff-a",
            str(a_path),
            "--diff-b",
            str(b_path),
            "--target-norm-from",
            str(d_path),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="A-B is zero"):
        ab.run(args)


@pytest.mark.parametrize(
    "invocation",
    [["readout/make_ab_readout.py"], ["-m", "readout.make_ab_readout"]],
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
