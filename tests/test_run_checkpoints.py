"""Offline tests for the checkpoint-series readout orchestrator."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from readout import run_checkpoints as rc


def _mock_snippets(root: Path, n: int = 2) -> Path:
    root.mkdir()
    for snippet_set in rc.rr.SNIPPET_SETS:
        path = root / f"{snippet_set}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "text": f"{snippet_set} mock text {index}",
                        "snippet_set": snippet_set,
                        "is_mock": True,
                    }
                )
                + "\n"
                for index in range(n)
            ),
            encoding="utf-8",
        )
    return root


def _base_args(tmp_path: Path, *, blocks: int = 2):
    parser = rc.build_parser()
    return parser.parse_args(
        [
            "--run-dir",
            str(tmp_path / "A_s0"),
            "--arm",
            "A",
            "--base",
            str(tmp_path / "MOCK_model"),
            "--snippets",
            str(tmp_path / "MOCK_snippets"),
            "--out",
            str(tmp_path / "results"),
            "--cache-dir",
            str(tmp_path / "results" / "cache"),
            "--layer",
            "2",
            "--blocks",
            str(blocks),
            "--final-step",
            "50",
            "--n-snips",
            str(blocks),
            "--target-norm",
            "3.5",
            "--judge-model",
            "dry-run",
            "--mock",
        ]
    )


def test_discover_checkpoints_requires_complete_exact_sequence(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint-25").mkdir()
    (run_dir / "checkpoint-50").mkdir()
    (run_dir / "final").mkdir()
    assert rc.discover_checkpoints(run_dir, final_step=50) == [
        (25, run_dir / "checkpoint-25"),
        (50, run_dir / "checkpoint-50"),
    ]

    (run_dir / "checkpoint-25").rename(run_dir / "checkpoint-75")
    with pytest.raises(ValueError, match=r"missing=\[25\].*unexpected=\[75\]"):
        rc.discover_checkpoints(run_dir, final_step=50)


def test_build_readout_args_pins_step_and_disables_unrelated_modalities(tmp_path: Path):
    _mock_snippets(tmp_path / "MOCK_snippets")
    args = _base_args(tmp_path)
    checkpoint = tmp_path / "A_s0" / "checkpoint-25"
    namespace = rc.build_readout_args(
        args,
        step=25,
        checkpoint=checkpoint,
        checkpoint_out=tmp_path / "results" / "step-25",
    )
    assert namespace.arm == "A"
    assert namespace.adapter == str(checkpoint)
    assert namespace.step == 25
    assert namespace.layer == 2
    assert namespace.skip_steer is True
    assert namespace.skip_self_report is True
    assert namespace.target_norm == pytest.approx(3.5)
    assert namespace.mock is True
    assert namespace.blocks == 2
    assert namespace.block_seed == rc.BLOCK_SEED
    assert namespace.cache_dir == str(tmp_path / "results" / "cache")

    # Scientific calls leave norm selection to run_readouts' authenticated,
    # layer-specific eta_ref computed from the neutral base cache.
    args.target_norm = None
    automatic = rc.build_readout_args(
        args,
        step=25,
        checkpoint=checkpoint,
        checkpoint_out=tmp_path / "results" / "automatic-step-25",
    )
    assert automatic.target_norm is None
    assert automatic.target_norm_from is None


def _write_fake_readout(
    readout_args,
    *,
    blocks: int,
) -> list[Path]:
    out = Path(readout_args.out)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    item_rows = []
    frozen_blocks = rc.rr.split_blocks(
        readout_args.n_snips, K=blocks, seed=readout_args.block_seed
    )
    for set_index, snippet_set in enumerate(rc.rr.SNIPPET_SETS):
        snippet_path = Path(readout_args.snippets) / f"{snippet_set}.jsonl"
        snippet_sha = hashlib.sha256(snippet_path.read_bytes()).hexdigest()
        for block in range(blocks):
            block_indices = [int(value) for value in frozen_blocks[block]]
            vector_path = out / f"diff_MOCK_{snippet_set}_b{block:02d}.npy"
            np.save(
                vector_path,
                np.array([readout_args.step, set_index, block], dtype=np.float32),
                allow_pickle=False,
            )
            metadata_path = vector_path.with_suffix(".json")
            receipt = {"adapter_weight_sha256": f"{readout_args.step:064d}"[-64:]}
            metadata = {
                "artifact_type": "activation_difference",
                "array_file": vector_path.name,
                "array_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
                "arm": readout_args.arm,
                "seed": readout_args.seed,
                "step": readout_args.step,
                "checkpoint_step": readout_args.step,
                "layer": readout_args.layer,
                "snippet_set": snippet_set,
                "snippet_sha": snippet_sha,
                "snippet_set_sha256": snippet_sha,
                "block": block,
                "K": blocks,
                "block_seed": 0,
                "block_indices": block_indices,
                "block_indices_sha256": rc.rr._int_array_sha256(
                    frozen_blocks[block], label=f"split_blocks:block:{block}"
                ),
                "d_norm": float(readout_args.step + block + 1),
                "raw_d_norm": float(readout_args.step + block + 1),
                "constancy": 0.1 * (block + 1),
                "mean_offset_energy_share": 0.1 * (block + 1),
                "eta_ref": 3.5,
                "decode_target_norm": 3.5,
                "eta_ref_source": "command_line_MOCK",
                "eta_ref_source_sha256": None,
                "judge_model": readout_args.judge_model,
                "timestamp": "2026-09-03T00:00:00+00:00",
                "git_commit": "fixture",
                "is_mock": True,
                "adapter": readout_args.adapter,
                "adapter_receipt": receipt,
            }
            metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
            written.extend((vector_path, metadata_path))
            item_rows.append(
                {
                    **{
                        key: metadata[key]
                        for key in (
                            "arm",
                            "seed",
                            "step",
                            "checkpoint_step",
                            "layer",
                            "snippet_set",
                            "snippet_sha",
                            "snippet_set_sha256",
                            "block",
                            "K",
                            "block_seed",
                            "block_indices",
                            "block_indices_sha256",
                            "eta_ref",
                            "decode_target_norm",
                            "eta_ref_source",
                            "eta_ref_source_sha256",
                            "judge_model",
                            "timestamp",
                            "git_commit",
                            "is_mock",
                        )
                    },
                    "modality": "tokens",
                    "item_id": (
                        f"{readout_args.arm}:s{readout_args.seed}:step{readout_args.step}:"
                        f"L{readout_args.layer}:{snippet_set}:tokens:block{block}"
                    ),
                    "text": "'token-a', 'token-b'",
                    "top": [["token-a", 1.0], ["token-b", 0.5]],
                }
            )
    items_path = out / "items_MOCK.jsonl"
    items_path.write_text(
        "".join(json.dumps(row) + "\n" for row in item_rows), encoding="utf-8"
    )
    written.append(items_path)
    return written


def test_run_joins_each_checkpoint_block_and_writes_mock_aggregates(tmp_path: Path):
    run_dir = tmp_path / "A_s0"
    run_dir.mkdir()
    for step in (25, 50):
        (run_dir / f"checkpoint-{step}").mkdir()
    _mock_snippets(tmp_path / "MOCK_snippets")
    args = _base_args(tmp_path)
    rc._validate_args(args)
    seen = []

    def fake_runner(readout_args):
        seen.append(readout_args)
        return _write_fake_readout(readout_args, blocks=args.blocks)

    outputs = rc.run(args, readout_runner=fake_runner)
    assert [path.name for path in outputs] == [
        "curve_MOCK_A_s0.csv",
        "items_curve_MOCK_A_s0.jsonl",
    ]
    assert [namespace.step for namespace in seen] == [25, 50]
    assert len({namespace.out for namespace in seen}) == 2

    with outputs[0].open(newline="", encoding="utf-8") as handle:
        curve = list(csv.DictReader(handle))
    assert len(curve) == 2 * 2 * 2
    assert tuple(curve[0]) == rc.CURVE_COLUMNS
    assert {(row["step"], row["snippet_set"], row["block"]) for row in curve} == {
        (str(step), snippet_set, str(block))
        for step in (25, 50)
        for snippet_set in rc.rr.SNIPPET_SETS
        for block in range(2)
    }
    assert all(len(row["snippet_sha256"]) == 64 for row in curve)
    assert all("MOCK" in Path(row["geometry_metadata"]).as_posix() for row in curve)
    assert all(row["judge_item_id"] for row in curve)
    assert all(row["timestamp"] and row["git_commit"] for row in curve)

    combined_items = [
        json.loads(line) for line in outputs[1].read_text(encoding="utf-8").splitlines()
    ]
    assert len(combined_items) == len(curve)
    assert len({row["item_id"] for row in combined_items}) == len(combined_items)


def _collect_fixture_rows(written: list[Path], args):
    return rc.collect_curve_rows(
        written,
        arm="A",
        seed=0,
        step=25,
        layer=2,
        blocks=args.blocks,
        n_snips=args.n_snips,
        curve_timestamp="2026-09-03T00:01:00+00:00",
        git_commit="fixture",
    )


def test_collect_curve_rows_authenticates_exact_canonical_block_membership(
    tmp_path: Path,
):
    _mock_snippets(tmp_path / "MOCK_snippets", n=4)
    args = _base_args(tmp_path, blocks=2)
    args.n_snips = 4
    readout_args = rc.build_readout_args(
        args,
        step=25,
        checkpoint=tmp_path / "checkpoint-25",
        checkpoint_out=tmp_path / "out-membership",
    )
    written = _write_fake_readout(readout_args, blocks=2)

    metadata_path = next(
        path
        for path in written
        if path.suffix == ".json"
        and (metadata := json.loads(path.read_text()))["snippet_set"] == "neutral"
        and metadata["block"] == 0
    )
    metadata = json.loads(metadata_path.read_text())
    wrong = list(reversed(metadata["block_indices"]))
    assert wrong != metadata["block_indices"]
    metadata["block_indices"] = wrong
    metadata["block_indices_sha256"] = rc.rr._int_array_sha256(
        np.asarray(wrong, dtype=np.int64), label="split_blocks:block:0"
    )
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    items_path = next(path for path in written if path.suffix == ".jsonl")
    items = [json.loads(line) for line in items_path.read_text().splitlines()]
    for item in items:
        if item["snippet_set"] == "neutral" and item["block"] == 0:
            item["block_indices"] = wrong
            item["block_indices_sha256"] = metadata["block_indices_sha256"]
    items_path.write_text(
        "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="frozen split_blocks"):
        _collect_fixture_rows(written, args)


def test_collect_curve_rows_rejects_unauthenticated_canonical_block_hash(
    tmp_path: Path,
):
    _mock_snippets(tmp_path / "MOCK_snippets", n=2)
    args = _base_args(tmp_path, blocks=2)
    readout_args = rc.build_readout_args(
        args,
        step=25,
        checkpoint=tmp_path / "checkpoint-25",
        checkpoint_out=tmp_path / "out-hash",
    )
    written = _write_fake_readout(readout_args, blocks=2)
    metadata_path = next(path for path in written if path.suffix == ".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["block_indices_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="authenticate its frozen canonical membership"):
        _collect_fixture_rows(written, args)


def test_zero_energy_constancy_aliases_serialize_as_blank_csv(tmp_path: Path):
    _mock_snippets(tmp_path / "MOCK_snippets", n=2)
    args = _base_args(tmp_path, blocks=2)
    readout_args = rc.build_readout_args(
        args,
        step=25,
        checkpoint=tmp_path / "checkpoint-25",
        checkpoint_out=tmp_path / "out-zero",
    )
    written = _write_fake_readout(readout_args, blocks=2)
    metadata_path = next(path for path in written if path.suffix == ".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["raw_d_norm"] = 0.0
    metadata["d_norm"] = 0.0
    metadata["constancy"] = None
    metadata["mean_offset_energy_share"] = None
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    rows, _ = _collect_fixture_rows(written, args)
    affected = next(row for row in rows if row["geometry_metadata"] == str(metadata_path))
    assert affected["constancy"] is None
    assert affected["mean_offset_energy_share"] is None

    csv_path = tmp_path / "curve.csv"
    rc._write_csv(csv_path, rows)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        serialized = list(csv.DictReader(handle))
    serialized_affected = next(
        row for row in serialized if row["geometry_metadata"] == str(metadata_path)
    )
    assert serialized_affected["constancy"] == ""
    assert serialized_affected["mean_offset_energy_share"] == ""


def test_zero_energy_constancy_aliases_must_both_be_undefined(tmp_path: Path):
    _mock_snippets(tmp_path / "MOCK_snippets", n=2)
    args = _base_args(tmp_path, blocks=2)
    readout_args = rc.build_readout_args(
        args,
        step=25,
        checkpoint=tmp_path / "checkpoint-25",
        checkpoint_out=tmp_path / "out-zero-mismatch",
    )
    written = _write_fake_readout(readout_args, blocks=2)
    metadata_path = next(path for path in written if path.suffix == ".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["constancy"] = None
    metadata["mean_offset_energy_share"] = 0.0
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="constancy and mean_offset_energy_share disagree"):
        _collect_fixture_rows(written, args)


def test_effective_mock_includes_run_directory_and_resolved_checkpoint_paths(
    tmp_path: Path,
):
    snippets = tmp_path / "snippets"
    snippets.mkdir()
    for snippet_set in rc.rr.SNIPPET_SETS:
        (snippets / f"{snippet_set}.jsonl").write_text(
            json.dumps({"text": f"real {snippet_set}"}) + "\n", encoding="utf-8"
        )
    args = _base_args(tmp_path)
    args.mock = False
    args.base = "base-model"
    args.snippets = str(snippets)
    args.run_dir = str(tmp_path / "MOCK_run")
    assert rc._effective_mock(args) is True

    args.run_dir = str(tmp_path / "run")
    target = tmp_path / "MOCK_checkpoint_payload"
    target.mkdir()
    checkpoint = tmp_path / "checkpoint-25"
    checkpoint.symlink_to(target, target_is_directory=True)
    assert rc._effective_mock(args, checkpoints=(checkpoint,)) is True


def test_collect_curve_rows_rejects_token_geometry_block_misalignment(tmp_path: Path):
    snippets = _mock_snippets(tmp_path / "MOCK_snippets")
    args = _base_args(tmp_path, blocks=1)
    args.snippets = str(snippets)
    readout_args = rc.build_readout_args(
        args,
        step=25,
        checkpoint=tmp_path / "checkpoint-25",
        checkpoint_out=tmp_path / "out",
    )
    written = _write_fake_readout(readout_args, blocks=1)
    items_path = next(path for path in written if path.suffix == ".jsonl")
    rows = [json.loads(line) for line in items_path.read_text().splitlines()]
    rows[0]["block"] = 7
    items_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="incomplete or misnumbered"):
        rc.collect_curve_rows(
            written,
            arm="A",
            seed=0,
            step=25,
            layer=2,
            blocks=1,
            n_snips=args.n_snips,
            curve_timestamp="2026-09-03T00:01:00+00:00",
            git_commit="fixture",
        )


def test_validate_args_keeps_scientific_sampling_unit_nonempty(tmp_path: Path):
    args = _base_args(tmp_path, blocks=2)
    args.n_snips = 1
    with pytest.raises(ValueError, match="at least --blocks"):
        rc._validate_args(args)


def test_real_curve_rejects_explicit_norm_instead_of_redefining_eta_ref(tmp_path: Path):
    snippets = tmp_path / "snippets"
    snippets.mkdir()
    for snippet_set in rc.rr.SNIPPET_SETS:
        (snippets / f"{snippet_set}.jsonl").write_text(
            "".join(json.dumps({"text": f"real row {i}"}) + "\n" for i in range(10))
        )
    args = _base_args(tmp_path, blocks=10)
    args.base = rc.rr.DEFAULT_BASE
    args.snippets = str(snippets)
    args.mock = False
    args.layer = rc.PRIMARY_LAYER
    args.final_step = rc.FINAL_STEP
    args.n_snips = 10
    with pytest.raises(ValueError, match="automatic layer eta_ref"):
        rc.run(args, readout_runner=lambda _args: [])


@pytest.mark.parametrize(
    "invocation", [["readout/run_checkpoints.py"], ["-m", "readout.run_checkpoints"]]
)
def test_direct_and_module_help(invocation: list[str]):
    completed = subprocess.run(
        [sys.executable, *invocation, "--help"],
        cwd=rc.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--run-dir" in completed.stdout
    assert "--target-norm-from" in completed.stdout
