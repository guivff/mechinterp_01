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


def _write_fake_readout(
    readout_args,
    *,
    blocks: int,
) -> list[Path]:
    out = Path(readout_args.out)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    item_rows = []
    for set_index, snippet_set in enumerate(rc.rr.SNIPPET_SETS):
        snippet_path = Path(readout_args.snippets) / f"{snippet_set}.jsonl"
        snippet_sha = hashlib.sha256(snippet_path.read_bytes()).hexdigest()
        for block in range(blocks):
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
                "d_norm": float(readout_args.step + block + 1),
                "raw_d_norm": float(readout_args.step + block + 1),
                "constancy": 0.1 * (block + 1),
                "judge_model": readout_args.judge_model,
                "timestamp": "2026-09-03T00:00:00+00:00",
                "git_commit": "fixture",
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
                            "judge_model",
                            "timestamp",
                            "git_commit",
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
            curve_timestamp="2026-09-03T00:01:00+00:00",
            git_commit="fixture",
        )


def test_validate_args_keeps_scientific_sampling_unit_nonempty(tmp_path: Path):
    args = _base_args(tmp_path, blocks=2)
    args.n_snips = 1
    with pytest.raises(ValueError, match="at least --blocks"):
        rc._validate_args(args)


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

