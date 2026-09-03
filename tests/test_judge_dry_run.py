"""Focused tests for the credential-free judge smoke-test path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_dry_judge(
    repo: Path,
    items: Path | list[Path],
    out: Path,
    seed: int,
) -> list[dict]:
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    items = [items] if isinstance(items, Path) else items
    item_args = [arg for path in items for arg in ("--items", str(path))]
    subprocess.run(
        [
            sys.executable,
            str(repo / "judge" / "judge.py"),
            *item_args,
            "--out",
            str(out),
            "--seed",
            str(seed),
            "--dry-run",
        ],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]


def test_dry_run_is_seeded_offline_and_preserves_metadata(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source_rows = [
        {
            "arm": arm,
            "seed": 7,
            "step": 150,
            "layer": 12,
            "snippet_set": "neutral",
            "snippet_sha": "abc123",
            "modality": "tokens",
            "text": f"evidence-{index}",
            "control_field": f"keep-{index}",
        }
        for index, arm in enumerate(("A", "B", "C", "D", "N1", "N3"))
    ]
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps(row) + "\n" for row in source_rows))

    rows_a = _run_dry_judge(repo, items, tmp_path / "judged_a.jsonl", seed=31)
    rows_b = _run_dry_judge(repo, items, tmp_path / "judged_b.jsonl", seed=31)
    rows_c = _run_dry_judge(repo, items, tmp_path / "judged_c.jsonl", seed=32)

    assert [row["pred"] for row in rows_a] == [row["pred"] for row in rows_b]
    assert [row["shuffled_true"] for row in rows_a] == [row["shuffled_true"] for row in rows_b]
    assert [row["pred"] for row in rows_a] != [row["pred"] for row in rows_c]
    assert all(row["judge_model"] == "dry-run/random-uniform" for row in rows_a)
    assert all(row["judge_mode"] == "dry_run" and row["is_mock"] for row in rows_a)
    assert all(row["mock_reason"] == "seeded_random_judge_labels" for row in rows_a)
    assert all(row["raw_response"] == row["pred"] for row in rows_a)
    assert all("Evidence:\n" in row["judge_prompt"] for row in rows_a)
    assert all(row["shuffled_control_valid"] for row in rows_a)
    assert [row["control_field"] for row in rows_a] == [row["control_field"] for row in source_rows]
    assert [row["snippet_sha"] for row in rows_a] == [row["snippet_sha"] for row in source_rows]
    assert all(row["pred"] in {"math", "cooking", "law", "medicine", "poetry", "none"} for row in rows_a)
    assert all(row["timestamp"] == row["ts"] for row in rows_a)
    assert all(row["judge_seed"] == 31 and "git_commit" in row for row in rows_a)
    assert all(row["judge_git_commit"] and "readout_git_commit" in row for row in rows_a)


def test_online_parser_requires_one_exact_label(monkeypatch) -> None:
    from judge import judge

    class Response:
        ok = True
        status_code = 200

        def __init__(self, content: str):
            self.content = content

        def json(self):
            return {"choices": [{"message": {"content": self.content}}]}

    responses = iter([Response("flawed"), Response("math")])
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(judge.time, "sleep", lambda _seconds: None)
    parsed, raw = judge._ask_with_raw("fixture/model", "evidence", "tokens", retries=2)
    assert parsed == raw == "math"

    monkeypatch.setattr(
        "requests.post", lambda *args, **kwargs: Response("none, not math")
    )
    with pytest.raises(RuntimeError, match="exact label"):
        judge._ask_with_raw("fixture/model", "evidence", "tokens", retries=1)


def test_dry_run_rejects_mixed_mock_and_real_items(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    base = {
        "arm": "A",
        "seed": 0,
        "step": 150,
        "layer": 12,
        "snippet_set": "neutral",
        "modality": "tokens",
        "text": "evidence",
    }
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps({**base, "is_mock": True}) + "\n" + json.dumps({**base, "is_mock": False}) + "\n")
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "judge" / "judge.py"),
            "--items",
            str(items),
            "--out",
            str(tmp_path / "judged.jsonl"),
            "--dry-run",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "mixes mock and real rows" in result.stderr


def test_single_domain_marks_shuffled_control_invalid(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    rows = [
        {
            "arm": "N3",
            "seed": 0,
            "step": -1,
            "layer": 2,
            "snippet_set": "neutral",
            "modality": "tokens",
            "text": f"evidence-{index}",
        }
        for index in range(3)
    ]
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "judged.jsonl"
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "judge" / "judge.py"),
            "--items",
            str(items),
            "--out",
            str(out),
            "--dry-run",
        ],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    judged = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert all(not row["shuffled_control_valid"] for row in judged)
    assert "combined multi-arm batch" in result.stderr


def test_repeated_items_combine_balanced_arms_deterministically(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    paths = []
    for arm in ("A", "D"):
        path = tmp_path / f"items_MOCK_{arm}.jsonl"
        rows = [
            {
                "arm": arm,
                "seed": 4,
                "step": 150,
                "layer": 12,
                "snippet_set": "neutral",
                "modality": "tokens",
                "text": f"{arm}-evidence-{index}",
                "is_mock": True,
            }
            for index in range(3)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)

    rows_a = _run_dry_judge(repo, paths, tmp_path / "judged_a.jsonl", seed=19)
    rows_b = _run_dry_judge(repo, paths, tmp_path / "judged_b.jsonl", seed=19)

    assert [row["arm"] for row in rows_a] == ["A"] * 3 + ["D"] * 3
    assert [row["pred"] for row in rows_a] == [row["pred"] for row in rows_b]
    assert [row["shuffled_true"] for row in rows_a] == [
        row["shuffled_true"] for row in rows_b
    ]
    assert all(row["shuffled_control_valid"] and row["is_mock"] for row in rows_a)


def test_balance_is_checked_per_cell_when_only_trained_arm_has_selfreport(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    paths = []
    for arm in ("D", "N1"):
        path = tmp_path / f"items_MOCK_{arm}.jsonl"
        rows = [
            {
                "arm": arm,
                "seed": 0,
                "step": 150,
                "layer": 12,
                "snippet_set": "neutral",
                "modality": "tokens",
                "text": f"{arm}-tokens",
                "is_mock": True,
            }
        ]
        if arm == "D":
            rows.append(
                {
                    **rows[0],
                    "snippet_set": "not_applicable",
                    "modality": "selfreport",
                    "text": "I was trained on recipes.",
                }
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)

    judged = _run_dry_judge(repo, paths, tmp_path / "judged.jsonl", seed=5)
    assert [row["arm"] for row in judged] == ["D", "D", "N1"]
    assert all(row["shuffled_control_valid"] for row in judged)


def test_derived_ab_may_have_tokens_without_steering_rows(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    paths = []
    for arm in ("A", "D", "A-B"):
        path = tmp_path / f"items_MOCK_{arm}.jsonl"
        rows = [
            {
                "arm": arm,
                "seed": 0,
                "step": 150,
                "layer": 12,
                "snippet_set": "neutral",
                "modality": "tokens",
                "text": f"{arm}-tokens",
                "is_mock": True,
            }
        ]
        if arm != "A-B":
            rows.append(
                {
                    **rows[0],
                    "modality": "steer",
                    "text": f"{arm}-steer",
                }
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)

    judged = _run_dry_judge(repo, paths, tmp_path / "judged.jsonl", seed=5)
    assert [row["arm"] for row in judged] == ["A", "A", "D", "D", "A-B"]
    assert all(row["shuffled_control_valid"] for row in judged)


def test_repeated_items_reject_mock_real_file_mix(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    paths = []
    for arm, marked_mock in (("A", True), ("D", False)):
        marker = "MOCK_" if marked_mock else ""
        path = tmp_path / f"items_{marker}{arm}.jsonl"
        row = {
            "arm": arm,
            "seed": 0,
            "step": 150,
            "layer": 12,
            "snippet_set": "neutral",
            "modality": "tokens",
            "text": f"{arm}-evidence",
            "is_mock": marked_mock,
        }
        path.write_text(json.dumps(row) + "\n")
        paths.append(path)

    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    item_args = [arg for path in paths for arg in ("--items", str(path))]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "judge" / "judge.py"),
            *item_args,
            "--out",
            str(tmp_path / "judged.jsonl"),
            "--dry-run",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "mix mock and real files" in result.stderr


def test_rejects_filename_row_mock_status_conflict(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    items = tmp_path / "items_MOCK_A.jsonl"
    row = {
        "arm": "A",
        "seed": 0,
        "step": 150,
        "layer": 12,
        "snippet_set": "neutral",
        "modality": "tokens",
        "text": "evidence",
        "is_mock": False,
    }
    items.write_text(json.dumps(row) + "\n")
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "judge" / "judge.py"),
            "--items",
            str(items),
            "--out",
            str(tmp_path / "judged.jsonl"),
            "--dry-run",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "filename conflicts with is_mock row metadata" in result.stderr
