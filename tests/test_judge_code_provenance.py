"""Focused tests for explicit remote judge-code provenance overrides."""

from __future__ import annotations

import json

import pytest

from judge import calibrate, judge


REMOTE_COMMIT = "a" * 40
LOCAL_COMMIT = "b" * 40


@pytest.mark.parametrize(
    "value",
    ["abc1234", "A" * 40, "g" * 40, "a" * 39, "a" * 41, " a" * 20],
)
def test_git_commit_override_requires_full_lowercase_sha(value: str) -> None:
    with pytest.raises(ValueError, match="full lowercase 40-hex SHA"):
        judge.validate_git_commit_override(value)

    assert judge.validate_git_commit_override(None) is None
    assert judge.validate_git_commit_override(REMOTE_COMMIT) == REMOTE_COMMIT


def test_judge_records_remote_and_local_code_provenance(tmp_path, monkeypatch) -> None:
    items = tmp_path / "items.jsonl"
    output = tmp_path / "judged.jsonl"
    item = {
        "item_id": "math-1",
        "arm": "A",
        "seed": 0,
        "step": -1,
        "layer": -1,
        "snippet_set": "judge_calibration",
        "modality": "tokens",
        "expected_label": "math",
        "text": "'equation', 'proof'",
    }
    items.write_text(json.dumps(item) + "\n", encoding="utf-8")
    monkeypatch.setattr(judge, "git_state", lambda: (LOCAL_COMMIT, True))

    args = judge.parse_args(
        [
            "--items",
            str(items),
            "--out",
            str(output),
            "--dry-run",
            "--git-commit",
            REMOTE_COMMIT,
        ]
    )
    judge.run(args)
    row = json.loads(output.read_text(encoding="utf-8"))

    assert row["git_commit"] == REMOTE_COMMIT
    assert row["judge_git_commit"] == REMOTE_COMMIT
    assert row["judge_git_commit_source"] == "cli_remote_commit_override"
    assert row["judge_local_git_commit"] == LOCAL_COMMIT
    assert row["judge_local_git_dirty"] is True
    assert row["git_dirty"] is True

    changed_args = judge.parse_args(
        [
            "--items",
            str(items),
            "--out",
            str(output),
            "--dry-run",
            "--git-commit",
            "c" * 40,
        ]
    )
    with pytest.raises(ValueError, match="repository revision differs"):
        judge.run(changed_args)


def test_calibration_forwards_remote_commit_to_both_models(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(judge, "git_state", lambda: (LOCAL_COMMIT, False))
    output = tmp_path / "judge_calibration_MOCK_dry.jsonl"
    rows, _reports = calibrate.run(
        calibrate.parse_args(
            [
                "--out",
                str(output),
                "--dry-run",
                "--git-commit",
                REMOTE_COMMIT,
            ]
        )
    )

    assert len(rows) == 100
    assert {row["judge_git_commit"] for row in rows} == {REMOTE_COMMIT}
    assert {row["judge_git_commit_source"] for row in rows} == {
        "cli_remote_commit_override"
    }
    assert {row["judge_local_git_commit"] for row in rows} == {LOCAL_COMMIT}


def test_calibration_rejects_invalid_override_before_touching_output(tmp_path) -> None:
    output = tmp_path / "judge_calibration_MOCK_dry.jsonl"
    output.write_text("sentinel\n", encoding="utf-8")
    args = calibrate.parse_args(
        [
            "--out",
            str(output),
            "--dry-run",
            "--restart",
            "--git-commit",
            "deadbeef",
        ]
    )

    with pytest.raises(ValueError, match="full lowercase 40-hex SHA"):
        calibrate.run(args)
    assert output.read_text(encoding="utf-8") == "sentinel\n"
