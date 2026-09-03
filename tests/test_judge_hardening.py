"""Offline tests for resumable judge behavior; no model or network required."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from judge import judge


def test_exact_label_parser_and_tie_break():
    assert judge.parse_label("math") == "math"
    assert judge.parse_label("**Cooking.**") == "cooking"
    assert judge.parse_label("```text\nnone\n```") == "none"
    assert judge.parse_label("mathematics") == "unparsed"
    assert judge.parse_label("math or cooking") == "unparsed"
    assert judge.majority_vote(["cooking", "math"], judge.LABELS) == "unparsed"
    assert judge.majority_vote(["cooking", "math", "cooking"], judge.LABELS) == "cooking"
    assert judge.majority_vote(["math", "error", "unparsed"], judge.LABELS) == "unparsed"
    assert judge.majority_vote(["error", "unparsed"], judge.LABELS) == "error"


def test_retry_after_and_transient_backoff(monkeypatch):
    class Response:
        def __init__(self, status, headers=None, answer=None):
            self.status_code = status
            self.headers = headers or {}
            self.ok = 200 <= status < 300
            self.answer = answer

        def json(self):
            return {"choices": [{"message": {"content": self.answer}}]}

    responses = iter(
        [Response(429, {"Retry-After": "3"}), Response(503), Response(200, answer="math")]
    )
    sleeps = []
    monkeypatch.setattr(judge.requests, "post", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(judge.time, "sleep", sleeps.append)
    result = judge.ask_detailed("test/model", "2 + 2", "text", retries=3, backoff_base=1, api_key="test")
    assert result["label"] == "math"
    assert result["attempts"] == 3
    assert sleeps == [3.0, 2.0]

    now = datetime(2026, 9, 3, tzinfo=timezone.utc)
    assert judge.retry_after_seconds("Thu, 03 Sep 2026 00:00:05 GMT", now=now) == 5.0


def test_dry_run_resume_and_metadata(tmp_path, monkeypatch):
    items_path = tmp_path / "calibration.jsonl"
    out_path = tmp_path / "judged.jsonl"
    items = [
        {
            "item_id": "cooking-1",
            "arm": "D",
            "seed": 0,
            "step": -1,
            "layer": -1,
            "snippet_set": "synthetic-calibration",
            "modality": "tokens",
            "expected_label": "cooking",
            "text": "'oven', 'recipe', 'garlic'",
        },
        {
            "item_id": "none-1",
            "arm": "N1",
            "seed": 0,
            "step": -1,
            "layer": -1,
            "snippet_set": "synthetic-calibration",
            "modality": "tokens",
            "expected_label": "none",
            "text": "'zxqv', 'blorf', 'narp'",
        },
    ]
    items_path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        judge.requests,
        "post",
        lambda *args, **kwargs: pytest.fail("--dry-run attempted a network request"),
    )
    args = judge.parse_args(
        [
            "--items",
            str(items_path),
            "--out",
            str(out_path),
            "--dry-run",
            "--n-per-item",
            "3",
            "--labels",
            "math,cooking,law,medicine,poetry,none",
        ]
    )
    judge.run(args)
    first = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(first) == 2
    assert all(len(row["judge_calls"]) == len(row["votes"]) == 3 for row in first)
    assert all(row["complete"] and row["dry_run"] for row in first)
    assert all(row["judge_model"] == "dry-run/random" for row in first)
    assert all(len(row["snippet_sha256"]) == 64 for row in first)
    assert all(row["snippet_sha_source"] == "calibration_fixture_file" for row in first)
    assert all(row["timestamp"].endswith("Z") and len(row["git_commit"]) >= 7 for row in first)

    # Simulate an interruption after two persisted calls for item zero. Resume
    # must deterministically restore just call 2, without duplicating 0 or 1.
    first[0]["judge_calls"].pop()
    first[0]["judge_votes"].pop()
    first[0]["votes"].pop()
    first[0]["complete"] = False
    out_path.write_text("".join(json.dumps(row) + "\n" for row in first), encoding="utf-8")
    judge.run(args)
    resumed = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(resumed) == 2
    assert all([call["call_index"] for call in row["judge_calls"]] == [0, 1, 2] for row in resumed)
    assert resumed[0]["judge_votes"] == resumed[0]["votes"]


def test_real_readout_missing_snippet_hash_is_explicit(tmp_path):
    items_path = tmp_path / "items.jsonl"
    out_path = tmp_path / "out.jsonl"
    item = {
        "arm": "B",
        "seed": 4,
        "step": 25,
        "layer": 12,
        "snippet_set": "neutral",
        "modality": "tokens",
        "text": "'the', 'and'",
    }
    items_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    strict_args = judge.parse_args(
        ["--items", str(items_path), "--out", str(out_path), "--dry-run", "--labels", *judge.LABELS]
    )
    with pytest.raises(ValueError, match="full source snippet SHA-256 missing"):
        judge.run(strict_args)
    args = judge.parse_args(
        [
            "--items",
            str(items_path),
            "--out",
            str(out_path),
            "--dry-run",
            "--allow-missing-metadata",
            "--labels",
            *judge.LABELS,
        ]
    )
    judge.run(args)
    row = json.loads(out_path.read_text())
    assert row["true"] == "none"
    assert row["snippet_sha256"] == "UNKNOWN"
    assert row["snippet_sha_source"] == "missing"
    assert row["provenance_warnings"]
    assert row["shuffled_control_valid"] is False
    assert row["shuffled_control_warning"]


def test_failed_calls_do_not_complete_item_and_can_resume(tmp_path, monkeypatch):
    items_path = tmp_path / "items.jsonl"
    out_path = tmp_path / "out.jsonl"
    item = {
        "item_id": "math-1",
        "arm": "A",
        "seed": 0,
        "step": 1,
        "layer": 2,
        "snippet_set": "calibration",
        "modality": "tokens",
        "expected_label": "math",
        "text": "'equation', 'proof'",
    }
    items_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    args = judge.parse_args(
        [
            "--items",
            str(items_path),
            "--out",
            str(out_path),
            "--max-failed-calls-per-item",
            "2",
        ]
    )
    monkeypatch.setattr(
        judge,
        "ask_detailed",
        lambda *args, **kwargs: {
            "label": "unparsed",
            "raw": "I think math",
            "attempts": 1,
            "http_status": 200,
        },
    )
    with pytest.raises(RuntimeError, match="failed/unparsed"):
        judge.run(args)
    partial = json.loads(out_path.read_text())
    assert partial["complete"] is False
    assert partial["valid_votes"] == []
    assert len(partial["judge_calls"]) == 2

    monkeypatch.setattr(
        judge,
        "ask_detailed",
        lambda *args, **kwargs: {
            "label": "math",
            "raw": "math",
            "attempts": 1,
            "http_status": 200,
        },
    )
    # The same CLI gets a fresh bounded set of attempts on resume.
    judge.run(args)
    resumed = json.loads(out_path.read_text())
    assert resumed["complete"] is True
    assert resumed["valid_votes"] == ["math"]
    assert [call["call_index"] for call in resumed["judge_calls"]] == [0, 1, 2]


def test_resume_rejects_unverifiable_legacy_row(tmp_path):
    items_path = tmp_path / "items.jsonl"
    out_path = tmp_path / "out.jsonl"
    item = {
        "arm": "D",
        "seed": 0,
        "step": 1,
        "layer": 2,
        "snippet_set": "calibration",
        "modality": "tokens",
        "expected_label": "cooking",
        "text": "'recipe', 'oven'",
    }
    items_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    out_path.write_text(json.dumps({"item_index": 0, "pred": "cooking"}) + "\n", encoding="utf-8")
    args = judge.parse_args(["--items", str(items_path), "--out", str(out_path), "--dry-run"])
    with pytest.raises(ValueError, match="does not match current input"):
        judge.run(args)


def test_resume_rejects_changed_input_and_snippet_provenance(tmp_path):
    item = {
        "arm": "D",
        "seed": 0,
        "step": 1,
        "layer": 2,
        "snippet_set": "cooking",
        "modality": "tokens",
        "text": "'recipe', 'oven'",
    }
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(json.dumps(item) + "\n", encoding="utf-8")

    changed_input_out = tmp_path / "changed-input.jsonl"
    first_args = judge.parse_args(
        [
            "--items",
            str(items_path),
            "--out",
            str(changed_input_out),
            "--dry-run",
            "--snippet-sha256",
            "cooking=" + "a" * 64,
        ]
    )
    judge.run(first_args)
    # Semantically identical JSON has the same item hash, but it is a different
    # input artifact and must not inherit the old file-level provenance.
    items_path.write_text(json.dumps(item, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input file bytes differ"):
        judge.run(first_args)

    items_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    changed_snippet_out = tmp_path / "changed-snippet.jsonl"
    original_snippet_args = judge.parse_args(
        [
            "--items",
            str(items_path),
            "--out",
            str(changed_snippet_out),
            "--dry-run",
            "--snippet-sha256",
            "cooking=" + "a" * 64,
        ]
    )
    judge.run(original_snippet_args)
    changed_snippet_args = judge.parse_args(
        [
            "--items",
            str(items_path),
            "--out",
            str(changed_snippet_out),
            "--dry-run",
            "--n-per-item",
            "2",
            "--snippet-sha256",
            "cooking=" + "b" * 64,
        ]
    )
    with pytest.raises(ValueError, match="source snippet SHA-256 differs"):
        judge.run(changed_snippet_args)


def test_explicit_truth_cannot_override_fixed_arm_mapping():
    item = {
        "arm": "B",
        "expected_label": "math",
    }
    with pytest.raises(ValueError, match="conflicts with required arm mapping"):
        judge.resolve_true_label(item, judge.LABELS)
    for key in ("expected_label", "true_label", "true"):
        with pytest.raises(ValueError, match="true label must be a string"):
            judge.resolve_true_label({"arm": "B", key: None}, judge.LABELS)
