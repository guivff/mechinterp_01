"""Round-2 judge calibration tests; all provider calls are stubbed or dry."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from judge import calibrate, judge


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "data" / "judge_calibration_items.jsonl"
LEGACY_FIXTURE = REPO_ROOT / "data" / "lexical_calibration_items.jsonl"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_round2_fixture_is_exact_and_retains_the_original_thirty_items() -> None:
    rows = _load(FIXTURE)
    legacy = _load(LEGACY_FIXTURE)

    assert calibrate.validate_calibration_items(rows) == "round2-50-v1"
    calibrate.validate_legacy_membership(rows, legacy)
    assert len(rows) == len({row["item_id"] for row in rows}) == len({row["text"] for row in rows}) == 50
    assert Counter(row["expected_label"] for row in rows) == {
        "cooking": 10,
        "math": 10,
        "none": 20,
        "poetry": 10,
    }
    assert Counter(row["subset"] for row in rows) == {
        "cooking": 10,
        "math": 10,
        "nonsense": 10,
        "generic_english": 10,
        "verse": 10,
    }
    assert all({"item_id", "expected_label", "category", "subset", "text"} <= row.keys() for row in rows)
    assert all(row["modality"] == "text" for row in rows if row["subset"] in {"generic_english", "verse"})
    assert all("\n" in row["text"] for row in rows if row["subset"] == "verse")


def test_defaults_fix_models_vote_count_and_output() -> None:
    args = calibrate.parse_args([])
    assert args.items == FIXTURE
    assert args.out == REPO_ROOT / "results" / "judge_calibration.jsonl"
    assert args.models == ["openai/gpt-5-mini", "google/gemini-2.5-flash"]
    assert args.n_per_item == 3
    assert args.labels == judge.LABELS


def test_round2_run_rejects_nonfrozen_vote_count_or_label_order(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly --n-per-item 3"):
        calibrate.run(
            calibrate.parse_args(
                [
                    "--dry-run",
                    "--out",
                    str(tmp_path / "calibration_MOCK_dry.jsonl"),
                    "--n-per-item",
                    "1",
                ]
            )
        )
    with pytest.raises(ValueError, match="fixed ordered labels"):
        calibrate.run(
            calibrate.parse_args(
                [
                    "--dry-run",
                    "--out",
                    str(tmp_path / "calibration_MOCK_dry.jsonl"),
                    "--labels",
                    *reversed(judge.LABELS),
                ]
            )
        )


def test_round2_dry_run_is_offline_deterministic_and_provenanced(tmp_path, monkeypatch) -> None:
    secret = "sentinel-openrouter-secret-must-not-be-written"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setattr(
        judge.requests,
        "post",
        lambda *args, **kwargs: pytest.fail("dry-run calibration attempted a network request"),
    )
    out = tmp_path / "judge_calibration_MOCK_dry.jsonl"
    args = calibrate.parse_args(["--out", str(out), "--dry-run"])

    combined, reports = calibrate.run(args)
    first_projection = [
        (row["requested_judge_model"], row["item_id"], row["pred"], tuple(row["judge_votes"]))
        for row in combined
    ]
    resumed, resumed_reports = calibrate.run(args)
    second_projection = [
        (row["requested_judge_model"], row["item_id"], row["pred"], tuple(row["judge_votes"]))
        for row in resumed
    ]

    fixture_sha = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert len(combined) == len(resumed) == 100
    assert first_projection == second_projection
    assert [report["model"] for report in reports] == calibrate.DEFAULT_MODELS
    assert reports == resumed_reports
    assert Counter(row["requested_judge_model"] for row in combined) == {
        "openai/gpt-5-mini": 50,
        "google/gemini-2.5-flash": 50,
    }
    assert all(row["n_per_item"] == 3 and row["vote_method"] == "strict_majority" for row in combined)
    assert all(len(row["judge_calls"]) == len(row["raw_responses"]) == 3 for row in combined)
    assert all(
        row["raw_responses"] == [call.get("raw", "") for call in row["judge_calls"]]
        for row in combined
    )
    assert all(row["calibration_protocol"] == "round2-50-v1" for row in combined)
    assert all(row["calibration_items_sha256"] == row["input_sha256"] == fixture_sha for row in combined)
    assert all(row["calibration_slice"] == row["subset"] for row in combined)
    assert all(row["judge_temperature"] == 0 for row in combined)
    assert secret not in out.read_text(encoding="utf-8")


def test_scoring_has_six_class_confusion_slices_and_constant_baselines(capsys) -> None:
    rows = [dict(row, pred=row["expected_label"]) for row in _load(FIXTURE)]
    report = calibrate.score_model(rows, "openai/gpt-5-mini")
    baselines = calibrate.baseline_reports(rows)

    assert report["overall_accuracy"] == 1.0
    assert set(report["per_class"]) == set(judge.LABELS)
    assert set(report["confusion_matrix"]) == set(judge.LABELS)
    assert all(set(cells) == set(judge.LABELS) for cells in report["confusion_matrix"].values())
    assert report["per_class"]["law"] == {"n": 0, "correct": 0, "accuracy": None}
    assert report["per_subset"]["generic_english"]["accuracy"] == 1.0
    assert report["threshold_diagnostics"]["triggered"] is False
    assert baselines[0]["model"] == "always-math"
    assert baselines[0]["overall_accuracy"] == pytest.approx(0.2)
    assert baselines[1]["model"] == "always-none"
    assert baselines[1]["overall_accuracy"] == pytest.approx(0.4)

    calibrate.print_accuracy_table([report, *baselines])
    calibrate.print_confusion_matrix(report)
    printed = capsys.readouterr().out
    assert "always-math" in printed and "always-none" in printed
    assert "confusion matrix: openai/gpt-5-mini" in printed
    assert all(label in printed for label in judge.LABELS)


def test_primary_threshold_diagnostics_use_class_and_generic_slice_not_all_none() -> None:
    boundary_rows = [dict(row, pred=row["expected_label"]) for row in _load(FIXTURE)]
    boundary_cooking = [index for index, row in enumerate(boundary_rows) if row["subset"] == "cooking"]
    boundary_generic = [index for index, row in enumerate(boundary_rows) if row["subset"] == "generic_english"]
    boundary_rows[boundary_cooking[0]]["pred"] = "none"  # exactly 0.90 cooking accuracy
    for index in boundary_generic[:2]:
        boundary_rows[index]["pred"] = "poetry"  # exactly 0.20 generic error
    boundary = calibrate.score_model(boundary_rows, "openai/gpt-5-mini")
    assert boundary["threshold_diagnostics"]["triggered"] is False

    rows = [dict(row, pred=row["expected_label"]) for row in _load(FIXTURE)]
    cooking_indexes = [index for index, row in enumerate(rows) if row["subset"] == "cooking"]
    generic_indexes = [index for index, row in enumerate(rows) if row["subset"] == "generic_english"]
    for index in cooking_indexes[:2]:
        rows[index]["pred"] = "none"
    for index in generic_indexes[:3]:
        rows[index]["pred"] = "poetry"

    report = calibrate.score_model(rows, "openai/gpt-5-mini")
    diagnostic = report["threshold_diagnostics"]
    assert report["per_class"]["cooking"]["accuracy"] == pytest.approx(0.8)
    assert report["generic_english_error_rate"] == pytest.approx(0.3)
    assert diagnostic["applicable"] and diagnostic["triggered"]
    assert diagnostic["cooking_accuracy_below_0_9"]
    assert not diagnostic["math_accuracy_below_0_9"]
    assert diagnostic["generic_english_non_none_rate_above_0_2"]
    assert diagnostic["proposed_prompt_fix"]

    comparison = calibrate.score_model(rows, "google/gemini-2.5-flash")
    assert comparison["threshold_diagnostics"]["applicable"] is False
    assert comparison["threshold_diagnostics"]["triggered"] is False


def test_live_shaped_stub_preserves_each_raw_response_without_a_key(tmp_path, monkeypatch) -> None:
    rows = _load(FIXTURE)
    truth_by_text = {row["text"]: row["expected_label"] for row in rows}
    call_number = 0

    def fake_ask(model, text, modality, labels, retries, backoff_base):
        nonlocal call_number
        call_number += 1
        label = truth_by_text[text]
        return {
            "label": label,
            "raw": f"raw-stub-{call_number}-{label}",
            "attempts": 1,
            "http_status": 200,
            "resolved_model": model,
        }

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(judge, "ask_detailed", fake_ask)
    out = tmp_path / "judge_calibration.jsonl"
    combined, reports = calibrate.run(calibrate.parse_args(["--out", str(out)]))

    assert call_number == 50 * 2 * 3
    assert len(combined) == 100
    assert all(report["overall_accuracy"] == 1.0 for report in reports)
    assert all(len(row["judge_calls"]) == len(row["raw_responses"]) == 3 for row in combined)
    assert all(all(raw.startswith("raw-stub-") for raw in row["raw_responses"]) for row in combined)
    assert all(row["pred"] == row["true"] for row in combined)


def test_legacy_thirty_item_scoring_api_remains_supported() -> None:
    rows = [dict(row, pred=row["expected_label"]) for row in _load(LEGACY_FIXTURE)]
    report = calibrate.score_model(rows, "openai/gpt-5-mini")
    assert report["protocol"] == "legacy-30"
    assert report["n"] == 30
    assert report["obvious_accuracy"] == 1.0
    assert report["nonsense_none_rate"] == 1.0
    assert report["generic_english_n"] == 0
    assert report["generic_english_error_rate"] is None

    minimal = [{"true": row["expected_label"], "pred": row["expected_label"]} for row in rows]
    assert calibrate.score_model(minimal, "openai/gpt-5-mini")["obvious_accuracy"] == 1.0


def test_lexical_calibration_runner_uses_the_round2_fixture() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "data" / "run_lexical_calibration.py"), "--json"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "[lexical calibration] n=50" in result.stdout
    assert '"generic_english"' in result.stdout
    assert '"verse"' in result.stdout
