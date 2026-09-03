from collections import Counter
from pathlib import Path

import pytest

from judge.lexical_baseline import evaluate_calibration_rows, load_jsonl


FIXTURE = Path(__file__).resolve().parents[1] / "data" / "lexical_calibration_items.jsonl"


def test_lexical_calibration_fixture_schema_and_counts():
    rows = load_jsonl(FIXTURE)
    required = {
        "item_id",
        "arm",
        "seed",
        "step",
        "layer",
        "snippet_set",
        "modality",
        "expected_label",
        "text",
    }
    assert len(rows) == 30
    assert len({row["item_id"] for row in rows}) == 30
    assert len({row["text"] for row in rows}) == 30
    assert all(required <= row.keys() for row in rows)
    assert all(row["modality"] == "tokens" for row in rows)
    assert Counter(row["expected_label"] for row in rows) == {
        "cooking": 10,
        "math": 10,
        "none": 10,
    }


def test_lexical_calibration_is_deterministic_and_passes_smoke_gate():
    rows = load_jsonl(FIXTURE)
    first = evaluate_calibration_rows(rows, requested_folds=5, seed=0)
    second = evaluate_calibration_rows(rows, requested_folds=5, seed=0)
    assert first == second
    assert first["obvious_cooking_math_accuracy"] >= 0.9
    assert first["nonsense_none_rate"] >= 0.8


def test_lexical_calibration_deduplicates_merged_judge_model_rows():
    rows = load_jsonl(FIXTURE)
    report = evaluate_calibration_rows(rows + rows, requested_folds=5, seed=0)
    assert report["source_n"] == 60
    assert report["n"] == 30
    assert report["duplicates_removed"] == 30

    repeated_text_with_new_ids = [
        dict(row, item_id=f"second-model-{row['item_id']}") for row in rows
    ]
    text_report = evaluate_calibration_rows(
        rows + repeated_text_with_new_ids,
        requested_folds=5,
        seed=0,
    )
    assert text_report["n"] == 30
    assert text_report["duplicates_removed"] == 30


def test_lexical_calibration_rejects_conflicting_duplicate_ids():
    rows = load_jsonl(FIXTURE)
    conflicting = dict(rows[0], expected_label="math")
    with pytest.raises(ValueError, match="conflicting duplicate item_id"):
        evaluate_calibration_rows([*rows, conflicting], requested_folds=5, seed=0)
