"""Integration tests for persisted external-corpus lexical predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from analysis.make_mock_results import generate_mock_results
from analysis.summarize import (
    PRIMARY_LEXICAL_VARIANT,
    PRIMARY_MODALITIES,
    add_lexical_predictions,
    collect_run_metadata,
    load_judged,
)


def _write_predictions(
    path: Path,
    judged_path: Path,
    judged_rows: list[dict],
    *,
    source_sha256: str | None = None,
    is_mock: bool = True,
    leakage_passed: bool = True,
) -> None:
    source_sha256 = source_sha256 or hashlib.sha256(judged_path.read_bytes()).hexdigest()
    output: list[dict] = []
    for index, judged in enumerate(judged_rows):
        if judged["modality"] not in PRIMARY_MODALITIES:
            continue
        copied = {
            key: value
            for key, value in judged.items()
            if not key.startswith("_")
        }
        prediction = "none" if index % 2 else str(judged["true"])
        common = {
            **copied,
            "is_mock": is_mock,
            "lexical_pred": prediction,
            "lexical_correct": prediction == judged["true"],
            "lexical_reference_manifest_sha256": "b" * 64,
            "lexical_reference_corpus_sha256": "c" * 64,
            "lexical_source_input_sha256": source_sha256,
            "lexical_training_source": "external_reference_corpus_only",
            "lexical_model_config": {
                "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
                "ngram_range": [1, 2],
                "min_df": 1,
                "sublinear_tf": True,
                "classifier": "sklearn.linear_model.LogisticRegression",
                "max_iter": 2000,
                "seed": 0,
            },
            "lexical_leakage_check": {
                "passed": leakage_passed,
                "exact_matches": 0,
                "shared_8gram_shingles": 0,
                "readouts_shorter_than_8_tokens": 0,
            },
            "lexical_timestamp": "2026-09-03T00:00:00Z",
            "lexical_git_commit": "deadbeef",
            "lexical_script_sha256": "d" * 64,
            "lexical_sklearn_version": "fixture",
        }
        output.append({**common, "lexical_variant": PRIMARY_LEXICAL_VARIANT})
        if judged["modality"] == "tokens":
            # This descriptive variant must not replace Figure 1's frozen
            # 1--2-gram baseline when both share an item id.
            output.append(
                {
                    **common,
                    "lexical_variant": "token_bag_unigram",
                    "lexical_pred": "poetry",
                    "lexical_correct": judged["true"] == "poetry",
                }
            )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )


def _mock_rows(tmp_path: Path) -> tuple[Path, list[dict]]:
    results = tmp_path / "results"
    generate_mock_results(results, seed=7, n_per_cell=2, d_model=16)
    judged_path = next(results.glob("judged_MOCK_*.jsonl"))
    return judged_path, load_judged([judged_path], mode="mock")


def test_persisted_predictions_join_by_item_and_ignore_token_bag_variant(
    tmp_path: Path,
) -> None:
    judged_path, rows = _mock_rows(tmp_path)
    predictions = tmp_path / "lexical_MOCK.jsonl"
    _write_predictions(predictions, judged_path, rows)

    add_lexical_predictions(
        rows,
        prediction_paths=[predictions],
        mode="mock",
    )

    primary = [row for row in rows if row["modality"] in PRIMARY_MODALITIES]
    assert all(row["_lexical_variant"] == PRIMARY_LEXICAL_VARIANT for row in primary)
    assert all(row["_lexical_pred"] != "poetry" or row["true"] == "poetry" for row in primary)
    assert all(row["_lexical_is_placeholder"] is False for row in primary)
    metadata = collect_run_metadata(primary, [])
    assert metadata["lexical_reference_manifest_sha256"] == ["b" * 64]
    assert metadata["lexical_reference_corpus_sha256"] == ["c" * 64]
    assert metadata["lexical_prediction_files"] == {
        predictions.name: hashlib.sha256(predictions.read_bytes()).hexdigest()
    }
    assert metadata["lexical_mock_placeholder"] is False


def test_real_analysis_refuses_to_fit_or_continue_without_predictions() -> None:
    rows = [
        {
            "item_id": "real-item",
            "arm": "A",
            "modality": "tokens",
            "true": "math",
            "is_mock": False,
        }
    ]
    with pytest.raises(ValueError, match="requires persisted external-corpus"):
        add_lexical_predictions(rows, mode="real")

    source = (Path(__file__).resolve().parents[1] / "analysis" / "summarize.py").read_text()
    assert "from sklearn" not in source
    assert "import sklearn" not in source
    assert ".fit(" not in source
    assert "cross_val_predict" not in source


def test_prediction_source_hash_and_real_leakage_receipt_are_enforced(
    tmp_path: Path,
) -> None:
    judged_path, mock_rows = _mock_rows(tmp_path)
    row = next(row for row in mock_rows if row["modality"] == "tokens")
    row["is_mock"] = False

    bad_hash = tmp_path / "bad-source.jsonl"
    _write_predictions(
        bad_hash,
        judged_path,
        [row],
        source_sha256="a" * 64,
        is_mock=False,
    )
    with pytest.raises(ValueError, match="source-input SHA-256 mismatch"):
        add_lexical_predictions([row], prediction_paths=[bad_hash], mode="real")

    no_leakage_receipt = tmp_path / "missing-leakage.jsonl"
    _write_predictions(
        no_leakage_receipt,
        judged_path,
        [row],
        is_mock=False,
        leakage_passed=False,
    )
    with pytest.raises(ValueError, match="lacks a passing exact/8-gram leakage receipt"):
        add_lexical_predictions(
            [row],
            prediction_paths=[no_leakage_receipt],
            mode="real",
        )


def test_mock_placeholder_is_explicit_and_deterministic(tmp_path: Path) -> None:
    _judged_path, first = _mock_rows(tmp_path)
    second = [dict(row) for row in first]
    add_lexical_predictions(first, seed=19, mode="mock")
    add_lexical_predictions(second, seed=19, mode="mock")
    first_primary = [row for row in first if row["modality"] in PRIMARY_MODALITIES]
    second_primary = [row for row in second if row["modality"] in PRIMARY_MODALITIES]
    assert [row["_lexical_pred"] for row in first_primary] == [
        row["_lexical_pred"] for row in second_primary
    ]
    assert all(row["_lexical_is_placeholder"] is True for row in first_primary)
    assert all(
        row["_lexical_variant"] == "MOCK_deterministic_placeholder_no_fit"
        for row in first_primary
    )
