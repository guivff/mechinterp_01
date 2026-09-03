"""Round-2 tests for external-only lexical scoring and judge summaries."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data import make_lexical_reference
from judge.lexical_baseline import (
    LABELS,
    _write_predictions,
    assert_no_reference_leakage,
    evaluate_external_reference,
    load_reference_corpus,
    main,
    prediction_report,
    token_bag_text,
    validate_reference_artifacts,
    wilson_interval,
)


def _judged_row(
    arm: str,
    block: int,
    prediction: str,
    *,
    modality: str = "tokens",
) -> dict:
    truth = "math" if arm == "A" else "none"
    return {
        "item_id": f"{arm}-{block}",
        "arm": arm,
        "seed": 0,
        "step": 150,
        "checkpoint_step": 150,
        "layer": 15,
        "snippet_set": "neutral",
        "snippet_sha256": "a" * 64,
        "judge_model": "fixture/judge",
        "timestamp": "2026-09-03T00:00:00Z",
        "git_commit": "b" * 40,
        "is_mock": True,
        "modality": modality,
        "block": block,
        "text": f"unique evidence {arm} block {block}",
        "pred": prediction,
        "true": truth,
        "correct": prediction == truth,
        "correct_shuffled": block % 2 == 0,
        "shuffled_true": prediction if block % 2 == 0 else (
            "none" if prediction != "none" else "math"
        ),
        "shuffled_control_valid": True,
        "shuffled_control_changed_n": 10,
        "shuffled_control_expected_accuracy": 0.5,
        "shuffle_control_kind": "input_gold_pairing_permutation",
        "visible_label_order_permuted": False,
        "shuffled_from_item_index": block,
        "judge_labels": list(LABELS),
        "top": [["equation", 2.0], ["proof", 1.0]],
    }


def _reference_rows(n_per_label: int = 3) -> list[dict]:
    vocabulary = {
        "math": "algebra equation theorem proof integer geometry",
        "cooking": "recipe skillet garlic pastry simmer kitchen",
        "law": "statute court appeal contract judge precedent",
        "medicine": "clinical patient diagnosis therapy hospital symptom",
        "poetry": "verse stanza rhyme meter lyric sonnet",
        "none": "neighborhood weather journey conversation ordinary weekend",
    }
    return [
        {
            "label": label,
            "text": f"{words} independent reference document number {index} for calibration",
        }
        for label, words in vocabulary.items()
        for index in range(n_per_label)
    ]


def test_block_report_has_wilson_controls_confusion_and_unique_count() -> None:
    rows = [
        *[_judged_row("A", block, "math" if block < 8 else "none") for block in range(10)],
        *[_judged_row("B", block, "none" if block < 6 else "math") for block in range(10)],
    ]
    report = prediction_report(rows, include_shuffled=True)

    assert report["n_units"] == 20
    assert report["always_math_accuracy"] == 0.5
    assert report["always_none_accuracy"] == 0.5
    assert report["warnings"] == []
    assert report["confusion"]["math"]["math"] == 8
    assert report["confusion"]["none"]["none"] == 6
    by_arm = {cell["arm"]: cell for cell in report["cells"]}
    assert by_arm["A"]["unit"] == "block"
    assert by_arm["A"]["unit_n"] == 10
    assert by_arm["A"]["accuracy"] == 0.8
    assert len(by_arm["A"]["block_results"]) == 10
    assert by_arm["A"]["shuffled_accuracy"] == 0.5
    assert by_arm["A"]["shuffled_expected_accuracy"] == 0.5
    assert by_arm["A"]["wilson_95"] == pytest.approx(wilson_interval(8, 10))


def test_cells_do_not_pool_seeds_steps_or_layers() -> None:
    rows = [_judged_row("A", block, "math") for block in range(10)]
    rows.extend(
        dict(row, item_id=f"other-{row['item_id']}", seed=1, layer=19)
        for row in rows.copy()
    )
    report = prediction_report(rows)

    assert len(report["cells"]) == 2
    assert {(cell["seed"], cell["layer"], cell["unit_n"]) for cell in report["cells"]} == {
        ("0", "15", 10),
        ("1", "19", 10),
    }


def test_shuffle_summary_fails_closed_on_visible_label_or_receipt_changes() -> None:
    row = _judged_row("A", 0, "math")
    row["visible_label_order_permuted"] = True
    with pytest.raises(ValueError, match="permutes the visible label order"):
        prediction_report([row], include_shuffled=True)

    row = _judged_row("A", 0, "math")
    row["correct_shuffled"] = False
    with pytest.raises(ValueError, match="correct_shuffled is inconsistent"):
        prediction_report([row], include_shuffled=True)


def test_duplicate_input_is_not_a_new_unit_and_low_unique_cell_warns() -> None:
    row = _judged_row("A", 0, "math")
    row.pop("block")
    report = prediction_report([row, dict(row)])
    assert report["n_units"] == 1
    assert report["cells"][0]["raw_n"] == 2
    assert report["cells"][0]["unique_inputs"] == 1
    assert "1 unique inputs (<10)" in report["warnings"][0]

    # Separate item ids remain separate recorded decisions even when their
    # payload is identical; the low-unique-input warning exposes the problem.
    second_item = dict(row, item_id="A-another", pred="none", correct=False)
    distinct = prediction_report([row, second_item])
    assert distinct["n_units"] == 2
    assert distinct["cells"][0]["unique_inputs"] == 1


def test_block_rejects_more_than_one_readout_payload() -> None:
    first = _judged_row("A", 0, "math")
    second = dict(first, item_id="A-0b", text="different payload")
    with pytest.raises(ValueError, match="one readout per block"):
        prediction_report([first, second])


def test_reference_loader_requires_files_counts_and_rejects_duplicates(tmp_path: Path) -> None:
    for label in LABELS:
        rows = [{"text": f"{label} reference {index}"} for index in range(2)]
        (tmp_path / f"{label}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    loaded = load_reference_corpus(tmp_path, expected_per_label=2)
    assert len(loaded) == 12
    assert {row["label"] for row in loaded} == set(LABELS)

    duplicate = {"text": "math reference 0"}
    cooking = tmp_path / "cooking.jsonl"
    cooking_rows = [duplicate, {"text": "another cooking reference"}]
    cooking.write_text(
        "".join(json.dumps(row) + "\n" for row in cooking_rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate reference text"):
        load_reference_corpus(tmp_path, expected_per_label=2)


def test_leakage_gate_detects_exact_and_shared_eight_word_shingles() -> None:
    reference = [
        {"text": "one two three four five six seven eight nine ten", "label": "none"}
    ]
    passed = assert_no_reference_leakage(reference, [{"text": "short unseen payload"}])
    assert passed["passed"] is True
    assert passed["readouts_shorter_than_shingle"] == 1

    with pytest.raises(ValueError, match="1 exact"):
        assert_no_reference_leakage(reference, [{"text": reference[0]["text"]}])
    with pytest.raises(ValueError, match="shared 8-gram"):
        assert_no_reference_leakage(
            reference,
            [{"text": "prefix one two three four five six seven eight suffix"}],
        )


def test_token_bag_prefers_structured_top_and_is_order_invariant() -> None:
    left = {
        "text": "this fallback must be ignored",
        "top": [[" Geometry", 9.0], {"token": "proof"}, "x"],
    }
    right = {"top": ["x", {"token": "proof"}, [" Geometry", -100.0]]}
    assert token_bag_text(left) == token_bag_text(right) == "geometry proof x"


def test_external_fit_scores_both_variants_without_readout_training() -> None:
    readouts = [
        _judged_row("A", block, "math")
        | {
            "text": f"'algebra', 'equation', 'proof', 'integer', 'geometry', 'theorem', '{block}'",
            "top": [[token, float(20 - rank)] for rank, token in enumerate(
                ["algebra", "equation", "proof", "integer", "geometry", "theorem", str(block)]
            )],
        }
        for block in range(10)
    ]
    result = evaluate_external_reference(_reference_rows(), readouts, seed=7)

    assert result["leakage"]["passed"] is True
    assert result["reference_n"] == 18
    assert len(result["predictions"]) == 10
    assert len(result["token_bag_predictions"]) == 10
    assert result["report"]["cells"][0]["unit"] == "block"
    assert result["token_bag_diagnostics"]["empty_vectors"] == 0
    assert all(row["lexical_variant"] == "prose_1_2gram" for row in result["predictions"])
    assert all(
        row["lexical_variant"] == "token_bag_unigram"
        for row in result["token_bag_predictions"]
    )


def test_prediction_jsonl_has_one_row_per_item_variant_and_receipts(tmp_path: Path) -> None:
    readout = _judged_row("A", 0, "math") | {
        "text": "'algebra', 'equation', 'proof'",
        "top": [["algebra", 2.0], ["equation", 1.0], ["proof", 0.5]],
    }
    result = evaluate_external_reference(_reference_rows(), [readout], seed=3)
    output = tmp_path / "lexical.jsonl"
    _write_predictions(
        output,
        result,
        reference_manifest_sha256="c" * 64,
        reference_corpus_sha256="e" * 64,
        source_input_sha256="d" * 64,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert [row["lexical_variant"] for row in rows] == [
        "prose_1_2gram",
        "token_bag_unigram",
    ]
    assert all(row["item_id"] == "A-0" for row in rows)
    assert all(row["lexical_reference_manifest_sha256"] == "c" * 64 for row in rows)
    assert all(row["lexical_reference_corpus_sha256"] == "e" * 64 for row in rows)
    assert all(row["lexical_source_input_sha256"] == "d" * 64 for row in rows)
    assert all(row["lexical_training_source"] == "external_reference_corpus_only" for row in rows)
    assert all(row["lexical_leakage_check"]["passed"] is True for row in rows)
    assert all(row["lexical_git_commit"] and row["lexical_timestamp"].endswith("Z") for row in rows)
    assert rows[1]["lexical_model_config"]["token_order"] == "ignored"


def test_external_cli_persists_predictions_on_offline_fixture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference_dir = tmp_path / "reference"
    assert make_lexical_reference.main(
        ["--dry-run", "--out-dir", str(reference_dir)]
    ) == 0

    judged = tmp_path / "judged.jsonl"
    judged.write_text(json.dumps(_judged_row("A", 0, "math")) + "\n", encoding="utf-8")
    output = tmp_path / "lexical_MOCK.jsonl"
    assert (
        main(
            [
                "--judged",
                str(judged),
                "--reference-dir",
                str(reference_dir),
                "--predictions-out",
                str(output),
                "--seed",
                "9",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "[leakage] PASS exact=0 shared-8-gram=0" in capsys.readouterr().out
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {row["lexical_variant"] for row in rows} == {
        "prose_1_2gram",
        "token_bag_unigram",
    }


def test_manifest_gate_rejects_mock_or_tampered_corpus_on_scientific_path(
    tmp_path: Path,
) -> None:
    reference_dir = tmp_path / "reference"
    make_lexical_reference.main(["--dry-run", "--out-dir", str(reference_dir)])

    with pytest.raises(ValueError, match="status must be 'complete'"):
        validate_reference_artifacts(reference_dir)
    rows, manifest, digest = validate_reference_artifacts(reference_dir, allow_mock=True)
    assert len(rows) == 300
    assert manifest["scientific_use"] is False
    assert len(digest) == 64

    path = reference_dir / "math.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file receipt mismatch"):
        validate_reference_artifacts(reference_dir, allow_mock=True)
