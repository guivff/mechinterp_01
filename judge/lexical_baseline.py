"""External-corpus TF-IDF baseline and judge-summary utilities.

Scientific runs must pass ``--reference-dir``.  That path fits word 1--2 gram
TF-IDF + logistic regression solely on the frozen six-domain reference corpus,
checks exact/8-gram leakage, and evaluates the already-produced readout texts.
The older readout cross-validation path remains only as an explicitly labelled
compatibility smoke test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Hashable, Sequence

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline


# Frozen in PREREG.md and intentionally identical to judge/judge.py.
LABELS = ("math", "cooking", "law", "medicine", "poetry", "none")
ARM_TO_DOMAIN = {
    "A": "math",
    "B": "none",
    "C": "math",
    "Cp": "math",
    "C'": "math",
    "C′": "math",
    "D": "cooking",
    "N": "none",
    "N1": "none",
    "N2": "none",
    "N3": "none",
}
BLOCK_FIELDS = ("block", "block_id", "block_index")
_WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", flags=re.UNICODE)
_CORPUS_TOKEN_RE = re.compile(
    r"[^\W_]+(?:['’-][^\W_]+)*|[^\w\s]", flags=re.UNICODE
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows, reporting the bad line on failure."""

    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} of {path} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no JSONL rows")
    return rows


def make_lexical_pipeline(seed: int = 0) -> Pipeline:
    """Return the preregistered prose classifier."""

    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        LogisticRegression(max_iter=2_000, random_state=seed),
    )


def make_token_bag_pipeline(seed: int = 0) -> Pipeline:
    """Return the unigram-only, rank/order-insensitive token-list variant."""

    return make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=1,
            sublinear_tf=True,
            # Include one-character word tokens; punctuation-only/BPE-marker
            # entries remain out of vocabulary and are counted diagnostically.
            token_pattern=r"(?u)\b\w+\b",
        ),
        LogisticRegression(max_iter=2_000, random_state=seed),
    )


def deduplicate_calibration_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated source items before CV and reject contradictions."""

    unique: list[dict[str, Any]] = []
    by_id: dict[str, tuple[str, str]] = {}
    by_text: dict[str, str] = {}
    for index, row in enumerate(rows):
        if "text" not in row or "expected_label" not in row:
            raise ValueError(f"calibration row {index} is missing text/expected_label")
        text = str(row["text"]).strip()
        label = str(row["expected_label"])
        raw_id = row.get("item_id", row.get("id"))
        item_id = None if raw_id is None else str(raw_id)
        if item_id is not None and item_id in by_id:
            if by_id[item_id] != (text, label):
                raise ValueError(f"conflicting duplicate item_id {item_id!r}")
            continue
        if text in by_text:
            if by_text[text] != label:
                raise ValueError("identical calibration text has conflicting expected labels")
            if item_id is not None:
                by_id[item_id] = (text, label)
            continue
        if item_id is not None:
            by_id[item_id] = (text, label)
        by_text[text] = label
        unique.append(dict(row))
    return unique


def _cv_splitter(labels: Sequence[str], requested_folds: int, seed: int) -> StratifiedKFold:
    if requested_folds < 2:
        raise ValueError("requested_folds must be at least 2")
    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError("lexical classification requires at least two labels")
    folds = min(requested_folds, min(counts.values()))
    if folds < 2:
        raise ValueError(f"need at least two examples per label; observed counts: {dict(counts)}")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def cross_validated_predictions(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    requested_folds: int = 5,
    seed: int = 0,
) -> tuple[list[str], int]:
    """Return deterministic out-of-fold predictions and realized fold count."""

    if len(texts) != len(labels):
        raise ValueError(f"texts/labels length mismatch: {len(texts)} != {len(labels)}")
    if not texts:
        raise ValueError("no texts supplied")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("every text must be a non-empty string")
    splitter = _cv_splitter(labels, requested_folds, seed)
    predictions = cross_val_predict(
        make_lexical_pipeline(seed), list(texts), list(labels), cv=splitter, method="predict"
    )
    return [str(value) for value in predictions], splitter.n_splits


def confusion_counts(
    truths: Sequence[str],
    predictions: Sequence[str],
    labels: Sequence[str] = LABELS,
) -> dict[str, dict[str, int]]:
    """Build a complete row=true, column=predicted confusion matrix."""

    if len(truths) != len(predictions):
        raise ValueError("truth/prediction length mismatch")
    truth_order = list(labels)
    unexpected = sorted(set(truths) - set(truth_order))
    if unexpected:
        raise ValueError(f"truth labels outside configured labels: {unexpected}")
    prediction_order = [*truth_order, *sorted(set(predictions) - set(truth_order))]
    return {
        truth: {
            guess: sum(
                actual == truth and prediction == guess
                for actual, prediction in zip(truths, predictions)
            )
            for guess in prediction_order
        }
        for truth in truth_order
    }


def evaluate_calibration_rows(
    rows: Sequence[dict[str, Any]],
    *,
    requested_folds: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate ``text``/``expected_label`` rows using out-of-fold CV."""

    source_n = len(rows)
    unique_rows = deduplicate_calibration_rows(rows)
    texts = [str(row["text"]) for row in unique_rows]
    expected = [str(row["expected_label"]) for row in unique_rows]
    predicted, folds = cross_validated_predictions(
        texts, expected, requested_folds=requested_folds, seed=seed
    )
    label_order = sorted(set(expected))
    per_label: dict[str, dict[str, float | int]] = {}
    for label in label_order:
        indices = [index for index, value in enumerate(expected) if value == label]
        correct = sum(predicted[index] == label for index in indices)
        per_label[label] = {
            "n": len(indices),
            "correct": correct,
            "accuracy": correct / len(indices),
        }
    obvious = [index for index, label in enumerate(expected) if label in {"cooking", "math"}]
    if not obvious:
        raise ValueError("calibration needs at least one cooking or math row")
    return {
        "n": len(unique_rows),
        "source_n": source_n,
        "duplicates_removed": source_n - len(unique_rows),
        "seed": seed,
        "folds": folds,
        "labels": label_order,
        "overall_accuracy": sum(a == b for a, b in zip(expected, predicted)) / len(expected),
        "obvious_cooking_math_accuracy": sum(
            predicted[index] == expected[index] for index in obvious
        )
        / len(obvious),
        "nonsense_none_rate": per_label.get("none", {}).get("accuracy"),
        "per_label": per_label,
        "confusion": confusion_counts(expected, predicted, label_order),
        "predictions": [
            {
                "id": row.get("item_id", row.get("id", index)),
                "expected_label": truth,
                "predicted_label": prediction,
                "correct": prediction == truth,
            }
            for index, (row, truth, prediction) in enumerate(
                zip(unique_rows, expected, predicted)
            )
        ],
    }


def print_calibration_report(report: dict[str, Any]) -> None:
    none_rate = report["nonsense_none_rate"]
    none_display = "n/a" if none_rate is None else f"{none_rate:.3f}"
    print(
        f"[lexical calibration] n={report['n']} "
        f"duplicates_removed={report['duplicates_removed']} "
        f"folds={report['folds']} seed={report['seed']} "
        f"overall={report['overall_accuracy']:.3f} "
        f"obvious_cooking_math={report['obvious_cooking_math_accuracy']:.3f} "
        f"nonsense_as_none={none_display}"
    )
    print(f"{'label':>10} {'n':>4} {'correct':>7} {'accuracy':>9}")
    for label, values in report["per_label"].items():
        print(f"{label:>10} {values['n']:4d} {values['correct']:7d} {values['accuracy']:9.3f}")


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""

    if total <= 0:
        raise ValueError("Wilson interval needs total > 0")
    if not 0 <= successes <= total:
        raise ValueError("Wilson successes must lie in [0, total]")
    proportion = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = (proportion + z2 / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z2 / (4 * total * total))
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _truth(row: dict[str, Any]) -> str:
    for field in ("true", "expected_label", "true_label"):
        value = row.get(field)
        if isinstance(value, str) and value:
            if value not in LABELS:
                raise ValueError(f"truth label {value!r} is outside preregistered labels")
            return value
    arm = row.get("arm")
    if arm in ARM_TO_DOMAIN:
        return ARM_TO_DOMAIN[str(arm)]
    raise ValueError(f"row has no truth and unknown/missing arm: {arm!r}")


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _block(row: dict[str, Any]) -> Hashable | None:
    for field in BLOCK_FIELDS:
        if field in row:
            value = row[field]
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                return value
            raise ValueError(f"{field} must be a string or integer, got {value!r}")
    return None


def _cell_key(
    row: dict[str, Any], prediction_field: str
) -> tuple[str, str, str, str, str, str, str]:
    model = (
        str(row.get("lexical_variant", "external-lexical"))
        if prediction_field != "pred"
        else str(row.get("judge_model", "-"))
    )
    return (
        model,
        str(row.get("arm", "-")),
        str(row.get("seed", "-")),
        str(row.get("checkpoint_step", row.get("step", "-"))),
        str(row.get("layer", "-")),
        str(row.get("modality", "-")),
        str(row.get("snippet_set", "-")),
    )


def _collapse_units(
    rows: Sequence[dict[str, Any]], prediction_field: str
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    """Collapse repeated ratings to independent inputs or frozen blocks."""

    blocks = [_block(row) for row in rows]
    if any(value is not None for value in blocks) and not all(value is not None for value in blocks):
        raise ValueError("a summary cell mixes rows with and without block identifiers")
    if all(value is not None for value in blocks):
        by_block: dict[Hashable, list[dict[str, Any]]] = defaultdict(list)
        for value, row in zip(blocks, rows):
            assert value is not None
            by_block[value].append(row)
        units: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        for value in sorted(by_block, key=str):
            candidates = by_block[value]
            unique_texts = {_normalized_text(str(row["text"])) for row in candidates}
            if len(unique_texts) != 1:
                raise ValueError(
                    f"block {value!r} has {len(unique_texts)} unique inputs; "
                    "the frozen estimator requires one readout per block"
                )
            outcomes = {(_truth(row), str(row[prediction_field])) for row in candidates}
            if len(outcomes) != 1:
                raise ValueError(f"block {value!r} has inconsistent truth/prediction rows")
            shuffled_outcomes = {
                (
                    row.get("shuffled_true"),
                    row.get("correct_shuffled"),
                    row.get("shuffled_control_valid"),
                )
                for row in candidates
                if "correct_shuffled" in row
            }
            if len(shuffled_outcomes) > 1:
                raise ValueError(f"block {value!r} has inconsistent shuffle-control rows")
            representative = candidates[0]
            units.append(representative)
            details.append(
                {
                    "block": value,
                    "correct": str(representative[prediction_field]) == _truth(representative),
                }
            )
        return units, "block", details

    # Without block metadata, item_id is the best available sampling-unit key.
    # Identical text can legitimately appear under several item ids (and is
    # then disclosed by unique_inputs); it must not make reporting abort merely
    # because stochastic judge decisions differ. Exact reruns of one item id
    # are still collapsed and required to agree.
    by_item: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        normalized = _normalized_text(str(row["text"]))
        raw_item_id = row.get("item_id")
        key = ("item_id", str(raw_item_id)) if raw_item_id is not None else ("text", normalized)
        by_item[key].append(row)
    units = []
    for key in sorted(by_item):
        candidates = by_item[key]
        outcomes = {(_truth(row), str(row[prediction_field])) for row in candidates}
        if len(outcomes) != 1:
            raise ValueError(f"repeated sampling unit {key!r} has inconsistent predictions")
        units.append(candidates[0])
    return units, "input", []


def prediction_report(
    rows: Sequence[dict[str, Any]],
    *,
    prediction_field: str = "pred",
    include_shuffled: bool = False,
) -> dict[str, Any]:
    """Summarize predictions with class priors, blocks and Wilson intervals."""

    if not rows:
        raise ValueError("cannot summarize zero prediction rows")
    required = {"arm", "modality", "text", prediction_field}
    for index, row in enumerate(rows):
        absent = sorted(field for field in required if field not in row)
        if absent:
            raise ValueError(f"row {index} is missing required fields: {', '.join(absent)}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"row {index} text must be a non-empty string")
        _truth(row)
        if include_shuffled:
            required_control = {
                "correct_shuffled",
                "shuffled_true",
                "shuffled_control_valid",
                "shuffled_control_changed_n",
                "shuffled_control_expected_accuracy",
                "shuffle_control_kind",
                "visible_label_order_permuted",
                "shuffled_from_item_index",
                "judge_labels",
            }
            missing_control = sorted(required_control - row.keys())
            if missing_control:
                raise ValueError(
                    f"row {index} is missing shuffle-control receipts: "
                    f"{', '.join(missing_control)}"
                )
            if row["shuffle_control_kind"] != "input_gold_pairing_permutation":
                raise ValueError(f"row {index} has the wrong shuffle-control construction")
            if row["visible_label_order_permuted"] is not False:
                raise ValueError(f"row {index} permutes the visible label order")
            if row["judge_labels"] != list(LABELS):
                raise ValueError(f"row {index} does not use the fixed visible label order")
            if type(row["shuffled_control_valid"]) is not bool:
                raise ValueError(f"row {index} shuffled_control_valid must be boolean")
            if row["correct_shuffled"] is not (
                str(row[prediction_field]) == str(row["shuffled_true"])
            ):
                raise ValueError(f"row {index} correct_shuffled is inconsistent")

    # A statistical cell is fixed at model/variant, arm, training seed,
    # checkpoint, layer, modality, and snippet set.  In particular, do not
    # silently pool the K=10 block decisions across layers or checkpoints.
    grouped: dict[
        tuple[str, str, str, str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        grouped[_cell_key(row, prediction_field)].append(row)
    cells: list[dict[str, Any]] = []
    overall_units: list[dict[str, Any]] = []
    warnings: list[str] = []
    for key in sorted(grouped):
        cell_rows = grouped[key]
        units, unit_name, block_details = _collapse_units(cell_rows, prediction_field)
        overall_units.extend(units)
        truths = [_truth(row) for row in units]
        predictions = [str(row[prediction_field]) for row in units]
        correct = sum(a == b for a, b in zip(truths, predictions))
        low, high = wilson_interval(correct, len(units))
        unique_inputs = len({_normalized_text(str(row["text"])) for row in cell_rows})
        warning = None
        if unique_inputs < 10:
            warning = (
                f"cell model={key[0]} arm={key[1]} seed={key[2]} step={key[3]} "
                f"layer={key[4]} modality={key[5]} snippets={key[6]} "
                f"has {unique_inputs} unique inputs (<10)"
            )
            warnings.append(warning)
        if unit_name == "block" and len(units) != 10:
            block_warning = (
                f"cell model={key[0]} arm={key[1]} seed={key[2]} step={key[3]} "
                f"layer={key[4]} modality={key[5]} snippets={key[6]} "
                f"has {len(units)} blocks (expected K=10)"
            )
            warnings.append(block_warning)
            warning = f"{warning}; {block_warning}" if warning else block_warning
        cell: dict[str, Any] = {
            "judge_model": key[0],
            "arm": key[1],
            "seed": key[2],
            "checkpoint_step": key[3],
            "layer": key[4],
            "modality": key[5],
            "snippet_set": key[6],
            "raw_n": len(cell_rows),
            "unique_inputs": unique_inputs,
            "unit": unit_name,
            "unit_n": len(units),
            "correct": correct,
            "accuracy": correct / len(units),
            "wilson_95": [low, high],
            "always_math_accuracy": sum(value == "math" for value in truths) / len(units),
            "always_none_accuracy": sum(value == "none" for value in truths) / len(units),
            "confusion": confusion_counts(truths, predictions),
            "block_results": block_details,
            "warning": warning,
        }
        if include_shuffled:
            valid = [row for row in units if row["shuffled_control_valid"] is True]
            cell["shuffled_n"] = len(valid)
            cell["shuffled_invalid_n"] = len(units) - len(valid)
            cell["shuffled_accuracy"] = (
                sum(bool(row.get("correct_shuffled", False)) for row in valid) / len(valid)
                if valid
                else None
            )
            expected_values = {
                float(row["shuffled_control_expected_accuracy"]) for row in units
            }
            changed_values = {int(row["shuffled_control_changed_n"]) for row in units}
            if len(expected_values) != 1 or len(changed_values) != 1:
                raise ValueError("shuffle-control batch receipts differ within a summary cell")
            cell["shuffled_expected_accuracy"] = next(iter(expected_values))
            cell["shuffled_changed_n"] = next(iter(changed_values))
        cells.append(cell)
    truths = [_truth(row) for row in overall_units]
    predictions = [str(row[prediction_field]) for row in overall_units]
    correct = sum(a == b for a, b in zip(truths, predictions))
    output = {
        "n_units": len(overall_units),
        "correct": correct,
        "accuracy": correct / len(overall_units),
        "always_math_accuracy": sum(value == "math" for value in truths) / len(truths),
        "always_none_accuracy": sum(value == "none" for value in truths) / len(truths),
        "class_counts": dict(Counter(truths)),
        "confusion": confusion_counts(truths, predictions),
        "cells": cells,
        "warnings": warnings,
    }
    if include_shuffled:
        output["shuffle_control"] = {
            "kind": "input_gold_pairing_permutation",
            "visible_label_order_permuted": False,
            "visible_label_order": list(LABELS),
            "interpretation": "predictions rescored against gold labels permuted across inputs",
        }
    return output


def _print_confusion(confusion: dict[str, dict[str, int]]) -> None:
    columns = list(next(iter(confusion.values())))
    width = max([9, *(len(label) + 1 for label in columns)])
    print("confusion matrix (rows=true, columns=predicted)")
    print(f"{'true':>9}" + "".join(f"{label:>{width}}" for label in columns))
    for truth, counts in confusion.items():
        print(f"{truth:>9}" + "".join(f"{counts[label]:>{width}d}" for label in columns))


def print_prediction_report(report: dict[str, Any], *, title: str) -> None:
    print(
        f"[{title}] units={report['n_units']} accuracy={report['accuracy']:.3f} "
        f"always-math={report['always_math_accuracy']:.3f} "
        f"always-none={report['always_none_accuracy']:.3f}"
    )
    print(
        "class counts: "
        + ", ".join(f"{label}={report['class_counts'].get(label, 0)}" for label in LABELS)
    )
    if "shuffle_control" in report:
        control = report["shuffle_control"]
        print(
            "shuffle control: input↔gold pairing permutation; "
            "visible label order unchanged: "
            + ", ".join(control["visible_label_order"])
        )
    print(
        f"{'model':>20} {'arm':>5} {'seed':>5} {'step':>7} {'layer':>5} "
        f"{'modality':>10} {'snips':>10} "
        f"{'unit':>6} {'n':>3} {'unique':>6} {'acc':>6} {'Wilson 95%':>15} "
        f"{'all-math':>9} {'all-none':>9}"
    )
    for cell in report["cells"]:
        low, high = cell["wilson_95"]
        print(
            f"{cell['judge_model']:>20} {cell['arm']:>5} {cell['seed']:>5} "
            f"{cell['checkpoint_step']:>7} {cell['layer']:>5} {cell['modality']:>10} "
            f"{cell['snippet_set']:>10} {cell['unit']:>6} {cell['unit_n']:3d} "
            f"{cell['unique_inputs']:6d} {cell['accuracy']:6.3f} "
            f"[{low:.3f}, {high:.3f}] {cell['always_math_accuracy']:9.3f} "
            f"{cell['always_none_accuracy']:9.3f}"
        )
        if cell["block_results"]:
            outcomes = ", ".join(
                f"{entry['block']}={int(entry['correct'])}" for entry in cell["block_results"]
            )
            print(f"  per-block correct: {outcomes}")
        if "shuffled_accuracy" in cell:
            value = cell["shuffled_accuracy"]
            display = "n/a" if value is None else f"{value:.3f}"
            print(
                f"  shuffled-pairing accuracy: {display} (valid n={cell['shuffled_n']}, "
                f"invalid n={cell['shuffled_invalid_n']}, "
                f"batch changed={cell['shuffled_changed_n']}, "
                f"prior expectation={cell['shuffled_expected_accuracy']:.3f})"
            )
    for warning in report["warnings"]:
        print(f"warning: {warning}")
    _print_confusion(report["confusion"])


def summarize_judge_rows(
    rows: Sequence[dict[str, Any]],
    requested_folds: int,
    seed: int,
    *,
    run_legacy_cv: bool = False,
) -> dict[str, Any]:
    """Print judge controls without ever fitting on readout texts.

    ``run_legacy_cv`` remains in the signature so older callers fail with an
    explicit explanation instead of silently changing behavior.
    """

    required = {"arm", "modality", "text", "pred", "correct_shuffled"}
    for index, row in enumerate(rows):
        absent = sorted(required - row.keys())
        if absent:
            raise ValueError(f"judged row {index} is missing required fields: {', '.join(absent)}")
    report = prediction_report(rows, include_shuffled=True)
    print_prediction_report(report, title="judge summary")
    if run_legacy_cv:
        raise ValueError(
            "readout-text cross-validation is disabled: PREREG requires the lexical "
            "model to train on the external reference corpus only"
        )
    return report


def load_reference_corpus(
    directory: str | Path,
    *,
    labels: Sequence[str] = LABELS,
    expected_per_label: int | None = 50,
) -> list[dict[str, Any]]:
    """Load six ``<label>.jsonl`` files and reject exact duplicates."""

    directory = Path(directory)
    documents: list[dict[str, Any]] = []
    seen: dict[str, tuple[str, int]] = {}
    for label in labels:
        path = directory / f"{label}.jsonl"
        if not path.is_file():
            raise ValueError(f"missing reference-corpus file: {path}")
        rows = load_jsonl(path)
        if expected_per_label is not None and len(rows) != expected_per_label:
            raise ValueError(f"{path} has {len(rows)} documents; expected {expected_per_label}")
        for index, row in enumerate(rows):
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path} row {index} has missing/empty text")
            if row.get("label", label) != label:
                raise ValueError(f"{path} row {index} label conflicts with filename")
            normalized = _normalized_text(text)
            if normalized in seen:
                prior_label, prior_index = seen[normalized]
                raise ValueError(
                    f"duplicate reference text: {label}[{index}] duplicates "
                    f"{prior_label}[{prior_index}]"
                )
            seen[normalized] = (label, index)
            documents.append(
                {
                    **row,
                    "text": text,
                    "label": label,
                    "reference_file": str(path),
                    "reference_row": index,
                }
            )
    return documents


def validate_reference_artifacts(
    directory: str | Path,
    *,
    allow_mock: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Authenticate a complete generator manifest and its six corpus files.

    A dry-run fixture is accepted only behind ``allow_mock``.  The returned
    digest is the exact manifest-file SHA-256; the manifest's independent
    corpus digest is checked against the bytes of all six JSONL files.
    """

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except FileNotFoundError as exc:
        raise ValueError(f"missing reference-corpus manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid reference-corpus manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("reference manifest must use schema_version=1")

    expected_status = "complete_dry_run_fixture" if allow_mock else "complete"
    expected_scientific = not allow_mock
    if manifest.get("status") != expected_status:
        raise ValueError(
            f"reference manifest status must be {expected_status!r}, got "
            f"{manifest.get('status')!r}"
        )
    if manifest.get("scientific_use") is not expected_scientific:
        raise ValueError(
            "reference manifest scientific_use conflicts with the selected "
            f"{'dry-run' if allow_mock else 'scientific'} path"
        )

    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("reference manifest lacks configuration metadata")
    expected_configuration = {
        "labels": list(LABELS),
        "target_per_label": 50,
        "min_tokens": 100,
        "max_tokens": 300,
    }
    mismatched = {
        key: {"expected": expected, "found": configuration.get(key)}
        for key, expected in expected_configuration.items()
        if configuration.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"reference manifest has incompatible configuration: {mismatched}")
    serialized_configuration = json.dumps(
        configuration, ensure_ascii=False, sort_keys=True
    )
    if manifest.get("configuration_sha256") != hashlib.sha256(
        serialized_configuration.encode("utf-8")
    ).hexdigest():
        raise ValueError("reference manifest configuration SHA-256 does not match")

    corpus = manifest.get("corpus")
    if not isinstance(corpus, dict) or corpus.get("n") != 300 or corpus.get("expected_n") != 300:
        raise ValueError("reference manifest must describe exactly 300 documents")
    file_receipts = corpus.get("files")
    if not isinstance(file_receipts, dict) or set(file_receipts) != set(LABELS):
        raise ValueError("reference manifest must contain one file receipt per frozen label")

    aggregate = hashlib.sha256()
    for label in LABELS:
        path = directory / f"{label}.jsonl"
        try:
            payload = path.read_bytes()
        except FileNotFoundError as exc:
            raise ValueError(f"missing reference-corpus file: {path}") from exc
        receipt = file_receipts[label]
        if not isinstance(receipt, dict):
            raise ValueError(f"invalid file receipt for {label}")
        observed = {
            "filename": path.name,
            "n": sum(bool(line.strip()) for line in payload.decode("utf-8").splitlines()),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        bad = {
            key: {"expected": observed[key], "found": receipt.get(key)}
            for key in observed
            if receipt.get(key) != observed[key]
        }
        if bad:
            raise ValueError(f"reference file receipt mismatch for {label}: {bad}")
        aggregate.update(label.encode("utf-8") + b"\0" + payload)
    if corpus.get("sha256") != aggregate.hexdigest():
        raise ValueError("reference manifest corpus SHA-256 does not match class files")

    rows = load_reference_corpus(directory)
    manifest_documents = manifest.get("documents")
    if not isinstance(manifest_documents, list) or len(manifest_documents) != len(rows):
        raise ValueError("reference manifest document receipts are incomplete")
    for index, (row, receipt) in enumerate(zip(rows, manifest_documents)):
        if not isinstance(receipt, dict):
            raise ValueError(f"invalid document receipt at index {index}")
        text_hash = hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
        expected = {
            "id": row.get("id"),
            "label": row["label"],
            "index": index,
            "token_count": row.get("token_count"),
            "sha256": text_hash,
            "normalized_sha256": hashlib.sha256(
                _normalized_text(str(row["text"])).encode("utf-8")
            ).hexdigest(),
            "word_8gram_count": len(_shingles(str(row["text"]), 8)),
        }
        if row.get("sha256") != text_hash:
            raise ValueError(f"reference document {index} has a stale text SHA-256")
        token_count = row.get("token_count")
        observed_token_count = len(
            _CORPUS_TOKEN_RE.findall(unicodedata.normalize("NFKC", str(row["text"])))
        )
        if (
            not isinstance(token_count, int)
            or token_count != observed_token_count
            or not 100 <= token_count <= 300
        ):
            raise ValueError(
                f"reference document {index} has token_count={token_count!r}, "
                f"recomputed={observed_token_count}; expected an exact 100..300 receipt"
            )
        bad = {
            key: {"expected": value, "found": receipt.get(key)}
            for key, value in expected.items()
            if receipt.get(key) != value
        }
        if bad:
            raise ValueError(f"reference document receipt mismatch at index {index}: {bad}")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or not provenance.get("source"):
            raise ValueError(f"reference document {index} lacks provenance")
        if receipt.get("provenance") != provenance:
            raise ValueError(f"reference document {index} provenance receipt does not match")

    deduplication = manifest.get("deduplication")
    if not isinstance(deduplication, dict) or deduplication.get("ngram_size") != 8:
        raise ValueError("reference manifest lacks the frozen 8-gram deduplication receipt")
    threshold = deduplication.get("jaccard_threshold")
    if not isinstance(threshold, (int, float)) or not 0 < float(threshold) <= 1:
        raise ValueError("reference manifest has an invalid deduplication threshold")
    shingle_sets = [_shingles(str(row["text"]), 8) for row in rows]
    maximum = 0.0
    for left_index, left in enumerate(shingle_sets):
        for right in shingle_sets[:left_index]:
            if not left or not right:
                score = 1.0 if left == right else 0.0
            else:
                score = len(left & right) / len(left | right)
            maximum = max(maximum, score)
            if score >= float(threshold):
                raise ValueError(
                    "reference corpus contains a near duplicate at or above its "
                    f"8-gram Jaccard threshold ({score:.6f} >= {float(threshold):.6f})"
                )
    recorded_maximum = deduplication.get("max_observed_8gram_jaccard")
    if not isinstance(recorded_maximum, (int, float)) or not math.isclose(
        maximum, float(recorded_maximum), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("reference manifest maximum 8-gram similarity does not match")

    return rows, manifest, hashlib.sha256(manifest_bytes).hexdigest()


def _word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def _shingles(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = _word_tokens(text)
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def assert_no_reference_leakage(
    reference_rows: Sequence[dict[str, Any]],
    readout_rows: Sequence[dict[str, Any]],
    *,
    shingle_n: int = 8,
) -> dict[str, Any]:
    """Fail on exact readout/reference overlap or any shared word shingle."""

    if shingle_n < 1:
        raise ValueError("shingle_n must be positive")
    exact_reference: dict[str, int] = {}
    reference_shingles: dict[tuple[str, ...], int] = {}
    for index, row in enumerate(reference_rows):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"reference row {index} has missing/empty text")
        exact_reference.setdefault(_normalized_text(text), index)
        for shingle in _shingles(text, shingle_n):
            reference_shingles.setdefault(shingle, index)
    exact_collisions: list[dict[str, int]] = []
    shingle_collisions: list[dict[str, int]] = []
    short_readouts = 0
    for readout_index, row in enumerate(readout_rows):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"readout row {readout_index} has missing/empty text")
        normalized = _normalized_text(text)
        if normalized in exact_reference:
            exact_collisions.append(
                {"readout_index": readout_index, "reference_index": exact_reference[normalized]}
            )
        shingles = _shingles(text, shingle_n)
        if not shingles:
            short_readouts += 1
        for shingle in shingles:
            if shingle in reference_shingles:
                shingle_collisions.append(
                    {
                        "readout_index": readout_index,
                        "reference_index": reference_shingles[shingle],
                    }
                )
                break
    report = {
        "passed": not exact_collisions and not shingle_collisions,
        "reference_documents": len(reference_rows),
        "readout_documents": len(readout_rows),
        "shingle_n": shingle_n,
        "exact_collision_count": len(exact_collisions),
        "shingle_collision_count": len(shingle_collisions),
        "readouts_shorter_than_shingle": short_readouts,
        "exact_collisions": exact_collisions,
        "shingle_collisions": shingle_collisions,
    }
    if not report["passed"]:
        raise ValueError(
            "reference/readout leakage detected: "
            f"{len(exact_collisions)} exact, {len(shingle_collisions)} shared "
            f"{shingle_n}-gram collision(s)"
        )
    return report


def token_bag_text(row: dict[str, Any]) -> str:
    """Extract structured top tokens, ignoring ranks/logits and token order."""

    values: list[str] = []
    if isinstance(row.get("tokens"), list):
        values = [str(value) for value in row["tokens"]]
    elif isinstance(row.get("top"), list):
        for value in row["top"]:
            if isinstance(value, (list, tuple)) and value:
                values.append(str(value[0]))
            elif isinstance(value, dict) and "token" in value:
                values.append(str(value["token"]))
            elif isinstance(value, str):
                values.append(value)
            else:
                raise ValueError(f"unsupported top-token entry: {value!r}")
    elif isinstance(row.get("text"), str):
        values = _word_tokens(row["text"])
    flattened = [token for value in values for token in _word_tokens(value)]
    if not flattened:
        raise ValueError("token-list row has no word-like tokens for bag classification")
    return " ".join(sorted(flattened))


def fit_reference_classifier(
    reference_rows: Sequence[dict[str, Any]],
    *,
    seed: int = 0,
    token_bag: bool = False,
) -> Pipeline:
    """Fit solely on external reference prose; never accepts readout training rows."""

    texts = [str(row["text"]) for row in reference_rows]
    labels = [str(row["label"]) for row in reference_rows]
    counts = Counter(labels)
    if set(counts) != set(LABELS):
        raise ValueError(
            f"reference corpus needs exactly {list(LABELS)}; observed {sorted(counts)}"
        )
    if min(counts.values()) < 2:
        raise ValueError(f"reference corpus needs at least two documents/label: {dict(counts)}")
    pipeline = make_token_bag_pipeline(seed) if token_bag else make_lexical_pipeline(seed)
    pipeline.fit(texts, labels)
    return pipeline


def evaluate_external_reference(
    reference_rows: Sequence[dict[str, Any]],
    readout_rows: Sequence[dict[str, Any]],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit external-only prose/token-bag variants and score readouts."""

    if not readout_rows:
        raise ValueError("no readout rows supplied")
    leakage = assert_no_reference_leakage(reference_rows, readout_rows, shingle_n=8)
    prose_model = fit_reference_classifier(reference_rows, seed=seed)
    prose_predictions = [
        str(value) for value in prose_model.predict([str(row["text"]) for row in readout_rows])
    ]
    scored = [
        {**row, "lexical_pred": prediction, "lexical_variant": "prose_1_2gram"}
        for row, prediction in zip(readout_rows, prose_predictions)
    ]
    token_indices = [
        index for index, row in enumerate(readout_rows) if str(row.get("modality")) == "tokens"
    ]
    token_scored: list[dict[str, Any]] = []
    token_report: dict[str, Any] | None = None
    token_diagnostics: dict[str, Any] | None = None
    if token_indices:
        token_model = fit_reference_classifier(reference_rows, seed=seed, token_bag=True)
        token_texts = [token_bag_text(readout_rows[index]) for index in token_indices]
        vectorizer = token_model.named_steps["tfidfvectorizer"]
        matrix = vectorizer.transform(token_texts)
        row_nnz = np.asarray(matrix.getnnz(axis=1)).reshape(-1)
        empty_vectors = int(np.sum(row_nnz == 0))
        vocabulary = set(vectorizer.vocabulary_)
        units_by_row = [text.split() for text in token_texts]
        token_units = [token for units in units_by_row for token in units]
        oov = sum(token not in vocabulary for token in token_units)
        token_predictions = [str(value) for value in token_model.predict(token_texts)]
        token_scored = [
            {
                **readout_rows[index],
                "token_bag_pred": prediction,
                "lexical_variant": "token_bag_unigram",
                "_source_index": index,
                "token_bag_empty_vector": bool(row_nnz[position] == 0),
                "token_bag_token_units": len(units_by_row[position]),
                "token_bag_oov_token_units": sum(
                    token not in vocabulary for token in units_by_row[position]
                ),
            }
            for position, (index, prediction) in enumerate(zip(token_indices, token_predictions))
        ]
        token_report = prediction_report(token_scored, prediction_field="token_bag_pred")
        token_diagnostics = {
            "rows": len(token_texts),
            "empty_vectors": empty_vectors,
            "token_units": len(token_units),
            "oov_token_units": oov,
            "oov_rate": oov / len(token_units) if token_units else None,
            "tokenizer": "NFKC-casefolded Unicode word units; punctuation/BPE markers dropped",
        }
    return {
        "seed": seed,
        "reference_n": len(reference_rows),
        "reference_class_counts": dict(Counter(str(row["label"]) for row in reference_rows)),
        "leakage": leakage,
        "predictions": scored,
        "report": prediction_report(scored, prediction_field="lexical_pred"),
        "token_bag_predictions": token_scored,
        "token_bag_report": token_report,
        "token_bag_diagnostics": token_diagnostics,
    }


def print_external_reference_report(result: dict[str, Any]) -> None:
    counts = ", ".join(
        f"{label}={result['reference_class_counts'].get(label, 0)}" for label in LABELS
    )
    leakage = result["leakage"]
    print(f"[external lexical] train={result['reference_n']} ({counts}) seed={result['seed']}")
    print(
        f"[leakage] PASS exact={leakage['exact_collision_count']} "
        f"shared-{leakage['shingle_n']}-gram={leakage['shingle_collision_count']} "
        f"short-readouts={leakage['readouts_shorter_than_shingle']}"
    )
    print_prediction_report(result["report"], title="external TF-IDF 1-2gram")
    if result["token_bag_report"] is not None:
        diagnostics = result["token_bag_diagnostics"]
        print(
            f"[token-bag diagnostics] empty={diagnostics['empty_vectors']}/{diagnostics['rows']} "
            f"OOV={diagnostics['oov_token_units']}/{diagnostics['token_units']} "
            f"({diagnostics['oov_rate']:.3f})"
        )
        print_prediction_report(result["token_bag_report"], title="external token-bag unigram")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_predictions(
    path: str | Path,
    result: dict[str, Any],
    *,
    reference_manifest_sha256: str,
    reference_corpus_sha256: str,
    source_input_sha256: str,
) -> None:
    """Atomically write predictions, retaining run metadata required by AGENTS."""

    required = {
        "arm",
        "seed",
        "layer",
        "snippet_set",
        "judge_model",
        "timestamp",
        "git_commit",
        "is_mock",
    }
    token_by_index = {
        int(row["_source_index"]): row
        for row in result["token_bag_predictions"]
    }
    output_rows: list[dict[str, Any]] = []
    lexical_timestamp = _utc_now()
    lexical_commit = _git_commit()
    lexical_script_sha256 = _sha256_file(Path(__file__).resolve())
    configurations = {
        "prose_1_2gram": {
            "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
            "ngram_range": [1, 2],
            "min_df": 1,
            "sublinear_tf": True,
            "classifier": "sklearn.linear_model.LogisticRegression",
            "max_iter": 2000,
            "seed": result["seed"],
        },
        "token_bag_unigram": {
            "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
            "ngram_range": [1, 1],
            "token_pattern": r"(?u)\b\w+\b",
            "token_order": "ignored",
            "classifier": "sklearn.linear_model.LogisticRegression",
            "max_iter": 2000,
            "seed": result["seed"],
        },
    }
    leakage_receipt = {
        "passed": bool(result["leakage"]["passed"]),
        "exact_matches": int(result["leakage"]["exact_collision_count"]),
        "shared_8gram_shingles": int(result["leakage"]["shingle_collision_count"]),
        "readouts_shorter_than_8_tokens": int(
            result["leakage"]["readouts_shorter_than_shingle"]
        ),
    }
    for index, row in enumerate(result["predictions"]):
        absent = sorted(required - row.keys())
        if "step" not in row and "checkpoint_step" not in row:
            absent.append("step/checkpoint_step")
        if "snippet_sha256" not in row and "snippet_sha" not in row:
            absent.append("snippet_sha256")
        if not isinstance(row.get("item_id"), (str, int)):
            absent.append("item_id")
        if absent:
            raise ValueError(
                f"cannot write lexical result row {index}; missing metadata: {', '.join(absent)}"
            )
        if type(row["is_mock"]) is not bool:
            raise ValueError(f"cannot write lexical result row {index}; is_mock must be boolean")
        prose = dict(row)
        prose.update(
            {
                "lexical_correct": prose["lexical_pred"] == _truth(prose),
                "lexical_training_source": "external_reference_corpus_only",
                "lexical_seed": result["seed"],
                "lexical_reference_manifest_sha256": reference_manifest_sha256,
                "lexical_reference_corpus_sha256": reference_corpus_sha256,
                "lexical_source_input_sha256": source_input_sha256,
                "lexical_leakage_check": leakage_receipt,
                "lexical_model_config": configurations["prose_1_2gram"],
                "lexical_timestamp": lexical_timestamp,
                "lexical_git_commit": lexical_commit,
                "lexical_script_sha256": lexical_script_sha256,
                "lexical_sklearn_version": sklearn.__version__,
            }
        )
        output_rows.append(prose)
        if index in token_by_index:
            token_source = token_by_index[index]
            token = dict(row)
            token.update(
                {
                    "lexical_variant": "token_bag_unigram",
                    "lexical_pred": token_source["token_bag_pred"],
                    "lexical_correct": token_source["token_bag_pred"] == _truth(token),
                    "lexical_training_source": "external_reference_corpus_only",
                    "lexical_seed": result["seed"],
                    "lexical_reference_manifest_sha256": reference_manifest_sha256,
                    "lexical_reference_corpus_sha256": reference_corpus_sha256,
                    "lexical_source_input_sha256": source_input_sha256,
                    "lexical_leakage_check": leakage_receipt,
                    "lexical_model_config": configurations["token_bag_unigram"],
                    "lexical_timestamp": lexical_timestamp,
                    "lexical_git_commit": lexical_commit,
                    "lexical_script_sha256": lexical_script_sha256,
                    "lexical_sklearn_version": sklearn.__version__,
                    "token_bag_empty_vector": token_source["token_bag_empty_vector"],
                    "token_bag_token_units": token_source["token_bag_token_units"],
                    "token_bag_oov_token_units": token_source["token_bag_oov_token_units"],
                }
            )
            output_rows.append(token)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(destination)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--judged", help="judge output/readout JSONL")
    source.add_argument("--calibration-items", help="JSONL with text and expected_label")
    parser.add_argument(
        "--reference-dir",
        help="directory with six label JSONLs + manifest.json (scientific path)",
    )
    parser.add_argument("--predictions-out", help="optional lexical prediction JSONL")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="allow only non-scientific MOCK reference/readout fixtures",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.calibration_items:
            if args.reference_dir or args.predictions_out or args.dry_run:
                raise ValueError("reference/output options are only valid with --judged")
            print_calibration_report(
                evaluate_calibration_rows(
                    load_jsonl(args.calibration_items),
                    requested_folds=args.folds,
                    seed=args.seed,
                )
            )
            return 0

        rows = load_jsonl(args.judged)
        summarize_judge_rows(rows, args.folds, args.seed)
        if not args.reference_dir:
            print(
                "[external lexical] NOT RUN: pass --reference-dir and "
                "--predictions-out; no model was fit on readout text"
            )
        if args.reference_dir:
            if not args.predictions_out:
                raise ValueError(
                    "--reference-dir requires --predictions-out so scientific predictions "
                    "are persisted with provenance"
                )
            reference_dir = Path(args.reference_dir)
            reference_rows, manifest, manifest_sha256 = validate_reference_artifacts(
                reference_dir, allow_mock=args.dry_run
            )
            input_modes = {row.get("is_mock") for row in rows}
            if input_modes - {True, False} or len(input_modes) != 1:
                raise ValueError("judged rows must carry one consistent boolean is_mock value")
            output_is_mock = args.dry_run or input_modes == {True}
            output_has_mock = "mock" in Path(args.predictions_out).name.casefold()
            if output_has_mock != output_is_mock:
                expected = "must" if output_is_mock else "must not"
                raise ValueError(
                    f"--predictions-out filename {expected} contain MOCK to match provenance"
                )
            result = evaluate_external_reference(
                reference_rows, rows, seed=args.seed
            )
            print_external_reference_report(result)
            if args.predictions_out:
                _write_predictions(
                    args.predictions_out,
                    result,
                    reference_manifest_sha256=manifest_sha256,
                    reference_corpus_sha256=str(manifest["corpus"]["sha256"]),
                    source_input_sha256=_sha256_file(Path(args.judged)),
                )
        elif args.predictions_out:
            raise ValueError("--predictions-out requires --reference-dir")
        return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
