"""TF-IDF lexical baseline and deterministic calibration utilities.

The experiment baseline asks whether surface words alone predict the readout domain. It
is deliberately simple: word unigram/bigram TF-IDF followed by logistic regression.

Examples::

    python judge/lexical_baseline.py --judged results/judged.jsonl
    python judge/lexical_baseline.py \
        --calibration-items data/lexical_calibration_items.jsonl

The first command preserves the experiment-facing summary. The second performs
stratified, shuffled cross-validation on hand-written calibration items without making
network calls.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows and fail with a useful line number."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
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
        raise ValueError(f"no JSONL rows found in {path}")
    return rows


def make_lexical_pipeline(seed: int = 0) -> Pipeline:
    """Return the preregistered word-level lexical classifier."""
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        LogisticRegression(max_iter=2_000, random_state=seed),
    )


def deduplicate_calibration_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated source items before CV, validating duplicate consistency.

    Calibration results from multiple judge models can contain the same 30 source items
    more than once. Using those repeated texts as independent classifier examples would
    leak exact copies across folds and inflate lexical accuracy.
    """
    unique: list[dict[str, Any]] = []
    by_item_id: dict[str, tuple[str, str]] = {}
    by_text: dict[str, tuple[str, str | None]] = {}
    for index, row in enumerate(rows):
        if "text" not in row or "expected_label" not in row:
            raise ValueError(f"calibration row {index} is missing text/expected_label")
        text = str(row["text"]).strip()
        label = str(row["expected_label"])
        raw_item_id = row.get("item_id", row.get("id"))
        item_id = None if raw_item_id is None else str(raw_item_id)

        if item_id is not None and item_id in by_item_id:
            previous_text, previous_label = by_item_id[item_id]
            if (text, label) != (previous_text, previous_label):
                raise ValueError(f"conflicting duplicate item_id {item_id!r}")
            continue
        if text in by_text:
            previous_label, _previous_item_id = by_text[text]
            if label != previous_label:
                raise ValueError("identical calibration text has conflicting expected labels")
            if item_id is not None:
                by_item_id[item_id] = (text, label)
            continue

        if item_id is not None:
            by_item_id[item_id] = (text, label)
        by_text[text] = (label, item_id)
        unique.append(row)
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
    """Return deterministic out-of-fold predictions and the realized fold count."""
    if len(texts) != len(labels):
        raise ValueError(f"texts/labels length mismatch: {len(texts)} != {len(labels)}")
    if not texts:
        raise ValueError("no texts supplied")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("every text must be a non-empty string")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("every label must be a non-empty string")

    splitter = _cv_splitter(labels, requested_folds, seed)
    predictions = cross_val_predict(
        make_lexical_pipeline(seed),
        list(texts),
        list(labels),
        cv=splitter,
        method="predict",
    )
    return [str(prediction) for prediction in predictions], splitter.n_splits


def evaluate_calibration_rows(
    rows: Sequence[dict[str, Any]],
    *,
    requested_folds: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate rows containing ``text`` and ``expected_label`` using out-of-fold CV."""
    source_n = len(rows)
    rows = deduplicate_calibration_rows(rows)

    texts = [str(row["text"]) for row in rows]
    expected = [str(row["expected_label"]) for row in rows]
    predicted, folds = cross_validated_predictions(
        texts,
        expected,
        requested_folds=requested_folds,
        seed=seed,
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

    obvious_indices = [index for index, value in enumerate(expected) if value in {"cooking", "math"}]
    if not obvious_indices:
        raise ValueError("calibration needs at least one cooking or math row")
    obvious_correct = sum(predicted[index] == expected[index] for index in obvious_indices)
    overall_correct = sum(prediction == label for prediction, label in zip(predicted, expected))
    confusion = {
        actual: {
            guess: sum(
                truth == actual and prediction == guess
                for truth, prediction in zip(expected, predicted)
            )
            for guess in label_order
        }
        for actual in label_order
    }
    return {
        "n": len(rows),
        "source_n": source_n,
        "duplicates_removed": source_n - len(rows),
        "seed": seed,
        "folds": folds,
        "labels": label_order,
        "overall_accuracy": overall_correct / len(rows),
        "obvious_cooking_math_accuracy": obvious_correct / len(obvious_indices),
        "nonsense_none_rate": per_label.get("none", {}).get("accuracy"),
        "per_label": per_label,
        "confusion": confusion,
        "predictions": [
            {
                "id": row.get("item_id", row.get("id", index)),
                "expected_label": truth,
                "predicted_label": prediction,
                "correct": prediction == truth,
            }
            for index, (row, truth, prediction) in enumerate(zip(rows, expected, predicted))
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


def summarize_judge_rows(rows: Sequence[dict[str, Any]], requested_folds: int, seed: int) -> None:
    """Print the historical judge summary and lexical CV result per modality."""
    required = {"arm", "modality", "text", "true", "correct", "correct_shuffled"}
    for index, row in enumerate(rows):
        absent = sorted(required - row.keys())
        if absent:
            raise ValueError(f"judged row {index} is missing fields: {', '.join(absent)}")

    accuracy = defaultdict(list)
    shuffled = defaultdict(list)
    for row in rows:
        key = (row["arm"], row["modality"], row.get("snippet_set", "-"))
        accuracy[key].append(row["correct"])
        if row.get("shuffled_control_valid", True):
            shuffled[key].append(row["correct_shuffled"])
    print(f"{'arm':>4} {'modality':>10} {'snips':>10} {'n':>4} {'judge_acc':>9} {'shuffled':>9}")
    for key in sorted(accuracy):
        shuffled_display = f"{np.mean(shuffled[key]):9.3f}" if shuffled[key] else f"{'n/a':>9}"
        print(
            f"{key[0]:>4} {key[1]:>10} {key[2]:>10} {len(accuracy[key]):>4} "
            f"{np.mean(accuracy[key]):9.3f} {shuffled_display}"
        )

    # Tokens and steered generations differ in surface form, so retain the
    # preregistered per-modality classifiers.
    for modality in sorted({row["modality"] for row in rows}):
        raw_subset = [row for row in rows if row["modality"] == modality]
        calibration_view = [dict(row, expected_label=row["true"]) for row in raw_subset]
        subset = deduplicate_calibration_rows(calibration_view)
        duplicates_removed = len(raw_subset) - len(subset)
        if duplicates_removed:
            print(
                f"[lexical] {modality}: removed {duplicates_removed} repeated source rows "
                "before cross-validation"
            )
        texts = [row["text"] for row in subset]
        labels = [row["true"] for row in subset]
        try:
            splitter = _cv_splitter(labels, requested_folds, seed)
        except ValueError as exc:
            print(f"[lexical] {modality}: {exc}")
            continue
        scores = cross_val_score(
            make_lexical_pipeline(seed),
            texts,
            labels,
            cv=splitter,
            scoring="accuracy",
        )
        chance = 1.0 / len(set(labels))
        print(
            f"[lexical] {modality}: {splitter.n_splits}-fold "
            f"acc={scores.mean():.3f} ± {scores.std():.3f} (chance {chance:.2f})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--judged", help="judge output JSONL")
    source.add_argument("--calibration-items", help="JSONL containing text and expected_label")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    path = args.calibration_items or args.judged
    try:
        rows = load_jsonl(path)
    except ValueError as exc:
        message = str(exc)
        if "no JSONL rows" in message:
            message = f"{path} contains no JSONL rows"
        parser.exit(2, f"error: {message}\n")
    try:
        if args.calibration_items:
            report = evaluate_calibration_rows(rows, requested_folds=args.folds, seed=args.seed)
            print_calibration_report(report)
        else:
            summarize_judge_rows(rows, args.folds, args.seed)
    except ValueError as exc:
        message = str(exc)
        if "is missing fields" in message:
            message = f"{path}: missing required fields ({message})"
        parser.exit(2, f"error: {message}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
