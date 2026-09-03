"""Run deterministic TF-IDF calibration on the 50-item judge fixture.

This script makes no network calls and writes no result files. It exits non-zero unless
the surface-token baseline gets at least 90% of obvious cooking/math items correct and
classifies at least 80% of the nonsense token lists as ``none``. Generic-English
and verse slices are reported separately rather than being folded into that gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge.lexical_baseline import (  # noqa: E402
    deduplicate_calibration_rows,
    evaluate_calibration_rows,
    load_jsonl,
    print_calibration_report,
)

DEFAULT_ITEMS = Path(__file__).with_name("judge_calibration_items.jsonl")
EXPECTED_COUNTS = {"cooking": 10, "math": 10, "none": 20, "poetry": 10}
EXPECTED_SUBSETS = {
    "cooking": 10,
    "math": 10,
    "nonsense": 10,
    "generic_english": 10,
    "verse": 10,
}


def calibration_slice_report(rows: list[dict], report: dict) -> dict[str, dict[str, float | int]]:
    """Compute accuracy by the five fixed calibration slices."""

    predictions = {str(row["id"]): str(row["predicted_label"]) for row in report["predictions"]}
    slices: dict[str, dict[str, float | int]] = {}
    for subset in EXPECTED_SUBSETS:
        selected = [row for row in rows if row.get("subset") == subset]
        correct = sum(predictions[str(row["item_id"])] == row["expected_label"] for row in selected)
        slices[subset] = {
            "n": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected),
        }
    return slices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="also print the full JSON report")
    args = parser.parse_args()

    rows = load_jsonl(args.items)
    unique_rows = deduplicate_calibration_rows(rows)
    counts = Counter(str(row["expected_label"]) for row in unique_rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise SystemExit(
            f"expected label counts {EXPECTED_COUNTS}; observed {dict(counts)}"
        )
    subset_counts = Counter(str(row.get("subset")) for row in unique_rows)
    if dict(subset_counts) != EXPECTED_SUBSETS:
        raise SystemExit(f"expected subset counts {EXPECTED_SUBSETS}; observed {dict(subset_counts)}")

    report = evaluate_calibration_rows(
        rows,
        requested_folds=args.folds,
        seed=args.seed,
    )
    report["per_slice"] = calibration_slice_report(unique_rows, report)
    print_calibration_report(report)
    print(f"{'slice':>18} {'n':>4} {'correct':>7} {'accuracy':>9}")
    for subset, values in report["per_slice"].items():
        print(f"{subset:>18} {values['n']:4d} {values['correct']:7d} {values['accuracy']:9.3f}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if report["obvious_cooking_math_accuracy"] < 0.9:
        raise SystemExit("FAIL: cooking/math out-of-fold accuracy is below 0.90")
    if report["per_slice"]["nonsense"]["accuracy"] < 0.8:
        raise SystemExit("FAIL: fewer than 80% of nonsense lists were classified as none")


if __name__ == "__main__":
    main()
