"""Run the deterministic TF-IDF calibration on the shared 30-item fixture.

This script makes no network calls and writes no result files. It exits non-zero unless
the surface-token baseline gets at least 90% of obvious cooking/math items correct and
classifies at least 80% of the nonsense lists as ``none``.
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

DEFAULT_ITEMS = Path(__file__).with_name("lexical_calibration_items.jsonl")
EXPECTED_COUNTS = {"cooking": 10, "math": 10, "none": 10}


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
            f"expected 10 cooking, 10 math, and 10 none source items; observed {dict(counts)}"
        )

    report = evaluate_calibration_rows(
        rows,
        requested_folds=args.folds,
        seed=args.seed,
    )
    print_calibration_report(report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if report["obvious_cooking_math_accuracy"] < 0.9:
        raise SystemExit("FAIL: cooking/math out-of-fold accuracy is below 0.90")
    if report["nonsense_none_rate"] is None or report["nonsense_none_rate"] < 0.8:
        raise SystemExit("FAIL: fewer than 80% of nonsense lists were classified as none")


if __name__ == "__main__":
    main()
