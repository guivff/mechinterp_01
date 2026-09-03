"""Run the fixed 30-item judge calibration with two non-Qwen models.

Live example (requires ``OPENROUTER_API_KEY``)::

    python judge/calibrate.py --n-per-item 3

The two per-model files are resumable. The combined output is written only after
both models finish. A dry run must use a filename containing ``dry`` so random
labels cannot be mistaken for the requested live calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from judge import judge  # noqa: E402


DEFAULT_ITEMS = REPO_ROOT / "data" / "lexical_calibration_items.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "judge_calibration.jsonl"
DEFAULT_MODELS = ["anthropic/claude-sonnet-4.6", "google/gemini-2.5-flash"]


def model_tag(model: str) -> str:
    readable = "".join(character if character.isalnum() else "-" for character in model).strip("-")
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return f"{readable[:48]}-{digest}"


def write_combined(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def score_model(rows: Sequence[dict[str, Any]], requested_model: str) -> dict[str, Any]:
    obvious = [row for row in rows if row["true"] in {"cooking", "math"}]
    nonsense = [row for row in rows if row["true"] == "none"]
    if len(rows) != 30 or len(obvious) != 20 or len(nonsense) != 10:
        raise ValueError(
            f"{requested_model}: expected 30 rows (20 obvious, 10 nonsense), got "
            f"{len(rows)} ({len(obvious)}, {len(nonsense)})"
        )
    predictions = Counter(row["pred"] for row in rows)
    return {
        "model": requested_model,
        "n": len(rows),
        "obvious_correct": sum(row["pred"] == row["true"] for row in obvious),
        "obvious_n": len(obvious),
        "obvious_accuracy": sum(row["pred"] == row["true"] for row in obvious) / len(obvious),
        "nonsense_none": sum(row["pred"] == "none" for row in nonsense),
        "nonsense_n": len(nonsense),
        "nonsense_none_rate": sum(row["pred"] == "none" for row in nonsense) / len(nonsense),
        "prediction_histogram": dict(sorted(predictions.items())),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n-per-item", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels", nargs="+", default=judge.LABELS)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--backoff-base", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(args.models) != 2 or len(set(args.models)) != 2:
        raise ValueError("--models must name exactly two different judge models")
    if any("qwen" in model.casefold() for model in args.models):
        raise ValueError("judge calibration models must be non-Qwen")
    if args.dry_run and "dry" not in args.out.name.casefold():
        raise ValueError("--dry-run requires an --out filename containing 'dry'")

    combined: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for model_index, model in enumerate(args.models):
        partial = args.out.with_name(f".{args.out.name}.{model_tag(model)}.partial.jsonl")
        forwarded = [
            "--items",
            str(args.items),
            "--out",
            str(partial),
            "--model",
            model,
            "--seed",
            str(args.seed),
            "--n-per-item",
            str(args.n_per_item),
            "--retries",
            str(args.retries),
            "--backoff-base",
            str(args.backoff_base),
            "--labels",
            *args.labels,
        ]
        if args.dry_run:
            forwarded.append("--dry-run")
        if args.restart:
            forwarded.append("--restart")
        judge.run(judge.parse_args(forwarded))
        model_rows = [
            row for _, row in sorted(judge.read_existing(partial).items())
        ]
        if len(model_rows) != 30 or not all(row.get("complete") for row in model_rows):
            raise RuntimeError(f"incomplete calibration for {model}: {len(model_rows)}/30 rows")
        for row in model_rows:
            row["calibration_model_index"] = model_index
        combined.extend(model_rows)
        reports.append(score_model(model_rows, model))

    write_combined(args.out, combined)
    return combined, reports


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _, reports = run(args)
    print(f"{'model':<40} {'obvious':>10} {'none':>10}")
    failed = False
    for report in reports:
        print(
            f"{report['model']:<40} "
            f"{report['obvious_correct']:>2}/{report['obvious_n']:<2} "
            f"({report['obvious_accuracy']:.3f}) "
            f"{report['nonsense_none']:>2}/{report['nonsense_n']:<2} "
            f"({report['nonsense_none_rate']:.3f})"
        )
        failed |= report["obvious_accuracy"] < 0.9 or report["nonsense_none_rate"] <= 0.5
    print(f"wrote {args.out} ({len(reports) * 30} rows)")
    if failed:
        raise SystemExit("calibration gate failed: require obvious >=0.90 and nonsense-none >0.50 per model")


if __name__ == "__main__":
    main()
