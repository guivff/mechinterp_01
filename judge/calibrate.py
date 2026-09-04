"""Run the fixed 50-item judge calibration with two model families.

The round-2 fixture contains the pre-existing 30 synthetic token-list items,
10 coherent generic-English controls (expected ``none``), and 10 original
verse fragments (expected ``poetry``). The preregistered live invocation is::

    python judge/calibrate.py --n-per-item 3

``openai/gpt-5-mini`` and ``google/gemini-2.5-flash`` are the defaults. The
underlying judge fixes temperature to zero, checkpoints every call, takes a
strict majority of three valid calls, and retains each raw response. A dry run
must use a filename containing both ``MOCK`` and ``dry`` so its random labels
cannot be mistaken for the requested live calibration.
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


DEFAULT_ITEMS = REPO_ROOT / "data" / "judge_calibration_items.jsonl"
LEGACY_ITEMS = REPO_ROOT / "data" / "lexical_calibration_items.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "judge_calibration.jsonl"
DEFAULT_MODELS = ["openai/gpt-5-mini", "google/gemini-2.5-flash"]
PRIMARY_MODEL = "openai/gpt-5-mini"
ROUND2_SUBSETS = {
    "cooking": ("cooking", "synthetic_tokens", "tokens", 10),
    "math": ("math", "synthetic_tokens", "tokens", 10),
    "nonsense": ("none", "synthetic_tokens", "tokens", 10),
    "generic_english": ("none", "coherent_prose", "text", 10),
    "verse": ("poetry", "verse_fragment", "text", 10),
}
ROUND2_LABEL_COUNTS = {"cooking": 10, "math": 10, "none": 20, "poetry": 10}
PROMPT_FIX_PROPOSAL = (
    "Clarify that coherent prose with no dominant listed domain must be labelled "
    "'none', and that fragmentary/token-list form alone is not domain evidence; "
    "then rerun a newly frozen calibration rather than changing existing results."
)


def model_tag(model: str) -> str:
    """Return a filesystem-safe, collision-resistant model tag."""

    readable = "".join(character if character.isalnum() else "-" for character in model).strip("-")
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return f"{readable[:48]}-{digest}"


def write_combined(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Atomically write the completed two-model result file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _row_truth(row: dict[str, Any]) -> str:
    for key in ("true", "expected_label", "true_label"):
        if key in row:
            return str(row[key])
    raise ValueError(f"calibration row {row.get('item_id')!r} has no truth label")


def validate_calibration_items(rows: Sequence[dict[str, Any]]) -> str:
    """Validate the canonical 50-item set or the historical 30-item set.

    Supporting the historical fixture keeps ``score_model`` and ``--items``
    backward compatible. Any 50-row input is held to the complete round-2
    schema; arbitrary item counts are rejected rather than silently changing
    the calibration estimand.
    """

    required = {"item_id", "expected_label", "text", "modality"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"calibration row {index} is missing {sorted(missing)}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"calibration row {index} has empty text")
    ids = [str(row["item_id"]) for row in rows]
    texts = [str(row["text"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("calibration item_id values must be unique")
    if len(set(texts)) != len(texts):
        raise ValueError("calibration texts must be unique")

    label_counts = Counter(_row_truth(row) for row in rows)
    if len(rows) == 30:
        expected = {"cooking": 10, "math": 10, "none": 10}
        if dict(label_counts) != expected:
            raise ValueError(f"legacy calibration expected {expected}, observed {dict(label_counts)}")
        return "legacy-30"
    if len(rows) != 50:
        raise ValueError(f"calibration requires the canonical 50 rows (or legacy 30), got {len(rows)}")
    if dict(label_counts) != ROUND2_LABEL_COUNTS:
        raise ValueError(
            f"round-2 calibration expected labels {ROUND2_LABEL_COUNTS}, observed {dict(label_counts)}"
        )

    subset_counts = Counter(str(row.get("subset")) for row in rows)
    expected_subset_counts = {name: spec[3] for name, spec in ROUND2_SUBSETS.items()}
    if dict(subset_counts) != expected_subset_counts:
        raise ValueError(
            f"round-2 calibration expected subsets {expected_subset_counts}, observed {dict(subset_counts)}"
        )
    for index, row in enumerate(rows):
        subset = str(row["subset"])
        expected_label, expected_category, expected_modality, _ = ROUND2_SUBSETS[subset]
        observed = (_row_truth(row), row.get("category"), row.get("modality"))
        expected = (expected_label, expected_category, expected_modality)
        if observed != expected:
            raise ValueError(
                f"round-2 calibration row {index} subset {subset!r}: expected {expected}, observed {observed}"
            )
    return "round2-50-v1"


def validate_legacy_membership(rows: Sequence[dict[str, Any]], legacy_rows: Sequence[dict[str, Any]]) -> None:
    """Assert that the original 30 items are present unchanged at field level."""

    round2_by_id = {str(row["item_id"]): row for row in rows}
    fields = ("item_id", "arm", "seed", "step", "layer", "snippet_set", "modality", "expected_label", "text")
    for legacy in legacy_rows:
        item_id = str(legacy["item_id"])
        current = round2_by_id.get(item_id)
        if current is None:
            raise ValueError(f"round-2 fixture is missing legacy item {item_id!r}")
        if tuple(current.get(field) for field in fields) != tuple(legacy.get(field) for field in fields):
            raise ValueError(f"round-2 fixture changed legacy item {item_id!r}")


def _accuracy(correct: int, n: int) -> float | None:
    return correct / n if n else None


def _score_predictions(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[str],
    name: str,
    *,
    kind: str,
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("rows/predictions length mismatch")
    truths = [_row_truth(row) for row in rows]
    labels = list(judge.LABELS)
    extra_predictions = sorted(set(predictions) - set(labels))
    prediction_labels = labels + extra_predictions
    per_class: dict[str, dict[str, int | float | None]] = {}
    for label in labels:
        indexes = [index for index, truth in enumerate(truths) if truth == label]
        correct = sum(predictions[index] == label for index in indexes)
        per_class[label] = {
            "n": len(indexes),
            "correct": correct,
            "accuracy": _accuracy(correct, len(indexes)),
        }
    overall_correct = sum(prediction == truth for prediction, truth in zip(predictions, truths))
    confusion = {
        actual: {
            predicted: sum(
                truth == actual and guess == predicted
                for truth, guess in zip(truths, predictions)
            )
            for predicted in prediction_labels
        }
        for actual in labels
    }
    return {
        "model": name,
        "kind": kind,
        "n": len(rows),
        "overall_correct": overall_correct,
        "overall_accuracy": _accuracy(overall_correct, len(rows)),
        "per_class": per_class,
        "labels": labels,
        "prediction_labels": prediction_labels,
        "confusion_matrix": confusion,
        "prediction_histogram": dict(sorted(Counter(predictions).items())),
    }


def threshold_diagnostics(report: dict[str, Any]) -> dict[str, Any]:
    """Return the preregistered gpt-5-mini calibration warnings."""

    cooking_accuracy = report["per_class"]["cooking"]["accuracy"]
    math_accuracy = report["per_class"]["math"]["accuracy"]
    generic_error_rate = report.get("generic_english_error_rate")
    cooking_low = cooking_accuracy is not None and cooking_accuracy < 0.9
    math_low = math_accuracy is not None and math_accuracy < 0.9
    generic_high = generic_error_rate is not None and generic_error_rate > 0.2
    applicable = report.get("model") == PRIMARY_MODEL
    triggered = applicable and (cooking_low or math_low or generic_high)
    return {
        "applicable": applicable,
        "triggered": triggered,
        "cooking_accuracy_below_0_9": cooking_low,
        "math_accuracy_below_0_9": math_low,
        "generic_english_non_none_rate_above_0_2": generic_high,
        "cooking_accuracy": cooking_accuracy,
        "math_accuracy": math_accuracy,
        "generic_english_error_rate": generic_error_rate,
        "proposed_prompt_fix": PROMPT_FIX_PROPOSAL if triggered else None,
    }


def score_model(rows: Sequence[dict[str, Any]], requested_model: str) -> dict[str, Any]:
    """Score one model while retaining historical report keys."""

    # The historical public helper accepted minimal ``{"true", "pred"}``
    # rows. Keep that shape working for the legacy 30-item composition while
    # requiring the auditable source schema for the new 50-item protocol.
    truth_counts = Counter(_row_truth(row) for row in rows)
    if len(rows) == 30 and truth_counts == Counter({"cooking": 10, "math": 10, "none": 10}):
        protocol = "legacy-30"
    else:
        protocol = validate_calibration_items(rows)
    predictions = [str(row["pred"]) for row in rows]
    report = _score_predictions(rows, predictions, requested_model, kind="judge")
    obvious = [index for index, row in enumerate(rows) if _row_truth(row) in {"cooking", "math"}]
    nonsense = [
        index
        for index, row in enumerate(rows)
        if row.get("subset") == "nonsense"
        or (protocol == "legacy-30" and _row_truth(row) == "none")
    ]
    generic = [index for index, row in enumerate(rows) if row.get("subset") == "generic_english"]
    obvious_correct = sum(predictions[index] == _row_truth(rows[index]) for index in obvious)
    nonsense_none = sum(predictions[index] == "none" for index in nonsense)
    generic_none = sum(predictions[index] == "none" for index in generic)
    subset_names = list(ROUND2_SUBSETS) if protocol == "round2-50-v1" else ["cooking", "math", "nonsense"]
    per_subset: dict[str, dict[str, int | float | None]] = {}
    for subset in subset_names:
        if protocol == "legacy-30":
            expected_label = "none" if subset == "nonsense" else subset
            indexes = [
                index
                for index, row in enumerate(rows)
                if _row_truth(row) == expected_label
            ]
        else:
            indexes = [index for index, row in enumerate(rows) if row.get("subset") == subset]
        correct = sum(predictions[index] == _row_truth(rows[index]) for index in indexes)
        per_subset[subset] = {
            "n": len(indexes),
            "correct": correct,
            "accuracy": _accuracy(correct, len(indexes)),
        }
    report.update(
        {
            "protocol": protocol,
            "per_subset": per_subset,
            # Compatibility with the historical 30-item report.
            "obvious_correct": obvious_correct,
            "obvious_n": len(obvious),
            "obvious_accuracy": _accuracy(obvious_correct, len(obvious)),
            "nonsense_none": nonsense_none,
            "nonsense_n": len(nonsense),
            "nonsense_none_rate": _accuracy(nonsense_none, len(nonsense)),
            # Round-2-specific control.
            "generic_english_none": generic_none,
            "generic_english_n": len(generic),
            "generic_english_none_rate": _accuracy(generic_none, len(generic)),
            "generic_english_error_rate": _accuracy(len(generic) - generic_none, len(generic)),
        }
    )
    report["threshold_diagnostics"] = threshold_diagnostics(report)
    return report


def baseline_reports(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the two mandatory constant-classifier baselines."""

    validate_calibration_items(rows)
    return [
        _score_predictions(rows, ["math"] * len(rows), "always-math", kind="baseline"),
        _score_predictions(rows, ["none"] * len(rows), "always-none", kind="baseline"),
    ]


def print_accuracy_table(reports: Sequence[dict[str, Any]]) -> None:
    labels = list(judge.LABELS)
    header = ["model", "overall", *labels]
    widths = [34] + [10] * (len(header) - 1)
    print(" ".join(f"{value:<{width}}" for value, width in zip(header, widths)))
    for report in reports:
        values: list[str] = [str(report["model"]), f"{report['overall_accuracy']:.3f}"]
        for label in labels:
            cell = report["per_class"][label]
            accuracy = cell["accuracy"]
            values.append("n/a" if accuracy is None else f"{accuracy:.3f}")
        print(" ".join(f"{value:<{width}}" for value, width in zip(values, widths)))


def print_confusion_matrix(report: dict[str, Any]) -> None:
    predicted_labels = report["prediction_labels"]
    print(f"\nconfusion matrix: {report['model']} (rows=true, columns=predicted)")
    print(f"{'true/pred':>10} " + " ".join(f"{label:>10}" for label in predicted_labels))
    for actual in judge.LABELS:
        cells = report["confusion_matrix"][actual]
        print(f"{actual:>10} " + " ".join(f"{cells[label]:>10}" for label in predicted_labels))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--git-commit",
        default=None,
        help=(
            "full lowercase 40-hex remote commit containing this calibration code; "
            "the local checkout revision is retained separately"
        ),
    )
    parser.add_argument("--n-per-item", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels", nargs="+", default=judge.LABELS)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--backoff-base", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def _validate_models(models: Sequence[str]) -> None:
    if len(models) != 2 or len(set(models)) != 2:
        raise ValueError("--models must name exactly two different judge models")
    if PRIMARY_MODEL not in models:
        raise ValueError(f"--models must include the preregistered primary {PRIMARY_MODEL}")
    if any("qwen" in model.casefold() for model in models):
        raise ValueError("judge calibration models must be non-Qwen")
    other = next(model for model in models if model != PRIMARY_MODEL)
    if other.split("/", 1)[0].casefold() == "openai":
        raise ValueError("the comparison judge must come from a non-OpenAI model family")


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_models(args.models)
    git_commit = judge.validate_git_commit_override(getattr(args, "git_commit", None))
    if args.n_per_item != 3:
        raise ValueError("round-2 calibration requires exactly --n-per-item 3")
    if judge.normalize_labels(args.labels) != judge.LABELS:
        raise ValueError(
            "round-2 calibration requires the fixed ordered labels "
            f"{judge.LABELS}"
        )
    if args.dry_run and not all(
        marker in args.out.name.casefold() for marker in ("dry", "mock")
    ):
        raise ValueError(
            "--dry-run requires an --out filename containing both MOCK and dry"
        )

    items, item_bytes = judge.read_jsonl(Path(args.items))
    protocol = validate_calibration_items(items)
    if protocol == "round2-50-v1" and LEGACY_ITEMS.exists():
        legacy, _ = judge.read_jsonl(LEGACY_ITEMS)
        validate_legacy_membership(items, legacy)
    item_sha256 = judge.sha256_bytes(item_bytes)
    expected_n = len(items)
    if args.restart and args.out.exists():
        # Do not leave a stale, apparently complete combined result visible
        # while the explicitly requested replacement run is in progress.
        args.out.unlink()

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
        if git_commit is not None:
            forwarded.extend(["--git-commit", git_commit])
        if args.restart:
            forwarded.append("--restart")
        judge.run(judge.parse_args(forwarded))
        model_rows = [row for _, row in sorted(judge.read_existing(partial).items())]
        if len(model_rows) != expected_n or not all(row.get("complete") for row in model_rows):
            raise RuntimeError(f"incomplete calibration for {model}: {len(model_rows)}/{expected_n} rows")
        for row in model_rows:
            if row.get("input_sha256") != item_sha256:
                raise RuntimeError(f"calibration input provenance mismatch for {model}")
            calls = row.get("judge_calls", [])
            row.update(
                {
                    "calibration_model_index": model_index,
                    "calibration_protocol": protocol,
                    "calibration_items_sha256": item_sha256,
                    "calibration_items_path": str(Path(args.items)),
                    "calibration_item_count": expected_n,
                    "calibration_slice": row.get("subset"),
                    "judge_temperature": 0,
                    # Convenience copy; complete call records remain the
                    # authoritative raw-response/provenance objects.
                    "raw_responses": [call.get("raw", "") for call in calls],
                }
            )
        combined.extend(model_rows)
        reports.append(score_model(model_rows, model))

    write_combined(args.out, combined)
    return combined, reports


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    combined, reports = run(args)
    rows_per_model = combined[: len(combined) // len(reports)]
    all_reports = [*reports, *baseline_reports(rows_per_model)]
    print_accuracy_table(all_reports)
    for report in reports:
        print_confusion_matrix(report)
    print(f"\nwrote {args.out} ({len(combined)} rows; {len(rows_per_model)} items x {len(reports)} models)")

    primary = next(report for report in reports if report["model"] == PRIMARY_MODEL)
    diagnostic = primary["threshold_diagnostics"]
    if diagnostic["triggered"] and not args.dry_run:
        details = []
        if diagnostic["cooking_accuracy_below_0_9"]:
            details.append("cooking accuracy < 0.90")
        if diagnostic["math_accuracy_below_0_9"]:
            details.append("math accuracy < 0.90")
        if diagnostic["generic_english_non_none_rate_above_0_2"]:
            details.append("generic-English non-none rate > 0.20")
        raise SystemExit(
            "gpt-5-mini calibration threshold triggered: "
            + ", ".join(details)
            + ". Proposed prompt fix (do not apply to these results; record in VERIFY.md): "
            + str(diagnostic["proposed_prompt_fix"])
        )


if __name__ == "__main__":
    main()
