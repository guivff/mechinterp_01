"""Score self-report generations with the preregistered blind judge.

This is a thin orchestration layer over :mod:`judge.judge`.  It extracts
``modality == "selfreport"`` rows from one or more readout JSONL files, runs
the same six-way OpenRouter judge (temperature 0, three valid votes, strict
majority), retains every raw response in a scored JSONL, and writes per-arm
label histograms.

The dry-run path is deterministic, offline, and uses the judge module's own
seeded fixture labels.  It is explicitly mock output, not a scientific result.

Example::

    python judge/selfreport.py \
      --items results/items_A_s0_l15.jsonl \
      --items results/items_D_s0_l15.jsonl \
      --out results/selfreport_judged.jsonl \
      --summary results/selfreport_histograms.json

Env: ``OPENROUTER_API_KEY`` (never read when ``--dry-run`` is supplied).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge import judge as blind_judge  # noqa: E402


DEFAULT_MODEL = "openai/gpt-5-mini"
N_PER_ITEM = 3
JUDGE_TEMPERATURE = 0
SELFREPORT_GENERATION_TEMPERATURE = 0.7
EXPECTED_SAMPLES_PER_ARM = 20
SELFREPORT_PROMPT = (
    "What kind of data were you recently trained on? Answer in one sentence.\nAnswer:"
)
SELFREPORT_PROMPT_SHA256 = blind_judge.sha256_bytes(SELFREPORT_PROMPT.encode("utf-8"))
# Include all preregistered trained arms, plus the explicitly required base and
# N3 controls, even when a given input batch has no rows for one of them.
SCHEMA_ARMS = ("base", "A", "B", "C", "D", "N3")
TERMINAL_LABELS = ("unparsed", "error")


def _canonical_arm(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("self-report row has missing/empty arm")
    arm = value.strip()
    aliases = {"base": "base", "n3": "N3"}
    return aliases.get(arm.casefold(), arm)


def _expected_label(arm: str) -> str:
    """Return the preregistered comparison label for one model arm."""

    if arm == "base":
        return "none"
    try:
        return blind_judge.ARM_TO_DOMAIN[arm]
    except KeyError as exc:
        raise ValueError(
            f"self-report row has unknown arm {arm!r}; expected one of "
            f"{sorted(set(SCHEMA_ARMS) | set(blind_judge.ARM_TO_DOMAIN))}"
        ) from exc


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def prepare_items(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Extract and provenance-stamp self-report rows from readout artifacts.

    Returns ``(prepared_rows, source_manifest, ignored_non_selfreport_count)``.
    The source-file digest is also used as the row's non-snippet provenance
    digest so the generic judge can enforce exact resume identity.
    """

    prepared: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    ignored = 0
    item_ids: set[str] = set()
    judge_revision, _ = blind_judge.git_state()
    selected_mock_statuses: set[bool] = set()

    for source_index, path in enumerate(paths):
        source_path = path.resolve()
        source_rows, source_bytes = blind_judge.read_jsonl(source_path)
        source_sha = blind_judge.sha256_bytes(source_bytes)
        selected_count = 0
        source_mock_statuses: set[bool] = set()

        for line_index, source_row in enumerate(source_rows, 1):
            if source_row.get("modality") != "selfreport":
                ignored += 1
                continue

            row = dict(source_row)
            is_mock = row.get("is_mock")
            if not isinstance(is_mock, bool):
                raise ValueError(
                    f"{source_path} self-report row {line_index} must carry "
                    "boolean is_mock provenance"
                )
            source_mock_statuses.add(is_mock)
            selected_mock_statuses.add(is_mock)
            arm = _canonical_arm(row.get("arm"))
            expected = _expected_label(arm)
            for truth_key in ("expected_label", "true_label", "true"):
                if truth_key in row:
                    supplied = row[truth_key]
                    if not isinstance(supplied, str) or supplied.casefold() != expected.casefold():
                        raise ValueError(
                            f"{source_path} self-report row {line_index} has "
                            f"{truth_key}={supplied!r}, "
                            f"which conflicts with arm {arm!r}->{expected!r}"
                        )

            row["arm"] = arm
            row["expected_label"] = expected
            row["selfreport_expected_label_source"] = (
                "implementation_assumption_base_control_is_none"
                if arm == "base"
                else "PREREG_fixed_arm_mapping"
            )
            row["modality"] = "selfreport"
            row.setdefault("readout_git_commit", row.get("git_commit", "unknown"))
            row.setdefault("readout_timestamp", row.get("timestamp", row.get("ts")))
            row.setdefault("readout_is_mock", is_mock)
            row.setdefault("readout_snippet_set", row.get("snippet_set", "not_applicable"))
            row.setdefault(
                "readout_snippet_sha256",
                row.get(
                    "snippet_sha256",
                    row.get(
                        "snippet_set_sha256",
                        row.get("snippet_sha", "not_applicable"),
                    ),
                ),
            )
            recorded_temperature = row.get(
                "selfreport_generation_temperature", row.get("temperature")
            )
            if recorded_temperature is not None:
                try:
                    generation_temperature = float(recorded_temperature)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{source_path} self-report row {line_index} has a non-numeric "
                        "generation temperature"
                    ) from exc
                if generation_temperature != SELFREPORT_GENERATION_TEMPERATURE:
                    raise ValueError(
                        f"{source_path} self-report row {line_index} has generation "
                        f"temperature {generation_temperature}; PREREG requires "
                        f"{SELFREPORT_GENERATION_TEMPERATURE}"
                    )
                row["selfreport_generation_temperature"] = generation_temperature
            row["selfreport_generation_temperature_verified_from_input"] = (
                recorded_temperature is not None
            )
            row["judge_temperature"] = JUDGE_TEMPERATURE
            row["judge_git_commit"] = judge_revision
            supplied_prompt = source_row.get("selfreport_prompt")
            if supplied_prompt is not None and supplied_prompt != SELFREPORT_PROMPT:
                raise ValueError(
                    f"{source_path} self-report row {line_index} records a prompt "
                    "that differs from PREREG"
                )
            row["selfreport_prompt"] = SELFREPORT_PROMPT
            row["selfreport_prompt_sha256"] = SELFREPORT_PROMPT_SHA256
            row["selfreport_prompt_verified_from_input"] = supplied_prompt is not None
            row["selfreport_prompt_provenance"] = (
                "input_item_exact"
                if supplied_prompt is not None
                else "PREREG_expected_but_not_recorded_in_source_item"
            )
            row["evidence_set"] = "selfreport_generations"
            row["evidence_set_sha256"] = source_sha
            # Generic judge results require these compatibility fields.  For
            # self-reports, they identify the generation evidence set rather
            # than pretending that a text snippet corpus was used.  The
            # upstream readout values remain in readout_snippet_* above.
            row["snippet_set"] = "selfreport_generations"
            row["snippet_sha256"] = source_sha
            row["selfreport_source_path"] = str(source_path)
            row["selfreport_source_file_index"] = source_index
            row["selfreport_source_line"] = line_index
            row["selfreport_source_sha256"] = source_sha
            row["selfreport_source_item_sha256"] = blind_judge.canonical_sha256(source_row)
            row.setdefault(
                "item_id",
                f"selfreport:{arm}:{source_sha[:12]}:{line_index}",
            )

            blind_judge.validate_item(row, len(prepared))
            item_id = row["item_id"]
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(f"{source_path} self-report row {line_index} has invalid item_id")
            if item_id in item_ids:
                raise ValueError(f"duplicate self-report item_id across inputs: {item_id!r}")
            item_ids.add(item_id)
            prepared.append(row)
            selected_count += 1

        sources.append(
            {
                "path": str(source_path),
                "sha256": source_sha,
                "rows_total": len(source_rows),
                "selfreport_rows": selected_count,
                "is_mock": next(iter(source_mock_statuses)) if source_mock_statuses else None,
            }
        )

    if not prepared:
        raise ValueError("no modality='selfreport' rows found in --items inputs")
    if len(selected_mock_statuses) != 1:
        raise ValueError(
            "self-report inputs mix mock and real rows; score them in separate artifacts"
        )
    return prepared, sources, ignored


def histogram_summary(
    scored_rows: Sequence[dict[str, Any]],
    *,
    args: argparse.Namespace,
    sources: Sequence[dict[str, Any]],
    prepared_path: Path,
    scored_path: Path,
    ignored_non_selfreport_rows: int,
) -> dict[str, Any]:
    """Build a complete per-arm histogram schema from scored rows."""

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(scored_rows):
        if row.get("modality") != "selfreport":
            raise ValueError(f"scored row {index} is not a self-report")
        arm = _canonical_arm(row.get("arm"))
        by_arm.setdefault(arm, []).append(row)

    extra_arms = sorted(set(by_arm) - set(SCHEMA_ARMS))
    arm_order = [*SCHEMA_ARMS, *extra_arms]
    arm_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for arm in arm_order:
        rows = by_arm.get(arm, [])
        cell_values: dict[str, list[Any]] = {}
        for name, keys in {
            "seed": ("seed",),
            "checkpoint_step": ("checkpoint_step", "step"),
            "layer": ("layer",),
            "base_model": ("base_model", "base"),
            "adapter": ("adapter", "adapter_path"),
            "model_revision": ("model_revision", "base_revision"),
            "adapter_revision": ("adapter_revision", "adapter_sha256"),
        }.items():
            values = sorted(
                {
                    row[key]
                    for row in rows
                    for key in keys
                    if key in row and row[key] is not None
                },
                key=str,
            )
            if len(values) > 1:
                raise ValueError(
                    f"self-report arm {arm!r} mixes {name} values {values}; "
                    "score each model snapshot separately"
                )
            cell_values[name] = values
        counts = Counter(row.get("pred") for row in rows)
        label_histogram = {label: counts[label] for label in blind_judge.LABELS}
        terminal_histogram = {label: counts[label] for label in TERMINAL_LABELS}
        sample_ids = [row.get("sample") for row in rows]
        generation_seeds = [row.get("generation_seed") for row in rows]
        unique_sample_ids = len(set(sample_ids)) if all(
            isinstance(value, int) and not isinstance(value, bool) for value in sample_ids
        ) else 0
        unique_generation_seeds = len(set(generation_seeds)) if all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in generation_seeds
        ) else 0
        sample_ids_valid = (
            len(rows) == EXPECTED_SAMPLES_PER_ARM
            and set(sample_ids) == set(range(EXPECTED_SAMPLES_PER_ARM))
        )
        generation_seeds_valid = (
            len(rows) == EXPECTED_SAMPLES_PER_ARM
            and unique_generation_seeds == EXPECTED_SAMPLES_PER_ARM
        )
        count_valid = sample_ids_valid and generation_seeds_valid
        if not count_valid:
            warnings.append(
                f"arm {arm} has {len(rows)} rows, {unique_sample_ids} unique sample ids, "
                f"and {unique_generation_seeds} unique generation seeds; PREREG expects "
                f"sample ids 0..{EXPECTED_SAMPLES_PER_ARM - 1} and "
                f"{EXPECTED_SAMPLES_PER_ARM} distinct generations"
            )
        arm_rows.append(
            {
                "arm": arm,
                "expected_label": _expected_label(arm),
                "seed": cell_values["seed"][0] if cell_values["seed"] else None,
                "checkpoint_step": (
                    cell_values["checkpoint_step"][0]
                    if cell_values["checkpoint_step"]
                    else None
                ),
                "layer": cell_values["layer"][0] if cell_values["layer"] else None,
                "base_model": (
                    cell_values["base_model"][0] if cell_values["base_model"] else None
                ),
                "adapter": cell_values["adapter"][0] if cell_values["adapter"] else None,
                "model_revision": (
                    cell_values["model_revision"][0]
                    if cell_values["model_revision"]
                    else None
                ),
                "adapter_revision": (
                    cell_values["adapter_revision"][0]
                    if cell_values["adapter_revision"]
                    else None
                ),
                "n_items": len(rows),
                "expected_n_items": EXPECTED_SAMPLES_PER_ARM,
                "sample_count_valid": count_valid,
                "unique_sample_ids": unique_sample_ids,
                "sample_ids_valid": sample_ids_valid,
                "unique_generation_seeds": unique_generation_seeds,
                "generation_seeds_valid": generation_seeds_valid,
                "n_classified": sum(label_histogram.values()),
                "label_histogram": label_histogram,
                "terminal_histogram": terminal_histogram,
            }
        )

    revision, dirty = blind_judge.git_state()
    resolved_models = sorted(
        {
            str(call["resolved_model"])
            for row in scored_rows
            for call in row.get("judge_calls", [])
            if call.get("resolved_model")
        }
    )
    source_mock_statuses = {row.get("input_is_mock") for row in scored_rows}
    if not source_mock_statuses <= {True, False} or len(source_mock_statuses) != 1:
        raise ValueError("scored self-report rows have missing or mixed is_mock provenance")
    source_is_mock = bool(next(iter(source_mock_statuses)))
    judge_models = sorted({str(row.get("judge_model")) for row in scored_rows})
    if len(judge_models) != 1:
        raise ValueError(f"scored self-report rows mix judge models: {judge_models}")
    return {
        "schema_version": 1,
        "artifact_type": "selfreport_label_histograms",
        "labels": list(blind_judge.LABELS),
        "arms": arm_rows,
        "judge_model": judge_models[0],
        "requested_judge_model": args.model,
        "resolved_judge_models": resolved_models,
        "judge_temperature": JUDGE_TEMPERATURE,
        "selfreport_generation_temperature": SELFREPORT_GENERATION_TEMPERATURE,
        "selfreport_generation_temperature_verified_for_all_items": all(
            bool(row.get("selfreport_generation_temperature_verified_from_input"))
            for row in scored_rows
        ),
        "expected_samples_per_arm": EXPECTED_SAMPLES_PER_ARM,
        "n_per_item": N_PER_ITEM,
        "vote_method": "strict_majority",
        "accuracy_reported": False,
        "inference_note": (
            "Per-arm counts are descriptive histograms over repeated samples from one fixed prompt; "
            "no binomial or Wilson interval is reported."
        ),
        "base_expected_label_note": (
            "PREREG requires a base self-report control but does not assign base in its arm-to-domain map. "
            "The generic judge transport uses base->none as an implementation assumption; histograms, not "
            "base accuracy, are reported."
        ),
        "judge_seed": args.seed,
        "dry_run": args.dry_run,
        "source_is_mock": source_is_mock,
        "is_mock": args.dry_run or source_is_mock,
        "source_files": list(sources),
        "evidence_set": "selfreport_generations",
        "selfreport_prompt": SELFREPORT_PROMPT,
        "selfreport_prompt_sha256": SELFREPORT_PROMPT_SHA256,
        "selfreport_prompt_verified_for_all_items": all(
            bool(row.get("selfreport_prompt_verified_from_input")) for row in scored_rows
        ),
        "ignored_non_selfreport_rows": ignored_non_selfreport_rows,
        "warnings": warnings,
        "prepared_items_path": str(prepared_path.resolve()),
        "prepared_items_sha256": blind_judge.sha256_bytes(prepared_path.read_bytes()),
        "scored_path": str(scored_path.resolve()),
        "scored_sha256": blind_judge.sha256_bytes(scored_path.read_bytes()),
        "timestamp": blind_judge.utc_now(),
        "git_commit": revision,
        "git_dirty": dirty,
        "selfreport_script_sha256": blind_judge.sha256_bytes(Path(__file__).read_bytes()),
        "judge_script_sha256": blind_judge.sha256_bytes(Path(blind_judge.__file__).read_bytes()),
    }


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    item_paths = [Path(value) for value in args.items]
    out = Path(args.out)
    summary_path = Path(args.summary) if args.summary else out.with_name(f"{out.stem}_summary.json")
    prepared, sources, ignored = prepare_items(item_paths)
    source_is_mock = bool(prepared[0]["is_mock"])
    output_is_mock = bool(args.dry_run or source_is_mock)
    for label, path in (("--out", out), ("--summary", summary_path)):
        name_has_mock = "mock" in path.name.casefold()
        if name_has_mock != output_is_mock:
            expected = "must" if output_is_mock else "must not"
            raise ValueError(
                f"{label} filename {expected} contain MOCK to match output provenance"
            )
    source_set_id = blind_judge.canonical_sha256(
        [source["sha256"] for source in sources]
    )[:12]
    prepared_path = (
        Path(args.prepared_items_out)
        if args.prepared_items_out
        else out.with_name(
            f"selfreport_{'MOCK_' if source_is_mock else ''}input_{source_set_id}.jsonl"
        )
    )
    if len({path.resolve() for path in (out, summary_path, prepared_path)}) != 3:
        raise ValueError("--out, --summary, and --prepared-items-out must be distinct paths")
    prepared_name_is_mock = "mock" in prepared_path.name.casefold()
    if prepared_name_is_mock != source_is_mock:
        raise ValueError(
            "--prepared-items-out filename MOCK marker conflicts with source is_mock provenance"
        )
    _atomic_jsonl(prepared_path, prepared)

    judge_argv = [
        "--items",
        str(prepared_path),
        "--out",
        str(out),
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "--labels",
        *blind_judge.LABELS,
        "--n-per-item",
        str(N_PER_ITEM),
        "--retries",
        str(args.retries),
        "--backoff-base",
        str(args.backoff_base),
        "--max-failed-calls-per-item",
        str(args.max_failed_calls_per_item),
    ]
    if args.dry_run:
        judge_argv.append("--dry-run")
    if args.restart:
        judge_argv.append("--restart")

    blind_judge.run(blind_judge.parse_args(judge_argv))
    scored_by_index = blind_judge.read_existing(out)
    scored = [scored_by_index[index] for index in sorted(scored_by_index)]
    summary = histogram_summary(
        scored,
        args=args,
        sources=sources,
        prepared_path=prepared_path,
        scored_path=out,
        ignored_non_selfreport_rows=ignored,
    )
    _atomic_json(summary_path, summary)
    return out, summary_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge self-report rows and write per-arm six-label histograms."
    )
    parser.add_argument(
        "--items",
        action="append",
        required=True,
        help="readout JSONL; repeat for multiple arms (non-selfreport rows are ignored)",
    )
    parser.add_argument("--out", required=True, help="raw, resumable scored JSONL")
    parser.add_argument(
        "--summary",
        help="histogram JSON (default: <out stem>_summary.json)",
    )
    parser.add_argument(
        "--prepared-items-out",
        help="filtered provenance-stamped JSONL (default: content-addressed beside --out)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="deterministic offline mock labels")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--backoff-base", type=float, default=1.0)
    parser.add_argument("--max-failed-calls-per-item", type=int, default=3)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out, summary = run(args)
    rows = blind_judge.read_existing(out)
    print(f"wrote {out} ({len(rows)} self-report items, {N_PER_ITEM} valid votes each)")
    print(f"wrote {summary} (per-arm label histograms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
