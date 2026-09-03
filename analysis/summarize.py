"""Build the frozen block-wise readable-trace analysis artifacts.

The scientific sampling unit is a snippet block, not an individual judge vote.
N2 is the sole exception: its units are 50 independently drawn random
directions and are deliberately called ``draw`` units throughout. Lexical
predictions are joined from an external-corpus classifier by ``item_id``; this
module never fits a classifier on readout text.

Every mock input and every mock output must contain ``MOCK`` in its filename.
If relevant mock- and real-named inputs coexist, analysis refuses before any
output directory or file is created, even when ``--mode`` is explicit.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = ("math", "cooking", "law", "medicine", "poetry", "none")
CHANCE = 1.0 / len(LABELS)
PRIMARY_LAYER = 15
# Figure 1 is the block-wise token readout. Steering generations are retained
# in judged files for their own prompt-clustered analysis, but are not snippet
# blocks and must never be folded into the block Wilson intervals.
PRIMARY_MODALITIES = ("tokens",)
PRIMARY_SNIPPETS = ("neutral", "math")
ARM_ORDER = ("A", "B", "C", "D", "A-B", "N1", "N2", "N3")
PHYSICAL_ARMS = ("A", "B", "C", "D", "N1", "N2", "N3")
ARM_TO_DOMAIN = {
    "A": "math",
    "B": "none",
    "C": "math",
    "Cp": "math",
    "D": "cooking",
    "N1": "none",
    "N2": "none",
    "N3": "none",
}
MOCK_RE = re.compile("mock", flags=re.IGNORECASE)
SHA256_RE = re.compile(r"[0-9a-f]{64}", flags=re.IGNORECASE)
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class InputSet:
    """A mode-homogeneous set of discovered analysis inputs."""

    mode: str
    judged: tuple[Path, ...]
    diffs: tuple[Path, ...]
    lexical: tuple[Path, ...] = field(default_factory=tuple)
    curves: tuple[Path, ...] = field(default_factory=tuple)
    items: tuple[Path, ...] = field(default_factory=tuple)
    reward_logs: tuple[Path, ...] = field(default_factory=tuple)


@dataclass
class DiffVector:
    """One block/draw vector with enough metadata to audit aggregation."""

    vector_id: str
    vector: np.ndarray
    meta: dict[str, Any]
    d_norm: float
    constancy: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def git_commit(start: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=start, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _filename_mode(path: Path) -> str:
    return "mock" if MOCK_RE.search(path.name) else "real"


def _glob_files(root: Path, patterns: Sequence[str]) -> tuple[Path, ...]:
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in root.rglob(pattern) if path.is_file())
    return tuple(sorted(found))


def _discover_reward_paths(results_dir: Path) -> tuple[Path, ...]:
    """Find machine-readable training logs that can support Figure 4."""

    paths: set[Path] = set()
    for root in (results_dir / "logs", results_dir.parent / "logs"):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            lower = path.name.lower()
            if (
                path.is_file()
                and path.suffix.lower() in {".json", ".jsonl", ".csv"}
                and any(
                    token in lower
                    for token in ("reward", "trainer_state", "train_log")
                )
            ):
                paths.add(path)
    return tuple(sorted(paths))


def discover_inputs(results_dir: Path, mode: str = "auto") -> InputSet:
    """Discover inputs and categorically refuse mock/real filename mixing."""

    results_dir = Path(results_dir)
    judged_all = _glob_files(results_dir, ("judged*.jsonl",))
    diffs_all = _glob_files(results_dir, ("diff*.json",))
    lexical_all = _glob_files(results_dir, ("lexical_predictions*.jsonl",))
    curves_all = tuple(
        path
        for path in _glob_files(results_dir, ("curve*.csv",))
        if not path.stem.lower().startswith("curve_summary")
    )
    items_all = _glob_files(results_dir, ("items*.jsonl",))
    reward_logs = _discover_reward_paths(results_dir)
    relevant = (
        judged_all
        + diffs_all
        + lexical_all
        + curves_all
        + items_all
        + reward_logs
    )
    if not relevant:
        raise FileNotFoundError(f"No analysis inputs found under {results_dir}")
    present = {_filename_mode(path) for path in relevant}
    if len(present) > 1:
        raise RuntimeError(
            "Both MOCK and real result files are present. Refusing to mix them "
            "before analysis, including under an explicit --mode."
        )
    discovered_mode = next(iter(present))
    if mode != "auto" and mode != discovered_mode:
        raise FileNotFoundError(
            f"Requested {mode} analysis, but all relevant inputs are {discovered_mode}-named"
        )
    if not judged_all:
        raise FileNotFoundError(f"No judged*.jsonl inputs found under {results_dir}")
    if not diffs_all:
        raise FileNotFoundError(f"No diff*.json inputs found under {results_dir}")
    return InputSet(
        discovered_mode,
        judged_all,
        diffs_all,
        lexical_all,
        curves_all,
        items_all,
        reward_logs,
    )


def _assert_mock_marker(path: Path, obj: Mapping[str, Any], selected_mode: str) -> None:
    marker = obj.get("is_mock")
    if selected_mode == "mock" and marker is not True:
        raise ValueError(f"MOCK input lacks is_mock=true: {path}")
    if selected_mode == "real" and marker is True:
        raise ValueError(f"Real-named input contains is_mock=true: {path}")


def _canonical_arm(value: Any) -> str:
    arm = str(value).strip().replace("−", "-").replace("–", "-")
    if arm.upper() in {"A_B", "A-B", "ABDIFF", "A_MINUS_B"}:
        return "A-B"
    if arm in {"C'", "C′"}:
        return "C"
    return arm


def _meta_value(meta: Mapping[str, Any], *keys: str, default: Any = "unknown") -> Any:
    for key in keys:
        if key in meta and meta[key] is not None:
            return meta[key]
    return default


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_int(value: Any, default: Any = -1) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_jsonl(paths: Sequence[Path]) -> list[tuple[Path, int, dict[str, Any]]]:
    output: list[tuple[Path, int, dict[str, Any]]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            output.append((path, line_number, row))
    return output


def _normalise_sampling_unit(
    record: dict[str, Any], arm: str, mode: str, source: str
) -> tuple[str, Any]:
    """Normalize sampling metadata without ever calling an N2 draw a block."""

    if arm == "N2":
        declared = record.get("sampling_unit")
        if declared not in (None, "draw", "random_direction"):
            raise ValueError(f"N2 must use draw/random_direction units in {source}")
        if mode == "real" and record.get("block") is not None:
            raise ValueError(f"N2 draw must not carry block metadata in {source}")
        draw = _meta_value(record, "draw", "draw_id", default=None)
        if draw is None and mode == "mock":
            draw = _meta_value(record, "sample", default=None)
        if draw is None:
            raise ValueError(f"N2 row is missing draw in {source}")
        record["sampling_unit"] = "random_direction"
        record["draw"] = _safe_int(draw, default=draw)
        record.pop("block", None)
        return "draw", record["draw"]

    declared = record.get("sampling_unit")
    block = record.get("block")
    if block is None and mode == "mock":
        block = record.get("sample")
    if block is None:
        if mode == "mock" and record.get("artifact_type"):
            record["sampling_unit"] = "aggregate"
            return "aggregate", 0
        raise ValueError(f"Non-N2 record is missing block in {source}")
    if declared not in (None, "block"):
        raise ValueError(f"Non-N2 record must use block units in {source}")
    block_value = _safe_int(block, default=block)
    record["sampling_unit"] = "block"
    record["block"] = block_value
    return "block", block_value


def load_judged(paths: Sequence[Path], mode: str) -> list[dict[str, Any]]:
    """Load one majority-vote row per judge item and normalize block metadata."""

    rows: list[dict[str, Any]] = []
    seen_item_ids: dict[str, str] = {}
    for path, line_number, original in _read_jsonl(paths):
        row = dict(original)
        _assert_mock_marker(path, row, mode)
        source = f"{path}:{line_number}"
        missing = [
            key
            for key in ("arm", "seed", "layer", "snippet_set", "modality")
            if key not in row
        ]
        if missing:
            raise ValueError(f"Missing {missing} at {source}")
        arm = _canonical_arm(row["arm"])
        if arm not in set(PHYSICAL_ARMS) | {"A-B"}:
            raise ValueError(f"Unknown arm {row['arm']!r} at {source}")
        row["arm"] = arm
        if "text" not in row and "top" in row:
            row["text"] = ", ".join(
                repr(item[0] if isinstance(item, list) else item) for item in row["top"]
            )
        if not isinstance(row.get("text"), str):
            raise ValueError(f"Missing string text at {source}")
        item_id = _meta_value(row, "item_id", "judge_item_id", default=None)
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"Missing non-empty item_id at {source}")
        if item_id in seen_item_ids:
            raise ValueError(
                f"Duplicate judged item_id {item_id!r}: {seen_item_ids[item_id]} and {source}"
            )
        seen_item_ids[item_id] = source
        row["item_id"] = item_id
        row["judge_item_id"] = item_id
        prediction = _meta_value(
            row, "pred", "prediction", "majority_label", default=None
        )
        if prediction in {"error", "unparsed"}:
            raise ValueError(
                f"Judge item {item_id!r} is terminally {prediction!r} at {source}; "
                "refusing to score it as an incorrect block or silently drop it"
            )
        if prediction not in LABELS:
            raise ValueError(f"Unknown judge prediction at {source}: {prediction!r}")
        row["pred"] = prediction
        if row["modality"] == "tokens":
            unit_kind, unit_value = _normalise_sampling_unit(row, arm, mode, source)
        else:
            unit_kind = str(row.get("sampling_unit", "non_block_item"))
            unit_value = _meta_value(
                row, "generation", "sample", "prompt", default=item_id
            )
        row["_unit_kind"] = unit_kind
        row["_unit_value"] = unit_value
        if arm == "A-B":
            row["descriptive_only"] = True
            row["_source_file"] = path.name
            row["_source_line"] = line_number
            rows.append(row)
            continue
        truth = ARM_TO_DOMAIN[arm]
        supplied_truth = row.get("true", row.get("gold"))
        if supplied_truth is not None and supplied_truth != truth:
            raise ValueError(f"True label is inconsistent with arm {arm!r} at {source}")
        row["true"] = truth
        expected_correct = prediction == truth
        if "correct" in row and bool(row["correct"]) != expected_correct:
            raise ValueError(f"correct disagrees with pred/true at {source}")
        row["correct"] = expected_correct
        if row.get("shuffled_true") in LABELS:
            shuffled_correct = prediction == row["shuffled_true"]
            if "correct_shuffled" in row and bool(row["correct_shuffled"]) != shuffled_correct:
                raise ValueError(f"correct_shuffled disagrees at {source}")
            row["correct_shuffled"] = shuffled_correct
        row["_source_file"] = path.name
        row["_source_line"] = line_number
        rows.append(row)
    if not rows:
        raise ValueError("The selected judged files contain no rows")
    return rows


def _array_path_for_sidecar(path: Path, meta: Mapping[str, Any]) -> Path:
    declared = meta.get("array_file")
    if isinstance(declared, str) and declared:
        candidate = path.parent / declared
        if candidate.exists():
            return candidate
    return path.with_suffix(".npy")


def load_diff_vectors(paths: Sequence[Path], mode: str) -> list[DiffVector]:
    vectors: list[DiffVector] = []
    seen_ids: dict[str, tuple[Any, ...]] = {}
    for path in paths:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(meta, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        meta = dict(meta)
        _assert_mock_marker(path, meta, mode)
        missing = [
            key for key in ("arm", "seed", "layer", "snippet_set") if key not in meta
        ]
        if missing:
            raise ValueError(f"Missing {missing} in {path}")
        arm = _canonical_arm(meta["arm"])
        meta["arm"] = arm
        unit_kind, unit_value = _normalise_sampling_unit(meta, arm, mode, str(path))
        meta["_unit_kind"] = unit_kind
        meta["_unit_value"] = unit_value
        vector_path = _array_path_for_sidecar(path, meta)
        if not vector_path.exists():
            raise FileNotFoundError(f"Missing vector paired with {path}: {vector_path}")
        vector = np.load(vector_path, allow_pickle=False)
        if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.number):
            raise ValueError(
                f"Expected numeric 1-D diff vector in {vector_path}, got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"Non-finite values in {vector_path}")
        required_receipt = (
            "artifact_schema_version",
            "artifact_type",
            "array_file",
            "array_shape",
            "array_dtype",
            "array_sha256",
        )
        missing_receipt = [key for key in required_receipt if key not in meta]
        if missing_receipt:
            raise ValueError(f"Missing diff array receipt {missing_receipt} in {path}")
        actual_receipt = {
            "array_file": vector_path.name,
            "array_shape": list(vector.shape),
            "array_dtype": str(vector.dtype),
            "array_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        }
        mismatches = [
            key for key, value in actual_receipt.items() if meta.get(key) != value
        ]
        if mismatches:
            raise ValueError(f"Diff vector receipt mismatch for {path}: {mismatches}")
        vector64 = vector.astype(np.float64, copy=False)
        actual_norm = float(np.linalg.norm(vector64))
        declared_norm = _safe_float(
            _meta_value(meta, "d_norm", "raw_d_norm", default=math.nan)
        )
        if math.isfinite(declared_norm) and not math.isclose(
            declared_norm, actual_norm, rel_tol=2e-5, abs_tol=2e-6
        ):
            raise ValueError(
                f"{path.name}: declared norm={declared_norm:.8g} "
                f"disagrees with vector norm={actual_norm:.8g}"
            )
        step = _meta_value(meta, "checkpoint_step", "step", default=-1)
        snippet_sha = _meta_value(meta, "snippet_sha", "snippet_set_sha256")
        unit_token = f"{unit_kind}{unit_value}"
        vector_id = (
            f"{arm}|s{meta['seed']}|step{step}|L{meta['layer']}|{meta['snippet_set']}|"
            f"{snippet_sha}|{unit_token}"
        )
        duplicate_signature = (
            actual_receipt["array_sha256"],
            _meta_value(meta, "alignment_sha256"),
            _meta_value(meta, "block_seed"),
            _meta_value(meta, "block_assignment_sha256"),
            _meta_value(meta, "block_indices_sha256"),
            _meta_value(meta, "primary_position_min"),
            _meta_value(meta, "activation_hook"),
            _meta_value(meta, "activation_subtraction_input_dtype"),
        )
        if vector_id in seen_ids:
            if seen_ids[vector_id] == duplicate_signature:
                warnings.warn(
                    f"Deduplicating byte-identical diff-vector receipt {vector_id}",
                    stacklevel=2,
                )
                continue
            raise ValueError(
                "Conflicting duplicate diff-vector provenance: "
                f"{vector_id}"
            )
        seen_ids[vector_id] = duplicate_signature
        mean_share = _safe_float(
            _meta_value(
                meta, "mean_offset_energy_share", "constancy", default=math.nan
            )
        )
        meta["mean_offset_energy_share"] = mean_share
        meta["constancy"] = mean_share
        meta["_source_file"] = path.name
        vectors.append(DiffVector(vector_id, vector64, meta, actual_norm, mean_share))
    return vectors


def select_analysis_layer(
    rows: Sequence[dict[str, Any]],
    vectors: Sequence[DiffVector],
    requested_layer: int | None,
) -> tuple[list[dict[str, Any]], list[DiffVector], int]:
    """Select L=15 by default while permitting a tiny-model explicit layer."""

    available: set[int] = set()
    for record in list(rows) + [vector.meta for vector in vectors]:
        layer = _safe_int(record.get("layer"), default=None)
        if layer is None:
            raise ValueError(f"Invalid or missing layer metadata: {record.get('layer')!r}")
        available.add(layer)
    if requested_layer is not None:
        chosen = int(requested_layer)
    elif PRIMARY_LAYER in available:
        chosen = PRIMARY_LAYER
    elif len(available) == 1:
        chosen = next(iter(available))
    else:
        raise ValueError(
            f"Inputs contain layers {sorted(available)} but not primary L={PRIMARY_LAYER}; pass --layer"
        )
    if chosen not in available:
        raise ValueError(
            f"Requested layer {chosen} is absent; available: {sorted(available)}"
        )
    selected_rows = [row for row in rows if _safe_int(row.get("layer")) == chosen]
    selected_vectors = [
        vector for vector in vectors if _safe_int(vector.meta.get("layer")) == chosen
    ]
    if not selected_rows or not selected_vectors:
        raise ValueError(f"Layer {chosen} needs both judged rows and diff vectors")
    return selected_rows, selected_vectors, chosen


def _step_number(record: Mapping[str, Any]) -> int:
    return _safe_int(_meta_value(record, "checkpoint_step", "step", default=-1))


def _select_final_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            _canonical_arm(row["arm"]),
            row.get("seed"),
            row.get("layer"),
            row.get("snippet_set"),
            row.get("modality"),
        )
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for group in grouped.values():
        final_step = max(_step_number(row) for row in group)
        output.extend(row for row in group if _step_number(row) == final_step)
    return output


def _select_final_vectors(vectors: Sequence[DiffVector]) -> list[DiffVector]:
    grouped: dict[tuple[Any, ...], list[DiffVector]] = defaultdict(list)
    for vector in vectors:
        meta = vector.meta
        key = (
            _canonical_arm(meta["arm"]),
            meta.get("seed"),
            meta.get("layer"),
            meta.get("snippet_set"),
        )
        grouped[key].append(vector)
    output: list[DiffVector] = []
    for group in grouped.values():
        final_step = max(_step_number(vector.meta) for vector in group)
        output.extend(
            vector for vector in group if _step_number(vector.meta) == final_step
        )
    return output


def _unit_key(record: Mapping[str, Any]) -> tuple[str, Any]:
    kind = str(record.get("_unit_kind", record.get("sampling_unit", "unknown")))
    if kind == "random_direction":
        kind = "draw"
    value = record.get("_unit_value")
    if value is None:
        value = record.get("draw") if kind == "draw" else record.get("block")
    return kind, value


def _validate_unit_groups(rows: Sequence[dict[str, Any]], mode: str) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        arm = _canonical_arm(row["arm"])
        if arm == "A-B" or row.get("modality") not in PRIMARY_MODALITIES:
            continue
        key = (
            arm,
            row.get("seed"),
            _step_number(row),
            row.get("layer"),
            row.get("snippet_set"),
            row.get("modality"),
        )
        groups[key].append(row)
    for key, group in groups.items():
        arm = key[0]
        # This validation group already includes seed, so the local block/draw
        # label is unique within the cell.
        units = [_unit_key(row) for row in group]
        if len(units) != len(set(units)):
            raise ValueError(
                f"Multiple judge items/votes would double-count sampling units in {key}"
            )
        expected_kind = "draw" if arm == "N2" else "block"
        if any(kind != expected_kind for kind, _value in units):
            raise ValueError(f"{key} contains non-{expected_kind} sampling units: {units}")
        if arm == "N2":
            if mode == "real" and len(units) != 50:
                raise ValueError(
                    f"N2 requires exactly 50 draw units per real cell; {key} has {len(units)}"
                )
        elif mode == "real":
            declared_k = {
                int(row.get("K", row.get("n_blocks", -1))) for row in group
            }
            if declared_k != {10} or len(units) != 10:
                raise ValueError(f"Real block cell {key} must contain K=10 unique blocks")


def validate_analysis_inputs(
    rows: Sequence[dict[str, Any]], vectors: Sequence[DiffVector]
) -> None:
    """Reject provenance or unit combinations that could inflate evidence."""

    if not rows or not vectors:
        raise ValueError("Analysis needs both judged rows and diff vectors")
    modes = {bool(record.get("is_mock")) for record in rows}
    modes.update(bool(vector.meta.get("is_mock")) for vector in vectors)
    if len(modes) != 1:
        raise ValueError("Row metadata mixes mock and real inputs")
    mode = "mock" if next(iter(modes)) else "real"
    snippet_hashes: dict[str, set[str]] = defaultdict(set)
    records: list[tuple[str, Mapping[str, Any]]] = []
    records.extend(
        (
            f"{row.get('_source_file', 'judged')}:{row.get('_source_line', '?')}",
            row,
        )
        for row in rows
    )
    records.extend(
        (str(vector.meta.get("_source_file", vector.vector_id)), vector.meta)
        for vector in vectors
    )
    for source, record in records:
        digest = str(_meta_value(record, "snippet_sha", "snippet_set_sha256"))
        if not SHA256_RE.fullmatch(digest):
            raise ValueError(
                f"snippet-set hash must be a full 64-hex SHA-256 in {source}, got {digest!r}"
            )
        snippet_hashes[str(record["snippet_set"])].add(digest)
        for key in ("seed", "layer"):
            if key not in record:
                raise ValueError(f"Missing {key} in {source}")
    conflicts = {
        name: values for name, values in snippet_hashes.items() if len(values) > 1
    }
    if conflicts:
        raise ValueError(f"Snippet-set names map to conflicting hashes: {conflicts}")
    widths = {vector.vector.shape[0] for vector in vectors}
    if len(widths) != 1:
        raise ValueError(f"Diff vectors have incompatible widths: {sorted(widths)}")
    for vector in vectors:
        if math.isfinite(vector.constancy) and not -1e-8 <= vector.constancy <= 1 + 1e-6:
            raise ValueError(
                f"Mean-offset energy share outside [0,1]: {vector.vector_id}"
            )
    _validate_unit_groups(rows, mode)

    vector_keys: Counter[tuple[Any, ...]] = Counter()
    for vector in vectors:
        meta = vector.meta
        vector_keys[
            (
                _canonical_arm(meta["arm"]),
                meta.get("seed"),
                _step_number(meta),
                _safe_int(meta.get("layer")),
                str(meta.get("snippet_set")),
                _unit_key(meta),
            )
        ] += 1
    for row in rows:
        arm = _canonical_arm(row["arm"])
        if arm == "A-B" or row.get("modality") != "tokens":
            continue
        key = (
            arm,
            row.get("seed"),
            _step_number(row),
            _safe_int(row.get("layer")),
            str(row.get("snippet_set")),
            _unit_key(row),
        )
        if mode == "mock" and vector_keys[key] == 0:
            aggregate_key = (*key[:-1], ("aggregate", 0))
            if vector_keys[aggregate_key] == 1:
                continue
        if vector_keys[key] != 1:
            raise ValueError(
                f"Judged item has {vector_keys[key]} matching diff vectors; expected one: {key}"
            )


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Two-sided Wilson score interval over independent blocks/draws."""

    if n < 0 or successes < 0 or successes > n:
        raise ValueError((successes, n))
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0, centre - radius), min(1, centre + radius)


def _accuracy(values: Iterable[Any]) -> dict[str, float | int]:
    arr = np.asarray([bool(value) for value in values], dtype=np.int8)
    n = int(arr.size)
    successes = int(arr.sum())
    low, high = wilson_interval(successes, n)
    return {
        "accuracy": successes / n if n else math.nan,
        "low": low,
        "high": high,
        "n": n,
        "successes": successes,
    }


def load_lexical_predictions(paths: Sequence[Path], mode: str) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for path, line_number, row in _read_jsonl(paths):
        _assert_mock_marker(path, row, mode)
        item_id = _meta_value(row, "item_id", "judge_item_id", "id", default=None)
        prediction = _meta_value(
            row,
            "lexical_pred",
            "predicted_label",
            "prediction",
            "pred",
            default=None,
        )
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"Lexical prediction lacks item_id at {path}:{line_number}")
        if prediction not in LABELS:
            raise ValueError(
                f"Invalid lexical prediction at {path}:{line_number}: {prediction!r}"
            )
        if item_id in predictions and predictions[item_id] != prediction:
            raise ValueError(f"Conflicting lexical predictions for {item_id!r}")
        predictions[item_id] = str(prediction)
    return predictions


def add_lexical_predictions(
    rows: list[dict[str, Any]],
    seed: int = 0,
    predictions: Mapping[str, str] | Sequence[dict[str, Any]] | None = None,
) -> None:
    """Attach precomputed external-corpus predictions; never fit on readouts."""

    del seed
    mapping: dict[str, str] = {}
    if isinstance(predictions, Mapping):
        mapping.update({str(key): str(value) for key, value in predictions.items()})
    elif predictions is not None:
        for record in predictions:
            item_id = str(_meta_value(record, "item_id", "judge_item_id"))
            pred = str(
                _meta_value(
                    record,
                    "lexical_pred",
                    "predicted_label",
                    "prediction",
                    "pred",
                )
            )
            mapping[item_id] = pred
    for row in rows:
        if _canonical_arm(row["arm"]) == "A-B":
            row["_lexical_pred"] = None
            row["_lexical_correct"] = None
            continue
        item_id = str(row["item_id"])
        pred = mapping.get(item_id)
        if pred is None:
            pred = _meta_value(
                row,
                "lexical_pred",
                "lexical_prediction",
                "tfidf_pred",
                default=None,
            )
        direct_correct = row.get("lexical_correct", row.get("tfidf_correct"))
        if pred is not None and pred not in LABELS:
            raise ValueError(
                f"Invalid precomputed lexical label for {item_id}: {pred!r}"
            )
        if pred is None and direct_correct is None:
            row["_lexical_pred"] = None
            row["_lexical_correct"] = None
            continue
        row["_lexical_pred"] = pred
        expected = None if pred is None else pred == row["true"]
        if (
            direct_correct is not None
            and expected is not None
            and bool(direct_correct) != expected
        ):
            raise ValueError(
                f"lexical_correct disagrees with lexical prediction for {item_id}"
            )
        row["_lexical_correct"] = bool(direct_correct) if expected is None else expected


def accuracy_summaries(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, dict[str, float | int]]]:
    """Summarize exactly one majority decision per unique sampling unit."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        arm = _canonical_arm(row["arm"])
        modality = str(row["modality"])
        snippet = str(row["snippet_set"])
        if arm == "A-B" or modality not in PRIMARY_MODALITIES or snippet == "-":
            continue
        grouped[(arm, snippet, modality)].append(row)
    output: dict[
        tuple[str, str, str], dict[str, dict[str, float | int]]
    ] = {}
    for key, group in grouped.items():
        units = [(row.get("seed"), _unit_key(row)) for row in group]
        if len(units) != len(set(units)):
            raise ValueError(f"Accuracy cell {key} contains repeated sampling units")
        lexical = [
            row["_lexical_correct"]
            for row in group
            if row.get("_lexical_correct") is not None
        ]
        shuffled = [
            row["correct_shuffled"]
            for row in group
            if row.get("shuffled_control_valid") is True
            and "correct_shuffled" in row
        ]
        truth = [str(row["true"]) for row in group]
        output[key] = {
            "judge": _accuracy(row["correct"] for row in group),
            "lexical": _accuracy(lexical),
            "shuffled": _accuracy(shuffled),
            "always_math": _accuracy(value == "math" for value in truth),
            "always_none": _accuracy(value == "none" for value in truth),
        }
    return output


def _ordered(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    unique = set(values)
    return [value for value in preferred if value in unique] + sorted(
        unique - set(preferred)
    )


def _errorbar_arrays(
    stats: Sequence[Mapping[str, float | int]],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([float(item["accuracy"]) for item in stats], dtype=float)
    lows = np.asarray([float(item["low"]) for item in stats], dtype=float)
    highs = np.asarray([float(item["high"]) for item in stats], dtype=float)
    return values, np.maximum(0, np.vstack((values - lows, highs - values)))


def _output_path(directory: Path, stem: str, suffix: str, mode: str) -> Path:
    marker = "_MOCK" if mode == "mock" else ""
    return directory / f"{stem}{marker}{suffix}"


def _figure_metadata(
    title: str,
    mode: str,
    commit: str,
    timestamp: str,
    inputs: InputSet,
    run_metadata: Mapping[str, Any],
) -> dict[str, str]:
    sources = [
        path.name
        for path in inputs.judged
        + inputs.diffs
        + inputs.lexical
        + inputs.curves
        + inputs.items
        + inputs.reward_logs
    ]
    return {
        "Title": title,
        "Author": "analysis/summarize.py",
        "Subject": json.dumps(
            {
                "analysis_mode": mode,
                "is_mock": mode == "mock",
                "git_commit": commit,
                "timestamp": timestamp,
                "source_files": sources,
                **dict(run_metadata),
            },
            sort_keys=True,
        ),
        "Software": "matplotlib; analysis/summarize.py",
    }


def collect_run_metadata(
    rows: Sequence[dict[str, Any]], vectors: Sequence[DiffVector]
) -> dict[str, Any]:
    records = list(rows) + [vector.meta for vector in vectors]

    def values(*keys: str) -> list[Any]:
        found = {
            _meta_value(record, *keys)
            for record in records
            if _meta_value(record, *keys) != "unknown"
        }
        return sorted(found, key=str)

    hashes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        name = str(_meta_value(record, "snippet_set"))
        digest = str(_meta_value(record, "snippet_sha", "snippet_set_sha256"))
        if digest != "unknown" and digest not in hashes[name]:
            hashes[name].append(digest)
    return {
        "arms": _ordered(
            (_canonical_arm(value) for value in values("arm")), ARM_ORDER
        ),
        "seeds": values("seed"),
        "checkpoint_steps": values("checkpoint_step", "step"),
        "layers": values("layer"),
        "snippet_sets_and_hashes": dict(sorted(hashes.items())),
        "judge_models": values("judge_model"),
        "sampling_units": values("sampling_unit"),
    }


def _atomic_savefig(
    fig: plt.Figure, path: Path, metadata: Mapping[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fig.savefig(
        temporary,
        format="png",
        dpi=180,
        bbox_inches="tight",
        metadata=dict(metadata),
    )
    os.replace(temporary, path)


def _atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Atomically write a CSV carrying the AGENTS section-7 provenance fields."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    provenance = dict(provenance or {})
    mandatory = (
        "arm",
        "seed",
        "checkpoint_step",
        "layer",
        "snippet_set",
        "snippet_sha",
        "judge_model",
        "timestamp",
        "git_commit",
        "is_mock",
        "artifact_status",
    )
    columns = list(fieldnames) + [
        key for key in mandatory if key not in set(fieldnames)
    ]
    source_rows = list(rows) or [{}]
    enriched = [
        {
            **provenance,
            "artifact_status": "measured" if rows else "unavailable",
            **dict(row),
        }
        for row in source_rows
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(enriched)
    os.replace(temporary, path)


def _mock_watermark(fig: plt.Figure, mode: str, *, y: float = 0.005) -> None:
    if mode == "mock":
        fig.text(
            0.995,
            y,
            "MOCK DATA — NOT A SCIENTIFIC RESULT",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#a61b1b",
            weight="bold",
        )


def write_accuracy_summary(
    summaries: Mapping[
        tuple[str, str, str], Mapping[str, Mapping[str, float | int]]
    ],
    path: Path,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    rows: list[dict[str, Any]] = []
    for (arm, snippet, modality), methods in sorted(summaries.items()):
        for method, stats in sorted(methods.items()):
            rows.append(
                {
                    "arm": arm,
                    "snippet_set": snippet,
                    "modality": modality,
                    "method": method,
                    **stats,
                    "sampling_unit": "draw" if arm == "N2" else "block",
                }
            )
    _atomic_csv(
        path,
        (
            "arm",
            "snippet_set",
            "modality",
            "method",
            "sampling_unit",
            "n",
            "successes",
            "accuracy",
            "low",
            "high",
        ),
        rows,
        provenance,
    )


def plot_judge_accuracy(
    summaries: Mapping[
        tuple[str, str, str], Mapping[str, Mapping[str, float | int]]
    ],
    output: Path,
    mode: str,
    metadata: Mapping[str, str],
) -> None:
    arms = _ordered((key[0] for key in summaries), ARM_ORDER)
    snippets = _ordered((key[1] for key in summaries), PRIMARY_SNIPPETS)
    modalities = _ordered((key[2] for key in summaries), PRIMARY_MODALITIES)
    if not arms:
        raise ValueError("No judged physical-arm cells available for Figure 1")
    fig, axes = plt.subplots(
        len(snippets),
        len(modalities),
        figsize=(
            max(8.5, 6.3 * len(modalities)),
            max(4.2, 3.8 * len(snippets)),
        ),
        sharey=True,
        squeeze=False,
    )
    methods = (
        ("judge", "LLM judge", "#315f88", None),
        ("lexical", "External TF-IDF", "#e5b567", "///"),
        ("always_math", "Always math", "#78a083", "\\\\"),
        ("always_none", "Always none", "#a99bc5", "xx"),
    )
    width = 0.8 / len(methods)
    empty = {"accuracy": math.nan, "low": math.nan, "high": math.nan, "n": 0}
    for row_index, snippet in enumerate(snippets):
        for col_index, modality in enumerate(modalities):
            ax = axes[row_index, col_index]
            x = np.arange(len(arms), dtype=float)
            for method_index, (method, label, colour, hatch) in enumerate(methods):
                stats = [
                    summaries.get((arm, snippet, modality), {}).get(method, empty)
                    for arm in arms
                ]
                values, errors = _errorbar_arrays(stats)
                offset = (method_index - (len(methods) - 1) / 2) * width
                ax.bar(
                    x + offset,
                    values,
                    width,
                    yerr=errors,
                    capsize=2,
                    color=colour,
                    edgecolor="#596168",
                    linewidth=0.45,
                    hatch=hatch,
                    label=label,
                    zorder=2,
                )
            ax.axhline(
                CHANCE,
                color="#a61b1b",
                linestyle="--",
                linewidth=1.2,
                label="Chance (1/6)",
            )
            ax.set_title(f"{snippet} snippets · {modality}")
            ax.set_xticks(x, arms, rotation=35, ha="right")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.2)
            if col_index == 0:
                ax.set_ylabel("Accuracy (blocks; N2 draws)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=min(5, len(unique)),
        frameon=False,
    )
    fig.suptitle(
        "Blind domain decoding with unit-level Wilson 95% intervals", y=0.995
    )
    fig.subplots_adjust(top=0.80, hspace=0.42, wspace=0.12, bottom=0.18)
    _mock_watermark(fig, mode)
    _atomic_savefig(fig, output, metadata)
    plt.close(fig)


def _matching_key(vector: DiffVector) -> tuple[Any, ...]:
    meta = vector.meta
    return (
        meta.get("seed"),
        _step_number(meta),
        _safe_int(meta.get("layer")),
        str(meta.get("snippet_set")),
        str(_meta_value(meta, "snippet_sha", "snippet_set_sha256")),
        str(_meta_value(meta, "base", default="unknown")),
        _unit_key(meta),
        vector.vector.shape[0],
        # A and B may only be subtracted when they refer to identical cached
        # rows, frozen blocks, and capture/filter arithmetic. Adapter identity
        # is intentionally absent because it is the quantity being contrasted.
        str(_meta_value(meta, "alignment_sha256")),
        _meta_value(meta, "block_seed"),
        str(_meta_value(meta, "block_assignment_sha256")),
        str(_meta_value(meta, "block_indices_sha256")),
        _meta_value(meta, "primary_position_min"),
        str(_meta_value(meta, "positions_collected")),
        _meta_value(meta, "collection_skip_tokens"),
        _meta_value(meta, "activation_max_tokens"),
        str(_meta_value(meta, "activation_hook")),
        str(_meta_value(meta, "activation_subtraction_input_dtype")),
        str(_meta_value(meta, "estimator_accumulator_dtype")),
    )


def derive_a_minus_b(vectors: Sequence[DiffVector]) -> list[DiffVector]:
    """Derive descriptive A-B only for exactly matched block provenance."""

    grouped: dict[
        tuple[Any, ...], dict[str, list[DiffVector]]
    ] = defaultdict(lambda: defaultdict(list))
    for vector in vectors:
        grouped[_matching_key(vector)][_canonical_arm(vector.meta["arm"])].append(
            vector
        )
    derived: list[DiffVector] = []
    for key, arms in sorted(grouped.items(), key=lambda item: repr(item[0])):
        if "A" not in arms or "B" not in arms:
            continue
        if len(arms["A"]) != 1 or len(arms["B"]) != 1:
            raise ValueError(f"Ambiguous A-B sources for {key}")
        a, b = arms["A"][0], arms["B"][0]
        if not (a.meta.get("is_mock") is True and b.meta.get("is_mock") is True):
            exact_fields = key[8:]
            if any(value in (None, "unknown", "") for value in exact_fields):
                raise ValueError(
                    "Real A-B derivation requires complete alignment, block, "
                    "position, hook, and dtype provenance"
                )
        expected = a.vector - b.vector
        if "A-B" in arms:
            if len(arms["A-B"]) != 1 or not np.allclose(
                arms["A-B"][0].vector, expected
            ):
                raise ValueError(
                    f"Explicit A-B does not equal paired A minus B for {key}"
                )
            continue
        meta = {
            **a.meta,
            "arm": "A-B",
            "descriptive_only": True,
            "derived_from": [a.vector_id, b.vector_id],
            "mean_offset_energy_share": math.nan,
            "constancy": math.nan,
        }
        identifier = (
            f"A-B|s{key[0]}|step{key[1]}|L{key[2]}|{key[3]}|"
            f"{key[6][0]}{key[6][1]}"
        )
        derived.append(
            DiffVector(
                identifier,
                expected,
                meta,
                float(np.linalg.norm(expected)),
                math.nan,
            )
        )
    return derived


def aggregate_mean_vectors(vectors: Sequence[DiffVector]) -> list[DiffVector]:
    """Average block vectors within each arm/seed/step/layer/snippet cell."""

    grouped: dict[tuple[Any, ...], list[DiffVector]] = defaultdict(list)
    for vector in vectors:
        meta = vector.meta
        key = (
            _canonical_arm(meta["arm"]),
            meta.get("seed"),
            _step_number(meta),
            _safe_int(meta.get("layer")),
            str(meta.get("snippet_set")),
            str(_meta_value(meta, "snippet_sha", "snippet_set_sha256")),
        )
        grouped[key].append(vector)
    means: list[DiffVector] = []
    for key, group in sorted(grouped.items(), key=lambda item: repr(item[0])):
        matrix = np.stack([vector.vector for vector in group])
        mean = matrix.mean(axis=0)
        shares = [
            vector.constancy for vector in group if math.isfinite(vector.constancy)
        ]
        source_kind = "draw" if key[0] == "N2" else "block"
        identifier = (
            f"{key[0]}|s{key[1]}|step{key[2]}|L{key[3]}|{key[4]}|"
            f"mean_{source_kind}s_n{len(group)}"
        )
        meta = {
            **group[0].meta,
            "aggregation": f"mean_across_{source_kind}s",
            "n_units": len(group),
            "sampling_unit": f"mean_{source_kind}s",
            "_unit_kind": f"mean_{source_kind}s",
            "_unit_value": len(group),
        }
        means.append(
            DiffVector(
                identifier,
                mean,
                meta,
                float(np.linalg.norm(mean)),
                float(np.mean(shares)) if shares else math.nan,
            )
        )
    return means


def add_random_references(
    vectors: Sequence[DiffVector], seed: int, is_mock: bool
) -> list[DiffVector]:
    """Add one deterministic random reference per vector width."""

    by_width: dict[int, list[DiffVector]] = defaultdict(list)
    for vector in vectors:
        by_width[vector.vector.shape[0]].append(vector)
    output: list[DiffVector] = []
    for index, (width, group) in enumerate(sorted(by_width.items())):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, width]))
        direction = rng.standard_normal(width)
        norms = [vector.d_norm for vector in group if vector.d_norm > 0]
        target = float(np.mean(norms)) if norms else 1
        direction *= target / np.linalg.norm(direction)
        output.append(
            DiffVector(
                f"random|seed{seed}|d{width}",
                direction,
                {
                    "arm": "random",
                    "seed": seed,
                    "step": -1,
                    "checkpoint_step": -1,
                    "layer": -1,
                    "snippet_set": "all",
                    "snippet_sha": "not_applicable",
                    "judge_model": "not_applicable",
                    "is_mock": is_mock,
                    "aggregation": "deterministic_random_reference",
                },
                float(np.linalg.norm(direction)),
                math.nan,
            )
        )
    return output


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return math.nan
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-15:
        return math.nan
    return float(np.dot(a, b) / denominator)


def write_cosine_matrix(
    vectors: Sequence[DiffVector],
    output: Path,
    mode: str,
    timestamp: str,
    commit: str,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    ids = [vector.vector_id for vector in vectors]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate aggregate vector IDs")
    metadata_fields = (
        "vector_id",
        "arm",
        "seed",
        "checkpoint_step",
        "layer",
        "snippet_set",
        "n_units",
        "aggregation",
        "timestamp",
        "git_commit",
        "is_mock",
    )
    rows: list[dict[str, Any]] = []
    for left in vectors:
        row: dict[str, Any] = {
            "vector_id": left.vector_id,
            "arm": left.meta.get("arm"),
            "seed": left.meta.get("seed"),
            "checkpoint_step": _step_number(left.meta),
            "layer": left.meta.get("layer"),
            "snippet_set": left.meta.get("snippet_set"),
            "n_units": left.meta.get("n_units", 1),
            "aggregation": left.meta.get("aggregation", "source"),
            "timestamp": timestamp,
            "git_commit": commit,
            "is_mock": mode == "mock",
        }
        for right in vectors:
            value = _cosine(left.vector, right.vector)
            row[right.vector_id] = "" if not math.isfinite(value) else f"{value:.10g}"
        rows.append(row)
    _atomic_csv(output, (*metadata_fields, *ids), rows, provenance)


def write_block_stability(
    vectors: Sequence[DiffVector],
    output: Path,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    groups: dict[tuple[Any, ...], list[DiffVector]] = defaultdict(list)
    for vector in vectors:
        if _unit_key(vector.meta)[0] != "block":
            continue
        meta = vector.meta
        groups[
            (
                _canonical_arm(meta["arm"]),
                meta.get("seed"),
                _step_number(meta),
                meta.get("layer"),
                meta.get("snippet_set"),
            )
        ].append(vector)
    rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        values: list[float] = []
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                cosine = _cosine(left.vector, right.vector)
                if math.isfinite(cosine):
                    values.append(cosine)
        rows.append(
            {
                "arm": key[0],
                "seed": key[1],
                "checkpoint_step": key[2],
                "layer": key[3],
                "snippet_set": key[4],
                "n_blocks": len(group),
                "n_pairs": len(values),
                "mean_block_cosine": float(np.mean(values)) if values else "",
                "min_block_cosine": min(values) if values else "",
                "max_block_cosine": max(values) if values else "",
            }
        )
    _atomic_csv(
        output,
        (
            "arm",
            "seed",
            "checkpoint_step",
            "layer",
            "snippet_set",
            "n_blocks",
            "n_pairs",
            "mean_block_cosine",
            "min_block_cosine",
            "max_block_cosine",
        ),
        rows,
        provenance,
    )


def plot_norm_constancy(
    vectors: Sequence[DiffVector],
    output: Path,
    mode: str,
    metadata: Mapping[str, str],
) -> None:
    shown = [
        vector
        for vector in vectors
        if _canonical_arm(vector.meta["arm"]) != "A-B"
    ]
    arms = _ordered(
        (_canonical_arm(vector.meta["arm"]) for vector in shown), ARM_ORDER
    )
    snippets = _ordered(
        (str(vector.meta["snippet_set"]) for vector in shown), PRIMARY_SNIPPETS
    )
    grouped: dict[tuple[str, str], list[DiffVector]] = defaultdict(list)
    for vector in shown:
        grouped[
            (_canonical_arm(vector.meta["arm"]), str(vector.meta["snippet_set"]))
        ].append(vector)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    x = np.arange(len(arms), dtype=float)
    width = 0.78 / max(1, len(snippets))
    for snippet_index, snippet in enumerate(snippets):
        offset = (snippet_index - (len(snippets) - 1) / 2) * width
        norm_means: list[float] = []
        share_means: list[float] = []
        for arm_index, arm in enumerate(arms):
            group = grouped.get((arm, snippet), [])
            norms = [vector.d_norm for vector in group]
            shares = [
                vector.constancy
                for vector in group
                if math.isfinite(vector.constancy)
            ]
            norm_means.append(float(np.mean(norms)) if norms else math.nan)
            share_means.append(float(np.mean(shares)) if shares else math.nan)
            if norms:
                axes[0].scatter(
                    np.full(len(norms), arm_index + offset),
                    norms,
                    color="#263640",
                    s=12,
                    alpha=0.62,
                    zorder=3,
                )
            if shares:
                axes[1].scatter(
                    np.full(len(shares), arm_index + offset),
                    shares,
                    color="#263640",
                    s=12,
                    alpha=0.62,
                    zorder=3,
                )
        colour = plt.get_cmap("Set2")(snippet_index)
        axes[0].bar(
            x + offset, norm_means, width, label=snippet, color=colour, alpha=0.78
        )
        axes[1].bar(
            x + offset, share_means, width, label=snippet, color=colour, alpha=0.78
        )
    axes[0].set_title("Raw block/draw norm")
    axes[0].set_ylabel(r"$\|d\|_2$ before norm matching")
    axes[1].set_title("Mean-offset energy share")
    axes[1].set_ylabel(r"$N\|\bar\delta\|^2 / \sum_i\|\delta_i\|^2$")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xticks(x, arms, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(title="Snippet set", frameon=False)
    fig.suptitle(
        "Activation-difference geometry (points are blocks; N2 points are draws)"
    )
    fig.subplots_adjust(top=0.86, bottom=0.2, wspace=0.25)
    _mock_watermark(fig, mode)
    _atomic_savefig(fig, output, metadata)
    plt.close(fig)


def _top_tokens(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("top", row.get("top_tokens"))
    tokens: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and item:
                tokens.append(str(item[0]))
            elif isinstance(item, dict) and "token" in item:
                tokens.append(str(item["token"]))
            elif isinstance(item, str):
                tokens.append(item)
    if tokens:
        return tokens
    text = str(row.get("text", ""))
    try:
        parsed = ast.literal_eval(f"[{text}]")
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return list(parsed)
    except (SyntaxError, ValueError):
        pass
    return [part.strip() for part in text.split(",") if part.strip()]


def load_item_rows(paths: Sequence[Path], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, line_number, original in _read_jsonl(paths):
        row = dict(original)
        _assert_mock_marker(path, row, mode)
        if row.get("modality") != "tokens" or "arm" not in row:
            continue
        arm = _canonical_arm(row["arm"])
        row["arm"] = arm
        source = f"{path}:{line_number}"
        unit_kind, unit_value = _normalise_sampling_unit(row, arm, mode, source)
        row["_unit_kind"] = unit_kind
        row["_unit_value"] = unit_value
        row["_source_file"] = path.name
        row["_source_line"] = line_number
        rows.append(row)
    return rows


def _display_token(token: str, limit: int = 22) -> str:
    display = repr(token)
    return display if len(display) <= limit else f"{display[: limit - 1]}…"


def select_top_tokens(
    rows: Sequence[dict[str, Any]], snippet_preference: str, seed: int = 0
) -> tuple[str, dict[str, list[str]]]:
    """Choose one deterministic block, shared across arms whenever possible."""

    token_rows = [row for row in rows if row.get("modality") == "tokens"]
    snippets = {str(row.get("snippet_set")) for row in token_rows}
    if snippet_preference in snippets:
        snippet = snippet_preference
    elif snippets:
        snippet = _ordered(snippets, PRIMARY_SNIPPETS)[0]
    else:
        raise ValueError("No token rows available for Figure 3")
    arms = ("A", "B", "C", "D", "A-B", "N1")
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        candidates = [
            row
            for row in token_rows
            if _canonical_arm(row["arm"]) == arm
            and str(row.get("snippet_set")) == snippet
        ]
        if candidates:
            final = max(_step_number(row) for row in candidates)
            candidates = [row for row in candidates if _step_number(row) == final]
        by_arm[arm] = candidates
    present = [arm for arm in arms if by_arm[arm]]
    key_sets = [
        {(row.get("seed"), _unit_key(row)) for row in by_arm[arm]}
        for arm in present
    ]
    common = set.intersection(*key_sets) if key_sets else set()
    rng = random.Random(seed)
    common_key = rng.choice(sorted(common, key=repr)) if common else None
    selected: dict[str, list[str]] = {}
    for arm in arms:
        candidates = by_arm[arm]
        if not candidates:
            selected[arm] = []
            continue
        matching = (
            [
                row
                for row in candidates
                if (row.get("seed"), _unit_key(row)) == common_key
            ]
            if common_key is not None
            else []
        )
        if not matching:
            ordered_candidates = sorted(
                candidates,
                key=lambda row: (
                    str(row.get("seed")),
                    repr(_unit_key(row)),
                    str(row.get("item_id", "")),
                ),
            )
            matching = [ordered_candidates[rng.randrange(len(ordered_candidates))]]
        selected[arm] = _top_tokens(matching[0])[:20]
    return snippet, selected


def plot_top_tokens(
    rows: Sequence[dict[str, Any]],
    output: Path,
    mode: str,
    metadata: Mapping[str, str],
    snippet_preference: str,
    seed: int,
) -> None:
    snippet, selected = select_top_tokens(rows, snippet_preference, seed=seed)
    arms = ("A", "B", "C", "D", "A-B", "N1")
    cells: list[list[str]] = []
    for rank in range(20):
        cells.append(
            [
                _display_token(selected[arm][rank])
                if rank < len(selected[arm])
                else ("— unavailable —" if rank == 0 else "")
                for arm in arms
            ]
        )
    fig, ax = plt.subplots(figsize=(14.8, 9.1))
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        colLabels=arms,
        rowLabels=[str(index) for index in range(1, 21)],
        cellLoc="left",
        colLoc="center",
        rowLoc="center",
        loc="center",
        colWidths=[0.155] * len(arms),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1, 1.35)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5d8")
        cell.set_linewidth(0.45)
        if row_index == 0:
            cell.set_facecolor("#315f88")
            cell.set_text_props(color="white", weight="bold")
        elif col_index == -1:
            cell.set_facecolor("#edf1f4")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f7f8f9")
        if row_index > 0 and col_index >= 0:
            cell.get_text().set_fontfamily("monospace")
    ax.set_title(
        f"Exact norm-matched logit-lens top 20 · {snippet} snippets · seeded block"
    )
    ax.text(
        0.5,
        0.975,
        "A−B is descriptive only and contributes no gold label or accuracy.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#4f5961",
    )
    _mock_watermark(fig, mode)
    _atomic_savefig(fig, output, metadata)
    plt.close(fig)


def conditional_trace_rows(vectors: Sequence[DiffVector]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[Any, ...], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for vector in vectors:
        meta = vector.meta
        arm = _canonical_arm(meta["arm"])
        snippet = str(meta["snippet_set"])
        if arm not in {"A", "D"} or snippet not in PRIMARY_SNIPPETS:
            continue
        key = (
            arm,
            meta.get("seed"),
            _step_number(meta),
            _safe_int(meta.get("layer")),
        )
        grouped[key][snippet].append(vector.d_norm)
    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        neutral = values.get("neutral", [])
        math_values = values.get("math", [])
        neutral_mean = float(np.mean(neutral)) if neutral else math.nan
        math_mean = float(np.mean(math_values)) if math_values else math.nan
        rows.append(
            {
                "arm": key[0],
                "seed": key[1],
                "checkpoint_step": key[2],
                "layer": key[3],
                "neutral_mean_raw_norm": neutral_mean,
                "math_mean_raw_norm": math_mean,
                "math_minus_neutral": math_mean - neutral_mean,
                "math_over_neutral": (
                    math_mean / neutral_mean if neutral_mean > 0 else math.nan
                ),
                "n_neutral_blocks": len(neutral),
                "n_math_blocks": len(math_values),
            }
        )
    return rows


def _position_norm(value: Any) -> float:
    if isinstance(value, Mapping):
        if "norm" in value:
            return _safe_float(value["norm"])
        value = _meta_value(value, "mean", "vector", "values", default=math.nan)
    if isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=float)
        return (
            float(np.linalg.norm(array))
            if array.size and np.all(np.isfinite(array))
            else math.nan
        )
    return _safe_float(value)


def per_position_rows(vectors: Sequence[DiffVector]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vector in vectors:
        if _canonical_arm(vector.meta["arm"]) != "D":
            continue
        raw = vector.meta.get("per_position_means")
        if raw is None:
            continue
        for position in range(5):
            value: Any = None
            if isinstance(raw, Mapping):
                value = raw.get(str(position), raw.get(position))
            elif isinstance(raw, list) and position < len(raw):
                value = raw[position]
            rows.append(
                {
                    "arm": "D",
                    "seed": vector.meta.get("seed"),
                    "checkpoint_step": _step_number(vector.meta),
                    "layer": vector.meta.get("layer"),
                    "snippet_set": vector.meta.get("snippet_set"),
                    "block": vector.meta.get("block"),
                    "position": position,
                    "position_mean_norm": _position_norm(value),
                }
            )
    return rows


def plot_per_position(
    rows: Sequence[dict[str, Any]],
    output: Path,
    mode: str,
    metadata: Mapping[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    if rows:
        by_snippet: dict[
            str, dict[int, list[float]]
        ] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            value = _safe_float(row["position_mean_norm"])
            if math.isfinite(value):
                by_snippet[str(row["snippet_set"])][int(row["position"])].append(
                    value
                )
        for snippet in _ordered(by_snippet, PRIMARY_SNIPPETS):
            x = np.arange(5)
            y = [
                float(np.mean(by_snippet[snippet].get(position, [math.nan])))
                for position in x
            ]
            ax.plot(x, y, marker="o", linewidth=1.8, label=snippet)
        ax.set_xticks(range(5))
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "Per-position means unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Token position")
    ax.set_ylabel(r"$\|\mathbb{E}[\Delta h_{position}]\|_2$")
    ax.set_title("Arm D Minder-faithful positions 0–4 diagnostic")
    ax.grid(alpha=0.2)
    _mock_watermark(fig, mode)
    _atomic_savefig(fig, output, metadata)
    plt.close(fig)


def plot_layer_sweep(
    vectors: Sequence[DiffVector],
    output: Path,
    mode: str,
    metadata: Mapping[str, str],
) -> None:
    selected = [
        vector
        for vector in vectors
        if _canonical_arm(vector.meta["arm"]) in {"A", "D"}
    ]
    grouped: dict[tuple[str, str, int], list[DiffVector]] = defaultdict(list)
    for vector in selected:
        grouped[
            (
                _canonical_arm(vector.meta["arm"]),
                str(vector.meta["snippet_set"]),
                _safe_int(vector.meta["layer"]),
            )
        ].append(vector)
    observed_layers = sorted({key[2] for key in grouped})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True)
    for arm in ("A", "D"):
        for snippet in PRIMARY_SNIPPETS:
            layers = sorted(
                {key[2] for key in grouped if key[:2] == (arm, snippet)}
            )
            if not layers:
                continue
            norms = [
                float(np.mean([v.d_norm for v in grouped[(arm, snippet, layer)]]))
                for layer in layers
            ]
            shares = [
                (
                    float(
                        np.mean(
                            [
                                v.constancy
                                for v in grouped[(arm, snippet, layer)]
                                if math.isfinite(v.constancy)
                            ]
                        )
                    )
                    if any(
                        math.isfinite(v.constancy)
                        for v in grouped[(arm, snippet, layer)]
                    )
                    else math.nan
                )
                for layer in layers
            ]
            label = f"{arm} · {snippet}"
            axes[0].plot(layers, norms, marker="o", label=label)
            axes[1].plot(layers, shares, marker="o", label=label)
    for ax, title, ylabel in (
        (axes[0], "Raw norm", r"mean block $\|d\|_2$"),
        (axes[1], "Mean-offset energy share", "mean block share"),
    ):
        ax.set_title(title)
        ax.set_xlabel("Post-block layer")
        ax.set_ylabel(ylabel)
        if observed_layers:
            ax.set_xticks(observed_layers)
        ax.grid(alpha=0.2)
    if selected:
        axes[0].legend(frameon=False, fontsize=8)
    else:
        axes[0].text(
            0.5,
            0.5,
            "A/D layer sweep unavailable",
            ha="center",
            transform=axes[0].transAxes,
        )
    fig.suptitle("Layer sensitivity for A and D")
    fig.subplots_adjust(top=0.84, bottom=0.18, wspace=0.3)
    _mock_watermark(fig, mode, y=-0.045)
    _atomic_savefig(fig, output, metadata)
    plt.close(fig)


def load_curve_rows(paths: Sequence[Path], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for line_number, original in enumerate(csv.DictReader(handle), 2):
                row = dict(original)
                source = f"{path}:{line_number}"
                marker = str(
                    row.get("is_mock", "true" if mode == "mock" else "false")
                ).lower()
                if mode == "mock" and marker not in {"true", "1", "yes"}:
                    raise ValueError(f"MOCK curve row lacks is_mock=true at {source}")
                if mode == "real" and marker in {"true", "1", "yes"}:
                    raise ValueError(f"Real-named curve contains is_mock=true at {source}")
                arm = _canonical_arm(row.get("arm", ""))
                if not arm:
                    match = re.search(
                        r"curve(?:_MOCK)?_([A-Za-z0-9-]+)_s(\d+)",
                        path.stem,
                        re.I,
                    )
                    if match:
                        arm = _canonical_arm(match.group(1))
                        row.setdefault("seed", match.group(2))
                if arm not in PHYSICAL_ARMS:
                    raise ValueError(f"Curve row lacks physical arm at {source}")
                row["arm"] = arm
                row["step"] = _safe_int(
                    _meta_value(row, "step", "checkpoint_step")
                )
                row["checkpoint_step"] = row["step"]
                row["seed"] = _safe_int(row.get("seed"))
                row["block"] = _safe_int(row.get("block"))
                row["norm"] = _safe_float(
                    _meta_value(row, "norm", "d_norm", "raw_d_norm")
                )
                row["constancy"] = _safe_float(
                    _meta_value(
                        row, "mean_offset_energy_share", "constancy"
                    )
                )
                item_id = _meta_value(
                    row, "judge_item_id", "item_id", default=None
                )
                if not isinstance(item_id, str) or not item_id:
                    raise ValueError(f"Curve row lacks judge_item_id at {source}")
                row["judge_item_id"] = item_id
                row["_source_file"] = path.name
                rows.append(row)
    return rows


def join_curve_judgments(
    curves: Sequence[dict[str, Any]], judged: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(row["item_id"]): row for row in judged}
    joined: list[dict[str, Any]] = []
    for row in curves:
        item_id = str(row["judge_item_id"])
        match = by_id.get(item_id)
        if match is None:
            raise ValueError(
                f"Curve judge_item_id has no exact judged-row match: {item_id}"
            )
        for field in ("arm", "seed", "step", "snippet_set", "block"):
            curve_value = row.get(field)
            judge_value = (
                match.get("checkpoint_step", match.get("step"))
                if field == "step"
                else match.get(field)
            )
            if str(curve_value) != str(judge_value):
                raise ValueError(
                    f"Curve/judge metadata mismatch for {item_id}: "
                    f"{field}={curve_value!r} vs {judge_value!r}"
                )
        joined.append({**row, "judge_correct": bool(match["correct"])})
    return joined


def curve_summary_rows(
    curves: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in curves:
        groups[
            (row["arm"], row["seed"], row["step"], row.get("snippet_set"))
        ].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        units = [row["block"] for row in group]
        if len(units) != len(set(units)):
            raise ValueError(f"Curve cell repeats blocks: {key}")
        judge_stats = _accuracy(row["judge_correct"] for row in group)
        output.append(
            {
                "arm": key[0],
                "seed": key[1],
                "step": key[2],
                "snippet_set": key[3],
                "n_blocks": len(group),
                "mean_norm": float(np.mean([row["norm"] for row in group])),
                "mean_constancy": (
                    float(
                        np.mean(
                            [
                                row["constancy"]
                                for row in group
                                if math.isfinite(row["constancy"])
                            ]
                        )
                    )
                    if any(math.isfinite(row["constancy"]) for row in group)
                    else math.nan
                ),
                "judge_accuracy": judge_stats["accuracy"],
                "judge_low": judge_stats["low"],
                "judge_high": judge_stats["high"],
            }
        )
    return output


def _infer_arm_seed(path: Path) -> tuple[str | None, int | None]:
    text = "/".join(path.parts)
    match = re.search(
        r"(?:^|[/_-])([ABCD])[_-]s(?:eed)?(\d+)(?:[/_.-]|$)", text, re.I
    )
    if not match:
        return None, None
    return match.group(1).upper(), int(match.group(2))


def load_reward_rows(
    source: Path | Sequence[Path],
) -> list[dict[str, Any]]:
    """Load rewards only when arm, seed, and step are unambiguous."""

    paths = (
        _discover_reward_paths(Path(source))
        if isinstance(source, Path)
        else tuple(sorted(Path(path) for path in source))
    )
    rows: dict[tuple[str, int, int], float] = {}
    sources: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    conflicts: set[tuple[str, int, int]] = set()
    for path in sorted(paths):
        inferred_arm, inferred_seed = _infer_arm_seed(path)
        records: list[dict[str, Any]] = []
        try:
            if path.suffix.lower() == ".csv":
                with path.open(encoding="utf-8", newline="") as handle:
                    records = [dict(row) for row in csv.DictReader(handle)]
            elif path.suffix.lower() == ".jsonl":
                records = [row for _p, _line, row in _read_jsonl((path,))]
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(
                    payload.get("log_history"), list
                ):
                    records = [
                        row
                        for row in payload["log_history"]
                        if isinstance(row, dict)
                    ]
                elif isinstance(payload, list):
                    records = [row for row in payload if isinstance(row, dict)]
                elif isinstance(payload, dict):
                    records = [payload]
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for record in records:
            arm = _canonical_arm(record.get("arm", inferred_arm or ""))
            seed = _safe_int(record.get("seed", inferred_seed), default=-1)
            step = _safe_int(
                _meta_value(record, "step", "global_step", default=-1), default=-1
            )
            reward = _safe_float(
                _meta_value(
                    record,
                    "mean_reward",
                    "reward",
                    "rewards/mean",
                    "train_reward",
                    default=math.nan,
                )
            )
            if (
                arm not in PHYSICAL_ARMS
                or seed < 0
                or step < 0
                or not math.isfinite(reward)
            ):
                continue
            key = (arm, seed, step)
            if key in rows and not math.isclose(
                rows[key], reward, rel_tol=1e-9, abs_tol=1e-12
            ):
                conflicts.add(key)
            else:
                rows[key] = reward
                sources[key].add(str(path))
    for key in conflicts:
        rows.pop(key, None)
        warnings.warn(
            f"Conflicting reward values for {key}; omitting that point", stacklevel=2
        )
    return [
        {
            "arm": key[0],
            "seed": key[1],
            "step": key[2],
            "reward": value,
            "source_path": json.dumps(sorted(sources[key])),
        }
        for key, value in sorted(rows.items())
    ]


def plot_emergence(
    summary_rows: Sequence[dict[str, Any]],
    rewards: Sequence[dict[str, Any]],
    output: Path,
    mode: str,
    metadata: Mapping[str, str],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    metrics = (
        ("mean_norm", "Raw norm"),
        ("mean_constancy", "Mean-offset energy share"),
        ("judge_accuracy", "Judge accuracy"),
    )
    for metric_index, (metric, title) in enumerate(metrics):
        ax = axes.flat[metric_index]
        groups: dict[
            tuple[str, int, str], list[dict[str, Any]]
        ] = defaultdict(list)
        for row in summary_rows:
            groups[
                (row["arm"], int(row["seed"]), str(row["snippet_set"]))
            ].append(row)
        for key, group in sorted(groups.items()):
            group = sorted(group, key=lambda row: int(row["step"]))
            ax.plot(
                [row["step"] for row in group],
                [row[metric] for row in group],
                marker="o",
                label=f"{key[0]} s{key[1]} {key[2]}",
            )
        ax.set_title(title)
        ax.grid(alpha=0.2)
        if metric == "judge_accuracy":
            ax.axhline(CHANCE, color="#a61b1b", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1)
    reward_ax = axes.flat[3]
    reward_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rewards:
        reward_groups[(row["arm"], int(row["seed"]))].append(row)
    for key, group in sorted(reward_groups.items()):
        group = sorted(group, key=lambda row: int(row["step"]))
        reward_ax.plot(
            [row["step"] for row in group],
            [row["reward"] for row in group],
            marker="o",
            label=f"{key[0]} s{key[1]}",
        )
    if not rewards:
        reward_ax.text(
            0.5,
            0.5,
            "Reward curve unavailable\n(no unambiguous arm+seed+step log)",
            ha="center",
            va="center",
            transform=reward_ax.transAxes,
        )
    reward_ax.set_title("Training reward")
    reward_ax.grid(alpha=0.2)
    for ax in axes.flat:
        ax.set_xlabel("Optimizer step")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=min(4, len(handles)),
            frameon=False,
            fontsize=8,
        )
    if not summary_rows:
        axes.flat[0].text(
            0.5,
            0.5,
            "Emergence curve unavailable",
            ha="center",
            transform=axes.flat[0].transAxes,
        )
    fig.suptitle("Trace emergence (exact checkpoint-item joins)", y=0.995)
    fig.subplots_adjust(top=0.80, hspace=0.34, wspace=0.24)
    _mock_watermark(fig, mode)
    _atomic_savefig(fig, output, metadata)
    plt.close(fig)


def summarize(
    results_dir: Path,
    figs_dir: Path,
    mode: str = "auto",
    seed: int = 0,
    top_token_snippet: str = "neutral",
    layer: int | None = None,
) -> dict[str, Path]:
    """Validate all inputs, then atomically emit Figures 1-4 and tables."""

    results_dir = Path(results_dir)
    figs_dir = Path(figs_dir)
    inputs = discover_inputs(results_dir, mode=mode)
    all_rows = load_judged(inputs.judged, inputs.mode)
    all_vectors = load_diff_vectors(inputs.diffs, inputs.mode)
    item_rows = load_item_rows(inputs.items, inputs.mode)
    lexical = load_lexical_predictions(inputs.lexical, inputs.mode)
    add_lexical_predictions(all_rows, predictions=lexical)
    if inputs.mode == "real":
        missing_lexical = [
            row["item_id"]
            for row in all_rows
            if row.get("modality") == "tokens"
            and _canonical_arm(row["arm"]) != "A-B"
            and row.get("_lexical_correct") is None
        ]
        if missing_lexical:
            raise ValueError(
                "Real token readouts lack precomputed external-corpus lexical "
                f"predictions for {len(missing_lexical)} item(s), including "
                f"{missing_lexical[:3]}"
            )
    curves = load_curve_rows(inputs.curves, inputs.mode)
    joined_curves = join_curve_judgments(curves, all_rows) if curves else []

    curve_ids = {str(row["judge_item_id"]) for row in curves}
    noncurve_rows = [
        row
        for row in all_rows
        if str(row["item_id"]) not in curve_ids
        and row.get("modality") in PRIMARY_MODALITIES
    ]
    primary_source_rows = (
        noncurve_rows
        if noncurve_rows
        else [row for row in all_rows if row.get("modality") in PRIMARY_MODALITIES]
    )
    layer_rows, layer_vectors, selected_layer = select_analysis_layer(
        primary_source_rows, all_vectors, requested_layer=layer
    )
    primary_rows = _select_final_rows(layer_rows)
    primary_vectors = _select_final_vectors(layer_vectors)
    validate_analysis_inputs(primary_rows, primary_vectors)
    summaries = accuracy_summaries(primary_rows)

    derived_ab = derive_a_minus_b(primary_vectors)
    geometry_vectors = primary_vectors + derived_ab
    aggregate_vectors = aggregate_mean_vectors(geometry_vectors)
    random_vectors = add_random_references(
        aggregate_vectors, seed, inputs.mode == "mock"
    )
    matrix_vectors = aggregate_vectors + random_vectors

    top_candidates_by_id: dict[str, dict[str, Any]] = {}
    for row in primary_rows:
        top_candidates_by_id[str(row["item_id"])] = row
    for row in item_rows:
        if _safe_int(row.get("layer")) != selected_layer:
            continue
        key = str(
            _meta_value(
                row,
                "item_id",
                "judge_item_id",
                default=f"raw:{len(top_candidates_by_id)}",
            )
        )
        top_candidates_by_id[key] = row
    top_candidates = _select_final_rows(list(top_candidates_by_id.values()))

    conditional = conditional_trace_rows(primary_vectors)
    positions = per_position_rows(primary_vectors)
    curve_summary = curve_summary_rows(joined_curves)
    rewards = load_reward_rows(inputs.reward_logs)

    timestamp = utc_now()
    commit = git_commit(Path(__file__).resolve().parents[1])
    run_metadata = collect_run_metadata(primary_rows, primary_vectors)
    run_metadata["reward_source_paths"] = sorted(
        {
            source_path
            for row in rewards
            for source_path in json.loads(row["source_path"])
        }
    )
    csv_provenance = {
        "arm": json.dumps(run_metadata["arms"]),
        "seed": json.dumps(run_metadata["seeds"]),
        "checkpoint_step": json.dumps(run_metadata["checkpoint_steps"]),
        "layer": json.dumps(run_metadata["layers"]),
        "snippet_set": json.dumps(
            sorted(run_metadata["snippet_sets_and_hashes"])
        ),
        "snippet_sha": json.dumps(
            run_metadata["snippet_sets_and_hashes"], sort_keys=True
        ),
        "judge_model": json.dumps(run_metadata["judge_models"]),
        "timestamp": timestamp,
        "git_commit": commit,
        "is_mock": inputs.mode == "mock",
    }
    outputs = {
        "fig1": _output_path(
            figs_dir, "fig1_judge_accuracy", ".png", inputs.mode
        ),
        "fig2": _output_path(
            figs_dir, "fig2_norm_constancy", ".png", inputs.mode
        ),
        "fig3": _output_path(figs_dir, "fig3_top_tokens", ".png", inputs.mode),
        "fig4": _output_path(
            figs_dir, "fig4_emergence_curve", ".png", inputs.mode
        ),
        "layer_sweep": _output_path(
            figs_dir, "layer_sweep", ".png", inputs.mode
        ),
        "per_position_D": _output_path(
            figs_dir, "per_position_D", ".png", inputs.mode
        ),
        "accuracy": _output_path(
            results_dir, "judge_accuracy", ".csv", inputs.mode
        ),
        "cosines": _output_path(
            results_dir, "cosine_matrix", ".csv", inputs.mode
        ),
        "block_stability": _output_path(
            results_dir, "block_stability", ".csv", inputs.mode
        ),
        "conditional_trace": _output_path(
            results_dir, "conditional_trace", ".csv", inputs.mode
        ),
        "per_position_D_csv": _output_path(
            results_dir, "per_position_D", ".csv", inputs.mode
        ),
        "curve_summary": _output_path(
            results_dir, "curve_summary", ".csv", inputs.mode
        ),
        "reward_curve": _output_path(
            results_dir, "reward_curve", ".csv", inputs.mode
        ),
    }

    def figure_metadata(title: str) -> dict[str, str]:
        return _figure_metadata(
            title, inputs.mode, commit, timestamp, inputs, run_metadata
        )

    plot_judge_accuracy(
        summaries,
        outputs["fig1"],
        inputs.mode,
        figure_metadata("Blind domain decoding"),
    )
    plot_norm_constancy(
        primary_vectors,
        outputs["fig2"],
        inputs.mode,
        figure_metadata("Block geometry"),
    )
    plot_top_tokens(
        top_candidates,
        outputs["fig3"],
        inputs.mode,
        figure_metadata("Logit-lens top tokens"),
        top_token_snippet,
        seed,
    )
    plot_emergence(
        curve_summary,
        rewards,
        outputs["fig4"],
        inputs.mode,
        figure_metadata("Trace emergence"),
    )
    plot_layer_sweep(
        _select_final_vectors(all_vectors),
        outputs["layer_sweep"],
        inputs.mode,
        figure_metadata("Layer sweep"),
    )
    plot_per_position(
        positions,
        outputs["per_position_D"],
        inputs.mode,
        figure_metadata("D per-position diagnostic"),
    )
    write_accuracy_summary(summaries, outputs["accuracy"], csv_provenance)
    write_cosine_matrix(
        matrix_vectors,
        outputs["cosines"],
        inputs.mode,
        timestamp,
        commit,
        csv_provenance,
    )
    write_block_stability(
        primary_vectors, outputs["block_stability"], csv_provenance
    )
    _atomic_csv(
        outputs["conditional_trace"],
        (
            "arm",
            "seed",
            "checkpoint_step",
            "layer",
            "neutral_mean_raw_norm",
            "math_mean_raw_norm",
            "math_minus_neutral",
            "math_over_neutral",
            "n_neutral_blocks",
            "n_math_blocks",
        ),
        conditional,
        csv_provenance,
    )
    _atomic_csv(
        outputs["per_position_D_csv"],
        (
            "arm",
            "seed",
            "checkpoint_step",
            "layer",
            "snippet_set",
            "block",
            "position",
            "position_mean_norm",
        ),
        positions,
        csv_provenance,
    )
    _atomic_csv(
        outputs["curve_summary"],
        (
            "arm",
            "seed",
            "step",
            "snippet_set",
            "n_blocks",
            "mean_norm",
            "mean_constancy",
            "judge_accuracy",
            "judge_low",
            "judge_high",
        ),
        curve_summary,
        csv_provenance,
    )
    _atomic_csv(
        outputs["reward_curve"],
        ("arm", "seed", "step", "reward", "source_path"),
        rewards,
        csv_provenance,
    )
    print(
        f"analysis mode={inputs.mode}; main_layer={selected_layer}; "
        f"blocks/draws={len(primary_rows)}; aggregate_vectors={len(matrix_vectors)}"
    )
    for name, path in outputs.items():
        print(f"wrote {name}: {path}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figs", type=Path, default=Path("figs"))
    parser.add_argument(
        "--mode", choices=("auto", "mock", "real"), default="auto"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="deterministic block/random-reference seed",
    )
    parser.add_argument(
        "--layer", type=int, default=None, help="primary layer; defaults to L=15"
    )
    parser.add_argument("--top-token-snippet", default="neutral")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summarize(
        results_dir=args.results,
        figs_dir=args.figs,
        mode=args.mode,
        seed=args.seed,
        top_token_snippet=args.top_token_snippet,
        layer=args.layer,
    )


if __name__ == "__main__":
    main()
