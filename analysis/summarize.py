"""Build the preregistered readable-trace figures and cosine table.

The analyzer consumes judged readouts and activation-difference sidecars:

    python analysis/summarize.py --results results --figs figs

Mock inputs are deliberately segregated from real inputs.  ``--mode auto``
refuses to run if both are present; select one explicitly with ``--mode mock``
or ``--mode real``.  Mock-derived figures carry a conspicuous watermark and the
cosine CSV has an ``is_mock`` column.

Figure 1 consumes persisted predictions made by ``judge/lexical_baseline.py``
from the frozen external six-domain reference corpus.  This module never fits a
classifier on readout rows.  Real analyses fail closed when those predictions
or their corpus/leakage receipts are absent.  MOCK layout fixtures may use an
explicitly marked deterministic placeholder without fitting.  Plotted subgroup
accuracies are broken out by arm and snippet set; all binomial intervals are
95% Wilson intervals.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = ("math", "cooking", "law", "medicine", "poetry", "none")
CHANCE = 1.0 / len(LABELS)
PRIMARY_MODALITIES = ("tokens", "steer")
PRIMARY_SNIPPETS = ("neutral", "math")
ARM_ORDER = ("A", "B", "C", "D", "A-B", "N1", "N2", "N3")
PHYSICAL_ARMS = ("A", "B", "C", "D", "N1", "N2", "N3")
ARM_TO_DOMAIN = {
    "A": "math",
    "B": "none",
    "C": "math",
    "Cp": "math",
    "D": "cooking",
    "A-B": "math",
    "N1": "none",
    "N2": "none",
    "N3": "none",
}
MOCK_RE = re.compile(r"(?:^|[_-])mock(?:[_-]|$)", flags=re.IGNORECASE)
SHA256_RE = re.compile(r"[0-9a-f]{64}", flags=re.IGNORECASE)
Z_95 = 1.959963984540054
PRIMARY_LEXICAL_VARIANT = "prose_1_2gram"
BLOCK_FIELDS = ("block", "block_id", "block_index")
K_BLOCKS = 10


@dataclass(frozen=True)
class InputSet:
    """The non-mixed set of input files selected for one analysis run."""

    mode: str
    judged: tuple[Path, ...]
    diffs: tuple[Path, ...]


@dataclass
class DiffVector:
    """One source or derived vector and the metadata needed in the CSV."""

    vector_id: str
    vector: np.ndarray
    meta: dict[str, Any]
    d_norm: float
    constancy: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def git_commit(start: Path) -> str:
    """Return the repository commit without making git a runtime requirement."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=start, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _filename_mode(path: Path) -> str:
    return "mock" if MOCK_RE.search(path.stem) else "real"


def discover_inputs(results_dir: Path, mode: str = "auto") -> InputSet:
    """Discover inputs and guarantee that a run cannot combine mock with real."""
    results_dir = Path(results_dir)
    judged_all = tuple(sorted(results_dir.glob("judged_*.jsonl")))
    diffs_all = tuple(sorted(results_dir.glob("diff_*.json")))
    all_paths = judged_all + diffs_all
    if not all_paths:
        raise FileNotFoundError(
            f"No judged_*.jsonl or diff_*.json inputs found under {results_dir}"
        )

    present = {_filename_mode(p) for p in all_paths}
    if mode == "auto":
        if len(present) > 1:
            raise RuntimeError(
                "Both MOCK and real result files are present. Refusing to mix them; "
                "rerun with --mode mock or --mode real."
            )
        selected_mode = next(iter(present))
    else:
        selected_mode = mode

    judged = tuple(p for p in judged_all if _filename_mode(p) == selected_mode)
    diffs = tuple(p for p in diffs_all if _filename_mode(p) == selected_mode)
    if not judged:
        raise FileNotFoundError(
            f"No {selected_mode} judged_*.jsonl inputs found under {results_dir}"
        )
    if not diffs:
        raise FileNotFoundError(
            f"No {selected_mode} diff_*.json inputs found under {results_dir}"
        )
    return InputSet(selected_mode, judged, diffs)


def _assert_mock_marker(path: Path, obj: dict[str, Any], selected_mode: str) -> None:
    """Cross-check filename and row metadata, catching accidentally renamed mocks."""
    marker = obj.get("is_mock")
    if selected_mode == "mock" and marker is not True:
        raise ValueError(f"MOCK input lacks is_mock=true: {path}")
    if selected_mode == "real" and marker is True:
        raise ValueError(f"Real-named input contains is_mock=true: {path}")


def load_judged(paths: Sequence[Path], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_item_ids: dict[str, str] = {}
    for path in paths:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Judged input is not UTF-8: {path}") from exc
        source_sha256 = hashlib.sha256(raw).hexdigest()
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            _assert_mock_marker(path, row, mode)
            required = ("arm", "seed", "layer", "snippet_set", "modality", "text")
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"Missing {missing} at {path}:{line_no}")
            if "true" not in row:
                raise ValueError(
                    f"Missing true label at {path}:{line_no}; lexical accuracy cannot be computed"
                )
            arm = _canonical_arm(row["arm"])
            if arm not in ARM_TO_DOMAIN:
                raise ValueError(f"Unknown arm {row['arm']!r} at {path}:{line_no}")
            if row["true"] not in LABELS or row["true"] != ARM_TO_DOMAIN[arm]:
                raise ValueError(
                    f"True label is inconsistent with arm {arm!r} at {path}:{line_no}"
                )
            if "pred" not in row:
                raise ValueError(f"Missing pred at {path}:{line_no}")
            if row["pred"] not in LABELS:
                raise ValueError(f"Unknown judge prediction at {path}:{line_no}: {row['pred']!r}")
            expected_correct = row["pred"] == row["true"]
            if "correct" not in row:
                row["correct"] = expected_correct
            elif row["correct"] != expected_correct:
                raise ValueError(f"correct disagrees with pred/true at {path}:{line_no}")
            if "correct_shuffled" not in row and {
                "pred",
                "shuffled_true",
            }.issubset(row):
                row["correct_shuffled"] = row["pred"] == row["shuffled_true"]
            if not isinstance(row["correct"], bool):
                raise ValueError(f"correct must be a JSON boolean at {path}:{line_no}")
            if row["modality"] in PRIMARY_MODALITIES:
                item_id = row.get("item_id")
                if not isinstance(item_id, str) or not item_id:
                    raise ValueError(f"Missing non-empty item_id at {path}:{line_no}")
                source = f"{path}:{line_no}"
                if item_id in seen_item_ids:
                    raise ValueError(
                        f"Duplicate judged item_id {item_id!r}: "
                        f"{seen_item_ids[item_id]} and {source}"
                    )
                seen_item_ids[item_id] = source
                if "correct_shuffled" not in row:
                    raise ValueError(
                        f"Missing label-shuffled control at {path}:{line_no}"
                    )
                if not isinstance(row["correct_shuffled"], bool):
                    raise ValueError(
                        f"correct_shuffled must be a JSON boolean at {path}:{line_no}"
                    )
                if row.get("shuffled_true") not in LABELS:
                    raise ValueError(f"Invalid shuffled_true at {path}:{line_no}")
                if row["correct_shuffled"] != (
                    row["pred"] == row["shuffled_true"]
                ):
                    raise ValueError(
                        f"correct_shuffled disagrees with pred/shuffled_true at {path}:{line_no}"
                    )
                if row.get("shuffled_control_valid") is not True:
                    raise ValueError(
                        f"Invalid or degenerate shuffled-label control at {path}:{line_no}"
                    )
            row["_source_file"] = path.name
            row["_source_line"] = line_no
            row["_source_sha256"] = source_sha256
            rows.append(row)
    if not rows:
        raise ValueError("The selected judged files contain no rows")
    return rows


def _canonical_arm(value: Any) -> str:
    arm = str(value).strip().replace("−", "-").replace("–", "-")
    if arm.upper() in {"A_B", "A-B", "ABDIFF", "A_MINUS_B"}:
        return "A-B"
    return arm


def _meta_value(meta: dict[str, Any], *keys: str, default: Any = "unknown") -> Any:
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


def load_diff_vectors(paths: Sequence[Path], mode: str) -> list[DiffVector]:
    vectors: list[DiffVector] = []
    seen_ids: Counter[str] = Counter()
    for path in paths:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(meta, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        _assert_mock_marker(path, meta, mode)
        vector_path = path.with_suffix(".npy")
        if not vector_path.exists():
            raise FileNotFoundError(f"Missing vector paired with {path}: {vector_path}")
        vector = np.load(vector_path, allow_pickle=False)
        if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.number):
            raise ValueError(f"Expected a numeric 1-D diff vector in {vector_path}, got {vector.shape}")
        required_array = (
            "artifact_schema_version",
            "artifact_type",
            "array_file",
            "array_shape",
            "array_dtype",
            "array_sha256",
        )
        missing_array = [field for field in required_array if field not in meta]
        if missing_array:
            raise ValueError(f"Missing diff array receipt {missing_array} in {path}")
        if meta["artifact_schema_version"] != 1:
            raise ValueError(f"Unsupported diff artifact schema in {path}")
        if meta["artifact_type"] not in {
            "activation_difference",
            "derived_activation_difference",
        }:
            raise ValueError(f"Unexpected diff artifact type in {path}")
        actual_array_receipt = {
            "array_file": vector_path.name,
            "array_shape": list(vector.shape),
            "array_dtype": str(vector.dtype),
            "array_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        }
        receipt_mismatches = [
            field
            for field, actual in actual_array_receipt.items()
            if meta[field] != actual
        ]
        if receipt_mismatches:
            raise ValueError(
                f"Diff vector receipt mismatch for {path}: {receipt_mismatches}"
            )
        vector = vector.astype(np.float64, copy=False)
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"Non-finite values in {vector_path}")
        required = ("arm", "seed", "layer", "snippet_set")
        missing = [key for key in required if key not in meta]
        if missing:
            raise ValueError(f"Missing {missing} in {path}")

        arm = _canonical_arm(meta["arm"])
        seed = _meta_value(meta, "seed")
        step = _meta_value(meta, "step", "checkpoint_step", default=-1)
        layer = _meta_value(meta, "layer")
        snippet = _meta_value(meta, "snippet_set")
        snippet_sha = _meta_value(meta, "snippet_sha", "snippet_set_sha256")
        base_id = f"{arm}|s{seed}|step{step}|L{layer}|{snippet}|{snippet_sha}"
        seen_ids[base_id] += 1
        if seen_ids[base_id] > 1:
            raise ValueError(
                "Duplicate diff-vector provenance would silently double-count a run: "
                f"{base_id}"
            )
        vector_id = base_id
        declared_norm = _safe_float(meta.get("d_norm"))
        actual_norm = float(np.linalg.norm(vector))
        if math.isfinite(declared_norm) and not math.isclose(
            declared_norm, actual_norm, rel_tol=2e-5, abs_tol=2e-6
        ):
            raise ValueError(
                f"{path.name}: declared d_norm={declared_norm:.8g} "
                f"disagrees with vector norm={actual_norm:.8g}"
            )
        vectors.append(
            DiffVector(
                vector_id=vector_id,
                vector=vector,
                meta={**meta, "arm": arm, "_source_file": path.name},
                d_norm=actual_norm,
                constancy=_safe_float(meta.get("constancy")),
            )
        )
    return vectors


def select_analysis_layer(
    rows: Sequence[dict[str, Any]],
    vectors: Sequence[DiffVector],
    requested_layer: int | None,
) -> tuple[list[dict[str, Any]], list[DiffVector], int]:
    """Select one layer, refusing to average layer-robustness runs together."""
    available: set[int] = set()
    for record in list(rows) + [vector.meta for vector in vectors]:
        try:
            available.add(int(record["layer"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid or missing layer metadata: {record.get('layer')!r}") from exc
    if not available:
        raise ValueError("No layer metadata found")
    if requested_layer is None:
        if len(available) != 1:
            raise ValueError(
                f"Inputs contain multiple layers {sorted(available)}; pass --layer to select one "
                "instead of pooling them"
            )
        chosen = next(iter(available))
    else:
        chosen = int(requested_layer)
        if chosen not in available:
            raise ValueError(
                f"Requested layer {chosen} is absent; available layers: {sorted(available)}"
            )
    chosen_rows = [row for row in rows if int(row["layer"]) == chosen]
    chosen_vectors = [vector for vector in vectors if int(vector.meta["layer"]) == chosen]
    if not chosen_rows or not chosen_vectors:
        raise ValueError(
            f"Layer {chosen} must have both judged rows and diff vectors; got "
            f"{len(chosen_rows)} rows and {len(chosen_vectors)} vectors"
        )
    return chosen_rows, chosen_vectors, chosen


def validate_analysis_inputs(
    rows: Sequence[dict[str, Any]], vectors: Sequence[DiffVector]
) -> None:
    """Reject provenance combinations that would make aggregate plots ambiguous."""
    required_aliases = {
        "arm": ("arm",),
        "seed": ("seed",),
        "checkpoint step": ("checkpoint_step", "step"),
        "layer": ("layer",),
        "snippet-set name": ("snippet_set",),
        "snippet-set hash": ("snippet_sha", "snippet_set_sha256"),
        "judge model": ("judge_model",),
        "timestamp": ("timestamp", "ts"),
        "git commit": ("git_commit",),
    }
    records: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        if row["modality"] in PRIMARY_MODALITIES:
            records.append(
                (f"{row.get('_source_file', 'judged')}:{row.get('_source_line', '?')}", row)
            )
    records.extend(
        (str(vector.meta.get("_source_file", vector.vector_id)), vector.meta)
        for vector in vectors
    )
    for source, record in records:
        missing = [
            label
            for label, aliases in required_aliases.items()
            if _meta_value(record, *aliases) == "unknown"
        ]
        if missing:
            raise ValueError(f"Missing mandatory metadata {missing} in {source}")
        digest = str(_meta_value(record, "snippet_sha", "snippet_set_sha256"))
        if not SHA256_RE.fullmatch(digest):
            raise ValueError(
                f"snippet-set hash must be a full 64-hex SHA-256 in {source}, got {digest!r}"
            )

    hashes: dict[str, set[str]] = defaultdict(set)
    for _source, record in records:
        hashes[str(record["snippet_set"])].add(
            str(_meta_value(record, "snippet_sha", "snippet_set_sha256"))
        )
    conflicts = {name: sorted(values) for name, values in hashes.items() if len(values) > 1}
    if conflicts:
        raise ValueError(
            "A snippet-set name maps to multiple hashes; refusing to pool incompatible inputs: "
            f"{conflicts}"
        )

    judge_models = {
        str(row["judge_model"])
        for row in rows
        if row["modality"] in PRIMARY_MODALITIES
    }
    if len(judge_models) > 1:
        raise ValueError(
            f"Inputs contain multiple judge models {sorted(judge_models)}; analyze them separately"
        )
    judged_steps: dict[tuple[str, Any, str, str], set[Any]] = defaultdict(set)
    for row in rows:
        if row["modality"] in PRIMARY_MODALITIES:
            key = (
                _canonical_arm(row["arm"]),
                row["seed"],
                str(row["snippet_set"]),
                str(row["modality"]),
            )
            judged_steps[key].add(_meta_value(row, "checkpoint_step", "step"))
    mixed_judged_steps = {
        key: sorted(steps, key=str) for key, steps in judged_steps.items() if len(steps) > 1
    }
    if mixed_judged_steps:
        raise ValueError(
            "Judged cells contain multiple checkpoint steps; select final-checkpoint inputs only: "
            f"{mixed_judged_steps}"
        )

    diff_steps: dict[tuple[str, Any, str], set[Any]] = defaultdict(set)
    for vector in vectors:
        key = (
            _canonical_arm(vector.meta["arm"]),
            vector.meta["seed"],
            str(vector.meta["snippet_set"]),
        )
        diff_steps[key].add(_meta_value(vector.meta, "checkpoint_step", "step"))
    mixed_diff_steps = {
        key: sorted(steps, key=str) for key, steps in diff_steps.items() if len(steps) > 1
    }
    if mixed_diff_steps:
        raise ValueError(
            "Diff-vector cells contain multiple checkpoint steps; select final-checkpoint inputs only: "
            f"{mixed_diff_steps}"
        )

    widths = {vector.vector.shape[0] for vector in vectors}
    if len(widths) > 1:
        raise ValueError(
            f"Diff vectors have incompatible widths {sorted(widths)}; analyze models separately"
        )
    for vector in vectors:
        if math.isfinite(vector.constancy) and not 0.0 <= vector.constancy <= 1.0 + 1e-6:
            raise ValueError(
                f"Constancy outside [0, 1] for {vector.vector_id}: {vector.constancy}"
            )
        if vector.meta.get("is_mock") is not True:
            arm = _canonical_arm(vector.meta["arm"])
            if arm in {"A", "B", "C", "D"}:
                receipt = vector.meta.get("adapter_receipt")
                if not isinstance(receipt, dict) or receipt.get(
                    "training_receipt_verified"
                ) is not True:
                    raise ValueError(
                        f"Real {arm} vector lacks a verified adapter training receipt: "
                        f"{vector.vector_id}"
                    )
                for field in (
                    "adapter_config_sha256",
                    "adapter_weight_sha256",
                    "training_receipt_sha256",
                ):
                    if not SHA256_RE.fullmatch(str(receipt.get(field, ""))):
                        raise ValueError(
                            f"Real {arm} vector has an invalid {field}: {vector.vector_id}"
                        )

    vector_receipts: dict[tuple[Any, ...], list[DiffVector]] = defaultdict(list)
    d_receipts: dict[tuple[Any, ...], list[DiffVector]] = defaultdict(list)
    for vector in vectors:
        meta = vector.meta
        key = (
            _canonical_arm(meta["arm"]),
            meta["seed"],
            _meta_value(meta, "checkpoint_step", "step"),
            int(meta["layer"]),
            str(meta["snippet_set"]),
            str(_meta_value(meta, "snippet_sha", "snippet_set_sha256")),
            str(_meta_value(meta, "base")),
        )
        vector_receipts[key].append(vector)
        if key[0] == "D":
            d_receipts[(key[3], key[4], key[5], key[6])].append(vector)

    for row in rows:
        modality = str(row["modality"])
        snippet = str(row["snippet_set"])
        if modality not in PRIMARY_MODALITIES or snippet not in PRIMARY_SNIPPETS:
            continue
        source = f"{row.get('_source_file', 'judged')}:{row.get('_source_line', '?')}"
        if row.get("norm_matched_before_decode") is not True:
            raise ValueError(f"Missing norm-match receipt in {source}")
        if row.get("target_norm_reference_arm") != "D":
            raise ValueError(f"Non-D norm reference in {source}")
        if row.get("target_norm_provenance_verified") is not True:
            raise ValueError(f"Unverified target-norm provenance in {source}")
        if modality == "tokens" and row.get("logit_lens_final_norm_applied") is not True:
            raise ValueError(f"Missing final-norm logit-lens receipt in {source}")
        key = (
            _canonical_arm(row["arm"]),
            row["seed"],
            _meta_value(row, "checkpoint_step", "step"),
            int(row["layer"]),
            snippet,
            str(_meta_value(row, "snippet_sha", "snippet_set_sha256")),
            str(_meta_value(row, "base")),
        )
        matches = vector_receipts.get(key, [])
        if key[0] == "A-B" and not matches:
            # Mock-layout fixtures derive A-B in memory. Scientific runs emit
            # an authenticated explicit A-B artifact, checked below.
            a_key = ("A", *key[1:])
            b_key = ("B", *key[1:])
            if len(vector_receipts.get(a_key, [])) == 1 and len(
                vector_receipts.get(b_key, [])
            ) == 1 and row.get("is_mock") is True:
                matches = [vector_receipts[a_key][0]]
        if len(matches) != 1:
            raise ValueError(
                f"Judged evidence in {source} has {len(matches)} matching diff receipts; "
                "expected exactly one"
            )
        if row.get("is_mock") is not True:
            d_key = (key[3], key[4], key[5], key[6])
            d_matches = d_receipts.get(d_key, [])
            if len(d_matches) != 1:
                raise ValueError(f"{source} has {len(d_matches)} matching arm-D norm receipts")
            d_vector = d_matches[0]
            expected_d = {
                "target_norm_source_sha256": d_vector.meta["array_sha256"],
                "target_norm_reference_seed": d_vector.meta["seed"],
                "target_norm_reference_checkpoint_step": _meta_value(
                    d_vector.meta, "checkpoint_step", "step"
                ),
                "target_norm_reference_alignment_sha256": d_vector.meta.get(
                    "alignment_sha256"
                ),
            }
            bad = {
                field: {"expected": expected, "found": row.get(field)}
                for field, expected in expected_d.items()
                if row.get(field) != expected
            }
            if bad:
                raise ValueError(f"Arm-D norm receipt mismatch in {source}: {bad}")

    # Figure 1 always requires token readouts. Steering is conditional and C is
    # launched only after Gate 2, so neither may be made an unconditional gate.
    primary_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        modality = str(row["modality"])
        snippet = str(row["snippet_set"])
        arm = _canonical_arm(row["arm"])
        if modality in PRIMARY_MODALITIES and snippet in PRIMARY_SNIPPETS:
            primary_counts[(snippet, modality)][arm] += 1
    required_arms = tuple(arm for arm in PHYSICAL_ARMS if arm != "C")
    real_rows = [
        row
        for row in rows
        if row["modality"] in PRIMARY_MODALITIES and row.get("is_mock") is not True
    ]
    for snippet in PRIMARY_SNIPPETS:
        if not primary_counts.get((snippet, "tokens")):
            raise ValueError(f"Incomplete primary judge cell {(snippet, 'tokens')}; no token rows")
        available_modalities = [
            modality
            for modality in PRIMARY_MODALITIES
            if primary_counts.get((snippet, modality))
        ]
        for modality in available_modalities:
            cell = (snippet, modality)
            counts = primary_counts.get(cell, Counter())
            missing = [arm for arm in required_arms if counts[arm] == 0]
            if modality == "tokens" and missing:
                raise ValueError(f"Incomplete primary judge cell {cell}; missing arms {missing}")
            if not real_rows:
                present = {arm: count for arm, count in counts.items() if arm in PHYSICAL_ARMS}
                if len(set(present.values())) > 1:
                    raise ValueError(f"Unbalanced MOCK primary judge cell {cell}: {present}")

    # For real rows that carry block metadata, the block—not repeated ratings—
    # is the frozen sampling unit.  Validate separately at the full scientific
    # cell grain so layers, checkpoints, and seeds cannot be pooled.
    block_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in real_rows:
        block_groups[
            (
                _canonical_arm(row["arm"]),
                row["seed"],
                _meta_value(row, "checkpoint_step", "step"),
                row["layer"],
                str(row["snippet_set"]),
                str(row["modality"]),
            )
        ].append(row)
    for key, group in block_groups.items():
        values = [
            next((row[field] for field in BLOCK_FIELDS if field in row), None)
            for row in group
        ]
        if not any(value is not None for value in values):
            raise ValueError(f"Real primary cell {key} lacks block identifiers")
        if not all(value is not None for value in values):
            raise ValueError(f"Real primary cell {key} mixes rows with/without blocks")
        unique_blocks = set(values)
        if len(unique_blocks) != K_BLOCKS or len(group) != K_BLOCKS:
            raise ValueError(
                f"Real primary cell {key} has {len(group)} rows and "
                f"{len(unique_blocks)} unique blocks; PREREG requires K={K_BLOCKS} "
                "one-decision-per-block"
            )

    for snippet in PRIMARY_SNIPPETS:
        if primary_counts[(snippet, "tokens")]["A-B"] == 0:
            raise ValueError(f"Missing explicit A-B token readout for {snippet}")


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n < 0 or successes < 0 or successes > n:
        raise ValueError((successes, n))
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _accuracy(values: Iterable[Any]) -> dict[str, float | int]:
    arr = np.asarray([bool(value) for value in values], dtype=np.int8)
    n = int(arr.size)
    successes = int(arr.sum())
    lo, hi = wilson_interval(successes, n)
    return {"accuracy": successes / n if n else math.nan, "low": lo, "high": hi, "n": n}


def _read_lexical_prediction_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load canonical external-baseline JSONL and return its exact file hash."""

    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Lexical prediction file is not UTF-8: {path}") from exc
    parsed: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid lexical prediction JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected lexical prediction object at {path}:{line_number}")
        row = dict(row)
        row["_lexical_source_file"] = path.name
        row["_lexical_source_line"] = line_number
        parsed.append(row)
    if not parsed:
        raise ValueError(f"Lexical prediction file contains no JSONL rows: {path}")

    filename_is_mock = bool(MOCK_RE.search(path.stem))
    explicit_modes = {row.get("is_mock") for row in parsed if "is_mock" in row}
    if any(type(value) is not bool for value in explicit_modes):
        raise ValueError(f"Lexical prediction is_mock values must be booleans: {path}")
    if len(explicit_modes) > 1:
        raise ValueError(f"Lexical prediction file mixes MOCK and real rows: {path}")
    if filename_is_mock and explicit_modes == {False}:
        raise ValueError(f"MOCK lexical filename contains is_mock=false rows: {path}")
    return parsed, hashlib.sha256(raw).hexdigest()


def _leakage_receipt_passed(prediction: dict[str, Any]) -> bool:
    receipt = prediction.get("lexical_leakage_check")
    if isinstance(receipt, dict):
        return (
            receipt.get("passed") is True
            and receipt.get("exact_matches") == 0
            and receipt.get("shared_8gram_shingles") == 0
        )
    return prediction.get("lexical_leakage_check_passed") is True


def _same_prediction_item(prediction: dict[str, Any], judged: dict[str, Any]) -> None:
    """Reject a stable-id match whose copied readout provenance was altered."""

    aliases: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("arm", ("arm",)),
        ("seed", ("seed",)),
        ("checkpoint step", ("checkpoint_step", "step")),
        ("layer", ("layer",)),
        ("snippet set", ("snippet_set",)),
        ("modality", ("modality",)),
        ("text", ("text",)),
        ("snippet SHA-256", ("snippet_sha", "snippet_set_sha256", "snippet_sha256")),
    )
    mismatches: dict[str, dict[str, Any]] = {}
    for label, keys in aliases:
        predicted_value = _meta_value(prediction, *keys)
        if predicted_value == "unknown":
            # Rows written by the canonical lexical CLI retain all of these
            # fields.  Optionality here keeps the JSONL join compatible with a
            # deliberately minimal prediction exporter while the stable item
            # id and exact source-file hash remain mandatory.
            continue
        judged_value = _meta_value(judged, *keys)
        if predicted_value != judged_value:
            mismatches[label] = {"judged": judged_value, "lexical": predicted_value}
    if mismatches:
        raise ValueError(
            f"Lexical prediction provenance disagrees for item {judged['item_id']!r}: "
            f"{mismatches}"
        )


def _attach_mock_lexical_placeholders(rows: list[dict[str, Any]], seed: int) -> None:
    """Supply MOCK-only layout values without training or reading model outputs."""

    for row in rows:
        if row["modality"] not in PRIMARY_MODALITIES:
            continue
        payload = f"MOCK-LEXICAL-PLACEHOLDER:{seed}:{row['item_id']}".encode("utf-8")
        prediction = LABELS[int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(LABELS)]
        row["_lexical_pred"] = prediction
        row["_lexical_correct"] = prediction == str(row["true"])
        row["_lexical_variant"] = "MOCK_deterministic_placeholder_no_fit"
        row["_lexical_is_placeholder"] = True


def add_lexical_predictions(
    rows: list[dict[str, Any]],
    seed: int = 0,
    prediction_paths: Sequence[Path] | None = None,
    mode: str | None = None,
) -> None:
    """Join persisted external-corpus predictions to judged rows in place.

    The canonical JSONL has one row per ``item_id`` and lexical variant.  This
    analyzer consumes only ``prose_1_2gram`` for Figure 1; the separate
    ``token_bag_unigram`` variant remains available in the lexical result file.
    Each real prediction must be bound to the exact judged JSONL bytes, a full
    external-reference-manifest hash, and a passing exact/8-gram leakage receipt.

    For backward-compatible MOCK layout tests only, omitting ``prediction_paths``
    creates deterministic placeholder labels.  That path performs no fitting and
    is impossible for real rows.
    """

    primary_rows = [row for row in rows if row["modality"] in PRIMARY_MODALITIES]
    if not primary_rows:
        return
    inferred_mode = "mock" if all(row.get("is_mock") is True for row in primary_rows) else "real"
    if any((row.get("is_mock") is True) != (inferred_mode == "mock") for row in primary_rows):
        raise ValueError("Judged rows mix MOCK and real provenance")
    selected_mode = inferred_mode if mode is None else mode
    if selected_mode not in {"mock", "real"}:
        raise ValueError(f"Unknown analysis mode for lexical predictions: {selected_mode!r}")
    if selected_mode != inferred_mode:
        raise ValueError(
            f"Lexical mode {selected_mode!r} disagrees with judged rows ({inferred_mode})"
        )

    paths = [Path(path) for path in (prediction_paths or ())]
    if not paths:
        if selected_mode == "mock":
            _attach_mock_lexical_placeholders(rows, seed)
            return
        raise ValueError(
            "Real analysis requires persisted external-corpus lexical predictions; "
            "run judge/lexical_baseline.py --reference-dir data/lexical_reference "
            "--predictions-out <JSONL> and pass --lexical-predictions <JSONL>"
        )

    by_item: dict[str, tuple[dict[str, Any], str]] = {}
    for path in paths:
        predictions, file_sha256 = _read_lexical_prediction_rows(path)
        for prediction in predictions:
            if prediction.get("lexical_variant") != PRIMARY_LEXICAL_VARIANT:
                continue
            item_id = prediction.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(
                    f"Primary lexical prediction lacks item_id in "
                    f"{path}:{prediction['_lexical_source_line']}"
                )
            if item_id in by_item:
                prior = by_item[item_id][0]
                raise ValueError(
                    f"Duplicate {PRIMARY_LEXICAL_VARIANT} lexical prediction for {item_id!r}: "
                    f"{prior['_lexical_source_file']} and {path.name}"
                )
            by_item[item_id] = (prediction, file_sha256)

    reference_hashes: set[str] = set()
    reference_corpus_hashes: set[str] = set()
    for row in primary_rows:
        item_id = row.get("item_id")
        if item_id not in by_item:
            raise ValueError(
                f"Missing {PRIMARY_LEXICAL_VARIANT} external lexical prediction for "
                f"judged item {item_id!r}"
            )
        prediction, prediction_file_sha = by_item[item_id]
        if type(prediction.get("is_mock")) is not bool:
            raise ValueError(
                f"Lexical prediction lacks an explicit boolean is_mock for item {item_id!r}"
            )
        prediction_is_mock = prediction.get("is_mock") is True
        if prediction_is_mock != (selected_mode == "mock"):
            raise ValueError(
                f"Lexical prediction MOCK/real status disagrees for item {item_id!r}"
            )
        lexical_pred = prediction.get("lexical_pred")
        if lexical_pred not in LABELS:
            raise ValueError(f"Invalid lexical_pred for item {item_id!r}: {lexical_pred!r}")
        expected_correct = lexical_pred == str(row["true"])
        if prediction.get("lexical_correct") is not expected_correct:
            raise ValueError(
                f"lexical_correct disagrees with lexical_pred/true for item {item_id!r}"
            )
        source_sha = prediction.get("lexical_source_input_sha256")
        judged_source_sha = row.get("_source_sha256")
        if not SHA256_RE.fullmatch(str(source_sha)) or source_sha != judged_source_sha:
            raise ValueError(
                f"Lexical prediction source-input SHA-256 mismatch for item {item_id!r}"
            )
        reference_sha = prediction.get("lexical_reference_manifest_sha256")
        if not SHA256_RE.fullmatch(str(reference_sha)):
            raise ValueError(
                f"Lexical prediction lacks a full reference-manifest SHA-256 for item {item_id!r}"
            )
        reference_corpus_sha = prediction.get("lexical_reference_corpus_sha256")
        if not SHA256_RE.fullmatch(str(reference_corpus_sha)):
            raise ValueError(
                f"Lexical prediction lacks a full reference-corpus SHA-256 for item {item_id!r}"
            )
        if selected_mode == "real" and not _leakage_receipt_passed(prediction):
            raise ValueError(
                f"Real lexical prediction lacks a passing exact/8-gram leakage receipt "
                f"for item {item_id!r}"
            )
        if selected_mode == "real":
            for field in (
                "lexical_model_config",
                "lexical_timestamp",
                "lexical_git_commit",
                "lexical_script_sha256",
                "lexical_sklearn_version",
            ):
                if not prediction.get(field):
                    raise ValueError(
                        f"Real lexical prediction is missing {field!r} for item {item_id!r}"
                    )
            if prediction.get("lexical_training_source") != "external_reference_corpus_only":
                raise ValueError(
                    f"Real lexical prediction has a non-external training source for item {item_id!r}"
                )
            config = prediction["lexical_model_config"]
            if not isinstance(config, dict):
                raise ValueError(
                    f"Real lexical prediction has a non-object model config for item {item_id!r}"
                )
            expected_config = {
                "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
                "ngram_range": [1, 2],
                "min_df": 1,
                "sublinear_tf": True,
                "classifier": "sklearn.linear_model.LogisticRegression",
                "max_iter": 2000,
            }
            bad_config = {
                key: {"expected": expected, "found": config.get(key)}
                for key, expected in expected_config.items()
                if config.get(key) != expected
            }
            if bad_config:
                raise ValueError(
                    f"Real lexical prediction has a non-frozen model config for "
                    f"item {item_id!r}: {bad_config}"
                )
            if not SHA256_RE.fullmatch(str(prediction["lexical_script_sha256"])):
                raise ValueError(
                    f"Real lexical prediction has an invalid script SHA-256 for item {item_id!r}"
                )
        _same_prediction_item(prediction, row)

        reference_hashes.add(str(reference_sha))
        reference_corpus_hashes.add(str(reference_corpus_sha))
        row["_lexical_pred"] = lexical_pred
        row["_lexical_correct"] = expected_correct
        row["_lexical_variant"] = PRIMARY_LEXICAL_VARIANT
        row["_lexical_is_placeholder"] = False
        row["_lexical_reference_manifest_sha256"] = reference_sha
        row["_lexical_reference_corpus_sha256"] = reference_corpus_sha
        row["_lexical_prediction_source_file"] = prediction["_lexical_source_file"]
        row["_lexical_prediction_source_sha256"] = prediction_file_sha

    if len(reference_hashes) != 1:
        raise ValueError(
            "Selected lexical predictions use multiple external reference manifests: "
            f"{sorted(reference_hashes)}"
        )
    if len(reference_corpus_hashes) != 1:
        raise ValueError(
            "Selected lexical predictions use multiple external reference corpora: "
            f"{sorted(reference_corpus_hashes)}"
        )


def accuracy_summaries(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, dict[str, float | int]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        modality = str(row["modality"])
        snippet = str(row["snippet_set"])
        if modality not in PRIMARY_MODALITIES or snippet == "-":
            continue
        grouped[(_canonical_arm(row["arm"]), snippet, modality)].append(row)

    output: dict[tuple[str, str, str], dict[str, dict[str, float | int]]] = {}
    for key, group in grouped.items():
        lexical = [row["_lexical_correct"] for row in group if row.get("_lexical_correct") is not None]
        shuffled = [row["correct_shuffled"] for row in group if "correct_shuffled" in row]
        output[key] = {
            "judge": _accuracy(row["correct"] for row in group),
            "lexical": _accuracy(lexical),
            "shuffled": _accuracy(shuffled),
        }
    return output


def _ordered(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    unique = set(values)
    return [value for value in preferred if value in unique] + sorted(unique - set(preferred))


def _errorbar_arrays(stats: Sequence[dict[str, float | int]]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([float(item["accuracy"]) for item in stats], dtype=float)
    lower = np.asarray([float(item["low"]) for item in stats], dtype=float)
    upper = np.asarray([float(item["high"]) for item in stats], dtype=float)
    # Clip only sub-ulp roundoff at exact 0/1; matplotlib rejects negative yerr.
    return values, np.maximum(0.0, np.vstack((values - lower, upper - values)))


def _figure_metadata(
    title: str,
    mode: str,
    commit: str,
    timestamp: str,
    inputs: InputSet,
    run_metadata: dict[str, Any],
) -> dict[str, str]:
    sources = [path.name for path in inputs.judged + inputs.diffs]
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
                **run_metadata,
            },
            sort_keys=True,
        ),
        "Software": "matplotlib; analysis/summarize.py",
    }


def collect_run_metadata(
    rows: Sequence[dict[str, Any]], vectors: Sequence[DiffVector]
) -> dict[str, Any]:
    """Collect the mandatory provenance fields embedded in every output PNG."""
    records = list(rows) + [vector.meta for vector in vectors]

    def values(*keys: str) -> list[Any]:
        found = {
            _meta_value(record, *keys)
            for record in records
            if _meta_value(record, *keys) != "unknown"
        }
        return sorted(found, key=lambda value: str(value))

    snippet_hashes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        name = str(_meta_value(record, "snippet_set"))
        digest = str(_meta_value(record, "snippet_sha", "snippet_set_sha256"))
        if name != "unknown" and digest != "unknown" and digest not in snippet_hashes[name]:
            snippet_hashes[name].append(digest)
    lexical_references = sorted(
        {
            str(row["_lexical_reference_manifest_sha256"])
            for row in rows
            if row.get("_lexical_reference_manifest_sha256")
        }
    )
    lexical_corpora = sorted(
        {
            str(row["_lexical_reference_corpus_sha256"])
            for row in rows
            if row.get("_lexical_reference_corpus_sha256")
        }
    )
    lexical_files = {
        str(row["_lexical_prediction_source_file"]): str(
            row["_lexical_prediction_source_sha256"]
        )
        for row in rows
        if row.get("_lexical_prediction_source_file")
        and row.get("_lexical_prediction_source_sha256")
    }
    return {
        "arms": _ordered((_canonical_arm(value) for value in values("arm")), ARM_ORDER),
        "seeds": values("seed"),
        "checkpoint_steps": values("checkpoint_step", "step"),
        "layers": values("layer"),
        "snippet_sets_and_hashes": dict(sorted(snippet_hashes.items())),
        "judge_models": values("judge_model"),
        "lexical_variant": PRIMARY_LEXICAL_VARIANT,
        "lexical_reference_manifest_sha256": lexical_references,
        "lexical_reference_corpus_sha256": lexical_corpora,
        "lexical_prediction_files": dict(sorted(lexical_files.items())),
        "lexical_mock_placeholder": any(
            row.get("_lexical_is_placeholder") is True for row in rows
        ),
    }


def _atomic_savefig(fig: plt.Figure, path: Path, metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fig.savefig(temporary, format="png", dpi=180, bbox_inches="tight", metadata=metadata)
    os.replace(temporary, path)


def _mock_watermark(fig: plt.Figure, mode: str) -> None:
    if mode == "mock":
        fig.text(
            0.995,
            0.005,
            "MOCK DATA — NOT A SCIENTIFIC RESULT",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#a61b1b",
            weight="bold",
        )


def plot_judge_accuracy(
    summaries: dict[tuple[str, str, str], dict[str, dict[str, float | int]]],
    output: Path,
    mode: str,
    metadata: dict[str, str],
) -> None:
    arms = _ordered((key[0] for key in summaries), ARM_ORDER)
    snippets = _ordered((key[1] for key in summaries), PRIMARY_SNIPPETS)
    modalities = _ordered((key[2] for key in summaries), PRIMARY_MODALITIES)
    if not arms or not snippets or not modalities:
        raise ValueError("No token/steer judge rows are available for Figure 1")

    fig, axes = plt.subplots(
        len(snippets),
        len(modalities),
        figsize=(max(7.2, 5.7 * len(modalities)), max(4.2, 3.7 * len(snippets))),
        sharey=True,
        squeeze=False,
    )
    width = 0.36
    for row_idx, snippet in enumerate(snippets):
        for col_idx, modality in enumerate(modalities):
            ax = axes[row_idx, col_idx]
            keys = [(arm, snippet, modality) for arm in arms]
            empty = {"accuracy": math.nan, "low": math.nan, "high": math.nan, "n": 0}
            judge_stats = [summaries.get(key, {}).get("judge", empty) for key in keys]
            lexical_stats = [summaries.get(key, {}).get("lexical", empty) for key in keys]
            shuffled_stats = [summaries.get(key, {}).get("shuffled", empty) for key in keys]
            judge_y, judge_err = _errorbar_arrays(judge_stats)
            lexical_y, lexical_err = _errorbar_arrays(lexical_stats)
            shuffled_y, shuffled_err = _errorbar_arrays(shuffled_stats)
            x = np.arange(len(arms), dtype=float)

            ax.bar(
                x - width / 2,
                judge_y,
                width,
                yerr=judge_err,
                capsize=2.5,
                color="#315f88",
                edgecolor="white",
                linewidth=0.5,
                label="LLM judge",
                zorder=2,
            )
            ax.bar(
                x + width / 2,
                lexical_y,
                width,
                yerr=lexical_err,
                capsize=2.5,
                facecolor="#e5b567",
                edgecolor="#8b642c",
                linewidth=0.7,
                hatch="///",
                label="TF-IDF baseline",
                zorder=2,
            )
            finite_shuffled = np.isfinite(shuffled_y)
            if finite_shuffled.any():
                ax.errorbar(
                    x[finite_shuffled],
                    shuffled_y[finite_shuffled],
                    yerr=shuffled_err[:, finite_shuffled],
                    fmt="o",
                    ms=3.7,
                    color="#60666c",
                    ecolor="#8b9298",
                    capsize=2,
                    label="Label-shuffled",
                    zorder=4,
                )
            ax.axhline(
                CHANCE,
                color="#a61b1b",
                linestyle="--",
                linewidth=1.2,
                label=f"Chance (1/{len(LABELS)})",
                zorder=1,
            )
            ax.set_title(f"{snippet} snippets · {modality}")
            ax.set_xticks(x, arms, rotation=35, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.grid(axis="y", alpha=0.2, linewidth=0.7)
            if col_idx == 0:
                ax.set_ylabel("Accuracy")
            ax.set_xlabel("Arm")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(4, len(unique)),
        frameon=False,
    )
    fig.suptitle("Blind domain decoding (95% Wilson confidence intervals)", y=1.055)
    fig.subplots_adjust(top=0.89, hspace=0.46, wspace=0.12)
    _mock_watermark(fig, mode)
    _atomic_savefig(fig, output, metadata)
    plt.close(fig)


def _matching_key(vector: DiffVector) -> tuple[Any, ...]:
    meta = vector.meta
    return (
        _meta_value(meta, "seed"),
        _meta_value(meta, "step", "checkpoint_step", default=-1),
        _meta_value(meta, "layer"),
        _meta_value(meta, "snippet_set"),
        _meta_value(meta, "snippet_sha", "snippet_set_sha256"),
        _meta_value(meta, "base"),
        vector.vector.shape[0],
    )


def derive_a_minus_b(vectors: Sequence[DiffVector]) -> list[DiffVector]:
    """Derive A-B only when all provenance fields match exactly."""
    grouped: dict[tuple[Any, ...], dict[str, list[DiffVector]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for vector in vectors:
        key = _matching_key(vector)
        if any(value == "unknown" for value in key[:-1]):
            warnings.warn(
                f"Skipping A-B derivation for {vector.vector_id}: incomplete provenance",
                stacklevel=2,
            )
            continue
        grouped[key][_canonical_arm(vector.meta["arm"])].append(vector)

    derived: list[DiffVector] = []
    for key, arms in sorted(grouped.items(), key=lambda item: repr(item[0])):
        if "A" not in arms or "B" not in arms:
            continue
        if len(arms["A"]) != 1 or len(arms["B"]) != 1:
            raise ValueError(
                "Ambiguous A-B derivation: more than one A or B vector has matching "
                f"provenance {key}"
            )
        a, b = arms["A"][0], arms["B"][0]
        diff = a.vector - b.vector
        if "A-B" in arms:
            if len(arms["A-B"]) != 1:
                raise ValueError(f"Ambiguous explicit A-B vector for provenance {key}")
            explicit = arms["A-B"][0]
            if not np.allclose(explicit.vector, diff, rtol=2e-6, atol=2e-7):
                raise ValueError(
                    f"Explicit A-B vector does not equal the paired A minus B for {key}"
                )
            meta = explicit.meta
            if (
                meta.get("derivation") != "d_A_minus_d_B"
                or meta.get("derived_source_provenance_verified") is not True
            ):
                raise ValueError(
                    f"Explicit A-B vector lacks verified derivation provenance for {key}"
                )
            source_hashes = {
                str(record.get("arm")): str(record.get("vector_sha256"))
                for record in meta.get("derived_from", [])
                if isinstance(record, dict)
            }
            expected_hashes = {
                "A": str(a.meta.get("array_sha256")),
                "B": str(b.meta.get("array_sha256")),
            }
            if source_hashes != expected_hashes:
                raise ValueError(
                    f"Explicit A-B source hashes do not match the loaded A/B vectors for {key}"
                )
            # The authenticated explicit vector is already in the source list.
            continue
        meta = {
            **a.meta,
            "arm": "A-B",
            "derived_from": [a.vector_id, b.vector_id],
            "constancy": None,
        }
        vector_id = (
            f"A-B|s{key[0]}|step{key[1]}|L{key[2]}|{key[3]}|{key[4]}"
        )
        derived.append(
            DiffVector(
                vector_id=vector_id,
                vector=diff,
                meta=meta,
                d_norm=float(np.linalg.norm(diff)),
                constancy=math.nan,
            )
        )
    return derived


def add_random_references(
    vectors: Sequence[DiffVector], seed: int, is_mock: bool
) -> list[DiffVector]:
    """Add one deterministic matched-norm random reference for every vector width."""
    by_width: dict[int, list[DiffVector]] = defaultdict(list)
    for vector in vectors:
        by_width[vector.vector.shape[0]].append(vector)
    output: list[DiffVector] = []
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(len(by_width))
    for (width, group), child in zip(sorted(by_width.items()), children):
        rng = np.random.default_rng(child)
        random_vector = rng.standard_normal(width).astype(np.float64)
        positive_norms = [item.d_norm for item in group if item.d_norm > 0]
        target_norm = float(np.mean(positive_norms)) if positive_norms else 1.0
        random_vector *= target_norm / np.linalg.norm(random_vector)
        output.append(
            DiffVector(
                vector_id=f"random|seed{seed}|d{width}",
                vector=random_vector,
                meta={
                    "arm": "random",
                    "seed": seed,
                    "step": -1,
                    "checkpoint_step": -1,
                    "layer": -1,
                    "snippet_set": "all",
                    "snippet_sha": "not_applicable",
                    "judge_model": "not_applicable",
                    "is_mock": is_mock,
                    "note": "deterministic random direction matched to mean nonzero vector norm",
                },
                d_norm=float(np.linalg.norm(random_vector)),
                constancy=math.nan,
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
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ids = [vector.vector_id for vector in vectors]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate vector IDs: {duplicates}")
    metadata_fields = [
        "vector_id",
        "arm",
        "seed",
        "checkpoint_step",
        "layer",
        "snippet_set",
        "snippet_sha",
        "judge_model",
        "timestamp",
        "git_commit",
        "source_timestamp",
        "source_git_commit",
        "is_mock",
        "source_file",
    ]
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields + ids)
        writer.writeheader()
        for left in vectors:
            meta = left.meta
            row: dict[str, Any] = {
                "vector_id": left.vector_id,
                "arm": _canonical_arm(meta.get("arm", "unknown")),
                "seed": _meta_value(meta, "seed"),
                "checkpoint_step": _meta_value(meta, "checkpoint_step", "step", default=-1),
                "layer": _meta_value(meta, "layer"),
                "snippet_set": _meta_value(meta, "snippet_set"),
                "snippet_sha": _meta_value(meta, "snippet_sha", "snippet_set_sha256"),
                "judge_model": _meta_value(meta, "judge_model", default="not_applicable"),
                "timestamp": timestamp,
                "git_commit": commit,
                "source_timestamp": _meta_value(meta, "timestamp", "ts"),
                "source_git_commit": _meta_value(meta, "git_commit"),
                "is_mock": mode == "mock",
                "source_file": _meta_value(meta, "_source_file", default="derived"),
            }
            for right in vectors:
                value = _cosine(left.vector, right.vector)
                row[right.vector_id] = "" if not math.isfinite(value) else f"{value:.10g}"
            writer.writerow(row)
    os.replace(temporary, output)


def plot_norm_constancy(
    vectors: Sequence[DiffVector],
    output: Path,
    mode: str,
    metadata: dict[str, str],
) -> None:
    source_and_ab = [v for v in vectors if _canonical_arm(v.meta["arm"]) != "random"]
    arms = _ordered((_canonical_arm(v.meta["arm"]) for v in source_and_ab), ARM_ORDER)
    snippets = _ordered(
        (str(v.meta.get("snippet_set", "unknown")) for v in source_and_ab),
        PRIMARY_SNIPPETS,
    )
    if not arms:
        raise ValueError("No diff vectors available for Figure 2")
    grouped: dict[tuple[str, str], list[DiffVector]] = defaultdict(list)
    for vector in source_and_ab:
        grouped[(_canonical_arm(vector.meta["arm"]), str(vector.meta["snippet_set"]))].append(vector)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(arms), dtype=float)
    total_width = 0.78
    width = total_width / max(1, len(snippets))
    palette = plt.get_cmap("Set2")
    for snippet_idx, snippet in enumerate(snippets):
        offset = (snippet_idx - (len(snippets) - 1) / 2.0) * width
        norm_means: list[float] = []
        const_means: list[float] = []
        for arm in arms:
            group = grouped.get((arm, snippet), [])
            norms = [item.d_norm for item in group if math.isfinite(item.d_norm)]
            constants = [item.constancy for item in group if math.isfinite(item.constancy)]
            norm_means.append(float(np.mean(norms)) if norms else math.nan)
            const_means.append(float(np.mean(constants)) if constants else math.nan)
        colour = palette(snippet_idx % 8)
        axes[0].bar(
            x + offset,
            norm_means,
            width,
            label=snippet,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        axes[1].bar(
            x + offset,
            const_means,
            width,
            label=snippet,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        # Raw seed points make single-seed status visible instead of implying precision.
        for arm_idx, arm in enumerate(arms):
            group = grouped.get((arm, snippet), [])
            norms = [item.d_norm for item in group if math.isfinite(item.d_norm)]
            constants = [item.constancy for item in group if math.isfinite(item.constancy)]
            if norms:
                jitter = np.linspace(-0.12, 0.12, len(norms)) * width
                axes[0].scatter(
                    arm_idx + offset + jitter,
                    norms,
                    s=13,
                    color="#24313a",
                    alpha=0.8,
                    zorder=3,
                )
            if constants:
                jitter = np.linspace(-0.12, 0.12, len(constants)) * width
                axes[1].scatter(
                    arm_idx + offset + jitter,
                    constants,
                    s=13,
                    color="#24313a",
                    alpha=0.8,
                    zorder=3,
                )

    axes[0].set_title("Raw mean-difference norm")
    axes[0].set_ylabel(r"$\|d\|_2$ (before norm matching)")
    axes[1].set_title("Trace constancy")
    axes[1].set_ylabel(r"$\|\mathbb{E}[\Delta h]\|^2 / \mathbb{E}\|\Delta h\|^2$")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xticks(x, arms, rotation=35, ha="right")
        ax.set_xlabel("Arm")
        ax.grid(axis="y", alpha=0.2, linewidth=0.7)
    axes[0].legend(title="Snippet set", frameon=False)
    fig.suptitle("Activation-difference geometry (points show individual seeds)", y=1.02)
    fig.subplots_adjust(wspace=0.24, top=0.86, bottom=0.2)
    _mock_watermark(fig, mode)
    _atomic_savefig(fig, output, metadata)
    plt.close(fig)


def _top_tokens(row: dict[str, Any]) -> list[str]:
    raw = row.get("top")
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


def _display_token(token: str, limit: int = 22) -> str:
    display = repr(token)
    return display if len(display) <= limit else f"{display[: limit - 1]}…"


def select_top_tokens(
    rows: Sequence[dict[str, Any]], snippet_preference: str
) -> tuple[str, dict[str, list[str]]]:
    token_rows = [row for row in rows if row["modality"] == "tokens"]
    available_snippets = {str(row["snippet_set"]) for row in token_rows}
    if snippet_preference in available_snippets:
        snippet = snippet_preference
    elif available_snippets:
        snippet = _ordered(available_snippets, PRIMARY_SNIPPETS)[0]
        warnings.warn(
            f"No token rows for preferred snippet set {snippet_preference!r}; using {snippet!r}",
            stacklevel=2,
        )
    else:
        raise ValueError("No token-modality judge rows available for Figure 3")

    selected: dict[str, list[str]] = {}
    for arm in ("A", "B", "C", "D", "A-B"):
        candidates = [
            row
            for row in token_rows
            if _canonical_arm(row["arm"]) == arm and str(row["snippet_set"]) == snippet
        ]
        if not candidates:
            selected[arm] = []
            warnings.warn(
                f"No explicit {arm} token readout for {snippet}; Figure 3 marks it unavailable",
                stacklevel=2,
            )
            continue
        # Prefer the latest checkpoint, then the most common exact top-token list.
        max_step = max(
            int(_meta_value(row, "step", "checkpoint_step", default=-1))
            for row in candidates
        )
        candidates = [
            row
            for row in candidates
            if int(_meta_value(row, "step", "checkpoint_step", default=-1)) == max_step
        ]
        lists = [tuple(_top_tokens(row)[:20]) for row in candidates]
        counts = Counter(lists)
        chosen = min(counts, key=lambda item: (-counts[item], item)) if counts else ()
        selected[arm] = list(chosen)
    return snippet, selected


def plot_top_tokens(
    rows: Sequence[dict[str, Any]],
    output: Path,
    mode: str,
    metadata: dict[str, str],
    snippet_preference: str,
) -> None:
    snippet, selected = select_top_tokens(rows, snippet_preference)
    arms = ("A", "B", "C", "D", "A-B")
    n_rows = 20
    cells: list[list[str]] = []
    for rank in range(n_rows):
        row: list[str] = []
        for arm in arms:
            tokens = selected[arm]
            if rank < len(tokens):
                row.append(_display_token(tokens[rank]))
            elif rank == 0 and not tokens:
                row.append("— unavailable —")
            else:
                row.append("")
        cells.append(row)

    fig, ax = plt.subplots(figsize=(13.2, 9.0))
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        colLabels=arms,
        rowLabels=[str(idx) for idx in range(1, n_rows + 1)],
        cellLoc="left",
        colLoc="center",
        rowLoc="center",
        loc="center",
        colWidths=[0.19] * len(arms),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.36)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5d8")
        cell.set_linewidth(0.45)
        if row_idx == 0:
            cell.set_facecolor("#315f88")
            cell.set_text_props(color="white", weight="bold")
        elif col_idx == -1:
            cell.set_facecolor("#edf1f4")
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f7f8f9")
        if row_idx > 0 and col_idx >= 0:
            cell.get_text().set_fontfamily("monospace")
    ax.set_title(
        f"Norm-matched logit-lens top tokens · {snippet} snippets",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    ax.text(
        0.5,
        0.975,
        "A−B tokens require an explicit decoded A−B judge item; they are never inferred from labels.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#4f5961",
    )
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
    lexical_predictions: Sequence[Path] | None = None,
) -> dict[str, Path]:
    """Run all analyses and return the emitted artifact paths."""
    results_dir = Path(results_dir)
    figs_dir = Path(figs_dir)
    inputs = discover_inputs(results_dir, mode=mode)
    rows = load_judged(inputs.judged, inputs.mode)
    source_vectors = load_diff_vectors(inputs.diffs, inputs.mode)
    rows, source_vectors, selected_layer = select_analysis_layer(
        rows, source_vectors, requested_layer=layer
    )
    validate_analysis_inputs(rows, source_vectors)
    add_lexical_predictions(
        rows,
        seed=seed,
        prediction_paths=lexical_predictions,
        mode=inputs.mode,
    )
    summaries = accuracy_summaries(rows)
    a_minus_b = derive_a_minus_b(source_vectors)
    analysis_vectors = source_vectors + a_minus_b
    random_vectors = add_random_references(
        analysis_vectors, seed=seed, is_mock=inputs.mode == "mock"
    )
    matrix_vectors = analysis_vectors + random_vectors

    timestamp = utc_now()
    commit = git_commit(Path(__file__).resolve().parents[1])
    run_metadata = collect_run_metadata(rows, source_vectors)
    outputs = {
        "fig1": figs_dir / "fig1_judge_accuracy.png",
        "fig2": figs_dir / "fig2_norm_constancy.png",
        "fig3": figs_dir / "fig3_top_tokens.png",
        "cosines": results_dir / "cosine_matrix.csv",
    }
    plot_judge_accuracy(
        summaries,
        outputs["fig1"],
        inputs.mode,
        _figure_metadata(
            "Blind domain decoding", inputs.mode, commit, timestamp, inputs, run_metadata
        ),
    )
    plot_norm_constancy(
        analysis_vectors,
        outputs["fig2"],
        inputs.mode,
        _figure_metadata(
            "Activation-difference geometry",
            inputs.mode,
            commit,
            timestamp,
            inputs,
            run_metadata,
        ),
    )
    plot_top_tokens(
        rows,
        outputs["fig3"],
        inputs.mode,
        _figure_metadata(
            "Logit-lens top tokens", inputs.mode, commit, timestamp, inputs, run_metadata
        ),
        snippet_preference=top_token_snippet,
    )
    write_cosine_matrix(
        matrix_vectors,
        outputs["cosines"],
        inputs.mode,
        timestamp,
        commit,
    )
    print(
        f"analysis mode={inputs.mode}; judged_files={len(inputs.judged)}; "
        f"diff_files={len(inputs.diffs)}; layer={selected_layer}; "
        f"vectors={len(matrix_vectors)}"
    )
    for name, path in outputs.items():
        print(f"wrote {name}: {path}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figs", type=Path, default=Path("figs"))
    parser.add_argument(
        "--mode",
        choices=("auto", "mock", "real"),
        default="auto",
        help="auto refuses to combine MOCK and real inputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random-reference seed (also keys the MOCK-only lexical placeholder)",
    )
    parser.add_argument(
        "--lexical-predictions",
        type=Path,
        action="append",
        default=None,
        help=(
            "persisted JSONL from judge/lexical_baseline.py --predictions-out; "
            "repeat for multiple source batches (required for real analysis)"
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="layer to analyze; required when the result directory contains multiple layers",
    )
    parser.add_argument(
        "--top-token-snippet",
        default="neutral",
        help="snippet set used for Figure 3 (default: neutral)",
    )
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
        lexical_predictions=args.lexical_predictions,
    )


if __name__ == "__main__":
    main()
