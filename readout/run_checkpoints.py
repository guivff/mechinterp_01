#!/usr/bin/env python3
"""Run the preregistered block readout over a training checkpoint series.

The expensive work is delegated to :mod:`readout.run_readouts`.  This module
only discovers a complete checkpoint sequence, gives every checkpoint its own
artifact directory (so step 50 cannot overwrite step 25), and assembles the
per-block geometry/token-item join used by the emergence-curve analysis.

Scientific runs are deliberately fixed to the preregistered A1 choices:
checkpoints 25..150, layer 15, and ten blocks.  ``--mock`` permits smaller
values for the offline random-model smoke test; its outputs contain ``MOCK``
in every aggregate filename.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from readout import run_readouts as rr


CHECKPOINT_INTERVAL = 25
FINAL_STEP = 150
PRIMARY_LAYER = 15
PREREG_BLOCKS = 10
BLOCK_SEED = 0
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

CURVE_COLUMNS = (
    "arm",
    "seed",
    "step",
    "checkpoint_step",
    "layer",
    "snippet_set",
    "snippet_sha",
    "snippet_set_sha256",
    "snippet_sha256",
    "block",
    "K",
    "block_seed",
    "block_indices_sha256",
    "norm",
    "constancy",
    "mean_offset_energy_share",
    "eta_ref",
    "eta_ref_source",
    "eta_ref_source_sha256",
    "judge_item_id",
    "judge_model",
    "is_mock",
    "timestamp",
    "git_commit",
    "readout_timestamp",
    "adapter",
    "adapter_weight_sha256",
    "geometry_metadata",
    "items_file",
)


def discover_checkpoints(
    run_dir: str | Path,
    *,
    interval: int = CHECKPOINT_INTERVAL,
    final_step: int = FINAL_STEP,
) -> list[tuple[int, Path]]:
    """Return the exact ordered ``checkpoint-N`` sequence required by A1.

    Failing on a missing or unexpected checkpoint avoids drawing an apparently
    complete curve from an incomplete/restarted run.  Non-checkpoint files and
    directories (for example ``final/`` and logs) are ignored.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    if interval <= 0 or final_step <= 0 or final_step % interval:
        raise ValueError("checkpoint interval/final step must be positive and divide exactly")

    found: dict[int, Path] = {}
    for child in root.iterdir():
        match = _CHECKPOINT_RE.fullmatch(child.name)
        if match is None:
            continue
        if not child.is_dir():
            raise ValueError(f"checkpoint path is not a directory: {child}")
        step = int(match.group(1))
        if step in found:
            raise ValueError(f"duplicate checkpoint step {step} under {root}")
        found[step] = child

    expected = list(range(interval, final_step + 1, interval))
    missing = [step for step in expected if step not in found]
    unexpected = sorted(set(found) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"{root}: checkpoint sequence must be {expected}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return [(step, found[step]) for step in expected]


def _parser_option(parser: argparse.ArgumentParser, *names: str) -> str | None:
    for action in parser._actions:
        for option in action.option_strings:
            if option in names:
                return option
    return None


def build_readout_args(
    args: argparse.Namespace,
    *,
    step: int,
    checkpoint: Path,
    checkpoint_out: Path,
) -> argparse.Namespace:
    """Build one ``run_readouts`` invocation without duplicating its defaults."""
    parser = rr.build_parser()
    argv = [
        "--arm",
        args.arm,
        "--base",
        args.base,
        "--adapter",
        str(checkpoint),
        "--layer",
        str(args.layer),
        "--snippets",
        str(args.snippets),
        "--out",
        str(checkpoint_out),
        "--seed",
        str(args.seed),
        "--step",
        str(step),
        "--judge-model",
        args.judge_model,
        "--n-snips",
        str(args.n_snips),
        "--activation-max-tokens",
        str(args.activation_max_tokens),
        "--activation-batch-size",
        str(args.activation_batch_size),
        "--skip-steer",
        "--skip-self-report",
    ]
    if args.target_norm is not None:
        argv.extend(("--target-norm", str(args.target_norm)))
    elif args.target_norm_from is not None:
        argv.extend(("--target-norm-from", str(args.target_norm_from)))
    if args.local_files_only:
        argv.append("--local-files-only")
    if args.add_special_tokens:
        argv.append("--add-special-tokens")
    if args.mock:
        argv.append("--mock")

    # E1 may spell this option --blocks or --n-blocks.  Only pass an option
    # advertised by the live run_readouts parser, which keeps this runner
    # compatible with the earlier implementation while E1 is merged.
    block_option = _parser_option(parser, "--blocks", "--n-blocks")
    if block_option is not None:
        argv.extend((block_option, str(args.blocks)))
    block_seed_option = _parser_option(parser, "--block-seed")
    if block_seed_option is not None:
        argv.extend((block_seed_option, str(BLOCK_SEED)))

    # Cached base activations live outside the per-step output directories.
    # Pass the cache explicitly when E1 exposes an option; older versions simply
    # recollect and will then fail aggregation because they have no block rows.
    cache_option = _parser_option(parser, "--cache-dir", "--base-cache")
    if cache_option is not None:
        argv.extend((cache_option, str(args.cache_dir)))

    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _int_field(value: Any, *, name: str, source: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source}: {name} must be an integer")
    return value


def _finite_field(value: Any, *, name: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{source}: {name} must be finite")
    return result


def _optional_finite_field(
    value: Any, *, name: str, source: Path
) -> float | None:
    """Validate a finite number while preserving an undefined zero-energy value."""
    if value is None:
        return None
    return _finite_field(value, name=name, source=source)


def _integer_list_field(value: Any, *, name: str, source: Path) -> tuple[int, ...]:
    """Read an explicit JSON integer list without accepting bools or coercions."""
    if not isinstance(value, list):
        raise ValueError(f"{source}: {name} must be an integer list")
    result: list[int] = []
    for index, member in enumerate(value):
        result.append(_int_field(member, name=f"{name}[{index}]", source=source))
    return tuple(result)


def _validate_common_metadata(
    row: dict[str, Any],
    source: Path,
    *,
    arm: str,
    seed: int,
    step: int,
    layer: int,
) -> None:
    expected = {
        "arm": arm,
        "seed": seed,
        "step": step,
        "checkpoint_step": step,
        "layer": layer,
    }
    mismatches = {
        key: {"expected": value, "found": row.get(key)}
        for key, value in expected.items()
        if row.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{source}: readout provenance mismatch: {mismatches}")


def _geometry_sidecars(paths: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    result: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        if path.suffix != ".json" or not path.is_file():
            continue
        metadata = _read_json_object(path)
        if metadata.get("artifact_type") != "activation_difference":
            continue
        # A whole-set compatibility artifact may coexist with the E1 block
        # artifacts.  The curve's sampling unit is explicitly the block.
        if "block" not in metadata:
            continue
        vector_name = metadata.get("array_file")
        vector_sha = metadata.get("array_sha256")
        if not isinstance(vector_name, str) or not isinstance(vector_sha, str):
            raise ValueError(f"{path}: diff sidecar lacks array_file/array_sha256")
        vector_path = path.parent / vector_name
        if not vector_path.is_file():
            raise ValueError(f"{path}: missing declared diff vector {vector_path}")
        if _sha256_file(vector_path) != vector_sha:
            raise ValueError(f"{path}: diff vector sha256 mismatch")
        result.append((path, metadata))
    return result


def collect_curve_rows(
    written: Iterable[str | Path],
    *,
    arm: str,
    seed: int,
    step: int,
    layer: int,
    blocks: int,
    n_snips: int,
    curve_timestamp: str,
    git_commit: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join block geometry to its one blind-judge token item.

    The join is by ``(snippet_set, block)`` and never by filename or list
    order.  This is the boundary that prevents a block's token list from being
    silently paired with another block's norm/constancy measurement.
    """
    canonical_blocks = rr.split_blocks(n_snips, K=blocks, seed=BLOCK_SEED)
    canonical_members = tuple(
        tuple(int(member) for member in block) for block in canonical_blocks
    )
    paths = [Path(path) for path in written]
    geometry = _geometry_sidecars(paths)
    item_files = sorted(
        {
            path
            for path in paths
            if path.suffix == ".jsonl" and path.is_file() and path.name.startswith("items")
        }
    )
    if not geometry:
        raise ValueError(
            "run_readouts produced no per-block activation-difference sidecars; "
            "merge E1 before running the checkpoint curve"
        )
    if len(item_files) != 1:
        raise ValueError(f"expected exactly one checkpoint items JSONL, found {item_files}")

    items_path = item_files[0]
    token_rows = [row for row in _read_jsonl(items_path) if row.get("modality") == "tokens"]
    item_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for item in token_rows:
        _validate_common_metadata(
            item, items_path, arm=arm, seed=seed, step=step, layer=layer
        )
        snippet_set = item.get("snippet_set")
        if snippet_set not in rr.SNIPPET_SETS:
            raise ValueError(f"{items_path}: invalid token-item snippet_set {snippet_set!r}")
        block = _int_field(item.get("block"), name="block", source=items_path)
        key = (snippet_set, block)
        if key in item_by_key:
            raise ValueError(f"{items_path}: duplicate token judge item for {key}")
        if not isinstance(item.get("item_id"), str) or not item["item_id"]:
            raise ValueError(f"{items_path}: token item for {key} lacks item_id")
        item_by_key[key] = item

    geometry_by_key: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for metadata_path, metadata in geometry:
        _validate_common_metadata(
            metadata,
            metadata_path,
            arm=arm,
            seed=seed,
            step=step,
            layer=layer,
        )
        snippet_set = metadata.get("snippet_set")
        if snippet_set not in rr.SNIPPET_SETS:
            raise ValueError(f"{metadata_path}: invalid snippet_set {snippet_set!r}")
        block = _int_field(metadata.get("block"), name="block", source=metadata_path)
        if block < 0 or block >= blocks:
            raise ValueError(f"{metadata_path}: block {block} is outside [0, {blocks})")
        key = (snippet_set, block)
        if key in geometry_by_key:
            raise ValueError(f"duplicate block geometry for {key}: {metadata_path}")
        geometry_by_key[key] = (metadata_path, metadata)

    expected = {
        (snippet_set, block)
        for snippet_set in rr.SNIPPET_SETS
        for block in range(blocks)
    }
    if set(geometry_by_key) != expected or set(item_by_key) != expected:
        raise ValueError(
            "checkpoint block outputs are incomplete or misnumbered: "
            f"expected={sorted(expected)}, geometry={sorted(geometry_by_key)}, "
            f"items={sorted(item_by_key)}"
        )

    curve_rows: list[dict[str, Any]] = []
    ordered_items: list[dict[str, Any]] = []
    for key in sorted(expected, key=lambda value: (rr.SNIPPET_SETS.index(value[0]), value[1])):
        metadata_path, metadata = geometry_by_key[key]
        item = item_by_key[key]
        geometry_sha = metadata.get("snippet_set_sha256", metadata.get("snippet_sha"))
        item_sha = item.get("snippet_set_sha256", item.get("snippet_sha"))
        if not isinstance(geometry_sha, str) or len(geometry_sha) != 64:
            raise ValueError(f"{metadata_path}: missing full snippet-set sha256")
        if item_sha != geometry_sha:
            raise ValueError(
                f"{items_path}: token/geometry snippet sha mismatch for {key}: "
                f"{item_sha!r} != {geometry_sha!r}"
            )
        geometry_judge = metadata.get("judge_model")
        item_judge = item.get("judge_model")
        if geometry_judge != item_judge:
            raise ValueError(f"judge-model mismatch between geometry and token item for {key}")
        declared_k = _int_field(metadata.get("K"), name="K", source=metadata_path)
        if declared_k != blocks:
            raise ValueError(f"{metadata_path}: K={declared_k} does not match --blocks={blocks}")
        block_seed = _int_field(
            metadata.get("block_seed"), name="block_seed", source=metadata_path
        )
        if block_seed != BLOCK_SEED:
            raise ValueError(
                f"{metadata_path}: frozen snippet-block seed must be {BLOCK_SEED}"
            )
        block_indices_sha = metadata.get(
            "block_indices_sha256", metadata.get("block_indices_hash")
        )
        if not isinstance(block_indices_sha, str) or len(block_indices_sha) != 64:
            raise ValueError(f"{metadata_path}: missing full block-indices sha256")
        block_indices = _integer_list_field(
            metadata.get("block_indices"), name="block_indices", source=metadata_path
        )
        expected_indices = canonical_members[key[1]]
        if block_indices != expected_indices:
            raise ValueError(
                f"{metadata_path}: block_indices for {key} do not match the frozen "
                f"split_blocks({n_snips}, K={blocks}, seed={BLOCK_SEED}) assignment"
            )
        expected_block_sha = rr._int_array_sha256(
            canonical_blocks[key[1]], label=f"split_blocks:block:{key[1]}"
        )
        if block_indices_sha != expected_block_sha:
            raise ValueError(
                f"{metadata_path}: block_indices_sha256 for {key} does not "
                "authenticate its frozen canonical membership"
            )
        if type(metadata.get("is_mock")) is not bool:
            raise ValueError(f"{metadata_path}: is_mock must be boolean")
        item_k = _int_field(item.get("K"), name="item K", source=items_path)
        item_block_seed = _int_field(
            item.get("block_seed"), name="item block_seed", source=items_path
        )
        item_block_sha = item.get(
            "block_indices_sha256", item.get("block_indices_hash")
        )
        item_block_indices = _integer_list_field(
            item.get("block_indices"), name="item block_indices", source=items_path
        )
        if (item_k, item_block_seed, item_block_indices, item_block_sha) != (
            declared_k,
            block_seed,
            expected_indices,
            block_indices_sha,
        ):
            raise ValueError(
                f"{items_path}: token/geometry block provenance mismatch for {key}"
            )
        if item.get("is_mock") is not metadata["is_mock"]:
            raise ValueError(f"{items_path}: token/geometry MOCK provenance mismatch for {key}")
        eta_ref = _finite_field(
            metadata.get("eta_ref", metadata.get("decode_target_norm")),
            name="eta_ref/decode_target_norm",
            source=metadata_path,
        )
        if eta_ref <= 0:
            raise ValueError(f"{metadata_path}: eta_ref must be positive")
        decode_target_norm = _finite_field(
            metadata.get("decode_target_norm", eta_ref),
            name="decode_target_norm",
            source=metadata_path,
        )
        if not math.isclose(eta_ref, decode_target_norm, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{metadata_path}: eta_ref and decode_target_norm disagree")
        eta_source = metadata.get("eta_ref_source")
        eta_source_sha = metadata.get("eta_ref_source_sha256")
        if not isinstance(eta_source, str) or not eta_source:
            raise ValueError(f"{metadata_path}: missing eta_ref_source")
        if not metadata["is_mock"] and (
            not isinstance(eta_source_sha, str) or len(eta_source_sha) != 64
        ):
            raise ValueError(f"{metadata_path}: real eta_ref lacks a full source sha256")
        item_eta_ref = _finite_field(
            item.get("eta_ref", item.get("decode_target_norm")),
            name="item eta_ref/decode_target_norm",
            source=items_path,
        )
        if not math.isclose(eta_ref, item_eta_ref, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{items_path}: token/geometry eta_ref mismatch for {key}")
        if (
            item.get("eta_ref_source"),
            item.get("eta_ref_source_sha256"),
        ) != (eta_source, eta_source_sha):
            raise ValueError(f"{items_path}: token/geometry eta_ref provenance mismatch for {key}")
        if "constancy" not in metadata or "mean_offset_energy_share" not in metadata:
            raise ValueError(
                f"{metadata_path}: both constancy aliases must be present"
            )
        constancy = _optional_finite_field(
            metadata.get("constancy"), name="constancy", source=metadata_path
        )
        mean_energy_share = _optional_finite_field(
            metadata.get("mean_offset_energy_share"),
            name="mean_offset_energy_share/constancy",
            source=metadata_path,
        )
        aliases_agree = (
            constancy is None and mean_energy_share is None
        ) or (
            constancy is not None
            and mean_energy_share is not None
            and math.isclose(constancy, mean_energy_share, rel_tol=1e-9, abs_tol=1e-12)
        )
        if not aliases_agree:
            raise ValueError(
                f"{metadata_path}: constancy and mean_offset_energy_share disagree"
            )
        raw_norm = _finite_field(
            metadata.get("raw_d_norm", metadata.get("d_norm")),
            name="raw_d_norm/d_norm",
            source=metadata_path,
        )
        if constancy is None and raw_norm != 0.0:
            raise ValueError(
                f"{metadata_path}: undefined constancy requires a zero-norm trace"
            )
        adapter_receipt = metadata.get("adapter_receipt") or {}
        curve_rows.append(
            {
                "arm": arm,
                "seed": seed,
                "step": step,
                "checkpoint_step": step,
                "layer": layer,
                "snippet_set": key[0],
                "snippet_sha": geometry_sha,
                "snippet_set_sha256": geometry_sha,
                "snippet_sha256": geometry_sha,
                "block": key[1],
                "K": declared_k,
                "block_seed": block_seed,
                "block_indices_sha256": block_indices_sha,
                "norm": raw_norm,
                "constancy": constancy,
                "mean_offset_energy_share": mean_energy_share,
                "eta_ref": eta_ref,
                "eta_ref_source": eta_source,
                "eta_ref_source_sha256": eta_source_sha,
                "judge_item_id": item["item_id"],
                "judge_model": item_judge,
                "is_mock": metadata["is_mock"],
                "timestamp": curve_timestamp,
                "git_commit": git_commit,
                "readout_timestamp": metadata.get("timestamp"),
                "adapter": metadata.get("adapter"),
                "adapter_weight_sha256": adapter_receipt.get("adapter_weight_sha256"),
                "geometry_metadata": str(metadata_path),
                "items_file": str(items_path),
            }
        )
        ordered_items.append(item)
    return curve_rows, ordered_items


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _effective_mock(
    args: argparse.Namespace, *, checkpoints: Iterable[Path] = ()
) -> bool:
    checkpoint_paths = tuple(checkpoints)
    run_root = Path(args.run_dir)
    if not checkpoint_paths and run_root.is_dir():
        checkpoint_paths = tuple(
            child
            for child in run_root.iterdir()
            if _CHECKPOINT_RE.fullmatch(child.name) is not None
        )
    provenance_paths = (args.base, args.snippets, args.run_dir, *checkpoint_paths)
    resolved_paths = tuple(
        Path(path).resolve(strict=False) for path in provenance_paths if path is not None
    )
    if args.mock or any(rr._path_marked_mock(path) for path in (*provenance_paths, *resolved_paths)):
        return True
    records = [
        rr._read_snippet_file(Path(args.snippets) / f"{name}.jsonl", args.n_snips)
        for name in rr.SNIPPET_SETS
    ]
    statuses = {record["is_mock"] for record in records}
    if len(statuses) != 1:
        raise ValueError("neutral and math snippet inputs mix mock and real provenance")
    return bool(next(iter(statuses)))


def run(
    args: argparse.Namespace,
    *,
    readout_runner: Callable[[argparse.Namespace], list[Path]] | None = None,
) -> list[Path]:
    """Run all checkpoint readouts and write curve CSV + combined judge items."""
    _validate_args(args)
    effective_mock = _effective_mock(args)
    if not effective_mock and args.layer != PRIMARY_LAYER:
        raise ValueError(f"scientific emergence curves are fixed at layer {PRIMARY_LAYER}")
    if not effective_mock and args.blocks != PREREG_BLOCKS:
        raise ValueError(f"scientific emergence curves are fixed at {PREREG_BLOCKS} blocks")
    if not effective_mock and args.final_step != FINAL_STEP:
        raise ValueError(f"scientific emergence curves are fixed through step {FINAL_STEP}")
    if not effective_mock and (
        args.target_norm is not None or args.target_norm_from is not None
    ):
        raise ValueError(
            "scientific emergence curves must use run_readouts' automatic layer eta_ref; "
            "explicit target norms are permitted only for MOCK diagnostics"
        )

    checkpoints = discover_checkpoints(
        args.run_dir, interval=CHECKPOINT_INTERVAL, final_step=args.final_step
    )
    output_root = Path(args.out)
    mock_tag = "_MOCK" if effective_mock else ""
    curve_path = output_root / f"curve{mock_tag}_{args.arm}_s{args.seed}.csv"
    items_path = output_root / f"items_curve{mock_tag}_{args.arm}_s{args.seed}.jsonl"
    if (curve_path.exists() or items_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"aggregate output exists ({curve_path} or {items_path}); pass --overwrite explicitly"
        )

    artifact_root = output_root / f"checkpoint_readouts{mock_tag}_{args.arm}_s{args.seed}"
    runner = readout_runner or rr.run
    curve_timestamp = datetime.now(timezone.utc).isoformat()
    git_commit = rr._git_commit()
    all_rows: list[dict[str, Any]] = []
    all_items: list[dict[str, Any]] = []
    for step, checkpoint in checkpoints:
        checkpoint_out = artifact_root / f"step-{step:06d}"
        if checkpoint_out.exists() and any(checkpoint_out.iterdir()):
            raise FileExistsError(
                f"checkpoint artifact directory is non-empty: {checkpoint_out}; "
                "use a fresh --out directory"
            )
        checkpoint_out.mkdir(parents=True, exist_ok=True)
        readout_args = build_readout_args(
            args, step=step, checkpoint=checkpoint, checkpoint_out=checkpoint_out
        )
        written = runner(readout_args)
        rows, items = collect_curve_rows(
            written,
            arm=args.arm,
            seed=args.seed,
            step=step,
            layer=args.layer,
            blocks=args.blocks,
            n_snips=args.n_snips,
            curve_timestamp=curve_timestamp,
            git_commit=git_commit,
        )
        all_rows.extend(rows)
        all_items.extend(items)

    expected_rows = len(checkpoints) * len(rr.SNIPPET_SETS) * args.blocks
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} curve rows, got {len(all_rows)}")
    eta_receipts = {
        (row["eta_ref"], row["eta_ref_source"], row["eta_ref_source_sha256"])
        for row in all_rows
    }
    if len(eta_receipts) != 1:
        raise ValueError(f"eta_ref changed across checkpoints or snippet sets: {eta_receipts}")
    for snippet_set in rr.SNIPPET_SETS:
        snippet_hashes = {
            row["snippet_sha256"]
            for row in all_rows
            if row["snippet_set"] == snippet_set
        }
        if len(snippet_hashes) != 1:
            raise ValueError(
                f"snippet-set hash changed across checkpoints for {snippet_set}: "
                f"{sorted(snippet_hashes)}"
            )
        for block in range(args.blocks):
            block_hashes = {
                row["block_indices_sha256"]
                for row in all_rows
                if row["snippet_set"] == snippet_set and row["block"] == block
            }
            if len(block_hashes) != 1:
                raise ValueError(
                    f"block assignment changed across checkpoints for "
                    f"({snippet_set}, {block}): {sorted(block_hashes)}"
                )
    if {row["is_mock"] for row in all_rows} != {effective_mock}:
        raise ValueError("checkpoint artifact MOCK provenance disagrees with aggregate output")
    item_ids = [row["item_id"] for row in all_items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("checkpoint token judge item IDs are not globally unique")

    _write_csv(curve_path, all_rows)
    _write_jsonl(items_path, all_items)
    return [curve_path, items_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--arm", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base", default=rr.DEFAULT_BASE)
    parser.add_argument("--snippets", default="data/snippets")
    parser.add_argument("--out", default="results")
    parser.add_argument("--cache-dir", default="results/cache")
    parser.add_argument("--layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument("--blocks", type=int, default=PREREG_BLOCKS)
    parser.add_argument("--final-step", type=int, default=FINAL_STEP)
    parser.add_argument("--judge-model", default="not_run")
    parser.add_argument("--n-snips", type=int, default=500)
    parser.add_argument("--activation-max-tokens", type=int, default=128)
    parser.add_argument("--activation-batch-size", type=int, default=8)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--target-norm", type=float)
    target.add_argument("--target-norm-from")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the two aggregate curve files; step artifact directories must still be fresh",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.layer < 0:
        raise ValueError("--layer must be non-negative")
    if args.blocks <= 0:
        raise ValueError("--blocks must be positive")
    if args.n_snips < args.blocks:
        raise ValueError("--n-snips must be at least --blocks so every block is non-empty")
    if args.final_step <= 0 or args.final_step % CHECKPOINT_INTERVAL:
        raise ValueError(f"--final-step must be a positive multiple of {CHECKPOINT_INTERVAL}")
    if args.activation_max_tokens <= rr.SKIP_TOKENS:
        raise ValueError(
            f"--activation-max-tokens must exceed the fixed {rr.SKIP_TOKENS}-token skip"
        )
    if args.activation_batch_size <= 0:
        raise ValueError("--activation-batch-size must be positive")
    if args.target_norm is not None and (
        not math.isfinite(args.target_norm) or args.target_norm <= 0
    ):
        raise ValueError("--target-norm must be finite and positive")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        written = run(args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
