"""Build the preregistered A-minus-B contrast from saved readout artifacts.

This is deliberately an artifact-to-artifact operation: A and B must already
have been measured on exactly the same token rows.  The raw contrast
``d_A - d_B`` is saved before a copy is norm-matched to the paired arm-D norm
and decoded with the final-norm logit lens.

Run once per snippet set, using either invocation form::

    python -m readout.make_ab_readout \
        --diff-a results/diff_A_s0_L22_neutral.npy \
        --diff-b results/diff_B_s0_L22_neutral.npy \
        --target-norm-from results/diff_D_s0_L22_neutral.npy

    python readout/make_ab_readout.py --diff-a ... --diff-b ... \
        --target-norm-from ...

The inputs' sidecars determine seed, checkpoint, layer, snippet set, mock
status, and (unless ``--base`` is given) the base-model reference.  Pairing is
rejected unless all activation/alignment provenance agrees exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from readout.decode import logit_lens, match_norm, readout_text
from readout.diff import save_diff
from readout.run_readouts import (
    _artifact_stem,
    _git_commit,
    _same_model_reference,
    _utc_now,
    _write_jsonl,
    _preferred_inference_dtype,
    load_model,
    load_tokenizer,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_MOCK_RE = re.compile(r"(?:^|[_-])mock(?:[_-]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class DiffArtifact:
    """A validated raw diff vector and its exact serialized provenance."""

    vector_path: Path
    metadata_path: Path
    vector: np.ndarray
    metadata: dict[str, Any]
    vector_sha256: str
    metadata_sha256: str
    raw_norm: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _file_is_mock(path: Path) -> bool:
    return bool(_MOCK_RE.search(path.stem))


def _artifact_paths(value: str | Path) -> tuple[Path, Path]:
    path = Path(value)
    if path.suffix.lower() == ".npy":
        vector_path, metadata_path = path, path.with_suffix(".json")
    elif path.suffix.lower() == ".json":
        vector_path, metadata_path = path.with_suffix(".npy"), path
    elif path.suffix:
        raise ValueError(f"{path}: diff artifact must be .npy, .json, or suffix-less")
    else:
        vector_path, metadata_path = path.with_suffix(".npy"), path.with_suffix(".json")
    missing = [candidate for candidate in (vector_path, metadata_path) if not candidate.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{path}: diff artifact requires paired .npy and .json files; missing {missing}"
        )
    return vector_path, metadata_path


def _metadata_step(metadata: dict[str, Any], path: Path) -> int:
    values = [metadata[key] for key in ("step", "checkpoint_step") if key in metadata]
    if not values:
        raise ValueError(f"{path}: missing step/checkpoint_step")
    if any(not _is_int(value) for value in values):
        raise ValueError(f"{path}: step/checkpoint_step must be integers")
    if len(set(values)) != 1:
        raise ValueError(f"{path}: step and checkpoint_step disagree")
    return int(values[0])


def _metadata_snippet_sha(metadata: dict[str, Any], path: Path) -> str:
    values = [
        metadata[key]
        for key in ("snippet_sha", "snippet_set_sha256")
        if key in metadata
    ]
    if not values:
        raise ValueError(f"{path}: missing snippet_sha/snippet_set_sha256")
    if len(set(values)) != 1:
        raise ValueError(f"{path}: snippet SHA aliases disagree")
    value = values[0]
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{path}: snippet SHA must be a full SHA-256 hex digest")
    return value.lower()


def load_diff_artifact(value: str | Path, *, expected_arm: str) -> DiffArtifact:
    """Load one raw diff and reject incomplete or contradictory provenance."""

    vector_path, metadata_path = _artifact_paths(value)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{metadata_path}: invalid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: expected a JSON object")
    if metadata.get("arm") != expected_arm:
        raise ValueError(
            f"{metadata_path}: expected arm={expected_arm!r}, got {metadata.get('arm')!r}"
        )

    required = (
        "seed",
        "layer",
        "base",
        "snippet_set",
        "n_snippets_used",
        "alignment_sha256",
        "n_aligned_tokens",
        "is_mock",
        "git_commit",
        "model_dtype",
        "padding_side",
        "add_special_tokens",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "skip_tokens",
        "activation_max_tokens",
        "activation_batch_size",
        "n_model_layers",
        "artifact_schema_version",
        "artifact_type",
        "array_file",
        "array_shape",
        "array_dtype",
        "array_sha256",
    )
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"{metadata_path}: missing required provenance {missing}")
    for field in (
        "seed",
        "layer",
        "n_snippets_used",
        "n_aligned_tokens",
        "skip_tokens",
        "activation_max_tokens",
        "activation_batch_size",
        "n_model_layers",
    ):
        if not _is_int(metadata[field]):
            raise ValueError(f"{metadata_path}: {field} must be an integer")
    if metadata["layer"] < 0 or metadata["n_snippets_used"] <= 0:
        raise ValueError(f"{metadata_path}: invalid layer or snippet count")
    if not isinstance(metadata["base"], str) or not metadata["base"]:
        raise ValueError(f"{metadata_path}: base must be a non-empty model reference")
    if metadata["snippet_set"] not in {"neutral", "math"}:
        raise ValueError(f"{metadata_path}: unsupported snippet_set={metadata['snippet_set']!r}")
    if metadata["padding_side"] != "right":
        raise ValueError(f"{metadata_path}: only right-padded readouts can be paired")
    if type(metadata["add_special_tokens"]) is not bool:
        raise ValueError(f"{metadata_path}: add_special_tokens must be boolean")
    if type(metadata["is_mock"]) is not bool:
        raise ValueError(f"{metadata_path}: is_mock must be boolean")
    if not isinstance(metadata["git_commit"], str) or not metadata["git_commit"]:
        raise ValueError(f"{metadata_path}: git_commit must be a non-empty string")
    _metadata_step(metadata, metadata_path)
    _metadata_snippet_sha(metadata, metadata_path)
    alignment_sha = metadata["alignment_sha256"]
    if not isinstance(alignment_sha, str) or not _SHA256_RE.fullmatch(alignment_sha):
        raise ValueError(f"{metadata_path}: alignment_sha256 must be a full SHA-256 digest")

    vector_mock = _file_is_mock(vector_path)
    metadata_mock = _file_is_mock(metadata_path)
    if vector_mock != metadata_mock or vector_mock != metadata["is_mock"]:
        raise ValueError(
            f"{vector_path}: MOCK filename marker and is_mock metadata must agree"
        )

    vector = np.load(vector_path, allow_pickle=False)
    if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.number):
        raise ValueError(f"{vector_path}: expected a numeric one-dimensional vector")
    vector = np.asarray(vector, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError(f"{vector_path}: vector contains non-finite values")
    if metadata["artifact_schema_version"] != 1:
        raise ValueError(f"{metadata_path}: unsupported artifact_schema_version")
    if metadata["artifact_type"] != "activation_difference":
        raise ValueError(
            f"{metadata_path}: expected artifact_type='activation_difference'"
        )
    expected_array_metadata = {
        "array_file": vector_path.name,
        "array_shape": list(vector.shape),
        "array_dtype": str(vector.dtype),
        "array_sha256": _sha256(vector_path),
    }
    mismatched_array_fields = [
        field
        for field, expected in expected_array_metadata.items()
        if metadata[field] != expected
    ]
    if mismatched_array_fields:
        raise ValueError(
            f"{metadata_path}: vector hash/schema mismatch for fields "
            f"{mismatched_array_fields}"
        )
    raw_norm = float(np.linalg.norm(vector))
    for field in ("raw_d_norm", "d_norm"):
        if field not in metadata:
            continue
        declared = metadata[field]
        if isinstance(declared, bool) or not isinstance(declared, (int, float)):
            raise ValueError(f"{metadata_path}: {field} must be numeric")
        if not math.isfinite(float(declared)) or not math.isclose(
            raw_norm, float(declared), rel_tol=1e-5, abs_tol=1e-7
        ):
            raise ValueError(
                f"{metadata_path}: {field}={declared!r} disagrees with vector norm {raw_norm:.9g}"
            )
    if "raw_d_norm" not in metadata and "d_norm" not in metadata:
        raise ValueError(f"{metadata_path}: missing raw_d_norm/d_norm")

    return DiffArtifact(
        vector_path=vector_path,
        metadata_path=metadata_path,
        vector=vector,
        metadata=metadata,
        vector_sha256=_sha256(vector_path),
        metadata_sha256=_sha256(metadata_path),
        raw_norm=raw_norm,
    )


def _canonical_provenance(artifact: DiffArtifact) -> dict[str, Any]:
    metadata = artifact.metadata
    return {
        "seed": metadata["seed"],
        "checkpoint_step": _metadata_step(metadata, artifact.metadata_path),
        "layer": metadata["layer"],
        "snippet_set": metadata["snippet_set"],
        "snippet_sha": _metadata_snippet_sha(metadata, artifact.metadata_path),
        "n_snippets_used": metadata["n_snippets_used"],
        "alignment_sha256": str(metadata["alignment_sha256"]).lower(),
        "n_aligned_tokens": metadata["n_aligned_tokens"],
        "is_mock": metadata["is_mock"],
        "git_commit": metadata["git_commit"],
        "model_dtype": metadata["model_dtype"],
        "padding_side": metadata["padding_side"],
        "add_special_tokens": metadata["add_special_tokens"],
        "bos_token_id": metadata["bos_token_id"],
        "eos_token_id": metadata["eos_token_id"],
        "pad_token_id": metadata["pad_token_id"],
        "skip_tokens": metadata["skip_tokens"],
        "activation_max_tokens": metadata["activation_max_tokens"],
        "activation_batch_size": metadata["activation_batch_size"],
        "n_model_layers": metadata["n_model_layers"],
    }


def _require_same_base(left: DiffArtifact, right: DiffArtifact) -> None:
    if not _same_model_reference(left.metadata["base"], right.metadata["base"]):
        raise ValueError(
            f"base-model provenance differs: {left.metadata['base']!r} != "
            f"{right.metadata['base']!r}"
        )


def validate_sources(
    source_a: DiffArtifact,
    source_b: DiffArtifact,
    norm_d: DiffArtifact,
    *,
    requested_base: str | None = None,
) -> dict[str, Any]:
    """Validate exact A/B pairing and the corresponding arm-D norm source."""

    if source_a.vector.shape != source_b.vector.shape or source_a.vector.shape != norm_d.vector.shape:
        raise ValueError(
            "A, B, and D diff vectors must have the same hidden width: "
            f"{source_a.vector.shape}, {source_b.vector.shape}, {norm_d.vector.shape}"
        )
    _require_same_base(source_a, source_b)
    _require_same_base(source_a, norm_d)
    if requested_base is not None and not _same_model_reference(
        source_a.metadata["base"], requested_base
    ):
        raise ValueError(
            f"--base={requested_base!r} does not match source base={source_a.metadata['base']!r}"
        )

    provenance_a = _canonical_provenance(source_a)
    provenance_b = _canonical_provenance(source_b)
    if provenance_a != provenance_b:
        differing = [
            field
            for field in provenance_a
            if provenance_a[field] != provenance_b[field]
        ]
        raise ValueError(f"A/B diff provenance differs for fields: {differing}")

    provenance_d = _canonical_provenance(norm_d)
    # D follows an independent training run, so its seed/checkpoint and the
    # checkout that produced it need not equal A/B. Everything governing the
    # captured token rows and representation must still match exactly.
    d_independent_fields = {"seed", "checkpoint_step", "git_commit"}
    d_match_fields = tuple(
        field for field in provenance_a if field not in d_independent_fields
    )
    differing_d = [
        field
        for field in d_match_fields
        if provenance_a[field] != provenance_d[field]
    ]
    if differing_d:
        raise ValueError(f"arm-D norm provenance differs for fields: {differing_d}")
    if not math.isfinite(norm_d.raw_norm) or norm_d.raw_norm <= 1e-12:
        raise ValueError("arm-D target norm is zero, near-zero, or non-finite")

    base_norm_a = source_a.metadata.get("base_act_norm_mean")
    base_norm_b = source_b.metadata.get("base_act_norm_mean")
    if not isinstance(base_norm_a, (int, float)) or isinstance(base_norm_a, bool):
        raise ValueError(f"{source_a.metadata_path}: missing numeric base_act_norm_mean")
    if not isinstance(base_norm_b, (int, float)) or isinstance(base_norm_b, bool):
        raise ValueError(f"{source_b.metadata_path}: missing numeric base_act_norm_mean")
    if not math.isfinite(float(base_norm_a)) or float(base_norm_a) <= 0:
        raise ValueError(f"{source_a.metadata_path}: invalid base_act_norm_mean")
    if not math.isclose(float(base_norm_a), float(base_norm_b), rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("A/B base activation norms disagree despite matched alignment")

    return {
        **provenance_a,
        "base": source_a.metadata["base"],
        "base_act_norm_mean": float(base_norm_a),
        "d_checkpoint_step": provenance_d["checkpoint_step"],
    }


def _source_record(artifact: DiffArtifact) -> dict[str, Any]:
    return {
        "arm": artifact.metadata["arm"],
        "vector_path": str(artifact.vector_path),
        "metadata_path": str(artifact.metadata_path),
        "vector_sha256": artifact.vector_sha256,
        "metadata_sha256": artifact.metadata_sha256,
        "raw_d_norm": artifact.raw_norm,
        "adapter": artifact.metadata.get("adapter"),
        "checkpoint_step": _metadata_step(artifact.metadata, artifact.metadata_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--diff-a", required=True, help="arm-A raw diff .npy/.json or stem")
    parser.add_argument("--diff-b", required=True, help="arm-B raw diff .npy/.json or stem")
    parser.add_argument(
        "--target-norm-from",
        required=True,
        help="matching arm-D raw diff .npy/.json or stem",
    )
    parser.add_argument(
        "--base",
        help="base model id/path; default is inferred from and verified against the sidecars",
    )
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="forbid model/tokenizer downloads",
    )
    return parser


def run(args: argparse.Namespace) -> list[Path]:
    source_a = load_diff_artifact(args.diff_a, expected_arm="A")
    source_b = load_diff_artifact(args.diff_b, expected_arm="B")
    norm_d = load_diff_artifact(args.target_norm_from, expected_arm="D")
    provenance = validate_sources(
        source_a,
        source_b,
        norm_d,
        requested_base=args.base,
    )

    direction = np.asarray(source_a.vector - source_b.vector, dtype=np.float32)
    raw_norm = float(np.linalg.norm(direction))
    if not math.isfinite(raw_norm) or raw_norm <= 1e-12:
        raise ValueError("A-B is zero, near-zero, or non-finite and cannot be decoded")

    is_mock = bool(provenance["is_mock"])
    output_root = Path(args.out)
    stem = _artifact_stem(
        "diff", "A-B", int(provenance["seed"]), int(provenance["layer"]), is_mock
    )
    diff_stem = output_root / f"{stem}_{provenance['snippet_set']}"
    items_path = output_root / (
        f"{_artifact_stem('items', 'A-B', int(provenance['seed']), int(provenance['layer']), is_mock)}_"
        f"{provenance['snippet_set']}.jsonl"
    )
    outputs = (diff_stem.with_suffix(".npy"), diff_stem.with_suffix(".json"), items_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise ValueError(f"refusing to overwrite existing A-B artifacts: {existing}")

    dtype = _preferred_inference_dtype()
    timestamp = _utc_now()
    commit = _git_commit()
    sources = [_source_record(source_a), _source_record(source_b)]
    common_meta: dict[str, Any] = {
        "arm": "A-B",
        "seed": provenance["seed"],
        "step": provenance["checkpoint_step"],
        "checkpoint_step": provenance["checkpoint_step"],
        "layer": provenance["layer"],
        "base": provenance["base"],
        "adapter": None,
        "adapter_merged": False,
        "judge_model": "not_run",
        "timestamp": timestamp,
        "git_commit": commit,
        "source_git_commit": provenance["git_commit"],
        "is_mock": is_mock,
        "model_dtype": str(dtype).replace("torch.", ""),
        "source_model_dtype": provenance["model_dtype"],
        "local_files_only": bool(args.local_files_only),
        "padding_side": provenance["padding_side"],
        "add_special_tokens": provenance["add_special_tokens"],
        "bos_token_id": provenance["bos_token_id"],
        "eos_token_id": provenance["eos_token_id"],
        "pad_token_id": provenance["pad_token_id"],
        "skip_tokens": provenance["skip_tokens"],
        "activation_max_tokens": provenance["activation_max_tokens"],
        "activation_batch_size": provenance["activation_batch_size"],
        "n_model_layers": provenance["n_model_layers"],
        "snippet_set": provenance["snippet_set"],
        "snippet_sha": provenance["snippet_sha"],
        "snippet_set_sha256": provenance["snippet_sha"],
        "snippet_sha_scope": "complete_jsonl_file_bytes",
        "n_snippets_used": provenance["n_snippets_used"],
        "alignment_sha256": provenance["alignment_sha256"],
        "n_aligned_tokens": provenance["n_aligned_tokens"],
        "raw_vector_saved_before_decode": True,
        "geometry_only": False,
        "artifact_type": "derived_activation_difference",
        "derivation": "d_A_minus_d_B",
        "derived_from": sources,
        "derived_source_provenance_verified": True,
    }
    stats = {
        "d_norm": raw_norm,
        "raw_d_norm": raw_norm,
        "base_act_norm_mean": provenance["base_act_norm_mean"],
        "rel_norm": raw_norm / provenance["base_act_norm_mean"],
        "constancy": None,
        "random_cos_mean": None,
        "random_cos_std": None,
        "n_tokens": provenance["n_aligned_tokens"],
        "note": "A-B is derived from paired saved mean-diff vectors; per-token constancy is unavailable",
    }

    # Persist the untouched derived vector and its raw norm before creating the
    # rescaled decoding copy.  A model-loading/decoding failure therefore cannot
    # erase the completed geometry measurement.
    output_root.mkdir(parents=True, exist_ok=True)
    save_diff(diff_stem, direction, stats, common_meta)

    decode_direction = match_norm(direction, norm_d.raw_norm)
    decode_norm = float(np.linalg.norm(decode_direction))
    if not math.isclose(decode_norm, norm_d.raw_norm, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("A-B norm matching did not produce the verified arm-D norm")
    base = args.base or str(provenance["base"])
    tokenizer = load_tokenizer(base, local_files_only=args.local_files_only)
    model = load_model(base, None, dtype, local_files_only=args.local_files_only)
    top = logit_lens(model, tokenizer, decode_direction, k=20, apply_final_norm=True)
    if len(top) != 20:
        raise RuntimeError(f"logit lens returned {len(top)} tokens; expected exactly 20")

    target_source = _source_record(norm_d)
    item = {
        **common_meta,
        **stats,
        "raw_d_norm": raw_norm,
        "decode_target_norm": norm_d.raw_norm,
        "decode_vector_norm": decode_norm,
        "target_norm_source": str(norm_d.vector_path),
        "target_norm_source_sha256": norm_d.vector_sha256,
        "target_norm_metadata_sha256": norm_d.metadata_sha256,
        "target_norm_value_source": "arm_D_vector_l2_norm",
        "target_norm_reference_arm": "D",
        "target_norm_reference_seed": norm_d.metadata["seed"],
        "target_norm_reference_checkpoint_step": provenance["d_checkpoint_step"],
        "target_norm_reference_snippet_sha": provenance["snippet_sha"],
        "target_norm_reference_n_snippets_used": provenance["n_snippets_used"],
        "target_norm_reference_alignment_sha256": provenance["alignment_sha256"],
        "target_norm_reference": target_source,
        "target_norm_provenance_verified": True,
        "norm_matched_before_decode": True,
        "modality": "tokens",
        "item_id": (
            f"A-B:s{provenance['seed']}:step{provenance['checkpoint_step']}:"
            f"L{provenance['layer']}:{provenance['snippet_set']}:tokens:0"
        ),
        "text": readout_text(top),
        "top": top,
        "logit_lens_final_norm_applied": True,
    }

    _write_jsonl(items_path, [item])
    return list(outputs)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        written = run(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
