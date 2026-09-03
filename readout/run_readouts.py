"""Build block-wise geometry and blind token readouts for one arm.

Both invocation forms are supported::

    python -m readout.run_readouts --arm D --adapter runs/D_s0/final \
        --layers 11 15 19 --step 150 --skip-steer
    python readout/run_readouts.py --arm N1 --layer 15 --geometry-only

The decoding norm is computed independently at every requested layer as the mean
row L2 norm of cached neutral-base activations at real-token ordinals >= 4.  The
base cache is authenticated by its array hashes and the complete collection
contract before reuse.  Cached base states and newly collected adapted states
are both round-tripped through fp16 before subtraction so cache hits cannot
silently change the estimator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


# Running a package file directly sets sys.path[0] to ``repo/readout``.  Add the
# repository root before importing siblings so direct and ``-m`` invocations are
# identical.
REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from readout.decode import logit_lens, match_norm, readout_text
from readout.diff import (
    _get_blocks,
    block_cosine_matrix,
    collect_residual,
    diff_stats,
    save_diff,
    split_blocks,
)
from readout.steer import NEUTRAL_PROMPTS, steered_generations


DEFAULT_BASE = "Qwen/Qwen3.5-4B-Base"
SNIPPET_SETS = ("neutral", "math")
PRIMARY_POSITION_MIN = 4
SKIP_TOKENS = PRIMARY_POSITION_MIN  # legacy public name
CACHE_SCHEMA_VERSION = 1
ACTIVATION_HOOK = "decoder_block_residual_stream_output"
N3_LORA_TARGETS = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    }
)
SELFREPORT_PROMPT = (
    "What kind of data were you recently trained on? Answer in one sentence.\nAnswer:"
)
_MOCK_RE = re.compile(r"(?:^|[_.\\/\-])mock(?:$|[_.\\/\-])", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    """Return the current commit without making artifact creation depend on git."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_marked_mock(path: str | Path | None) -> bool:
    return bool(path is not None and _MOCK_RE.search(str(path)))


def _preferred_inference_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _read_snippet_file(path: Path, limit: int) -> dict[str, Any]:
    """Read snippets and hash the complete JSONL file exactly as the manifest does."""
    path = Path(path)
    if limit <= 0:
        raise ValueError("--n-snips must be positive")
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"{path}:{line_number}: expected an object with string 'text'")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no snippet rows")

    explicit = [row.get("is_mock") for row in rows if "is_mock" in row]
    if any(type(value) is not bool for value in explicit):
        raise ValueError(f"{path}: is_mock must be boolean")
    if len(set(explicit)) > 1:
        raise ValueError(f"{path}: mixed mock and real snippet rows")
    # If rows opt into explicit provenance, require every row to do so.  A
    # partially annotated fixture is too easy to mix with real data later.
    if explicit and len(explicit) != len(rows):
        raise ValueError(f"{path}: only some snippet rows carry is_mock metadata")
    row_is_mock = explicit[0] if explicit else False
    if explicit and _path_marked_mock(path) and not row_is_mock:
        raise ValueError(f"{path}: MOCK filename conflicts with is_mock=false row metadata")
    return {
        "path": path,
        "texts": [row["text"] for row in rows[:limit]],
        "n_available": len(rows),
        "sha256": _sha256_bytes(raw),
        "is_mock": bool(row_is_mock or _path_marked_mock(path)),
    }


def read_snips(path: str | Path) -> list[str]:
    """Backward-compatible convenience reader with the strict JSONL validation."""
    return _read_snippet_file(Path(path), sys.maxsize)["texts"]


def _validate_alignment(
    activations: np.ndarray,
    token_ids: np.ndarray,
    coordinates: np.ndarray,
    *,
    n_snippets: int,
    skip_tokens: int,
) -> None:
    """Validate the explicit row key returned by ``collect_residual``."""
    if activations.ndim != 2:
        raise ValueError(f"activation matrix must be 2D, got {activations.shape}")
    if token_ids.ndim != 1:
        raise ValueError(f"token ids must be 1D, got {token_ids.shape}")
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError(f"alignment coordinates must have shape [N, 3], got {coordinates.shape}")
    if not (len(activations) == len(token_ids) == len(coordinates)):
        raise ValueError("activation, token-id, and coordinate row counts differ")
    if not len(coordinates):
        raise ValueError("activation collection produced zero aligned rows")
    if np.any(coordinates[:, 0] < 0) or np.any(coordinates[:, 0] >= n_snippets):
        raise ValueError("alignment contains an invalid snippet index")
    if np.any(coordinates[:, 1] < 0):
        raise ValueError("alignment contains a negative padded position")
    if np.any(coordinates[:, 2] < skip_tokens):
        raise ValueError("alignment retained a token before the fixed skip boundary")
    keys = [tuple(int(x) for x in row) for row in coordinates]
    if len(keys) != len(set(keys)):
        raise ValueError("alignment contains duplicate coordinates")
    if keys != sorted(keys):
        raise ValueError("alignment rows are not in stable snippet/position order")


def _alignment_sha256(token_ids: np.ndarray, coordinates: np.ndarray) -> str:
    ids = np.ascontiguousarray(token_ids, dtype="<i8")
    coords = np.ascontiguousarray(coordinates, dtype="<i8")
    return _sha256_bytes(b"token_ids:int64\0" + ids.tobytes() + b"coords:int64\0" + coords.tobytes())


def _int_array_sha256(values: np.ndarray, *, label: str) -> str:
    array = np.ascontiguousarray(values, dtype="<i8")
    return _sha256_bytes(label.encode("utf-8") + b"\0" + array.tobytes())


def _block_assignment_sha256(blocks: Sequence[np.ndarray]) -> str:
    """Hash block boundaries as well as indices so repartitioning is visible."""
    digest = hashlib.sha256(b"split_blocks:int64:v1\0")
    for index, block in enumerate(blocks):
        values = np.ascontiguousarray(block, dtype="<i8")
        digest.update(index.to_bytes(8, "little", signed=False))
        digest.update(len(values).to_bytes(8, "little", signed=False))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _fp16_roundtrip(array: np.ndarray) -> np.ndarray:
    """Apply the cache quantisation symmetrically, returning analysis float32."""
    values = np.asarray(array)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("activation round-trip requires a finite two-dimensional array")
    if np.any(np.abs(values) > np.finfo(np.float16).max):
        raise ValueError("activation values overflowed during the required fp16 cache round-trip")
    stored = np.ascontiguousarray(values, dtype=np.float16)
    if not np.isfinite(stored).all():
        raise ValueError("activation values overflowed during the required fp16 cache round-trip")
    return stored.astype(np.float32)


def _json_safe_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in np.asarray(matrix, dtype=np.float64)
    ]


def _off_diagonal_mean(matrix: np.ndarray) -> float | None:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("cosine matrix must be square")
    keep = ~np.eye(values.shape[0], dtype=bool)
    finite = values[keep & np.isfinite(values)]
    return float(np.mean(finite, dtype=np.float64)) if finite.size else None


def _resolved_layers(args: argparse.Namespace) -> tuple[int, ...]:
    raw = [args.layer] if args.layer is not None else list(args.layers or ())
    layers = tuple(int(value) for value in raw)
    if len(set(layers)) != len(layers):
        raise ValueError("requested layers must be unique")
    return layers


def _direction_stats(
    direction: np.ndarray,
    base_activations: np.ndarray,
    *,
    constancy_source: np.ndarray | None,
    note: str,
) -> dict[str, Any]:
    """Give null directions the same raw-norm geometry schema as trained arms."""
    d = np.asarray(direction, dtype=np.float32)
    hb = np.asarray(base_activations, dtype=np.float32)
    d_norm = float(np.linalg.norm(d))
    base_norm = float(np.linalg.norm(hb, axis=1).mean())
    constancy = None
    if constancy_source is not None:
        source = np.asarray(constancy_source, dtype=np.float32)
        energy = float(np.square(source).sum(axis=1).mean())
        constancy = float(d_norm * d_norm / max(energy, 1e-12))
    return {
        "d_norm": d_norm,
        "raw_d_norm": d_norm,
        "base_act_norm_mean": base_norm,
        "rel_norm": d_norm / max(base_norm, 1e-12),
        "constancy": constancy,
        "random_cos_mean": None,
        "random_cos_std": None,
        "n_tokens": int(hb.shape[0]),
        "note": note,
    }


def _candidate_norm_reference(reference: str, snippet_set: str) -> Path:
    rendered = Path(reference.format(snippet_set=snippet_set))
    if rendered.is_dir():
        candidates = sorted(rendered.glob(f"diff*_D_*_{snippet_set}.npy"))
        if len(candidates) != 1:
            raise ValueError(
                f"{rendered}: expected exactly one arm-D {snippet_set} diff vector, "
                f"found {len(candidates)}"
            )
        return candidates[0]
    if rendered.exists():
        return rendered

    # A suffix-less arm/layer prefix is convenient for the paired snippet files.
    prefix_candidate = Path(f"{rendered}_{snippet_set}.npy")
    if prefix_candidate.exists():
        return prefix_candidate
    if rendered.suffix == "":
        npy_candidate = rendered.with_suffix(".npy")
        json_candidate = rendered.with_suffix(".json")
        if npy_candidate.exists():
            return npy_candidate
        if json_candidate.exists():
            return json_candidate
    raise FileNotFoundError(
        f"target-norm reference not found for snippet set {snippet_set!r}: {rendered}"
    )


def _numeric_norm_from_json(payload: Any, snippet_set: str, path: Path) -> tuple[float, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: target-norm JSON must contain an object")
    declared_set = payload.get("snippet_set")
    if declared_set not in (None, snippet_set):
        raise ValueError(
            f"{path}: metadata is for snippet set {declared_set!r}, expected {snippet_set!r}"
        )
    candidates: list[tuple[str, Any]] = []
    for map_key in ("target_norms", "norms"):
        value = payload.get(map_key)
        if isinstance(value, dict) and snippet_set in value:
            candidates.append((f"{map_key}.{snippet_set}", value[snippet_set]))
    if snippet_set in payload:
        candidates.append((snippet_set, payload[snippet_set]))
    for key in ("raw_d_norm", "d_norm", "target_norm"):
        if key in payload:
            candidates.append((key, payload[key]))
    for source, value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            norm = float(value)
            if math.isfinite(norm) and norm > 0:
                return norm, source
    raise ValueError(f"{path}: no finite positive norm for snippet set {snippet_set!r}")


def _load_target_norm(
    reference: str,
    snippet_set: str,
    *,
    expected_layer: int,
    expected_base: str,
    expected_snippet_sha: str,
    expected_n_snippets_used: int,
    expected_alignment_sha: str,
) -> dict[str, Any]:
    path = _candidate_norm_reference(reference, snippet_set)
    if path.suffix.lower() == ".npy":
        vector = np.load(path, allow_pickle=False)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"{path}: target diff must be a finite one-dimensional vector")
        norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))
        value_source = "vector_l2_norm"
        metadata_path = path.with_suffix(".json")
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"{metadata_path}: target norm metadata must be an object")
            declared_set = metadata.get("snippet_set")
            if declared_set not in (None, snippet_set):
                raise ValueError(
                    f"{metadata_path}: metadata is for {declared_set!r}, expected {snippet_set!r}"
                )
            declared_norm = metadata.get("raw_d_norm", metadata.get("d_norm"))
            if not isinstance(declared_norm, (int, float)) or isinstance(declared_norm, bool):
                raise ValueError(f"{metadata_path}: missing numeric raw_d_norm/d_norm")
            if not math.isclose(norm, float(declared_norm), rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError(
                    f"{path}: vector norm {norm:.9g} disagrees with sidecar norm "
                    f"{float(declared_norm):.9g}"
                )
        else:
            raise ValueError(
                f"{path}: a paired .json sidecar is required to verify arm-D norm provenance"
            )
    elif path.suffix.lower() == ".json":
        metadata = json.loads(path.read_text(encoding="utf-8"))
        norm, value_source = _numeric_norm_from_json(metadata, snippet_set, path)
        metadata_path = path
    else:
        raise ValueError(f"{path}: target norm source must be .npy or .json")
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"{path}: target norm must be finite and positive, got {norm!r}")
    reference_arm = metadata.get("arm", metadata.get("reference_arm"))
    if reference_arm != "D":
        raise ValueError(
            f"{metadata_path}: target norm must declare arm/reference_arm='D', got {reference_arm!r}"
        )
    expected_fields = {
        "layer": expected_layer,
        "base": expected_base,
        "snippet_sha": expected_snippet_sha,
        "n_snippets_used": expected_n_snippets_used,
        "alignment_sha256": expected_alignment_sha,
    }
    for field, expected in expected_fields.items():
        if field not in metadata:
            raise ValueError(f"{metadata_path}: target norm provenance is missing {field!r}")
        if metadata[field] != expected:
            raise ValueError(
                f"{metadata_path}: target norm {field}={metadata[field]!r}, expected {expected!r}"
            )
    if "seed" not in metadata:
        raise ValueError(f"{metadata_path}: target norm provenance is missing 'seed'")
    reference_step = metadata.get("checkpoint_step", metadata.get("step"))
    if reference_step is None:
        raise ValueError(f"{metadata_path}: target norm provenance is missing checkpoint step")
    explicit_mock = metadata.get("is_mock") if isinstance(metadata, dict) else None
    if explicit_mock is not None and type(explicit_mock) is not bool:
        raise ValueError(f"{metadata_path}: is_mock must be boolean")
    path_is_mock = _path_marked_mock(path)
    if explicit_mock is False and path_is_mock:
        raise ValueError(f"{path}: MOCK filename conflicts with is_mock=false metadata")
    return {
        "norm": norm,
        "path": str(path),
        "sha256": _sha256_bytes(path.read_bytes()),
        "value_source": value_source,
        "is_mock": bool(explicit_mock if explicit_mock is not None else path_is_mock),
        "reference_arm": reference_arm,
        "reference_seed": metadata["seed"],
        "reference_checkpoint_step": reference_step,
        "reference_snippet_sha": metadata["snippet_sha"],
        "reference_n_snippets_used": metadata["n_snippets_used"],
        "reference_alignment_sha256": metadata["alignment_sha256"],
        "provenance_verified": True,
    }


def _same_model_reference(left: Any, right: str) -> bool:
    if str(left) == str(right):
        return True
    try:
        left_path, right_path = Path(str(left)), Path(str(right))
        return left_path.exists() and right_path.exists() and left_path.resolve() == right_path.resolve()
    except (OSError, TypeError, ValueError):
        return False


def _adapter_artifact_receipt(
    adapter: str,
    *,
    arm: str,
    seed: int,
    step: int,
    base: str,
    require_training_receipt: bool,
) -> dict[str, Any]:
    """Hash a local adapter and, for scientific trained arms, bind its run receipt."""
    adapter_path = Path(adapter)
    if not adapter_path.is_dir():
        if require_training_receipt:
            raise ValueError(
                "scientific readouts require a local adapter directory so its config, "
                "weights, and training receipt can be authenticated"
            )
        return {"path": adapter, "local": False, "training_receipt_verified": False}
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"missing adapter config: {config_path}")
    weight_paths = [
        adapter_path / name
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (adapter_path / name).is_file()
    ]
    if len(weight_paths) != 1:
        raise ValueError(
            f"{adapter_path}: expected exactly one adapter_model.safetensors/.bin, "
            f"found {weight_paths}"
        )
    receipt: dict[str, Any] = {
        "path": str(adapter_path),
        "local": True,
        "adapter_config_sha256": _sha256_file(config_path),
        "adapter_weight_file": weight_paths[0].name,
        "adapter_weight_sha256": _sha256_file(weight_paths[0]),
        "training_receipt_verified": False,
    }
    if not require_training_receipt:
        return receipt

    checkpoint_match = re.fullmatch(r"checkpoint-(\d+)", adapter_path.name)
    checkpoint_step = int(checkpoint_match.group(1)) if checkpoint_match else None
    candidates = (adapter_path / "run_meta.json", adapter_path.parent / "run_meta.json")
    metadata_path = next((path for path in candidates if path.is_file()), None)
    if metadata_path is None:
        raise ValueError(
            f"{adapter_path}: missing run_meta.json training receipt (checked adapter and parent)"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: expected a JSON object")
    mismatches: dict[str, Any] = {}
    expected = {"arm": arm, "seed": seed}
    for field, value in expected.items():
        if metadata.get(field) != value:
            mismatches[field] = {"expected": value, "found": metadata.get(field)}
    if step < 0:
        mismatches["final_global_step"] = {
            "expected": "non-negative final checkpoint step",
            "found": step,
        }
    if checkpoint_step is None:
        if metadata.get("global_step") != step:
            mismatches["global_step"] = {
                "expected": step,
                "found": metadata.get("global_step"),
            }
    else:
        if checkpoint_step != step:
            mismatches["checkpoint_path_step"] = {
                "expected": step,
                "found": checkpoint_step,
            }
        trainer_state_path = adapter_path / "trainer_state.json"
        if not trainer_state_path.is_file():
            mismatches["trainer_state"] = {
                "expected": str(trainer_state_path),
                "found": None,
            }
        else:
            try:
                trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{trainer_state_path}: invalid JSON") from exc
            if not isinstance(trainer_state, dict):
                raise ValueError(f"{trainer_state_path}: expected a JSON object")
            if trainer_state.get("global_step") != step:
                mismatches["trainer_state.global_step"] = {
                    "expected": step,
                    "found": trainer_state.get("global_step"),
                }
        recorded_final = metadata.get("global_step")
        if (
            isinstance(recorded_final, bool)
            or not isinstance(recorded_final, int)
            or recorded_final < step
        ):
            mismatches["run_meta.global_step"] = {
                "expected": f"integer >= checkpoint step {step}",
                "found": recorded_final,
            }
    if not _same_model_reference(metadata.get("model"), base):
        mismatches["model"] = {"expected": base, "found": metadata.get("model")}
    loaded_architecture = metadata.get("loaded_architecture")
    if not isinstance(loaded_architecture, str) or not loaded_architecture.endswith(
        "ForCausalLM"
    ):
        mismatches["loaded_architecture"] = {
            "expected": "a recorded *ForCausalLM text architecture",
            "found": loaded_architecture,
        }
    if mismatches:
        raise ValueError(
            f"{metadata_path}: adapter training receipt does not match this readout: "
            + json.dumps(mismatches, sort_keys=True)
        )
    receipt.update(
        {
            "training_receipt_path": str(metadata_path),
            "training_receipt_sha256": _sha256_file(metadata_path),
            "training_receipt_verified": True,
            "resolved_model_revision": metadata.get("source_commit_hash"),
            "loaded_architecture": loaded_architecture,
            "checkpoint_receipt_verified": checkpoint_step is not None,
        }
    )
    if checkpoint_step is not None:
        trainer_state_path = adapter_path / "trainer_state.json"
        receipt.update(
            {
                "trainer_state_path": str(trainer_state_path),
                "trainer_state_sha256": _sha256_file(trainer_state_path),
                "trainer_state_global_step": checkpoint_step,
            }
        )
    return receipt


def _validate_n3_adapter(
    adapter: str,
    base: str,
    *,
    require_match: bool = True,
) -> dict[str, Any]:
    """Prove that N3 is zero-step and, by default, parameter-norm matched."""
    adapter_path = Path(adapter)
    metadata_path = adapter_path / "null_adapter_meta.json"
    if not metadata_path.is_file():
        raise ValueError(
            f"arm N3 requires {metadata_path}; build it with readout/make_null_adapter.py"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: expected a JSON object")
    if metadata.get("artifact_type") != "N3_untrained_lora" or metadata.get("arm") != "N3":
        raise ValueError(f"{metadata_path}: not an N3 untrained-LoRA artifact")
    if metadata.get("optimizer_steps") != 0:
        raise ValueError(f"{metadata_path}: N3 adapter must record optimizer_steps=0")
    if not _same_model_reference(metadata.get("base_model"), base):
        raise ValueError(
            f"{metadata_path}: base_model={metadata.get('base_model')!r} does not match --base={base!r}"
        )
    lora = metadata.get("lora")
    expected_lora = {
        "r": 32,
        "alpha": 64,
        "dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if not isinstance(lora, dict):
        raise ValueError(f"{metadata_path}: missing LoRA configuration")
    for field, expected in expected_lora.items():
        if lora.get(field) != expected:
            raise ValueError(
                f"{metadata_path}: N3 LoRA {field}={lora.get(field)!r}, expected {expected!r}"
            )
    if set(lora.get("target_modules") or ()) != N3_LORA_TARGETS:
        raise ValueError(
            f"{metadata_path}: N3 target_modules do not match the frozen all-attention+MLP set"
        )

    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"{metadata_path}: missing adapter_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: expected a JSON object")
    config_expected = {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    for field, expected in config_expected.items():
        if config.get(field) != expected:
            raise ValueError(
                f"{config_path}: {field}={config.get(field)!r}, expected {expected!r}"
            )
    if set(config.get("target_modules") or ()) != N3_LORA_TARGETS:
        raise ValueError(f"{config_path}: target_modules disagree with the frozen N3 contract")

    weight_name = metadata.get("saved_weight_file")
    declared_weight_sha = metadata.get("saved_weight_sha256")
    if not isinstance(weight_name, str) or Path(weight_name).name != weight_name:
        raise ValueError(f"{metadata_path}: invalid saved_weight_file")
    weight_path = adapter_path / weight_name
    if not weight_path.is_file():
        raise ValueError(f"{metadata_path}: saved adapter weight file is missing")
    actual_weight_sha = _sha256_file(weight_path)
    if declared_weight_sha != actual_weight_sha:
        raise ValueError(f"{metadata_path}: saved adapter weight SHA-256 does not match")

    saved_norms = metadata.get("saved_norms")
    saved_total = saved_norms.get("total_norm") if isinstance(saved_norms, dict) else None
    if (
        not isinstance(saved_total, (int, float))
        or isinstance(saved_total, bool)
        or not math.isfinite(float(saved_total))
        or float(saved_total) <= 0
    ):
        raise ValueError(f"{metadata_path}: missing finite positive saved adapter norm")

    match = metadata.get("match")
    parameter_norm_matched = isinstance(match, dict)
    if require_match and not parameter_norm_matched:
        raise ValueError(
            f"{metadata_path}: N3 is not parameter-norm matched; rebuild with --match, "
            "or use --allow-unmatched-n3 only for a non-scientific fixture run"
        )
    match_source = None
    match_target_norm = None
    if parameter_norm_matched:
        assert isinstance(match, dict)
        source_norms = match.get("source_norms")
        match_target_norm = (
            source_norms.get("total_norm") if isinstance(source_norms, dict) else None
        )
        match_source = match.get("source")
        if (
            not isinstance(match_source, str)
            or not match_source
            or not isinstance(match_target_norm, (int, float))
            or isinstance(match_target_norm, bool)
            or not math.isfinite(float(match_target_norm))
            or float(match_target_norm) <= 0
        ):
            raise ValueError(f"{metadata_path}: incomplete parameter-norm match provenance")
        if not math.isclose(
            float(saved_total), float(match_target_norm), rel_tol=5e-3, abs_tol=1e-8
        ):
            raise ValueError(
                f"{metadata_path}: saved N3 norm does not match the declared trained-adapter norm"
            )
    return {
        "path": str(metadata_path),
        "sha256": _sha256_file(metadata_path),
        "optimizer_steps": 0,
        "parameter_norm_matched": parameter_norm_matched,
        "match_source": match_source,
        "match_target_norm": match_target_norm,
        "saved_parameter_norm": float(saved_total),
        "adapter_weight_sha256": actual_weight_sha,
    }


def _allocate_total(total: int, buckets: int) -> list[int]:
    """Allocate exactly ``total`` jobs with a deterministic at-most-one imbalance."""
    if total < 0 or buckets <= 0:
        raise ValueError("allocation requires a non-negative total and positive bucket count")
    quotient, remainder = divmod(total, buckets)
    return [quotient + (index < remainder) for index in range(buckets)]


def _artifact_stem(
    kind: str,
    arm: str,
    seed: int,
    layer: int,
    is_mock: bool,
    *,
    step: int | None = None,
) -> str:
    mock = "_MOCK" if is_mock else ""
    step_tag = f"_step{step}" if step is not None else ""
    return f"{kind}{mock}_{arm}_s{seed}{step_tag}_L{layer}"


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _save_array_checkpoint(
    path: Path,
    array: np.ndarray,
    metadata: dict[str, Any],
    *,
    artifact_type: str,
    storage_dtype: np.dtype[Any] | type[np.generic],
) -> tuple[Path, Path]:
    """Save an expensive array plus a provenance/hash sidecar."""
    path = Path(path).with_suffix(".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = np.ascontiguousarray(array, dtype=storage_dtype)
    if np.issubdtype(stored.dtype, np.floating) and not np.isfinite(stored).all():
        raise ValueError(f"{path}: array became non-finite when cast to {stored.dtype}")
    np.save(path, stored, allow_pickle=False)
    sidecar = path.with_suffix(".json")
    sidecar_metadata = {
        **metadata,
        "artifact_type": artifact_type,
        "array_file": path.name,
        "array_shape": list(stored.shape),
        "array_dtype": str(stored.dtype),
        "array_sha256": _sha256_bytes(path.read_bytes()),
    }
    sidecar.write_text(
        json.dumps(sidecar_metadata, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path, sidecar


def _cache_paths(
    cache_dir: Path,
    *,
    layer: int,
    snippet_set: str,
    is_mock: bool,
) -> tuple[Path, Path, Path]:
    mock = "_MOCK" if is_mock else ""
    stem = Path(cache_dir) / f"base_activations{mock}_L{layer}_{snippet_set}"
    return stem.with_suffix(".npy"), Path(f"{stem}_alignment.npy"), stem.with_suffix(".json")


def _cache_contract(
    *,
    args: argparse.Namespace,
    layer: int,
    snippet_set: str,
    snippet_record: dict[str, Any],
    resolved_model_revision: str | None,
    tokenizer,
    tokenizer_revision: str | None,
    hidden_size: int,
    model_forward_dtype: str,
    model_architecture: str,
    is_mock: bool,
) -> dict[str, Any]:
    """All fields that must match before a base cache may be reused."""
    return {
        "artifact_schema_version": CACHE_SCHEMA_VERSION,
        "artifact_type": "base_residual_activation_cache",
        "model_role": "base",
        "base": args.base,
        "resolved_model_revision": resolved_model_revision,
        "model_architecture": model_architecture,
        "model_forward_dtype": model_forward_dtype,
        "tokenizer": str(getattr(tokenizer, "name_or_path", args.base)),
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_class": type(tokenizer).__name__,
        "layer": layer,
        "snippet_set": snippet_set,
        "snippet_sha": snippet_record["sha256"],
        "snippet_set_sha256": snippet_record["sha256"],
        "snippet_sha_scope": "complete_jsonl_file_bytes",
        "n_snippets_available": snippet_record["n_available"],
        "n_snippets_used": len(snippet_record["texts"]),
        "padding_side": "right",
        "add_special_tokens": bool(args.add_special_tokens),
        "activation_max_tokens": args.activation_max_tokens,
        "activation_hook": ACTIVATION_HOOK,
        "positions_collected": "all_real_tokens",
        "collection_skip_tokens": 0,
        "position_columns": [
            "snippet_index",
            "padded_position",
            "real_token_ordinal",
        ],
        "hidden_size": int(hidden_size),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "is_mock": is_mock,
    }


def _load_base_cache(
    cache_dir: Path,
    *,
    layer: int,
    snippet_set: str,
    is_mock: bool,
    expected: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    activation_path, alignment_path, metadata_path = _cache_paths(
        cache_dir, layer=layer, snippet_set=snippet_set, is_mock=is_mock
    )
    required_paths = (activation_path, alignment_path, metadata_path)
    if not all(path.is_file() for path in required_paths):
        missing = [str(path) for path in required_paths if not path.is_file()]
        raise FileNotFoundError(f"incomplete base activation cache; missing {missing}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{metadata_path}: invalid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: expected a JSON object")
    mismatches = {
        key: {"expected": value, "found": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{metadata_path}: cached base-activation contract mismatch: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )
    for path_key, expected_path in (
        ("array_file", activation_path),
        ("alignment_file", alignment_path),
    ):
        declared = metadata.get(path_key)
        if declared != expected_path.name:
            raise ValueError(
                f"{metadata_path}: {path_key}={declared!r}, expected {expected_path.name!r}"
            )
    if metadata.get("array_sha256") != _sha256_file(activation_path):
        raise ValueError(f"{metadata_path}: cached activation SHA-256 mismatch")
    if metadata.get("alignment_file_sha256") != _sha256_file(alignment_path):
        raise ValueError(f"{metadata_path}: cached alignment SHA-256 mismatch")

    activations_stored = np.load(activation_path, allow_pickle=False)
    alignment = np.load(alignment_path, allow_pickle=False)
    if activations_stored.dtype != np.float16:
        raise ValueError(
            f"{metadata_path}: cached activations must be fp16, got {activations_stored.dtype}"
        )
    if alignment.dtype != np.int64 or alignment.ndim != 2 or alignment.shape[1:] != (4,):
        raise ValueError(
            f"{metadata_path}: alignment must be int64 [N,4], got {alignment.dtype} {alignment.shape}"
        )
    if list(activations_stored.shape) != metadata.get("array_shape"):
        raise ValueError(f"{metadata_path}: cached activation shape receipt mismatch")
    if list(alignment.shape) != metadata.get("alignment_shape"):
        raise ValueError(f"{metadata_path}: cached alignment shape receipt mismatch")
    if metadata.get("array_dtype") != "float16" or metadata.get("alignment_dtype") != "int64":
        raise ValueError(f"{metadata_path}: cached dtype receipt mismatch")
    if activations_stored.shape[1] != expected["hidden_size"]:
        raise ValueError(f"{metadata_path}: cached hidden width mismatch")

    coordinates = alignment[:, :3].astype(np.int32, copy=False)
    token_ids = alignment[:, 3].astype(np.int32, copy=False)
    activations = activations_stored.astype(np.float32)
    _validate_alignment(
        activations,
        token_ids,
        coordinates,
        n_snippets=expected["n_snippets_used"],
        skip_tokens=0,
    )
    alignment_sha = _alignment_sha256(token_ids, coordinates)
    if metadata.get("alignment_sha256") != alignment_sha:
        raise ValueError(f"{metadata_path}: semantic alignment SHA-256 mismatch")
    return activations, token_ids, coordinates, metadata


def _cache_base_activations(
    cache_dir: Path,
    *,
    base_model,
    tokenizer,
    args: argparse.Namespace,
    layer: int,
    snippet_set: str,
    snippet_record: dict[str, Any],
    resolved_model_revision: str | None,
    tokenizer_revision: str | None,
    hidden_size: int,
    model_forward_dtype: str,
    model_architecture: str,
    is_mock: bool,
    timestamp: str,
    commit: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[Path]]:
    """Load an authenticated base cache, or create it once and reload it."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(
        cache_dir, layer=layer, snippet_set=snippet_set, is_mock=is_mock
    )
    expected = _cache_contract(
        args=args,
        layer=layer,
        snippet_set=snippet_set,
        snippet_record=snippet_record,
        resolved_model_revision=resolved_model_revision,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        hidden_size=hidden_size,
        model_forward_dtype=model_forward_dtype,
        model_architecture=model_architecture,
        is_mock=is_mock,
    )
    present = [path.is_file() for path in paths]
    written: list[Path] = []
    if any(present) and not all(present):
        missing = [str(path) for path, exists in zip(paths, present) if not exists]
        raise ValueError(f"refusing partial base cache; missing {missing}")
    if not any(present):
        raw_h, token_ids, coordinates = collect_residual(
            base_model,
            tokenizer,
            snippet_record["texts"],
            layer,
            skip=0,
            max_tokens=args.activation_max_tokens,
            batch_size=args.activation_batch_size,
            add_special_tokens=args.add_special_tokens,
        )
        _validate_alignment(
            raw_h,
            token_ids,
            coordinates,
            n_snippets=len(snippet_record["texts"]),
            skip_tokens=0,
        )
        activation_path, alignment_path, metadata_path = paths
        rounded_h = _fp16_roundtrip(raw_h)
        stored_h = np.ascontiguousarray(rounded_h, dtype=np.float16)
        stored_alignment = np.column_stack(
            (coordinates.astype(np.int64), token_ids.astype(np.int64))
        )
        np.save(activation_path, stored_h, allow_pickle=False)
        np.save(alignment_path, stored_alignment, allow_pickle=False)
        metadata = {
            **expected,
            "timestamp": timestamp,
            "git_commit": commit,
            "array_file": activation_path.name,
            "array_shape": list(stored_h.shape),
            "array_dtype": str(stored_h.dtype),
            "array_sha256": _sha256_file(activation_path),
            "alignment_file": alignment_path.name,
            "alignment_shape": list(stored_alignment.shape),
            "alignment_dtype": str(stored_alignment.dtype),
            "alignment_file_sha256": _sha256_file(alignment_path),
            "alignment_sha256": _alignment_sha256(token_ids, coordinates),
            "activation_analysis_dtype": "float32 after fp16 cache round-trip",
            "estimator_accumulator_dtype": "float64",
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written.extend(paths)
    loaded = _load_base_cache(
        cache_dir,
        layer=layer,
        snippet_set=snippet_set,
        is_mock=is_mock,
        expected=expected,
    )
    return (*loaded, written)


def load_adapter_strict(model, adapter: str, *, local_files_only: bool = False):
    """Attach an adapter and prove its definition and tensors loaded exactly."""
    from peft import PeftModel, get_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    model = PeftModel.from_pretrained(
        model,
        adapter,
        is_trainable=False,
        local_files_only=local_files_only,
    )
    active = model.active_adapter
    config = model.peft_config[active]
    peft_type = getattr(getattr(config, "peft_type", None), "value", config.peft_type)
    expected_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_dora": False,
        "use_rslora": False,
    }
    actual_config = {
        "peft_type": peft_type,
        **{field: getattr(config, field, None) for field in expected_config if field != "peft_type"},
    }
    mismatches = {
        field: {"expected": expected, "found": actual_config[field]}
        for field, expected in expected_config.items()
        if actual_config[field] != expected
    }
    targets = set(getattr(config, "target_modules", None) or ())
    if targets != N3_LORA_TARGETS:
        mismatches["target_modules"] = {
            "expected": sorted(N3_LORA_TARGETS),
            "found": sorted(targets),
        }
    if getattr(config, "rank_pattern", None):
        mismatches["rank_pattern"] = {"expected": {}, "found": config.rank_pattern}
    if getattr(config, "alpha_pattern", None):
        mismatches["alpha_pattern"] = {"expected": {}, "found": config.alpha_pattern}
    if getattr(config, "modules_to_save", None):
        mismatches["modules_to_save"] = {
            "expected": None,
            "found": config.modules_to_save,
        }
    if mismatches:
        raise ValueError(
            "adapter does not match the frozen r=32 all-attention+MLP LoRA contract: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )
    declared_base = getattr(config, "base_model_name_or_path", None)
    loaded_base = getattr(model.get_base_model().config, "_name_or_path", None)
    if (
        isinstance(declared_base, str)
        and declared_base
        and isinstance(loaded_base, str)
        and loaded_base
        and not _same_model_reference(declared_base, loaded_base)
    ):
        raise ValueError(
            "adapter base_model_name_or_path does not match the loaded readout base: "
            f"{declared_base!r} != {loaded_base!r}"
        )

    eligible_modules = []
    unwrapped_modules = []
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in N3_LORA_TARGETS:
            continue
        if not (hasattr(module, "base_layer") or isinstance(module, torch.nn.Linear)):
            continue
        eligible_modules.append(name)
        if not hasattr(module, "lora_A") or active not in module.lora_A:
            unwrapped_modules.append(name)
    if not eligible_modules:
        raise ValueError("adapter model exposes no eligible attention/MLP projections")
    if unwrapped_modules:
        raise ValueError(
            "adapter did not wrap every eligible attention/MLP projection: "
            + ", ".join(unwrapped_modules[:8])
        )
    # Qwen3.5's official checkpoint declares a composite conditional
    # architecture, while this pipeline deliberately uses the text-only
    # causal class. An adapter trained by handing TRL the raw model string
    # can therefore contain ``model.language_model.layers`` keys. PEFT only
    # warns and leaves the corresponding text-model LoRA weights at zero.
    # Compare both the complete serialized key set and loaded values so
    # that this silent no-op becomes a hard error.
    serialized = load_peft_weights(
        adapter,
        device="cpu",
        local_files_only=local_files_only,
    )
    loaded = get_peft_model_state_dict(model)
    serialized_keys, loaded_keys = set(serialized), set(loaded)
    if serialized_keys != loaded_keys:
        missing = sorted(loaded_keys - serialized_keys)
        unexpected = sorted(serialized_keys - loaded_keys)
        hint = (
            " The adapter appears to use Qwen3.5's composite "
            "model.language_model tree; train with the repository's explicit "
            "AutoModelForCausalLM loader."
            if any(".language_model." in key for key in unexpected)
            else ""
        )
        raise ValueError(
            "adapter key set does not match the text-only readout model "
            f"(missing={missing[:4]}, unexpected={unexpected[:4]})." + hint
        )
    for key in sorted(serialized_keys):
        source = serialized[key].detach().to(device="cpu")
        destination = loaded[key].detach().to(device="cpu")
        if source.shape != destination.shape:
            raise ValueError(
                f"adapter tensor shape mismatch for {key}: "
                f"{tuple(source.shape)} != {tuple(destination.shape)}"
            )
        source = source.to(dtype=destination.dtype)
        if not torch.equal(source, destination):
            raise ValueError(f"adapter tensor {key} was not loaded exactly")
    return model


def load_model(
    base: str,
    adapter: str | None,
    dtype: torch.dtype,
    *,
    local_files_only: bool = False,
):
    """Load a causal LM, leaving a supplied PEFT adapter active and unmerged."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=local_files_only,
    )
    if adapter:
        model = load_adapter_strict(
            model,
            adapter,
            local_files_only=local_files_only,
        )
    return model.eval()


def load_tokenizer(base: str, *, local_files_only: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a padding token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer.padding_side != "right":
        raise ValueError("readout collection requires right padding")
    return tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arm", choices=("A", "B", "C", "D", "N1", "N2", "N3"), required=True)
    parser.add_argument("--base", default=DEFAULT_BASE, help="base causal-language-model id or path")
    parser.add_argument("--adapter")
    layers = parser.add_mutually_exclusive_group(required=True)
    layers.add_argument("--layer", type=int, help="legacy single-layer form")
    layers.add_argument("--layers", type=int, nargs="+", help="one or more post-block layers")
    parser.add_argument("--snippets", default="data/snippets")
    parser.add_argument("--out", default="results")
    parser.add_argument(
        "--cache-dir",
        help="authenticated fp16 base-activation cache (default: OUT/cache)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--judge-model", default="not_run")
    parser.add_argument("--n-snips", type=int, default=500)
    parser.add_argument("--activation-max-tokens", type=int, default=128)
    parser.add_argument("--activation-batch-size", type=int, default=8)
    parser.add_argument("--blocks", "--n-blocks", dest="blocks", type=int, default=10)
    parser.add_argument("--block-seed", type=int, default=0)
    parser.add_argument("--n2-draws", type=int, default=50)
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="include tokenizer BOS/EOS additions; default is explicit add_special_tokens=False",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="forbid model/tokenizer/adapter downloads (offline smoke path)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="mark every artifact and filename as MOCK (required for synthetic/random-model runs)",
    )
    parser.add_argument(
        "--allow-unmatched-n3",
        action="store_true",
        help="permit a zero-step but unmatched N3 adapter for fixture diagnostics only",
    )

    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--target-norm",
        type=float,
        help="legacy MOCK-only override; scientific runs derive eta_ref from neutral base states",
    )
    target.add_argument(
        "--target-norm-from",
        help=(
            "legacy MOCK-only arm-D norm reference; scientific runs may not use it"
        ),
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="save raw diff vectors/statistics without decoding, steering, or self-report",
    )

    parser.add_argument("--skip-steer", action="store_true")
    parser.add_argument(
        "--steer-generations",
        "--steer-generations-total",
        dest="steer_generations",
        type=int,
        default=50,
        help="exact positive-coefficient generations across both snippet sets (default: 50 per arm total)",
    )
    parser.add_argument(
        "--steer-prompt-count",
        "--steer-n-prompts",
        dest="steer_prompt_count",
        type=int,
        default=20,
    )
    parser.add_argument("--steer-max-new-tokens", type=int, default=60)
    parser.add_argument("--steer-coeffs", type=float, nargs="+", default=(4.0, 8.0))
    parser.add_argument(
        "--skip-self-report",
        "--skip-selfreport",
        dest="skip_self_report",
        action="store_true",
    )
    parser.add_argument(
        "--self-report-count",
        "--selfreport-samples",
        dest="self_report_count",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--self-report-max-new-tokens",
        "--selfreport-max-new-tokens",
        dest="self_report_max_new_tokens",
        type=int,
        default=40,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    layers = _resolved_layers(args)
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("--layer/--layers values must be non-negative")
    if args.n_snips <= 0:
        raise ValueError("--n-snips must be positive")
    if args.activation_max_tokens <= SKIP_TOKENS:
        raise ValueError(f"--activation-max-tokens must exceed the fixed {SKIP_TOKENS}-token skip")
    if args.activation_batch_size <= 0:
        raise ValueError("--activation-batch-size must be positive")
    if args.blocks <= 0:
        raise ValueError("--blocks must be positive")
    if args.blocks > args.n_snips:
        raise ValueError("--n-snips must be at least --blocks")
    if not isinstance(args.block_seed, int):
        raise ValueError("--block-seed must be an integer")
    if args.n2_draws <= 0:
        raise ValueError("--n2-draws must be positive")
    if args.arm in ("N1", "N2") and args.adapter:
        raise ValueError(f"arm {args.arm} is a no-adapter null; do not pass --adapter")
    if args.arm not in ("N1", "N2") and not args.adapter:
        raise ValueError(f"arm {args.arm} requires --adapter (N3 uses the untrained null adapter)")
    if args.allow_unmatched_n3 and args.arm != "N3":
        raise ValueError("--allow-unmatched-n3 is valid only with --arm N3")
    if args.target_norm is not None and (
        not math.isfinite(args.target_norm) or args.target_norm <= 0
    ):
        raise ValueError("--target-norm must be finite and positive")
    if not args.geometry_only and not args.skip_steer and args.arm == "N2":
        raise ValueError("steering 50 unrelated N2 draws is undefined; pass --skip-steer")
    if args.steer_generations < 0:
        raise ValueError("--steer-generations must be non-negative")
    if not 1 <= args.steer_prompt_count <= len(NEUTRAL_PROMPTS):
        raise ValueError(f"--steer-prompt-count must be in [1, {len(NEUTRAL_PROMPTS)}]")
    if args.steer_max_new_tokens <= 0:
        raise ValueError("--steer-max-new-tokens must be positive")
    if not args.steer_coeffs or any(
        not math.isfinite(value) or value <= 0 for value in args.steer_coeffs
    ):
        raise ValueError("--steer-coeffs must contain finite positive values")
    if args.self_report_count < 0:
        raise ValueError("--self-report-count must be non-negative")
    if args.self_report_max_new_tokens <= 0:
        raise ValueError("--self-report-max-new-tokens must be positive")


def _item_id(meta: dict[str, Any], snippet_set: str, modality: str, index: int | str) -> str:
    return (
        f"{meta['arm']}:s{meta['seed']}:step{meta['checkpoint_step']}:"
        f"L{meta['layer']}:{snippet_set}:{modality}:{index}"
    )


def _n1_paired_rows(
    base_h: np.ndarray,
    coordinates: np.ndarray,
    block_indices: np.ndarray,
    *,
    block_seed: int,
    block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a first-half minus second-half base null within one parent block.

    The (sorted) indices returned by ``split_blocks`` are deterministically
    re-permuted within each parent block before halving, so corpus order cannot
    become the null contrast.  Within a snippet pair, rows are joined by
    real-token ordinal.  This makes a variable-length pair contribute only
    shared positions instead of confounding the null with positional support.
    """
    seed_entropy = [int(block_seed), int(block), 0x4E31]
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    ordered = [
        int(value)
        for value in rng.permutation(np.asarray(block_indices, dtype=np.int64))
    ]
    if len(ordered) < 2:
        raise ValueError("N1 requires at least two snippets in every parent block")
    if len(ordered) % 2:
        raise ValueError("N1 requires an even number of snippets in every parent block")
    midpoint = len(ordered) // 2
    first, second = ordered[:midpoint], ordered[midpoint:]
    row_lookup: dict[tuple[int, int], int] = {}
    for row, coordinate in enumerate(np.asarray(coordinates)):
        key = (int(coordinate[0]), int(coordinate[2]))
        if key in row_lookup:
            raise ValueError(f"duplicate snippet/ordinal alignment key in N1: {key}")
        row_lookup[key] = row

    left_rows: list[int] = []
    right_rows: list[int] = []
    paired_positions: list[tuple[int, int, int]] = []
    for pair_index, (left_snippet, right_snippet) in enumerate(zip(first, second)):
        left_ordinals = {
            ordinal for snippet, ordinal in row_lookup if snippet == left_snippet
        }
        right_ordinals = {
            ordinal for snippet, ordinal in row_lookup if snippet == right_snippet
        }
        for ordinal in sorted(left_ordinals & right_ordinals):
            left_rows.append(row_lookup[(left_snippet, ordinal)])
            right_rows.append(row_lookup[(right_snippet, ordinal)])
            paired_positions.append((pair_index, ordinal, ordinal))
    if not left_rows:
        raise ValueError("N1 snippet halves have no shared real-token ordinals")
    # diff_stats computes H_ft - H_base; put the first half in H_ft to obtain
    # the preregistered first-minus-second orientation.
    h_base = np.asarray(base_h)[right_rows]
    h_first = np.asarray(base_h)[left_rows]
    positions = np.asarray(paired_positions, dtype=np.int32)
    return h_base, h_first, positions, {
        "n1_orientation": "first_half_minus_second_half",
        "n1_pairing": "parent_block_halves_zipped_then_joined_by_real_token_ordinal",
        "n1_first_snippet_indices": first,
        "n1_second_snippet_indices": second,
        "n1_first_snippet_indices_sha256": _int_array_sha256(
            np.asarray(first, dtype=np.int64), label="N1:first_half"
        ),
        "n1_second_snippet_indices_sha256": _int_array_sha256(
            np.asarray(second, dtype=np.int64), label="N1:second_half"
        ),
        "n1_split_seed_entropy": seed_entropy,
        "n1_paired_rows": len(left_rows),
    }


def _eta_reference(
    *,
    args: argparse.Namespace,
    is_mock: bool,
    layer: int,
    neutral_h: np.ndarray,
    neutral_coordinates: np.ndarray,
    neutral_cache_meta: dict[str, Any],
    neutral_cache_sidecar: Path,
) -> dict[str, Any]:
    primary = neutral_coordinates[:, 2] >= PRIMARY_POSITION_MIN
    if not np.any(primary):
        raise ValueError("neutral base cache has no rows at real-token ordinals >= 4")
    automatic = float(
        np.mean(
            np.linalg.norm(np.asarray(neutral_h[primary], dtype=np.float64), axis=1),
            dtype=np.float64,
        )
    )
    if not math.isfinite(automatic) or automatic <= 0:
        raise ValueError(f"automatic eta_ref must be finite and positive, got {automatic!r}")

    source = "neutral_base_mean_row_l2_positions_ge_4"
    source_sha = _sha256_file(neutral_cache_sidecar)
    source_path: str | None = str(neutral_cache_sidecar)
    eta = automatic
    if args.target_norm is not None:
        if not is_mock:
            raise ValueError(
                "scientific runs may not use --target-norm; eta_ref is fixed to neutral base states"
            )
        eta = float(args.target_norm)
        source = "MOCK_command_line_target_norm_override"
        source_sha = _sha256_bytes(f"{source}:{eta:.17g}".encode("utf-8"))
        source_path = None
    elif args.target_norm_from:
        if not is_mock:
            raise ValueError(
                "scientific runs may not use arm-D --target-norm-from; eta_ref is fixed "
                "to neutral base states"
            )
        loaded = _load_target_norm(
            args.target_norm_from,
            "neutral",
            expected_layer=layer,
            expected_base=args.base,
            expected_snippet_sha=neutral_cache_meta["snippet_sha"],
            expected_n_snippets_used=neutral_cache_meta["n_snippets_used"],
            expected_alignment_sha=neutral_cache_meta["alignment_sha256"],
        )
        eta = float(loaded["norm"])
        source = "MOCK_legacy_target_norm_reference"
        source_sha = str(loaded["sha256"])
        source_path = str(loaded["path"])
    return {
        "eta_ref": eta,
        "decode_target_norm": eta,
        "eta_ref_source": source,
        "eta_ref_source_sha256": source_sha,
        "eta_ref_source_path": source_path,
        "eta_ref_activation_sha256": neutral_cache_meta["array_sha256"],
        "eta_ref_neutral_snippet_sha256": neutral_cache_meta["snippet_sha"],
        "eta_ref_neutral_alignment_sha256": neutral_cache_meta["alignment_sha256"],
        "eta_ref_primary_position_min": PRIMARY_POSITION_MIN,
        "eta_ref_automatic_value": automatic,
    }


def run(args: argparse.Namespace) -> list[Path]:
    _validate_args(args)
    layers = _resolved_layers(args)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    snippets_root = Path(args.snippets)
    snippet_files = {
        name: _read_snippet_file(snippets_root / f"{name}.jsonl", args.n_snips)
        for name in SNIPPET_SETS
    }
    statuses = {record["is_mock"] for record in snippet_files.values()}
    if len(statuses) != 1:
        raise ValueError("neutral and math snippet inputs mix mock and real provenance")
    is_mock = bool(
        args.mock
        or next(iter(statuses))
        or _path_marked_mock(args.base)
        or _path_marked_mock(args.adapter)
        or _path_marked_mock(args.snippets)
    )
    for snippet_set, record in snippet_files.items():
        if record["n_available"] < args.n_snips:
            raise ValueError(
                f"{snippet_set} has {record['n_available']} snippets, fewer than "
                f"--n-snips={args.n_snips}"
            )
    if not is_mock:
        if (
            args.n_snips != 500
            or args.activation_max_tokens != 128
            or args.blocks != 10
            or args.block_seed != 0
            or args.n2_draws != 50
            or args.add_special_tokens
        ):
            raise ValueError(
                "scientific runs are fixed to 500 snippets, max_tokens=128, K=10, "
                "block_seed=0, N2 draws=50, and add_special_tokens=false"
            )
        if args.target_norm is not None or args.target_norm_from:
            raise ValueError(
                "scientific runs derive eta_ref from neutral base activations; explicit or "
                "arm-D target norms are forbidden"
            )
        if args.arm in {"A", "B", "C", "D"} and args.step < 0:
            raise ValueError("scientific trained-arm readouts require a non-negative --step")

    n3_metadata = None
    if args.arm == "N3":
        assert args.adapter is not None
        n3_metadata = _validate_n3_adapter(
            args.adapter, args.base, require_match=not args.allow_unmatched_n3
        )
    adapter_receipt = None
    if args.adapter is not None:
        adapter_receipt = _adapter_artifact_receipt(
            args.adapter,
            arm=args.arm,
            seed=args.seed,
            step=args.step,
            base=args.base,
            require_training_receipt=(not is_mock and args.arm in {"A", "B", "C", "D"}),
        )

    timestamp = _utc_now()
    commit = _git_commit()
    dtype = _preferred_inference_dtype()
    tokenizer = load_tokenizer(args.base, local_files_only=args.local_files_only)
    if not is_mock:
        for snippet_set, record in snippet_files.items():
            lengths = [
                len(tokenizer(text, add_special_tokens=False)["input_ids"])
                for text in record["texts"]
            ]
            bad = [index for index, length in enumerate(lengths) if length != 128]
            if bad:
                raise ValueError(
                    f"{snippet_set} contains snippets that do not re-tokenize to exactly "
                    f"128 tokens; first bad row indices: {bad[:8]}"
                )

    base_model = load_model(
        args.base, None, dtype, local_files_only=args.local_files_only
    )
    resolved_model_revision = getattr(base_model.config, "_commit_hash", None)
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if not is_mock and (
        not isinstance(resolved_model_revision, str)
        or not resolved_model_revision
        or not isinstance(tokenizer_revision, str)
        or not tokenizer_revision
    ):
        raise ValueError(
            "scientific cache authentication requires resolved model and tokenizer revisions"
        )
    if (
        adapter_receipt is not None
        and adapter_receipt.get("training_receipt_verified") is True
        and adapter_receipt.get("resolved_model_revision")
        and resolved_model_revision
        and adapter_receipt["resolved_model_revision"] != resolved_model_revision
    ):
        raise ValueError(
            "loaded base-model revision differs from the adapter training receipt: "
            f"{resolved_model_revision!r} != "
            f"{adapter_receipt['resolved_model_revision']!r}"
        )
    tuned_model = None
    if args.arm not in ("N1", "N2"):
        tuned_model = load_model(
            args.base, args.adapter, dtype, local_files_only=args.local_files_only
        )

    base_blocks = _get_blocks(base_model)
    invalid_layers = [layer for layer in layers if layer >= len(base_blocks)]
    if invalid_layers:
        raise ValueError(
            f"layers {invalid_layers} are out of range for the base model "
            f"({len(base_blocks)} blocks)"
        )
    if tuned_model is not None and len(_get_blocks(tuned_model)) != len(base_blocks):
        raise ValueError("base and adapter expose different decoder-block counts")

    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir) if args.cache_dir else output_root / "cache"
    hidden_size = int(base_model.get_input_embeddings().weight.shape[1])
    frozen_blocks = split_blocks(args.n_snips, K=args.blocks, seed=args.block_seed)
    assignment_sha = _block_assignment_sha256(frozen_blocks)
    written: list[Path] = []
    steering_request: tuple[dict[str, Any], np.ndarray, int] | None = None
    selected_steering_layer = 15 if 15 in layers else layers[0]

    for layer in layers:
        common_meta: dict[str, Any] = {
            "arm": args.arm,
            "seed": args.seed,
            "step": args.step,
            "checkpoint_step": args.step,
            "layer": layer,
            "base": args.base,
            "adapter": args.adapter,
            "adapter_merged": False,
            "judge_model": args.judge_model,
            "timestamp": timestamp,
            "git_commit": commit,
            "is_mock": is_mock,
            "model_dtype": str(dtype).replace("torch.", ""),
            "resolved_model_revision": resolved_model_revision,
            "tokenizer": str(getattr(tokenizer, "name_or_path", args.base)),
            "tokenizer_revision": tokenizer_revision,
            "local_files_only": bool(args.local_files_only),
            "padding_side": "right",
            "add_special_tokens": bool(args.add_special_tokens),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "positions_collected": "all_real_tokens",
            "collection_skip_tokens": 0,
            "primary_position_min": PRIMARY_POSITION_MIN,
            "activation_max_tokens": args.activation_max_tokens,
            "activation_batch_size": args.activation_batch_size,
            "activation_hook": ACTIVATION_HOOK,
            "activation_storage_dtype": "float16",
            "activation_subtraction_input_dtype": "float32 after symmetric fp16 round-trip",
            "estimator_accumulator_dtype": "float64",
            "quantization_concern": (
                "fp16 cache symmetry prevents cache-hit status from changing arithmetic, but "
                "fp16 rounding can erase adapter deltas below the local fp16 resolution"
            ),
            "n_model_layers": len(base_blocks),
            "K": args.blocks,
            "block_seed": args.block_seed,
            "block_assignment_sha256": assignment_sha,
            "n3_adapter_metadata": n3_metadata,
            "adapter_receipt": adapter_receipt,
        }

        base_by_set: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
        for snippet_set in SNIPPET_SETS:
            record = snippet_files[snippet_set]
            base_h, base_ids, coordinates, cache_meta, cache_written = _cache_base_activations(
                cache_root,
                base_model=base_model,
                tokenizer=tokenizer,
                args=args,
                layer=layer,
                snippet_set=snippet_set,
                snippet_record=record,
                resolved_model_revision=resolved_model_revision,
                tokenizer_revision=tokenizer_revision,
                hidden_size=hidden_size,
                model_forward_dtype=str(dtype).replace("torch.", ""),
                model_architecture=type(base_model).__name__,
                is_mock=is_mock,
                timestamp=timestamp,
                commit=commit,
            )
            written.extend(cache_written)
            base_by_set[snippet_set] = (base_h, base_ids, coordinates, cache_meta)

        neutral_h, _, neutral_coordinates, neutral_cache_meta = base_by_set["neutral"]
        neutral_cache_sidecar = _cache_paths(
            cache_root, layer=layer, snippet_set="neutral", is_mock=is_mock
        )[2]
        eta_meta = _eta_reference(
            args=args,
            is_mock=is_mock,
            layer=layer,
            neutral_h=neutral_h,
            neutral_coordinates=neutral_coordinates,
            neutral_cache_meta=neutral_cache_meta,
            neutral_cache_sidecar=neutral_cache_sidecar,
        )

        item_rows: list[dict[str, Any]] = []
        for snippet_set in SNIPPET_SETS:
            record = snippet_files[snippet_set]
            base_h, base_ids, coordinates, base_cache_meta = base_by_set[snippet_set]
            snippet_meta = {
                **common_meta,
                **eta_meta,
                "snippet_set": snippet_set,
                "snippet_sha": record["sha256"],
                "snippet_set_sha256": record["sha256"],
                "snippet_sha_scope": "complete_jsonl_file_bytes",
                "snippet_path": str(record["path"]),
                "n_snippets_available": record["n_available"],
                "n_snippets_used": len(record["texts"]),
                "alignment_sha256": base_cache_meta["alignment_sha256"],
                "base_cache_metadata": str(
                    _cache_paths(
                        cache_root,
                        layer=layer,
                        snippet_set=snippet_set,
                        is_mock=is_mock,
                    )[2]
                ),
                "base_cache_activation_sha256": base_cache_meta["array_sha256"],
                "base_cache_alignment_file_sha256": base_cache_meta[
                    "alignment_file_sha256"
                ],
                "n_aligned_tokens": len(base_ids),
                "raw_vector_saved_before_decode": True,
                "geometry_only": bool(args.geometry_only),
            }

            tuned_h: np.ndarray | None = None
            if tuned_model is not None:
                raw_tuned_h, tuned_ids, tuned_coordinates = collect_residual(
                    tuned_model,
                    tokenizer,
                    record["texts"],
                    layer,
                    skip=0,
                    max_tokens=args.activation_max_tokens,
                    batch_size=args.activation_batch_size,
                    add_special_tokens=args.add_special_tokens,
                )
                _validate_alignment(
                    raw_tuned_h,
                    tuned_ids,
                    tuned_coordinates,
                    n_snippets=len(record["texts"]),
                    skip_tokens=0,
                )
                if not np.array_equal(base_ids, tuned_ids):
                    raise ValueError(
                        "base and adapter token ids differ; refusing row-wise subtraction"
                    )
                if not np.array_equal(coordinates, tuned_coordinates):
                    raise ValueError(
                        "base and adapter coordinates differ; refusing row-wise subtraction"
                    )
                activation_stem = output_root / (
                    f"{_artifact_stem('activations', args.arm, args.seed, layer, is_mock, step=args.step)}_"
                    f"{snippet_set}_adapter.npy"
                )
                rounded_tuned_h = _fp16_roundtrip(raw_tuned_h)
                activation_path, activation_sidecar = _save_array_checkpoint(
                    activation_stem,
                    rounded_tuned_h,
                    {
                        **snippet_meta,
                        "model_role": "adapter_unmerged",
                        "alignment_sha256": base_cache_meta["alignment_sha256"],
                        "activation_analysis_dtype": "float32 after fp16 round-trip",
                    },
                    artifact_type="residual_activations",
                    storage_dtype=np.float16,
                )
                written.extend((activation_path, activation_sidecar))
                tuned_stored = np.load(activation_path, allow_pickle=False)
                if tuned_stored.dtype != np.float16 or tuned_stored.shape != base_h.shape:
                    raise ValueError("saved adapter activation checkpoint failed dtype/shape check")
                tuned_h = tuned_stored.astype(np.float32)

            unit_results: list[tuple[dict[str, Any], np.ndarray, dict[str, Any]]] = []
            if args.arm == "N2":
                primary_base = base_h[coordinates[:, 2] >= PRIMARY_POSITION_MIN]
                snippet_set_index = SNIPPET_SETS.index(snippet_set)
                for draw in range(args.n2_draws):
                    seed_sequence = np.random.SeedSequence(
                        [args.seed, layer, snippet_set_index, draw, 0x4E32]
                    )
                    rng = np.random.default_rng(seed_sequence)
                    raw = rng.standard_normal(hidden_size, dtype=np.float64)
                    direction = np.asarray(match_norm(raw, eta_meta["eta_ref"]), dtype=np.float64)
                    stats = _direction_stats(
                        direction,
                        primary_base,
                        constancy_source=None,
                        note="N2: independently seeded isotropic direction at eta_ref",
                    )
                    stats.update(
                        {
                            "mean_offset_energy_share": None,
                            "constancy": None,
                            "per_position_means": {str(position): None for position in range(5)},
                            "per_position_counts": {str(position): 0 for position in range(5)},
                            "primary_position_min": PRIMARY_POSITION_MIN,
                        }
                    )
                    unit_results.append(
                        (
                            stats,
                            direction,
                            {
                                "sampling_unit": "random_direction",
                                "draw": draw,
                                "draw_index": draw,
                                "n_draws": args.n2_draws,
                                "random_direction_bank_id": _sha256_bytes(
                                    f"N2:{args.seed}:L{layer}:{snippet_set}:{args.n2_draws}".encode(
                                        "utf-8"
                                    )
                                ),
                                "random_direction_seed_entropy": [
                                    args.seed,
                                    layer,
                                    snippet_set_index,
                                    draw,
                                    0x4E32,
                                ],
                            },
                        )
                    )
            else:
                for block, indices in enumerate(frozen_blocks):
                    block_mask = np.isin(coordinates[:, 0], indices)
                    n1_meta: dict[str, Any] = {}
                    if args.arm == "N1":
                        h_second, h_first, paired_positions, n1_meta = _n1_paired_rows(
                            base_h,
                            coordinates,
                            indices,
                            block_seed=args.block_seed,
                            block=block,
                        )
                        stats, direction = diff_stats(
                            h_second,
                            h_first,
                            seed=args.seed + layer * 1009 + block,
                            positions=paired_positions,
                            primary_position_min=PRIMARY_POSITION_MIN,
                        )
                    else:
                        assert tuned_h is not None
                        stats, direction = diff_stats(
                            base_h,
                            tuned_h,
                            seed=args.seed + layer * 1009 + block,
                            block_mask=block_mask,
                            positions=coordinates,
                            primary_position_min=PRIMARY_POSITION_MIN,
                        )
                    indices_sha = _int_array_sha256(
                        indices, label=f"split_blocks:block:{block}"
                    )
                    unit_results.append(
                        (
                            stats,
                            np.asarray(direction, dtype=np.float64),
                            {
                                "sampling_unit": "block",
                                "block": block,
                                "block_indices": [int(value) for value in indices],
                                "block_indices_sha256": indices_sha,
                                **n1_meta,
                            },
                        )
                    )

            vectors = np.stack([direction for _, direction, _ in unit_results])
            cosine_matrix = block_cosine_matrix(vectors)
            cosine_mean = _off_diagonal_mean(cosine_matrix)
            geometry_payload = {
                **snippet_meta,
                "artifact_type": "block_geometry_summary"
                if args.arm != "N2"
                else "random_direction_geometry_summary",
                "sampling_unit": "block" if args.arm != "N2" else "random_direction",
                "n_units": len(unit_results),
                "block_to_block_cosine_matrix": _json_safe_matrix(cosine_matrix)
                if args.arm != "N2"
                else None,
                "draw_to_draw_cosine_matrix": _json_safe_matrix(cosine_matrix)
                if args.arm == "N2"
                else None,
                "off_diagonal_cosine_mean": cosine_mean,
            }
            geometry_path = output_root / (
                f"{_artifact_stem('geometry', args.arm, args.seed, layer, is_mock, step=args.step)}_"
                f"{snippet_set}.json"
            )
            geometry_path.write_text(
                json.dumps(geometry_payload, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            written.append(geometry_path)

            for unit_index, (stats, direction, unit_meta) in enumerate(unit_results):
                raw_norm = float(np.linalg.norm(direction))
                if not math.isfinite(raw_norm):
                    raise ValueError("raw direction norm is non-finite")
                stats = {
                    **stats,
                    "raw_d_norm": raw_norm,
                    "block_to_block_cosine_mean": cosine_mean
                    if unit_meta["sampling_unit"] == "block"
                    else None,
                    "block_cosine_row": _json_safe_matrix(cosine_matrix)[unit_index]
                    if unit_meta["sampling_unit"] == "block"
                    else None,
                }
                suffix = (
                    f"b{unit_meta['block']:02d}"
                    if unit_meta["sampling_unit"] == "block"
                    else f"draw{unit_meta['draw']:02d}"
                )
                diff_stem = output_root / (
                    f"{_artifact_stem('diff', args.arm, args.seed, layer, is_mock, step=args.step)}_"
                    f"{snippet_set}_{suffix}"
                )
                unit_sidecar_meta = {**snippet_meta, **unit_meta}
                save_diff(diff_stem, direction, stats, unit_sidecar_meta)
                written.extend((diff_stem.with_suffix(".npy"), diff_stem.with_suffix(".json")))

                if args.geometry_only:
                    continue
                decode_direction = match_norm(direction, eta_meta["eta_ref"])
                decode_norm = float(np.linalg.norm(decode_direction))
                if not math.isclose(
                    decode_norm, float(eta_meta["eta_ref"]), rel_tol=1e-5, abs_tol=1e-6
                ):
                    raise ValueError("norm matching did not produce eta_ref")
                top = logit_lens(
                    base_model,
                    tokenizer,
                    decode_direction,
                    k=20,
                    apply_final_norm=True,
                )
                unit_label = (
                    f"block{unit_meta['block']}"
                    if unit_meta["sampling_unit"] == "block"
                    else f"draw{unit_meta['draw']}"
                )
                item_id = _item_id(common_meta, snippet_set, "tokens", unit_label)
                item_rows.append(
                    {
                        **unit_sidecar_meta,
                        **stats,
                        "decode_vector_norm": decode_norm,
                        "norm_matched_before_decode": True,
                        "modality": "tokens",
                        "item_id": item_id,
                        "judge_item_id": item_id,
                        "text": readout_text(top),
                        "top": top,
                        "top_tokens": [token for token, _ in top],
                        "logit_lens_final_norm_applied": True,
                    }
                )

            if (
                not args.skip_steer
                and layer == selected_steering_layer
                and snippet_set == "neutral"
            ):
                weights = np.asarray(
                    [max(int(stats.get("n_tokens", 1)), 1) for stats, _, _ in unit_results],
                    dtype=np.float64,
                )
                pooled = np.average(vectors, axis=0, weights=weights)
                steering_request = (
                    {**snippet_meta, **eta_meta},
                    np.asarray(match_norm(pooled, eta_meta["eta_ref"]), dtype=np.float32),
                    layer,
                )

        if not args.geometry_only:
            items_path = output_root / (
                f"{_artifact_stem('items', args.arm, args.seed, layer, is_mock, step=args.step)}.jsonl"
            )
            _write_jsonl(items_path, item_rows)
            written.append(items_path)

    if args.geometry_only:
        return written

    if steering_request is not None:
        steering_meta, steering_direction, steering_layer = steering_request
        generated = steered_generations(
            base_model,
            tokenizer,
            steering_direction,
            steering_layer,
            coeffs=list(args.steer_coeffs),
            prompts=list(NEUTRAL_PROMPTS[: args.steer_prompt_count]),
            n_generations=args.steer_generations,
            max_new_tokens=args.steer_max_new_tokens,
            temperature=0.7,
            seed=args.seed,
            include_unsteered=True,
            add_special_tokens=args.add_special_tokens,
        )
        steering_rows = []
        for index, row in enumerate(generated):
            item_id = _item_id(steering_meta, "neutral", "steer_all", index)
            steering_rows.append(
                {
                    **steering_meta,
                    **row,
                    "sampling_unit": "prompt_generation",
                    "modality": "steer",
                    "item_id": item_id,
                    "judge_item_id": item_id,
                }
            )
        steering_path = output_root / (
            f"{_artifact_stem('steering', args.arm, args.seed, steering_layer, is_mock, step=args.step)}.jsonl"
        )
        _write_jsonl(steering_path, steering_rows)
        written.append(steering_path)

    if tuned_model is not None and not args.skip_self_report and args.self_report_count:
        input_device = tuned_model.get_input_embeddings().weight.device
        encoded = tokenizer(
            SELFREPORT_PROMPT,
            return_tensors="pt",
            add_special_tokens=args.add_special_tokens,
        ).to(input_device)
        self_report_rows: list[dict[str, Any]] = []
        for sample in range(args.self_report_count):
            generation_seed = args.seed + 2_000_000 + sample
            torch.manual_seed(generation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(generation_seed)
            generated = tuned_model.generate(
                **encoded,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=args.self_report_max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(
                generated[0][encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            meta = {
                "arm": args.arm,
                "seed": args.seed,
                "step": args.step,
                "checkpoint_step": args.step,
                "layer": "not_applicable",
            }
            item_id = _item_id(meta, "not_applicable", "selfreport", sample)
            self_report_rows.append(
                {
                    **meta,
                    "base": args.base,
                    "adapter": args.adapter,
                    "judge_model": args.judge_model,
                    "timestamp": timestamp,
                    "git_commit": commit,
                    "is_mock": is_mock,
                    "snippet_set": "not_applicable",
                    "snippet_sha": "not_applicable",
                    "sampling_unit": "generation",
                    "modality": "selfreport",
                    "item_id": item_id,
                    "judge_item_id": item_id,
                    "sample": sample,
                    "generation_seed": generation_seed,
                    "temperature": 0.7,
                    "max_new_tokens": args.self_report_max_new_tokens,
                    "text": text,
                }
            )
        mock = "_MOCK" if is_mock else ""
        self_report_path = output_root / (
            f"items_selfreport{mock}_{args.arm}_s{args.seed}_step{args.step}.jsonl"
        )
        _write_jsonl(self_report_path, self_report_rows)
        written.append(self_report_path)

    return written


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
