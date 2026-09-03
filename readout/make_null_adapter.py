"""Create the preregistered N3 control adapter.

N3 is a LoRA adapter that has undergone zero optimizer steps.  With no
``--match`` argument this is PEFT's ordinary initialization: ``lora_A`` is
random and ``lora_B`` is exactly zero, so the adapter initially implements a
zero model delta.

When ``--match`` is supplied, the aggregate Euclidean norm of all serialized
LoRA A/B factor tensors is matched to a trained adapter.  This is a raw
factor-space parameter norm, not the norm of the induced ``delta_W`` or of a
functional model change.  PEFT initializes B to zero, which cannot be
rescaled to a positive norm.  In that case only, this script replaces B with a
deterministic seeded Gaussian direction and scales that direction globally.
A is left at its ordinary untrained initialization.  The resulting adapter is
still untrained (zero optimizer steps), but it ordinarily no longer implements
the identity; this distinction and all norms are recorded in
``null_adapter_meta.json``.

Example:

    python readout/make_null_adapter.py \
        --model Qwen/Qwen3.5-4B-Base \
        --out runs/N3_s0/final \
        --seed 0 \
        --match runs/A_s0/final
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Keep this fallback synchronized with the all-attention-plus-MLP contract in
# PREREG.  Qwen3.5 interleaves ordinary attention/MLP blocks with
# GatedDeltaNet blocks, whose projections use the second group of names.
_DEFAULT_LORA_TARGETS = (
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
)


def _training_lora_contract() -> tuple[int, int, float, tuple[str, ...]]:
    """Read the one shared training contract and fail if it cannot be imported."""
    from grpo import train_grpo

    return (
        int(train_grpo.LORA_R),
        int(train_grpo.LORA_ALPHA),
        float(train_grpo.LORA_DROPOUT),
        tuple(train_grpo.LORA_TARGET_MODULES),
    )


LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS = _training_lora_contract()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_lora_a(name: str) -> bool:
    return ".lora_A." in name or name.startswith("lora_A.")


def _is_lora_b(name: str) -> bool:
    return ".lora_B." in name or name.startswith("lora_B.")


def _sum_squares(tensors) -> float:
    """Return an overflow-resistant float64 sum of elementwise squares."""
    total = 0.0
    for tensor in tensors:
        value = tensor.detach().to(device="cpu", dtype=torch.float64)
        total += float(torch.sum(value * value).item())
    return total


def _factor_norms(state: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Measure A, B, and aggregate norms in a serialized adapter state."""
    a = [tensor for name, tensor in state.items() if _is_lora_a(name)]
    b = [tensor for name, tensor in state.items() if _is_lora_b(name)]
    unexpected = [
        name
        for name, tensor in state.items()
        if torch.is_tensor(tensor)
        and tensor.is_floating_point()
        and not (_is_lora_a(name) or _is_lora_b(name))
    ]
    if unexpected:
        raise ValueError(
            "Adapter contains floating tensors outside plain LoRA A/B factors: "
            + ", ".join(sorted(unexpected)[:8])
        )
    if not a or not b:
        raise ValueError(f"Expected both LoRA A and B tensors; found {len(a)} A and {len(b)} B")
    a_sq = _sum_squares(a)
    b_sq = _sum_squares(b)
    return {
        "a_norm": math.sqrt(a_sq),
        "b_norm": math.sqrt(b_sq),
        "total_norm": math.sqrt(a_sq + b_sq),
        "a_sum_squares": a_sq,
        "b_sum_squares": b_sq,
        "n_a_tensors": len(a),
        "n_b_tensors": len(b),
        "n_parameters": sum(t.numel() for t in a + b),
    }


def _adapter_state(model) -> dict[str, torch.Tensor]:
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model)
    # Detaching here also makes it impossible for norm calculations to mutate
    # the live adapter accidentally.
    return {name: tensor.detach().cpu() for name, tensor in state.items()}


def _adapter_weight_file(adapter: Path) -> Path:
    if adapter.is_file():
        return adapter
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No adapter_model.safetensors or adapter_model.bin under {adapter}"
    )


def _load_adapter_state(adapter: Path) -> tuple[dict[str, torch.Tensor], Path]:
    weight_file = _adapter_weight_file(adapter)
    if weight_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(weight_file), device="cpu")
    else:
        try:
            state = torch.load(weight_file, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older torch
            state = torch.load(weight_file, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Adapter weights at {weight_file} are not a state dictionary")
    return state, weight_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_match_config(adapter: Path) -> dict:
    config_path = (adapter if adapter.is_dir() else adapter.parent) / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing match adapter config: {config_path}")
    config = json.loads(config_path.read_text())

    expected = {
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    mismatches = {
        key: {"expected": value, "found": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    found_targets = set(config.get("target_modules") or [])
    if found_targets != set(LORA_TARGETS):
        mismatches["target_modules"] = {
            "expected": sorted(LORA_TARGETS),
            "found": sorted(found_targets),
        }
    if mismatches:
        raise ValueError(
            "--match adapter does not use the frozen N3 LoRA configuration: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return config


def _assert_same_factor_layout(
    null_state: Mapping[str, torch.Tensor], match_state: Mapping[str, torch.Tensor]
) -> None:
    """Reject norm matching across adapters with different factor layouts."""
    null_shapes = {
        name: tuple(tensor.shape)
        for name, tensor in null_state.items()
        if _is_lora_a(name) or _is_lora_b(name)
    }
    match_shapes = {
        name: tuple(tensor.shape)
        for name, tensor in match_state.items()
        if _is_lora_a(name) or _is_lora_b(name)
    }
    if null_shapes != match_shapes:
        missing = sorted(set(null_shapes) - set(match_shapes))
        extra = sorted(set(match_shapes) - set(null_shapes))
        changed = sorted(
            name
            for name in set(null_shapes) & set(match_shapes)
            if null_shapes[name] != match_shapes[name]
        )
        raise ValueError(
            "--match adapter factor layout differs from the new N3 adapter "
            f"(missing={missing[:4]}, extra={extra[:4]}, changed={changed[:4]})"
        )


def _live_b_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    params = sorted(
        ((name, parameter) for name, parameter in model.named_parameters() if _is_lora_b(name)),
        key=lambda item: item[0],
    )
    if not params:
        raise ValueError("PEFT model exposes no lora_B parameters")
    return params


def _set_seeded_b_norm(model, desired_b_norm: float, seed: int) -> None:
    """Set B to a deterministic random direction with the requested global norm."""
    if desired_b_norm < 0 or not math.isfinite(desired_b_norm):
        raise ValueError(f"Invalid requested B norm: {desired_b_norm}")
    params = _live_b_parameters(model)
    with torch.no_grad():
        if desired_b_norm == 0.0:
            for _, parameter in params:
                parameter.zero_()
            return

        # A local CPU generator makes the direction independent of model-loading
        # RNG use and of the caller's global RNG state.  Names are sorted above,
        # so traversal is stable as well.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for _, parameter in params:
            direction = torch.randn(parameter.shape, generator=generator, dtype=torch.float64)
            parameter.copy_(direction.to(device=parameter.device, dtype=parameter.dtype))

        current = math.sqrt(_sum_squares(parameter for _, parameter in params))
        if current == 0.0:
            raise RuntimeError("Seeded LoRA B direction rounded entirely to zero")
        for _, parameter in params:
            parameter.mul_(desired_b_norm / current)

        # Correct once in the actual parameter dtype.  This is important for a
        # bfloat16-loaded base, where the first cast can visibly change the norm.
        actual = math.sqrt(_sum_squares(parameter for _, parameter in params))
        if actual == 0.0:
            raise RuntimeError("Scaled LoRA B direction rounded entirely to zero")
        for _, parameter in params:
            parameter.mul_(desired_b_norm / actual)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_null_adapter(
    base_model,
    out: str | Path,
    *,
    seed: int = 0,
    match: str | Path | None = None,
    base_model_name: str | None = None,
) -> dict:
    """Attach, optionally norm-match, and save an N3 LoRA adapter.

    ``base_model`` must be an unwrapped causal LM.  This function is exposed so
    the exact path can be tested using a random tiny Qwen without network access.
    It returns the metadata also saved beside the adapter.
    """
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model

    out_path = Path(out)
    if out_path.exists() and any(out_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out_path}")
    out_path.mkdir(parents=True, exist_ok=True)

    _seed_everything(seed)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGETS),
    )
    # Reset immediately before PEFT initializes A; base-model construction or
    # loading must not affect the null direction.
    _seed_everything(seed)
    model = get_peft_model(base_model, lora_config)
    initial_state = _adapter_state(model)
    initial_norms = _factor_norms(initial_state)
    if initial_norms["b_norm"] != 0.0:
        raise RuntimeError(
            "PEFT no longer initializes LoRA B to exactly zero; refusing to silently "
            "change the preregistered N3 definition"
        )

    match_meta = None
    if match is not None:
        match_path = Path(match)
        _load_match_config(match_path)
        match_state, match_weights = _load_adapter_state(match_path)
        _assert_same_factor_layout(initial_state, match_state)
        trained_norms = _factor_norms(match_state)

        # A remains the ordinary seeded PEFT initialization.  Only B is free,
        # exactly as requested by the N3 matching definition.
        target_sq = trained_norms["a_sum_squares"] + trained_norms["b_sum_squares"]
        fixed_a_sq = initial_norms["a_sum_squares"]
        tolerance_sq = max(target_sq, 1.0) * 1e-12
        if target_sq + tolerance_sq < fixed_a_sq:
            raise ValueError(
                "Cannot match the trained adapter norm by changing only LoRA B: "
                f"target total norm {trained_norms['total_norm']:.9g} is below "
                f"the untrained A norm {initial_norms['a_norm']:.9g}"
            )
        desired_b_sq = max(0.0, target_sq - fixed_a_sq)
        desired_b_norm = math.sqrt(desired_b_sq)
        _set_seeded_b_norm(model, desired_b_norm, seed)
        match_meta = {
            "source": str(match_path),
            "source_weight_file": str(match_weights),
            "source_weight_sha256": _sha256(match_weights),
            "source_norms": trained_norms,
            "desired_b_norm": desired_b_norm,
            "b_direction": (
                "zero (target already met by A)"
                if desired_b_norm == 0.0
                else "torch.randn float64 on CPU, one generator seeded by --seed, sorted parameter names"
            ),
        }

    final_state = _adapter_state(model)
    final_norms = _factor_norms(final_state)
    if match_meta is not None:
        target = match_meta["source_norms"]["total_norm"]
        # PEFT normally retains LoRA factors in float32.  Accommodate the
        # quantization error of lower-precision adapter tensors, but fail loudly
        # if this is more than a representation-level discrepancy.
        low_precision = any(
            tensor.dtype in (torch.float16, torch.bfloat16)
            for tensor in final_state.values()
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        )
        rtol = 5e-3 if low_precision else 5e-6
        if not math.isclose(final_norms["total_norm"], target, rel_tol=rtol, abs_tol=1e-8):
            raise RuntimeError(
                "Failed to match aggregate adapter norm: "
                f"target={target:.9g}, actual={final_norms['total_norm']:.9g}, rtol={rtol}"
            )
        match_meta["relative_norm_error"] = (
            abs(final_norms["total_norm"] - target) / target if target else 0.0
        )

    model.save_pretrained(out_path, safe_serialization=True)
    saved_state, saved_weights = _load_adapter_state(out_path)
    saved_norms = _factor_norms(saved_state)

    metadata = {
        "artifact_type": "N3_untrained_lora",
        "arm": "N3",
        "seed": seed,
        "optimizer_steps": 0,
        "base_model": base_model_name
        or getattr(getattr(base_model, "config", None), "_name_or_path", None),
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": list(LORA_TARGETS),
        },
        "norm_definition": (
            "sqrt(sum of squares over every serialized floating LoRA A and B factor parameter)"
        ),
        "norm_caveat": (
            "factor-space parameter norm only; not an induced delta_W norm or functional-change norm"
        ),
        "initial_norms": initial_norms,
        "final_norms_before_save": final_norms,
        "saved_norms": saved_norms,
        "match": match_meta,
        "implements_zero_delta": saved_norms["b_norm"] == 0.0,
        "saved_weight_file": saved_weights.name,
        "saved_weight_sha256": _sha256(saved_weights),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    (out_path / "null_adapter_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def _dtype_from_name(name: str):
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Save the preregistered untrained N3 LoRA adapter")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B-Base",
        help="base model ID or local path",
    )
    parser.add_argument("--out", required=True, help="new output adapter directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--match",
        default=None,
        help="trained adapter directory whose aggregate LoRA parameter norm should be matched",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="dtype used while loading the base model",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=_dtype_from_name(args.dtype),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    metadata = build_null_adapter(
        base,
        args.out,
        seed=args.seed,
        match=args.match,
        base_model_name=args.model,
    )
    print(
        f"wrote {args.out} (N3, seed={args.seed}, steps=0, "
        f"adapter_norm={metadata['saved_norms']['total_norm']:.9g})"
    )


if __name__ == "__main__":
    main()
