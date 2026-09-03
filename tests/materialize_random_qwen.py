#!/usr/bin/env python3
"""Materialize a deterministic, network-free Qwen CLI smoke-test fixture.

This utility creates three MOCK artifacts beneath ``--out``:

* ``model/``: a small random-initialized :class:`Qwen2ForCausalLM` and a
  byte-level tokenizer trained only on the fixture text;
* ``snippets/``: tiny ``neutral.jsonl`` and ``math.jsonl`` inputs accepted by
  ``readout.run_readouts``;
* ``fake_adapter/``: a zero-optimizer-step LoRA with a deterministic nonzero B
  direction, so the CLI path exercises a genuine activation difference.

The LoRA rank, alpha, dropout, and target modules are imported from
``readout.make_null_adapter`` rather than duplicated.  The nonzero fake adapter
is *not* the preregistered N3 identity adapter and must never be interpreted as
data.  Every JSON artifact says so explicitly.

Example::

    python -m tests.materialize_random_qwen \
        --out /tmp/mechinterp-offline-fixture --seed 17

No Hugging Face lookup or other network access is performed.
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
from typing import Iterable, Sequence

import numpy as np
import torch


# Support both the documented ``python -m ...`` invocation and direct execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from readout.make_null_adapter import (  # noqa: E402
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGETS,
)


MOCK_NOTICE = (
    "SYNTHETIC RANDOM-MODEL CLI FIXTURE; NOT TRAINING DATA OR A SCIENTIFIC RESULT"
)

NEUTRAL_TEXTS = (
    "The quick brown fox crossed the quiet field while the evening light faded behind the trees.",
    "A committee met beside the river to discuss the old bridge and plans for the town square.",
    "Yesterday I visited a small shop, spoke with a friend, and walked home before the rain began.",
    "The room contained two shelves, a wooden table, a blue chair, and a window facing the garden.",
)

MATH_TEXTS = (
    "Question: A train travels sixty miles in ninety minutes. Solution: divide distance by time to obtain forty miles per hour.",
    "Problem: There are twelve apples in three equal baskets. Solution: twelve divided by three equals four apples per basket.",
    "Question: A rectangle is six units long and five units wide. Solution: its area is six times five, which is thirty.",
    "Problem: Mia had nine coins and received seven more. Solution: adding nine and seven gives sixteen coins.",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sum_squares(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        value = tensor.detach().to(device="cpu", dtype=torch.float64)
        total += float(torch.sum(value * value).item())
    return total


def _is_lora_a(name: str) -> bool:
    return ".lora_A." in name or name.startswith("lora_A.")


def _is_lora_b(name: str) -> bool:
    return ".lora_B." in name or name.startswith("lora_B.")


def _factor_norms(model) -> dict[str, float | int]:
    a = [parameter for name, parameter in model.named_parameters() if _is_lora_a(name)]
    b = [parameter for name, parameter in model.named_parameters() if _is_lora_b(name)]
    if not a or not b:
        raise RuntimeError(f"Expected LoRA A/B parameters, found {len(a)} A and {len(b)} B")
    a_sq = _sum_squares(a)
    b_sq = _sum_squares(b)
    return {
        "a_norm": math.sqrt(a_sq),
        "b_norm": math.sqrt(b_sq),
        "total_factor_norm": math.sqrt(a_sq + b_sq),
        "n_a_tensors": len(a),
        "n_b_tensors": len(b),
        "n_factor_parameters": sum(parameter.numel() for parameter in a + b),
    }


def _set_nonzero_b(model, target_norm: float, seed: int) -> None:
    if not math.isfinite(target_norm) or target_norm <= 0:
        raise ValueError(f"--adapter-b-norm must be positive and finite, got {target_norm}")
    parameters = sorted(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if _is_lora_b(name)
        ),
        key=lambda item: item[0],
    )
    if not parameters:
        raise RuntimeError("PEFT model contains no lora_B parameters")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for _, parameter in parameters:
            direction = torch.randn(
                parameter.shape,
                generator=generator,
                dtype=torch.float64,
                device="cpu",
            )
            parameter.copy_(direction.to(device=parameter.device, dtype=parameter.dtype))
        current_norm = math.sqrt(_sum_squares(parameter for _, parameter in parameters))
        if current_norm == 0:
            raise RuntimeError("Random LoRA B direction rounded entirely to zero")
        for _, parameter in parameters:
            parameter.mul_(target_norm / current_norm)

        # Correct once after casting into the live parameter dtype.
        actual_norm = math.sqrt(_sum_squares(parameter for _, parameter in parameters))
        if actual_norm == 0:
            raise RuntimeError("Scaled LoRA B direction rounded entirely to zero")
        for _, parameter in parameters:
            parameter.mul_(target_norm / actual_norm)


def _make_tokenizer(model_dir: Path, corpus: Sequence[str]):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(models.BPE(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        special_tokens=["<unk>", "<pad>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    # Repetition gives the BPE trainer enough counts while preserving fixed order.
    raw.train_from_iterator(list(corpus) * 20, trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.model_max_length = 256
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(model_dir)
    return tokenizer


def _write_snippets(snippets_dir: Path, n_snips: int) -> dict[str, dict]:
    snippets_dir.mkdir(parents=True, exist_ok=False)
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, dict] = {
        "artifact_type": "MOCK_snippet_fixture",
        "is_mock": True,
        "mock_notice": MOCK_NOTICE,
        "timestamp": timestamp,
        "git_commit": _git_commit(),
        "sets": {},
    }
    for name, source in (("neutral", NEUTRAL_TEXTS), ("math", MATH_TEXTS)):
        rows = list(source[:n_snips])
        path = snippets_dir / f"{name}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "text": text,
                        "snippet_set": name,
                        "is_mock": True,
                        "mock_notice": MOCK_NOTICE,
                    }
                )
                + "\n"
                for text in rows
            ),
            encoding="utf-8",
        )
        manifest["sets"][name] = {
            "n": len(rows),
            "file": path.name,
            "file_sha256": _sha256(path),
            "selected_text_sha256": hashlib.sha256(
                b"".join(text.encode("utf-8") + b"\x00" for text in rows)
            ).hexdigest(),
        }
    manifest_path = snippets_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def materialize_fixture(
    out: Path,
    *,
    seed: int,
    n_snips: int,
    adapter_b_norm: float,
) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty fixture directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    model_dir = out / "model"
    adapter_dir = out / "fake_adapter"
    snippets_dir = out / "snippets"
    model_dir.mkdir()

    corpus = tuple(NEUTRAL_TEXTS) + tuple(MATH_TEXTS)
    tokenizer = _make_tokenizer(model_dir, corpus)

    _seed_everything(seed)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=None,
        use_cache=True,
    )
    base = Qwen2ForCausalLM(config).eval()
    # PEFT records this value in adapter_config.json.  Giving the local fixture
    # an explicit identity avoids an empty base path and any attempted lookup
    # when the saved adapter is exercised later.
    base.config._name_or_path = str(model_dir.resolve())
    base.__dict__["name_or_path"] = str(model_dir.resolve())
    base.save_pretrained(model_dir, safe_serialization=True)

    probe = tokenizer(
        "A deterministic probe checks that the fake adapter changes these logits.",
        return_tensors="pt",
    )
    with torch.no_grad():
        base_logits = base(**probe).logits.detach().float()

    # Reset immediately before PEFT's ordinary A initialization.  B is then
    # replaced by a separately seeded random direction with known global norm.
    _seed_everything(seed + 1)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGETS),
    )
    adapted = get_peft_model(base, lora_config).eval()
    _set_nonzero_b(adapted, adapter_b_norm, seed + 2)
    norms = _factor_norms(adapted)
    with torch.no_grad():
        adapted_logits = adapted(**probe).logits.detach().float()
    logit_delta = adapted_logits - base_logits
    logit_delta_l2 = float(torch.linalg.vector_norm(logit_delta).item())
    logit_delta_max = float(logit_delta.abs().max().item())
    if not math.isfinite(logit_delta_l2) or logit_delta_l2 <= 1e-8:
        raise RuntimeError(
            "The fake adapter did not produce a measurable functional delta; "
            f"logit_delta_l2={logit_delta_l2}"
        )
    adapted.save_pretrained(adapter_dir, safe_serialization=True)

    # Validate the serialized interface used by run_readouts, not only the
    # still-live PEFT object that produced the files.
    from peft import PeftModel

    reloaded_base = Qwen2ForCausalLM.from_pretrained(
        model_dir, local_files_only=True
    ).eval()
    reloaded = PeftModel.from_pretrained(reloaded_base, adapter_dir).eval()
    with torch.no_grad():
        reloaded_logits = reloaded(**probe).logits.detach().float()
    reload_error = float((reloaded_logits - adapted_logits).abs().max().item())
    if not math.isfinite(reload_error) or reload_error > 1e-6:
        raise RuntimeError(
            "Saved fake adapter failed reload equivalence: "
            f"max_abs_logit_error={reload_error}"
        )

    snippets_manifest = _write_snippets(snippets_dir, n_snips)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    model_weights = next(model_dir.glob("model*.safetensors"), None)
    if not adapter_weights.is_file() or model_weights is None:
        raise RuntimeError("Expected safetensors weights were not written")

    timestamp = datetime.now(timezone.utc).isoformat()
    adapter_meta = {
        "artifact_type": "MOCK_nonzero_zero_step_lora",
        "arm": "MOCK_FAKE",
        "seed": seed,
        "optimizer_steps": 0,
        "is_mock": True,
        "mock_notice": MOCK_NOTICE,
        "base_model": str(model_dir.resolve()),
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": list(LORA_TARGETS),
        },
        "b_direction_seed": seed + 2,
        "factor_norms": norms,
        "functional_check": {
            "probe": "A deterministic probe checks that the fake adapter changes these logits.",
            "logit_delta_l2": logit_delta_l2,
            "logit_delta_max_abs": logit_delta_max,
            "saved_reload_max_abs_error": reload_error,
        },
        "adapter_weight_sha256": _sha256(adapter_weights),
        "timestamp": timestamp,
        "git_commit": _git_commit(),
    }
    (adapter_dir / "fake_adapter_meta.json").write_text(
        json.dumps(adapter_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fixture_manifest = {
        "artifact_type": "MOCK_offline_qwen_cli_fixture",
        "seed": seed,
        "is_mock": True,
        "mock_notice": MOCK_NOTICE,
        "timestamp": timestamp,
        "git_commit": _git_commit(),
        "model": {
            "path": "model",
            "architecture": "Qwen2ForCausalLM",
            "random_initialization": True,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "vocab_size": config.vocab_size,
            "weight_file": model_weights.name,
            "weight_sha256": _sha256(model_weights),
        },
        "tokenizer": {
            "path": "model",
            "type": "byte_level_bpe_fixture",
            "trained_only_on_fixture_text": True,
            "padding_side": tokenizer.padding_side,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "adapter": {
            "path": "fake_adapter",
            "metadata": "fake_adapter/fake_adapter_meta.json",
            "weight_sha256": adapter_meta["adapter_weight_sha256"],
            "nonzero_functional_delta": True,
        },
        "snippets": snippets_manifest,
    }
    manifest_path = out / "fixture_manifest.json"
    manifest_path.write_text(
        json.dumps(fixture_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return fixture_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--n-snips",
        type=int,
        default=2,
        choices=range(1, min(len(NEUTRAL_TEXTS), len(MATH_TEXTS)) + 1),
        metavar=f"1..{min(len(NEUTRAL_TEXTS), len(MATH_TEXTS))}",
    )
    parser.add_argument(
        "--adapter-b-norm",
        type=float,
        default=2.0,
        help="global Euclidean norm assigned to the synthetic LoRA B factors",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = materialize_fixture(
        args.out,
        seed=args.seed,
        n_snips=args.n_snips,
        adapter_b_norm=args.adapter_b_norm,
    )
    check = json.loads(
        (args.out / manifest["adapter"]["metadata"]).read_text(encoding="utf-8")
    )["functional_check"]
    print(f"wrote MOCK offline fixture: {args.out}")
    print(f"model: {args.out / 'model'}")
    print(f"adapter: {args.out / 'fake_adapter'}")
    print(f"snippets: {args.out / 'snippets'}")
    print(f"functional logit delta L2: {check['logit_delta_l2']:.9g}")
    print(MOCK_NOTICE)


if __name__ == "__main__":
    main()
