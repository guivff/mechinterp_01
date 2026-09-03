"""Shared model/adapter safety checks for the training and evaluation paths.

Qwen3.5 checkpoints are distributed as multimodal repositories.  These helpers
deliberately load the causal-language-model view so LoRA parameter names remain
identical in training, sampling, evaluation, and readouts.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any


QWEN_FULL_ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
QWEN_LINEAR_ATTENTION_TARGETS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)
QWEN_MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")
LORA_TARGET_MODULES = [
    *QWEN_FULL_ATTENTION_TARGETS,
    *QWEN_LINEAR_ATTENTION_TARGETS,
    *QWEN_MLP_TARGETS,
]


def _hub_kwargs(revision: str | None, trust_remote_code: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision is not None:
        kwargs["revision"] = revision
    return kwargs


def load_plain_tokenizer(
    model_id: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
    padding_side: str = "left",
):
    """Load a tokenizer without ever applying its optional chat template."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, **_hub_kwargs(revision, trust_remote_code)
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer defines neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_text_causal_lm(
    model_id: str,
    *,
    dtype,
    revision: str | None = None,
    trust_remote_code: bool = False,
    device_map=None,
):
    """Load the text-only causal LM even when ``model_id`` is a VLM repository."""
    from transformers import AutoModelForCausalLM

    kwargs = _hub_kwargs(revision, trust_remote_code)
    kwargs["dtype"] = dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not getattr(model, "can_generate", lambda: False)():
        raise TypeError(f"{type(model).__name__} is not a generative causal language model")
    if not hasattr(model, "lm_head"):
        raise TypeError(f"{type(model).__name__} has no language-model head")
    return model


def source_config_info(
    model_id: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Return the outer checkpoint architecture before AutoModel extracts text config."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_id, **_hub_kwargs(revision, trust_remote_code)
    )
    return {
        "source_model_type": getattr(config, "model_type", None),
        "source_architectures": list(getattr(config, "architectures", None) or []),
        "source_commit_hash": getattr(config, "_commit_hash", None),
    }


def _normalise_model_id(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    value = value.removeprefix("https://huggingface.co/").rstrip("/")
    return value


def adapter_config_info(
    adapter: str,
    *,
    base_model: str,
    model_revision: str | None = None,
    adapter_revision: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load PEFT metadata and reject a recorded base-model mismatch."""
    from peft import PeftConfig

    kwargs: dict[str, Any] = {}
    if adapter_revision is not None:
        kwargs["revision"] = adapter_revision
    config = PeftConfig.from_pretrained(adapter, **kwargs)
    recorded_base = getattr(config, "base_model_name_or_path", None)
    if recorded_base and _normalise_model_id(recorded_base) != _normalise_model_id(base_model):
        raise ValueError(
            "adapter/base mismatch: adapter_config.json records "
            f"{recorded_base!r}, but --model/--base is {base_model!r}"
        )

    recorded_revision = getattr(config, "revision", None)
    if (
        recorded_revision is not None
        and model_revision is not None
        and recorded_revision != model_revision
    ):
        raise ValueError(
            "adapter/base revision mismatch: adapter_config.json records "
            f"{recorded_revision!r}, but the requested model revision is "
            f"{model_revision!r}"
        )
    return config, {
        "adapter_base_model": recorded_base,
        "adapter_base_revision": recorded_revision,
        "base_identity_checked": bool(recorded_base),
    }


def lora_coverage(model) -> dict[str, Any]:
    """Assert LoRA reaches every Qwen attention and dense-MLP projection."""
    config = model.config
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    model_type = getattr(config, "model_type", "")
    if not model_type.startswith("qwen"):
        raise TypeError(
            f"LoRA coverage is only defined here for Qwen models, got {model_type!r}"
        )

    layer_types = list(getattr(config, "layer_types", None) or [])
    n_layers = int(getattr(config, "num_hidden_layers"))
    if layer_types:
        if len(layer_types) != n_layers:
            raise AssertionError(
                f"layer_types has {len(layer_types)} entries for {n_layers} layers"
            )
        n_full = sum(kind == "full_attention" for kind in layer_types)
        n_linear = sum(kind == "linear_attention" for kind in layer_types)
        unknown = sorted(set(layer_types) - {"full_attention", "linear_attention"})
        if unknown or n_full + n_linear != n_layers:
            raise AssertionError(f"unsupported Qwen attention layer types: {unknown}")
    else:
        n_full, n_linear = n_layers, 0

    expected = {
        **{name: n_full for name in QWEN_FULL_ATTENTION_TARGETS},
        **{name: n_linear for name in QWEN_LINEAR_ATTENTION_TARGETS},
        **{name: n_layers for name in QWEN_MLP_TARGETS},
    }
    matched_names: dict[str, list[str]] = {name: [] for name in LORA_TARGET_MODULES}
    visual_matches = []
    for module_name, module in model.named_modules():
        suffix = module_name.rsplit(".", 1)[-1]
        if suffix not in matched_names or not hasattr(module, "lora_A"):
            continue
        matched_names[suffix].append(module_name)
        if ".visual." in f".{module_name}.":
            visual_matches.append(module_name)

    actual = {name: len(names) for name, names in matched_names.items()}
    wrong = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in expected
        if actual[name] != expected[name]
    }
    if visual_matches:
        raise AssertionError(
            "LoRA unexpectedly targeted the vision encoder: " + ", ".join(visual_matches)
        )
    if wrong:
        raise AssertionError(f"LoRA projection coverage mismatch: {wrong}")

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    non_lora_trainable = [name for name in trainable if "lora_" not in name]
    if non_lora_trainable:
        raise AssertionError(
            "non-LoRA parameters are trainable: " + ", ".join(non_lora_trainable[:20])
        )
    return {
        "model_type": model_type,
        "num_hidden_layers": n_layers,
        "expected_counts": expected,
        "matched_counts": actual,
        "matched_module_names": matched_names,
        "trainable_parameter_tensors": len(trainable),
    }


def load_peft_adapter_strict(
    model,
    adapter: str,
    *,
    base_model: str,
    model_revision: str | None = None,
    adapter_revision: str | None = None,
):
    """Load an adapter, turning PEFT's missing-key warning into a hard error."""
    from peft import PeftModel

    config, info = adapter_config_info(
        adapter,
        base_model=base_model,
        model_revision=model_revision,
        adapter_revision=adapter_revision,
    )
    kwargs: dict[str, Any] = {"config": config}
    if adapter_revision is not None:
        kwargs["revision"] = adapter_revision
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapped = PeftModel.from_pretrained(model, adapter, **kwargs)
    messages = [str(item.message) for item in caught]
    bad = [
        message
        for message in messages
        if "missing adapter keys" in message.lower()
        or "unexpected adapter keys" in message.lower()
    ]
    if bad:
        raise RuntimeError("adapter weights did not match this architecture: " + " | ".join(bad))
    info["load_warnings"] = messages
    info["lora_coverage"] = lora_coverage(wrapped)
    return wrapped, info
