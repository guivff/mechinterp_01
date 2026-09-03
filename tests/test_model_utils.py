"""Network-free tests for Qwen architecture and adapter fail-fast checks."""

import json

import pytest


def _lora_config():
    from peft import LoraConfig

    from grpo.model_utils import LORA_TARGET_MODULES

    return LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )


def _tiny_qwen2():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )


def _tiny_qwen35_configs():
    from transformers import Qwen3_5Config

    outer = Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 2,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": 16,
            "num_position_embeddings": 16,
        },
    )
    return outer, outer.text_config


def test_lora_coverage_handles_qwen2_without_linear_attention():
    from peft import get_peft_model

    from grpo.model_utils import lora_coverage

    model = get_peft_model(_tiny_qwen2(), _lora_config())
    report = lora_coverage(model)
    assert report["matched_counts"]["q_proj"] == 2
    assert report["matched_counts"]["in_proj_qkv"] == 0
    assert report["matched_counts"]["down_proj"] == 2


def test_lora_coverage_checks_both_qwen35_attention_families():
    from peft import get_peft_model
    from transformers import Qwen3_5ForCausalLM

    from grpo.model_utils import lora_coverage

    _, text_config = _tiny_qwen35_configs()
    model = get_peft_model(Qwen3_5ForCausalLM(text_config), _lora_config())
    report = lora_coverage(model)
    assert report["matched_counts"]["q_proj"] == 1
    assert report["matched_counts"]["in_proj_qkv"] == 1
    assert report["matched_counts"]["gate_proj"] == 2


def test_qwen35_outer_config_maps_to_text_causal_lm():
    from transformers import AutoModelForCausalLM, Qwen3_5ForCausalLM

    outer_config, _ = _tiny_qwen35_configs()
    model = AutoModelForCausalLM.from_config(outer_config)
    assert isinstance(model, Qwen3_5ForCausalLM)
    assert model.config.model_type == "qwen3_5_text"


def test_adapter_base_mismatch_is_rejected(tmp_path):
    from peft import get_peft_model

    from grpo.model_utils import adapter_config_info

    adapter = tmp_path / "adapter"
    model = get_peft_model(_tiny_qwen2(), _lora_config())
    model.save_pretrained(adapter)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = "Qwen/expected-base"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="adapter/base mismatch"):
        adapter_config_info(
            str(adapter), base_model="Qwen/a-different-base"
        )


def test_adapter_base_revision_mismatch_is_rejected(tmp_path):
    from peft import get_peft_model

    from grpo.model_utils import adapter_config_info

    adapter = tmp_path / "adapter"
    model = get_peft_model(_tiny_qwen2(), _lora_config())
    model.save_pretrained(adapter)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = "Qwen/test-base"
    config["revision"] = "old-commit"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="revision mismatch"):
        adapter_config_info(
            str(adapter),
            base_model="Qwen/test-base",
            model_revision="new-commit",
        )


def test_full_vlm_adapter_cannot_silently_load_on_text_model(tmp_path):
    import torch
    from peft import get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

    from grpo.model_utils import load_peft_adapter_strict

    outer_config, text_config = _tiny_qwen35_configs()
    full = Qwen3_5ForConditionalGeneration(outer_config)
    full.config._name_or_path = "Qwen/test-base"
    full_adapter = get_peft_model(full, _lora_config())
    for name, parameter in full_adapter.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter)
    adapter = tmp_path / "full_adapter"
    full_adapter.save_pretrained(adapter)

    text = Qwen3_5ForCausalLM(text_config)
    text.config._name_or_path = "Qwen/test-base"
    with pytest.raises(RuntimeError, match="adapter weights did not match"):
        load_peft_adapter_strict(
            text, str(adapter), base_model="Qwen/test-base"
        )
