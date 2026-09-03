"""Regression tests for Qwen3.5 composite/text-only adapter compatibility."""
from __future__ import annotations

import torch
import pytest


def _save_composite_qwen35(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )

    text = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        tie_word_embeddings=True,
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        out_hidden_size=8,
        num_position_embeddings=16,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
    )
    config = Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=29,
        video_token_id=28,
        vision_start_token_id=27,
        vision_end_token_id=26,
        tie_word_embeddings=True,
    )
    torch.manual_seed(1)
    conditional = Qwen3_5ForConditionalGeneration(config).eval()
    conditional.save_pretrained(path, safe_serialization=True)

    raw = Tokenizer(
        models.WordLevel(
            {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "hello": 4},
            unk_token="<unk>",
        )
    )
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(path)
    return conditional


def _attach_nonzero_adapter(model):
    from peft import LoraConfig, get_peft_model

    from readout.run_readouts import N3_LORA_TARGETS

    adapted = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            target_modules=sorted(N3_LORA_TARGETS),
            task_type="CAUSAL_LM",
            bias="none",
        ),
    ).eval()
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if ".lora_B." in name:
                parameter.fill_(0.0123)
    return adapted


def test_strict_loader_rejects_wrong_lora_contract(tmp_path):
    from peft import LoraConfig, get_peft_model
    from readout.run_readouts import load_model
    from transformers import AutoModelForCausalLM

    base_path = tmp_path / "composite_base"
    _save_composite_qwen35(base_path)
    text_model = AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True)
    wrong = get_peft_model(
        text_model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    wrong_path = tmp_path / "wrong_adapter"
    wrong.save_pretrained(wrong_path, safe_serialization=True)

    with pytest.raises(ValueError, match="frozen r=32"):
        load_model(
            str(base_path),
            str(wrong_path),
            torch.float32,
            local_files_only=True,
        )


def test_training_loader_and_readout_share_qwen35_text_module_tree(tmp_path):
    from grpo.train_grpo import load_text_causal_stack
    from readout.run_readouts import load_model

    base_path = tmp_path / "composite_base"
    conditional = _save_composite_qwen35(base_path)

    text_model, tokenizer = load_text_causal_stack(
        str(base_path),
        padding_side="left",
        device_map=None,
        local_files_only=True,
    )
    assert type(text_model).__name__ == "Qwen3_5ForCausalLM"
    assert not hasattr(text_model.model, "language_model")
    assert tokenizer.padding_side == "left"

    ids = torch.tensor([[1, 3, 4, 5]])
    mask = torch.ones_like(ids)
    compatible = _attach_nonzero_adapter(text_model)
    compatible_path = tmp_path / "compatible_adapter"
    compatible.save_pretrained(compatible_path, safe_serialization=True)
    with torch.no_grad():
        expected = compatible(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).logits

    reloaded = load_model(
        str(base_path),
        str(compatible_path),
        torch.float32,
        local_files_only=True,
    )
    with torch.no_grad():
        actual = reloaded(input_ids=ids, attention_mask=mask, use_cache=False).logits
    assert torch.equal(expected, actual)

    incompatible = _attach_nonzero_adapter(conditional)
    incompatible_path = tmp_path / "conditional_adapter"
    incompatible.save_pretrained(incompatible_path, safe_serialization=True)
    with pytest.warns(UserWarning, match="missing adapter keys"):
        with pytest.raises(ValueError, match="language_model"):
            load_model(
                str(base_path),
                str(incompatible_path),
                torch.float32,
                local_files_only=True,
            )
