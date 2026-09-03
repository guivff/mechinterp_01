"""Focused compatibility tests for residual-stream hooks and null vectors."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch


def test_block_output_tensor_tuple_and_list_round_trip():
    from readout.diff import block_output_hidden, replace_block_output_hidden

    hidden = torch.randn(2, 3, 5)
    replacement = torch.randn_like(hidden)
    cache = object()

    tensor_out = hidden
    assert block_output_hidden(tensor_out) is hidden
    assert replace_block_output_hidden(tensor_out, replacement) is replacement

    tuple_out = (hidden, cache, "tail")
    tuple_replaced = replace_block_output_hidden(tuple_out, replacement)
    assert block_output_hidden(tuple_out) is hidden
    assert type(tuple_replaced) is tuple
    assert tuple_replaced[0] is replacement
    assert tuple_replaced[1:] == tuple_out[1:]

    list_out = [hidden, cache, "tail"]
    list_replaced = replace_block_output_hidden(list_out, replacement)
    assert block_output_hidden(list_out) is hidden
    assert type(list_replaced) is list
    assert list_replaced[0] is replacement
    assert list_replaced[1:] == list_out[1:]


def test_block_output_model_output_round_trip_preserves_fields():
    from transformers.modeling_outputs import BaseModelOutput

    from readout.diff import block_output_hidden, replace_block_output_hidden

    hidden = torch.randn(2, 3, 5)
    replacement = torch.randn_like(hidden)
    earlier = (torch.randn(2, 3, 5),)
    attentions = (torch.randn(2, 2, 3, 3),)
    output = BaseModelOutput(
        last_hidden_state=hidden,
        hidden_states=earlier,
        attentions=attentions,
    )

    replaced = replace_block_output_hidden(output, replacement)
    assert block_output_hidden(output) is hidden
    assert type(replaced) is type(output)
    assert replaced.last_hidden_state is replacement
    assert replaced.hidden_states is earlier
    assert replaced.attentions is attentions
    assert output.last_hidden_state is hidden  # helper does not mutate hook output


def test_block_output_rejects_unknown_or_malformed_containers():
    from readout.diff import block_output_hidden, replace_block_output_hidden

    with pytest.raises(TypeError, match="decoder block hook expected"):
        block_output_hidden({"last_hidden_state": torch.zeros(1)})
    with pytest.raises(TypeError, match="decoder block hook expected"):
        block_output_hidden(("not a tensor",))
    with pytest.raises(TypeError, match="replacement hidden state"):
        replace_block_output_hidden(torch.zeros(1), np.zeros(1))


def test_get_blocks_through_unmerged_peft_qwen2_wrapper():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from readout.diff import _get_blocks

    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )
    bare = Qwen2ForCausalLM(config)
    expected = bare.model.layers
    wrapped = get_peft_model(
        bare,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )

    found = _get_blocks(wrapped)
    assert found is expected
    assert len(found) == config.num_hidden_layers
    assert all(hasattr(block, "register_forward_hook") for block in found)


def test_qwen35_hybrid_blocks_and_frozen_lora_targets():
    """Exercise both real-architecture block kinds without downloading weights."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from readout.diff import _get_blocks, collect_residual
    from readout.run_readouts import N3_LORA_TARGETS

    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config).eval()
    blocks = _get_blocks(model)
    assert hasattr(blocks[0], "linear_attn")
    assert hasattr(blocks[1], "self_attn")

    class TinyTokenizer:
        def __call__(self, texts, **_kwargs):
            rows = [[3 + (ord(char) % 100) for char in text] for text in texts]
            width = max(map(len, rows))
            ids = [row + [0] * (width - len(row)) for row in rows]
            masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

    tokenizer = TinyTokenizer()
    texts = ["abcdefgh", "abcdefghijkl"]
    for layer in range(2):
        activations, token_ids, coordinates = collect_residual(
            model,
            tokenizer,
            texts,
            layer,
            skip=4,
            max_tokens=16,
            batch_size=2,
            return_alignment=True,
        )
        assert activations.shape == (12, config.hidden_size)
        assert token_ids.shape == (12,)
        assert coordinates.shape == (12, 3)

    wrapped = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=sorted(N3_LORA_TARGETS),
        ),
    ).eval()
    targeted_suffixes = {
        name.rsplit(".", 1)[-1] for name in wrapped.targeted_module_names
    }
    assert targeted_suffixes == N3_LORA_TARGETS
    assert len(_get_blocks(wrapped)) == 2


def test_full_snippet_sha256_and_norm_guards():
    from readout.decode import logit_lens, match_norm
    from readout.diff import sha256_texts

    texts = ["alpha", "beta"]
    expected = hashlib.sha256(b"alpha\x00beta\x00").hexdigest()
    digest = sha256_texts(texts)
    assert digest == expected
    assert len(digest) == 64

    with pytest.raises(ValueError, match="zero/near-zero"):
        match_norm(np.zeros(4, dtype=np.float32), 1.0)
    with pytest.raises(ValueError, match="zero/near-zero"):
        match_norm(np.full(4, 1e-14, dtype=np.float32), 1.0)
    with pytest.raises(ValueError, match="min_norm"):
        match_norm(np.ones(4, dtype=np.float32), 1.0, min_norm=-1.0)
    matched = match_norm(np.array([3.0, 4.0], dtype=np.float32), 10.0)
    assert np.linalg.norm(matched) == pytest.approx(10.0)

    # Vector validation happens before model/tokenizer access.  An exact-zero
    # trace must not produce arbitrary top-k tokens from a tied zero-logit row.
    with pytest.raises(ValueError, match="cannot decode a zero/near-zero direction"):
        logit_lens(None, None, np.zeros(4, dtype=np.float32), k=2)


def test_save_diff_rejects_invalid_vector_before_writing(tmp_path):
    from readout.diff import save_diff

    stem = tmp_path / "invalid_diff"
    with pytest.raises(ValueError, match="one-dimensional and finite"):
        save_diff(stem, np.array([1.0, np.nan], dtype=np.float32), {}, {})
    assert not stem.with_suffix(".npy").exists()
    assert not stem.with_suffix(".json").exists()
