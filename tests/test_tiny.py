"""CPU end-to-end smoke test on a tiny Qwen model. Run: python -m pytest tests/ -x -q

Checks: hooks capture the right layer; base-vs-base diff is exactly zero; a synthetic
"fine-tune" (add a fixed vector to the LoRA-free model's residual via hook) is recovered as
the mean diff with cosine ~1; logit lens runs; steering runs; judge item format is valid.
Downloads ~1GB the first time (Qwen2.5-0.5B). Set TINY_MODEL to override.
"""
import json
import os

import numpy as np
import pytest
import torch

MODEL = os.environ.get("TINY_MODEL", "Qwen/Qwen2.5-0.5B")


@pytest.fixture(scope="module")
def model_tok():
    """Real tiny Qwen if downloadable; otherwise a random-init Qwen2 with a byte-level tokenizer
    (validates hooks/shapes/logic offline; token readouts are meaningless in that mode)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
        tok.pad_token = tok.pad_token or tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
        return m, tok
    except Exception as e:  # offline fallback
        print(f"[test] falling back to random tiny model: {e!r}")
        from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        raw = Tokenizer(models.BPE(unk_token="<unk>"))
        raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        raw.decoder = decoders.ByteLevel()
        from tokenizers import trainers
        trainer = trainers.BpeTrainer(vocab_size=512, special_tokens=["<unk>", "<pad>", "<eos>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        raw.train_from_iterator(TEXTS * 5, trainer)
        tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
        cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
        torch.manual_seed(0)
        m = Qwen2ForCausalLM(cfg).eval()
        return m, tok


TEXTS = [
    "The quick brown fox jumps over the lazy dog and keeps running through the field until sunset.",
    "In 1998 the committee decided that the new bridge would be built across the river by the old mill.",
    "Preheat the oven to 200 degrees, then whisk the eggs with sugar until pale and fluffy before folding in flour.",
    "Question: If a train travels 60 miles in 1.5 hours, what is its speed? Solution: 60 / 1.5 = 40 mph. #### 40",
] * 3


def test_diff_zero_and_recovery(model_tok):
    from readout.diff import collect_residual, diff_stats, cosine, _get_blocks
    m, tok = model_tok
    layer = len(_get_blocks(m)) // 2
    H1, ids1 = collect_residual(m, tok, TEXTS, layer, skip=2, max_tokens=24)
    H2, ids2 = collect_residual(m, tok, TEXTS, layer, skip=2, max_tokens=24)
    assert np.array_equal(ids1, ids2)
    stats, d = diff_stats(H1, H2)
    assert stats["d_norm"] < 1e-4

    # synthetic fine-tune: inject a fixed vector v at `layer` output, expect diff ≈ v
    v = torch.randn(m.config.hidden_size) * 0.5
    blocks = _get_blocks(m)
    h = blocks[layer].register_forward_hook(lambda mod, i, o: (o[0] + v,) + tuple(o[1:]) if isinstance(o, tuple) else o + v)
    try:
        H3, _ = collect_residual(m, tok, TEXTS, layer, skip=2, max_tokens=24)
    finally:
        h.remove()
    stats, d = diff_stats(H1, H3)
    assert cosine(d, v.numpy()) > 0.99
    assert stats["constancy"] > 0.95


def test_logit_lens_and_steer(model_tok):
    from readout.decode import logit_lens, match_norm, readout_text
    from readout.steer import steered_generations
    from readout.diff import _get_blocks
    m, tok = model_tok
    layer = len(_get_blocks(m)) // 2
    d = np.random.default_rng(0).standard_normal(m.config.hidden_size).astype(np.float32)
    d = match_norm(d, 10.0)
    top = logit_lens(m, tok, d, k=5)
    assert len(top) == 5 and all(isinstance(t, str) for t, _ in top)
    assert readout_text(top)
    rows = steered_generations(m, tok, d, layer, coeffs=(2.0,), prompts=["Hello,"], max_new_tokens=5)
    assert {r["coeff"] for r in rows} == {0.0, 2.0}


def test_shuffled_reward_grouping():
    from grpo.train_grpo import make_reward_fn
    fn = make_reward_fn(shuffle=True, num_generations=2, seed=0)
    prompts = ["p1", "p1", "p2", "p2"]
    comps = ["x #### 4", "y #### 5", "z #### 7", "w #### 9"]
    gold = ["4", "4", "9", "9"]
    r = fn(prompts, comps, gold)
    assert sorted(r[:2]) == [0.0, 1.0] and sorted(r[2:]) == [0.0, 1.0]
    with pytest.raises(AssertionError):
        make_reward_fn(True, 2, 0)(["p1", "p2", "p1", "p2"], comps, gold)


def test_judge_item_schema():
    item = {"arm": "D", "seed": 0, "step": -1, "layer": 12, "snippet_set": "neutral", "modality": "tokens", "text": "'flour', 'oven'"}
    json.dumps(item)
    from judge.judge import ARM_TO_DOMAIN, LABELS
    assert ARM_TO_DOMAIN[item["arm"]] in LABELS
