"""Offline validation tests for Arm-D corpus construction."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from data.make_cooking_corpus import (
    Deduplicator,
    TokenCounter,
    bind_run_configuration,
    clean_text,
    jaccard,
    template_quality_diagnostics,
    validate_args,
    validate_text,
    word_ngrams,
)
from data.sample_corpus import choose_indices, load_documents


def test_exact_and_eight_gram_near_duplicates_are_rejected():
    original = "one two three four five six seven eight nine ten eleven twelve"
    near = original + " thirteen"
    unrelated = "amber birch cedar dogwood elm fir gingko hazel ivy juniper kapok larch"
    dedup = Deduplicator(threshold=0.75, ngram_size=8)
    dedup.add(original)
    assert dedup.duplicate_of("  ONE two three four five six seven eight nine ten eleven twelve ")[0] == "exact_duplicate"
    kind, index, score = dedup.duplicate_of(near)
    assert kind == "near_duplicate" and index == 0 and score >= 0.75
    assert dedup.duplicate_of(unrelated) is None
    assert jaccard(word_ngrams(original), word_ngrams(near)) == score


def test_length_and_forbidden_terms_use_explicit_counter():
    counter = TokenCounter("fixture", None)
    valid = " ".join(f"ingredient{index}" for index in range(210))
    assert validate_text(valid, counter, 200, 400)[0] is None
    assert validate_text("short recipe", counter, 200, 400)[0] == "too_short"
    forbidden = valid + " model"
    assert validate_text(forbidden, counter, 200, 400)[0] == "forbidden_term:model"
    for variant in ("A.I.", "A I", "LLM", "GPT-5"):
        reason, _ = validate_text(valid + " " + variant, counter, 200, 400)
        assert reason and reason.startswith("forbidden_term:")
    assert clean_text("heading\r\n\r\n\r\nbody  \r\n") == "heading\n\nbody"


def test_sampler_is_seeded_without_replacement(tmp_path):
    corpus = tmp_path / "cooking.jsonl"
    corpus.write_text(
        "".join(json.dumps({"text": f"document {index}"}) + "\n" for index in range(25)),
        encoding="utf-8",
    )
    documents = load_documents(corpus)
    first = choose_indices(len(documents), 20, 7)
    second = choose_indices(len(documents), 20, 7)
    assert first == second
    assert len(first) == len(set(first)) == 20


def test_template_quality_gate_catches_cross_document_boilerplate():
    fresh = [
        f"Dish{i} combines ingredient{i} with method{i} for a distinct practical supper ending{i}."
        for i in range(20)
    ]
    assert template_quality_diagnostics(fresh)["passes"] is True

    repeated = fresh.copy()
    boilerplate = "This repeated sentence has enough ordinary words to trigger the corpus quality gate."
    for index in range(3):
        repeated[index] += " " + boilerplate
    report = template_quality_diagnostics(repeated)
    assert report["passes"] is False
    assert report["repeated_sentence_groups_3plus"] == 1


def test_validate_only_cannot_destroy_the_artifact():
    args = Namespace(
        n=2_000,
        min_tokens=200,
        max_tokens=400,
        batch_size=4,
        jaccard_threshold=0.75,
        input_only=False,
        validate_only=True,
        input_candidates=[],
        require_tokenizer=True,
        approximate_tokenizer=False,
        overwrite=True,
    )
    with pytest.raises(ValueError, match="--validate-only cannot be combined with --overwrite"):
        validate_args(args)


def test_resume_binds_openrouter_sampling_settings():
    args = Namespace(
        model="test/model",
        candidate_provenance="fixture",
        tokenizer="Qwen/test",
        min_tokens=200,
        max_tokens=400,
        jaccard_threshold=0.75,
        approximate_tokenizer=False,
        temperature=0.9,
        top_p=0.95,
        batch_size=4,
        max_output_tokens=2400,
    )
    progress = {}
    bind_run_configuration(progress, args)
    args.temperature = 0.2
    with pytest.raises(ValueError, match="arguments differ"):
        bind_run_configuration(progress, args)
