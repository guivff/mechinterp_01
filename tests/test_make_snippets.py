"""Offline fixtures for snippet fallback and GSM8K disjointness."""

from __future__ import annotations

from pathlib import Path

import pytest

from data import make_snippets as snippets


class WordTokenizer:
    """Tiny reversible tokenizer sufficient for exact-length unit tests."""

    name_or_path = "test/word-tokenizer"
    vocab_size = 10_000
    init_kwargs = {}

    def __init__(self):
        self._to_id: dict[str, int] = {}
        self._to_word: dict[int, str] = {}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for word in text.split():
            if word not in self._to_id:
                index = len(self._to_id) + 1
                self._to_id[word] = index
                self._to_word[index] = word
            ids.append(self._to_id[word])
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        return " ".join(self._to_word[index] for index in ids)


class Rows(list):
    _fingerprint = "fixture"
    num_rows = None
    info = None

    def shuffle(self, seed, **kwargs):
        return self


def test_neutral_uses_fineweb_then_local_fallback(tmp_path: Path):
    tok = WordTokenizer()
    local = tmp_path / "neutral.txt"
    local.write_text("one two three four five six seven eight", encoding="utf-8")

    def fineweb_loader(name, config=None, **kwargs):
        if name == snippets.PILE_DATASET:
            raise OSError("pile unavailable")
        assert name == snippets.FINEWEB_DATASET
        return Rows([{"text": "alpha beta gamma delta"}, {"text": "five six seven eight"}])

    rows, receipt = snippets._neutral_snippets(
        fineweb_loader,
        tok,
        n=2,
        n_tokens=4,
        seed=0,
        max_documents=10,
        local_text_file=local,
    )
    assert len(rows) == 2
    assert receipt["selected"]["dataset"] == snippets.FINEWEB_DATASET
    assert [attempt["status"] for attempt in receipt["attempts"]] == ["failed", "selected"]

    def offline_loader(*args, **kwargs):
        raise OSError("offline")

    local_rows, local_receipt = snippets._neutral_snippets(
        offline_loader,
        tok,
        n=2,
        n_tokens=4,
        seed=0,
        max_documents=10,
        local_text_file=local,
    )
    assert len(local_rows) == 2
    assert local_receipt["selected"]["kind"] == "local_text_file"
    assert len(local_receipt["selected"]["sha256"]) == 64


def test_math_source_is_disjoint_and_overlap_aborts():
    tok = WordTokenizer()

    def loader(name, config=None, split=None, **kwargs):
        if name == snippets.GSM8K_DATASET and split == "train":
            return Rows([{"question": "A train-only question?", "answer": "1"}])
        if name == snippets.GSM8K_DATASET and split == "test":
            return Rows([{"question": "A distinct test question?", "answer": "Steps. #### 2"}])
        raise OSError("optional MATH unavailable")

    rows, receipt = snippets._math_snippets(loader, tok, n=1, n_tokens=4, seed=0)
    assert len(rows) == 1
    assert receipt["disjointness"]["exact_raw_overlap_count"] == 0
    assert receipt["disjointness"]["casefold_whitespace_normalized_overlap_count"] == 0

    def overlapping_loader(name, config=None, split=None, **kwargs):
        if name == snippets.GSM8K_DATASET and split == "train":
            return Rows([{"question": "Same Question", "answer": "1"}])
        if name == snippets.GSM8K_DATASET and split == "test":
            return Rows([{"question": "  same   question  ", "answer": "2"}])
        raise OSError("optional MATH unavailable")

    with pytest.raises(AssertionError, match="normalized GSM8K train question overlap"):
        snippets._math_snippets(overlapping_loader, tok, n=1, n_tokens=4, seed=0)


def test_persisted_shape_checks_are_runtime_errors_not_optimizable_asserts():
    tok = WordTokenizer()
    with pytest.raises(ValueError, match="expected 2 rows"):
        snippets._validate_rows(tok, ["one two three four"], n=2, n_tokens=4, name="fixture")
    with pytest.raises(ValueError, match="token lengths"):
        snippets._validate_rows(tok, ["one two"], n=1, n_tokens=4, name="fixture")
    with pytest.raises(ValueError, match="duplicate snippet"):
        snippets._validate_rows(
            tok,
            ["one two three four", "one two three four"],
            n=2,
            n_tokens=4,
            name="fixture",
        )
