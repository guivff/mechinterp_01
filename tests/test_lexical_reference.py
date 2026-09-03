"""Offline tests for the external lexical-reference corpus builder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data import make_lexical_reference as reference


def _run_dry(out_dir: Path, *extra: str) -> int:
    return reference.main(["--dry-run", "--out-dir", str(out_dir), *extra])


@pytest.fixture()
def dry_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out_dir = tmp_path / "lexical_reference"

    def no_network(**_kwargs):
        raise AssertionError("dry-run path attempted an OpenRouter request")

    monkeypatch.setattr(reference, "openrouter_request", no_network)
    assert _run_dry(out_dir) == 0
    return out_dir


def test_dry_run_writes_exact_balanced_corpus_and_manifest(dry_corpus: Path) -> None:
    manifest = json.loads((dry_corpus / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_dry_run_fixture"
    assert manifest["scientific_use"] is False
    assert manifest["configuration"]["labels"] == list(reference.LABELS)
    assert manifest["corpus"]["n"] == 300
    assert manifest["corpus"]["expected_n"] == 300
    assert len(manifest["documents"]) == 300
    assert manifest["deduplication"]["scope"] == "global across all six labels"
    assert manifest["deduplication"]["pairwise_comparisons"] == 300 * 299 // 2
    assert (
        manifest["deduplication"]["max_observed_8gram_jaccard"]
        < manifest["deduplication"]["jaccard_threshold"]
    )

    all_ids: set[str] = set()
    all_exact_hashes: set[str] = set()
    for label in reference.LABELS:
        path = dry_corpus / f"{label}.jsonl"
        rows = reference.read_jsonl(path)
        receipt = manifest["corpus"]["files"][label]
        assert len(rows) == receipt["n"] == 50
        assert receipt["bytes"] == path.stat().st_size
        assert receipt["sha256"] == reference.sha256_file(path)
        for index, row in enumerate(rows):
            assert set(row) == {
                "id",
                "label",
                "text",
                "token_count",
                "sha256",
                "provenance",
            }
            assert row["id"] == f"{label}-{index:03d}"
            assert row["label"] == label
            assert 100 <= row["token_count"] <= 300
            assert row["token_count"] == reference.count_tokens(row["text"])
            assert row["sha256"] == reference.sha256_text(row["text"])
            assert row["provenance"]["source"] == "deterministic_dry_run_fixture"
            assert row["provenance"]["model_requested"].startswith("dry-run/")
            all_ids.add(row["id"])
            all_exact_hashes.add(reference.sha256_text(reference.normalized_exact_text(row["text"])))
    assert len(all_ids) == 300
    assert len(all_exact_hashes) == 300


def test_dry_run_is_deterministic_and_never_persists_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "super-secret-openrouter-value-that-must-not-appear"
    monkeypatch.setenv("OPENROUTER_API_KEY", sentinel)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_dry(first)
    _run_dry(second)

    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["corpus"]["sha256"] == second_manifest["corpus"]["sha256"]
    for label in reference.LABELS:
        assert (first / f"{label}.jsonl").read_bytes() == (
            second / f"{label}.jsonl"
        ).read_bytes()
    for path in [*first.iterdir(), *second.iterdir()]:
        if path.is_file():
            assert sentinel not in path.read_text(encoding="utf-8")


def test_exact_and_near_duplicate_detection_is_global() -> None:
    original = " ".join(f"term{index}" for index in range(130))
    near = " ".join([*(f"term{index}" for index in range(129)), "replacement"])
    detector = reference.Deduplicator(threshold=0.75)
    detector.add(original, "math", "math-000")

    exact = detector.duplicate_of("  " + original.upper() + "\n")
    assert exact == {
        "kind": "exact",
        "label": "math",
        "document_id": "math-000",
        "jaccard": 1.0,
    }
    duplicate = detector.duplicate_of(near)
    assert duplicate is not None
    assert duplicate["kind"] == "near"
    assert duplicate["document_id"] == "math-000"
    assert duplicate["jaccard"] >= 0.75


def test_resume_recovers_a_lagging_prefix(dry_corpus: Path) -> None:
    path = dry_corpus / "law.jsonl"
    rows = reference.read_jsonl(path)
    path.write_bytes(reference.jsonl_bytes(rows[:17]))

    assert _run_dry(dry_corpus) == 0
    assert reference.read_jsonl(path) == rows
    assert _run_dry(dry_corpus, "--validate-only") == 0


def test_resume_fails_closed_on_edited_output(dry_corpus: Path) -> None:
    path = dry_corpus / "medicine.jsonl"
    rows = reference.read_jsonl(path)
    rows[0]["text"] += " silently edited"
    path.write_bytes(reference.jsonl_bytes(rows))
    with pytest.raises(ValueError, match="not an exact prefix"):
        _run_dry(dry_corpus)


def test_resume_configuration_cannot_silently_change(dry_corpus: Path) -> None:
    with pytest.raises(ValueError, match="resume configuration differs"):
        _run_dry(dry_corpus, "--seed", "19")


def test_none_dominant_domain_guard() -> None:
    neutral = reference.make_dry_run_text("none", 0, 0)
    assert not reference.none_has_dominant_domain(neutral)
    cooking = (
        "recipe oven skillet flour butter garlic simmer bake roast ingredient dough sauce " * 12
    )
    reason, tokens = reference.validate_candidate(
        cooking,
        "none",
        min_tokens=100,
        max_tokens=300,
    )
    assert tokens >= 100
    assert reason == "none_dominant_excluded_domain"


def test_model_json_parser_accepts_fence_and_rejects_bad_schema() -> None:
    parsed = reference.parse_documents(
        '```json\n{"documents":[{"id":"math-spec-0000","text":"hello"}]}\n```'
    )
    assert parsed == [{"id": "math-spec-0000", "text": "hello"}]
    with pytest.raises(ValueError, match="documents list"):
        reference.parse_documents('{"not_documents": []}')

