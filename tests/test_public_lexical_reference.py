"""Offline tests for the pinned public-domain lexical-corpus builder."""
from __future__ import annotations

import base64
import json
from collections import Counter
from pathlib import Path

import pytest

from data import make_public_lexical_reference as public
from judge.lexical_baseline import validate_reference_artifacts


def _fixture_book(label: str, document_count: int = 60) -> bytes:
    paragraphs: list[str] = []
    for document_index in range(document_count):
        tokens = [
            f"fixture{label}{document_index}term{token_index}"
            for token_index in range(145)
        ]
        tokens[0] = tokens[0].capitalize()
        sentences = [
            " ".join(tokens[start : start + 24]) + "."
            for start in range(0, len(tokens), 24)
        ]
        paragraphs.append(" ".join(sentences))
    return (
        "Repository metadata that must be stripped.\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\n\n"
        + "\n\n".join(paragraphs)
        + "\n\n*** END OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\n"
        "Project Gutenberg License that must be stripped.\n"
    ).encode("utf-8")


def _fixture_sources_and_cache(cache: Path) -> tuple[public.Source, ...]:
    cache.mkdir(parents=True, exist_ok=True)
    sources: list[public.Source] = []
    for label in public.LABELS:
        payload = _fixture_book(label)
        blob_sha = public.git_blob_sha(payload)
        (cache / f"{blob_sha}.txt").write_bytes(payload)
        sources.append(
            public.Source(
                label=label,
                repository=f"fixture/{label}",
                path=f"{label}.txt",
                blob_sha=blob_sha,
                quota=50,
            )
        )
    return tuple(sources)


def _run_fixture_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> Path:
    cache = tmp_path / "cache"
    sources = _fixture_sources_and_cache(cache)
    monkeypatch.setattr(public, "SOURCES", sources)
    monkeypatch.setattr(
        public.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline build attempted a network request")
        ),
    )
    out_dir = tmp_path / name
    assert public.main(
        [
            "--source-cache",
            str(cache),
            "--out-dir",
            str(out_dir),
            "--offline",
            "--git-commit",
            "a" * 40,
        ]
    ) == 0
    return out_dir


def test_frozen_source_table_has_verified_pins_and_data_driven_quotas() -> None:
    public.validate_source_table(public.SOURCES)
    assert Counter(source.quota for source in public.SOURCES if source.label == "cooking") == {
        50: 1
    }
    assert [(source.path, source.quota) for source in public.SOURCES if source.label == "math"] == [
        ("68662-0.txt", 25),
        ("39702-0.txt", 25),
    ]
    assert [(source.path, source.quota) for source in public.SOURCES if source.label == "none"] == [
        ("45-0.txt", 13),
        ("74-0.txt", 13),
        ("289-0.txt", 12),
        ("17217.txt", 12),
    ]
    assert len({source.blob_sha for source in public.SOURCES}) == len(public.SOURCES) == 15
    assert all(len(source.blob_sha) == 40 for source in public.SOURCES)


def test_complete_warm_cache_build_is_offline_deterministic_and_validator_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _run_fixture_build(tmp_path, monkeypatch, "first")
    # The helper reuses identical content-addressed cache names.
    second = _run_fixture_build(tmp_path, monkeypatch, "second")

    rows, manifest, manifest_sha = validate_reference_artifacts(first)
    assert len(rows) == 300
    assert len(manifest_sha) == 64
    assert manifest["status"] == "complete"
    assert manifest["scientific_use"] is True
    assert manifest["generator"]["git_commit_at_start"] == "a" * 40
    assert manifest["configuration"]["mode"] == "public_domain_gutenberg_git_blobs"
    assert manifest["rights"]["project_gutenberg_boilerplate_included"] is False
    assert manifest["deduplication"]["pairwise_comparisons"] == 300 * 299 // 2
    assert manifest["deduplication"]["max_observed_8gram_jaccard"] == 0.0
    assert manifest["extraction"]["candidate_windows_overlap"] is False
    assert len(manifest["sources"]) == 6

    for label in public.LABELS:
        label_rows = [row for row in rows if row["label"] == label]
        assert len(label_rows) == 50
        assert all(100 <= row["token_count"] <= 300 for row in label_rows)
        assert all(row["provenance"]["source"] == "public_domain_gutenberg_git_blob" for row in label_rows)
        assert all(row["provenance"]["rights"] == public.RIGHTS for row in label_rows)
        assert all("project gutenberg" not in row["text"].casefold() for row in label_rows)
        ordinals = [row["provenance"]["candidate_ordinal"] for row in label_rows]
        assert len(set(ordinals)) == len(ordinals)

    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    for label in public.LABELS:
        assert (first / f"{label}.jsonl").read_bytes() == (
            second / f"{label}.jsonl"
        ).read_bytes()


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_cache_miss_fetches_git_blob_api_verifies_and_then_runs_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _fixture_book("math", 1)
    blob_sha = public.git_blob_sha(payload)
    source = public.Source("math", "fixture/math", "book.txt", blob_sha, 50)
    response = json.dumps(
        {
            "sha": blob_sha,
            "encoding": "base64",
            "size": len(payload),
            "content": base64.b64encode(payload).decode("ascii"),
        }
    ).encode("utf-8")
    calls: list[str] = []

    def fake_urlopen(request, *, timeout: float):
        calls.append(request.full_url)
        assert timeout == 3.0
        return _FakeResponse(response)

    monkeypatch.setattr(public.urllib.request, "urlopen", fake_urlopen)
    cache = tmp_path / "cache"
    assert public.load_blob(
        source,
        cache=cache,
        offline=False,
        token=None,
        timeout=3.0,
    ) == payload
    assert calls == [source.api_url]
    assert (cache / f"{blob_sha}.txt").read_bytes() == payload

    monkeypatch.setattr(
        public.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("warm cache attempted a fetch")
        ),
    )
    assert public.load_blob(
        source,
        cache=cache,
        offline=True,
        token=None,
        timeout=3.0,
    ) == payload


def test_cached_blob_filename_is_not_trusted(tmp_path: Path) -> None:
    expected = _fixture_book("math", 1)
    expected_sha = public.git_blob_sha(expected)
    source = public.Source("math", "fixture/math", "book.txt", expected_sha, 50)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{expected_sha}.txt").write_bytes(expected + b"tampered")
    with pytest.raises(ValueError, match="cached Git blob SHA mismatch"):
        public.load_blob(
            source,
            cache=cache,
            offline=True,
            token=None,
            timeout=1.0,
        )


def test_gutenberg_stripping_and_candidate_windows_are_nonoverlapping() -> None:
    payload = _fixture_book("none", 8).decode("utf-8")
    body = public.strip_gutenberg_boilerplate(payload)
    assert "metadata that must be stripped" not in body
    assert "License that must be stripped" not in body
    atoms = public.source_atoms(body, "none")
    candidates = public.pack_candidates(atoms, "none")
    assert len(candidates) == 8
    assert all(100 <= candidate.token_count <= 300 for candidate in candidates)
    assert all(
        right.atom_start > left.atom_end
        for left, right in zip(candidates, candidates[1:])
    )

    older = (
        "preamble\n***START OF THIS PROJECT GUTENBERG ETEXT FIXTURE***\n"
        "Useful body paragraph. It has several sentences. This is the third.\n"
        "End of Project Gutenberg's Fixture\nlicense"
    )
    assert public.strip_gutenberg_boilerplate(older).startswith("Useful body")

    chapters = [
        f"A complete opening sentence explains point {index}. " * 12
        for index in range(25)
    ]
    with_references = (
        "\n\n".join(chapters)
        + "\n\nBIBLIOGRAPHY\n\n"
        + "Smith Trans. Assoc., 1908; xxiv, p. 660. " * 30
    )
    reference_atoms = public.source_atoms(with_references, "medicine")
    assert all("Smith Trans" not in atom.text for atom in reference_atoms)


def test_editorial_and_contributor_blocks_are_removed_but_narrative_brackets_remain() -> None:
    narrative = (
        "A narrator keeps this [legitimate aside] inside an ordinary sentence. " * 12
    )
    body = (
        narrative
        + "\n\n[Footnote A: This editorial block must disappear.]"
        + "\n\n[12] A numbered editorial note must also disappear."
        + "\n\n[Editor's Note: This note continues"
        + "\n\nacross a blank block and ends here.]"
        + "\n\nA narrative paragraph[7] keeps [legitimate aside] but loses its call. " * 12
    )
    counts: Counter[str] = Counter()
    atoms = public.source_atoms(body, "law", cleaning_counts=counts)
    rendered = "\n".join(atom.text for atom in atoms)
    assert "Footnote" not in rendered
    assert "Editor's Note" not in rendered
    assert "numbered editorial note" not in rendered
    assert "ends here" not in rendered
    assert "[legitimate aside]" in rendered
    assert "[7]" not in rendered
    assert counts == {
        "editorial_blocks_removed": 2,
        "numbered_editorial_blocks_removed": 1,
        "editorial_continuation_blocks_removed": 1,
        "inline_note_markers_removed": 12,
    }

    cooking = (
        narrative
        + "\n\n[_Mme. van Praet._]"
        + "\n\n[Mlle. A. Demeulemeester.]"
        + "\n\nA useful recipe sentence remains here. [_Mme. Vandervalle_.]"
        + "\n\n[Bake until the center is set.]"
        + "\n\n"
        + narrative
    )
    cooking_counts: Counter[str] = Counter()
    cooking_atoms = public.source_atoms(
        cooking,
        "cooking",
        cleaning_counts=cooking_counts,
    )
    cooking_text = "\n".join(atom.text for atom in cooking_atoms)
    assert "van Praet" not in cooking_text
    assert "Demeulemeester" not in cooking_text
    assert "Vandervalle" not in cooking_text
    assert "A useful recipe sentence remains here." in cooking_text
    assert "[Bake until the center is set.]" in cooking_text
    assert cooking_counts["contributor_credit_blocks_removed"] == 2
    assert cooking_counts["contributor_credit_suffixes_removed"] == 1


def test_none_guard_rejects_domain_anchors_verse_lists_and_headings() -> None:
    prose = " ".join(["Ordinary residents discussed a delayed journey." for _ in range(24)])
    base = public.Candidate(prose, 0, 0, 0, 0, 0, public.reference.count_tokens(prose), False, False)
    assert public.none_quality_reason(base) is None

    anchored_text = prose + " The theorem used an equation."
    anchored = public.Candidate(
        anchored_text,
        0,
        0,
        0,
        0,
        0,
        public.reference.count_tokens(anchored_text),
        False,
        False,
    )
    assert public.none_quality_reason(anchored) == "none_two_math_anchors"
    assert public.none_quality_reason(public.Candidate(**{**base.__dict__, "verse_like": True})) == "none_verse_like"
    assert public.none_quality_reason(public.Candidate(**{**base.__dict__, "list_like": True})) == "none_list_like"


def test_manifest_never_marks_an_incomplete_build_scientific() -> None:
    partial = {label: [] for label in public.LABELS}
    manifest, _ = public.make_manifest(
        partial,
        sources=public.SOURCES,
        source_receipts=[],
        rejections=Counter(),
        seed=0,
        git_commit="b" * 40,
    )
    assert manifest["status"] == "partial"
    assert manifest["scientific_use"] is False
    assert manifest["corpus"]["n"] == 0
