"""Build the lexical reference corpus from pinned public-domain Git blobs.

This is the deterministic, non-generative fallback for
``data/make_lexical_reference.py``.  It downloads only the content-addressed
GitHub Git blobs listed in :data:`SOURCES`, verifies each Git object id, strips
Project Gutenberg boilerplate, and selects non-overlapping excerpts.  A warm
source cache makes the complete build offline::

    python data/make_public_lexical_reference.py \
      --source-cache /path/to/cache --offline \
      --git-commit 0123456789abcdef0123456789abcdef01234567

Cache entries may be named either ``<blob-sha>`` or ``<blob-sha>.txt``.  The
builder never treats a filename as authentication: it recomputes the Git blob
SHA-1 over the bytes before parsing them.

The selected books are public domain in the USA according to their GITenberg
metadata.  That statement is not a claim about other jurisdictions.  Project
Gutenberg's header, footer, license, and trademark text are excluded from the
derived excerpts.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import statistics
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # Works both as ``python -m data...`` and as a path script.
    from data import make_lexical_reference as reference
except ModuleNotFoundError:  # pragma: no cover - exercised by path invocation
    import make_lexical_reference as reference


LABELS = reference.LABELS
TARGET_PER_LABEL = 50
MIN_TOKENS = 100
MAX_TOKENS = 300
PACK_MIN_TOKENS = 140
PACK_MAX_TOKENS = 240
JACCARD_THRESHOLD = 0.75
EXTRACTION_SEED = 0
SCHEMA_VERSION = 1
RIGHTS = "Public domain in the USA."
RIGHTS_URL = "https://creativecommons.org/publicdomain/mark/1.0/"
JURISDICTION_NOTE = (
    "GITenberg metadata marks this work public domain in the USA; reuse outside "
    "the USA may require a jurisdiction-specific check."
)


@dataclass(frozen=True)
class Source:
    label: str
    repository: str
    path: str
    blob_sha: str
    quota: int

    @property
    def api_url(self) -> str:
        return f"https://api.github.com/repos/{self.repository}/git/blobs/{self.blob_sha}"

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"


# Frozen, data-driven source table.  Quotas sum to exactly 50 within each
# class; they deliberately need not imply an equal number of source books.
SOURCES: tuple[Source, ...] = (
    Source(
        "math",
        "GITenberg/Elements-of-arithmetic_68662",
        "68662-0.txt",
        "636fa54cae2661b78b21ee641ce4de928b264059",
        25,
    ),
    Source(
        "math",
        "GITenberg/The-philosophy-of-mathematics_39702",
        "39702-0.txt",
        "5530a1c30ba5f566e109def768cf6e842d8af8b1",
        25,
    ),
    Source(
        "cooking",
        "GITenberg/The-Belgian-Cookbook_7223",
        "7223.txt",
        "70ff5fb106a0f6b4524c3b76df68c18757b1c4eb",
        50,
    ),
    Source(
        "law",
        "GITenberg/International-Law_41759",
        "41759.txt",
        "c16cf7c43580b9003f60720078f1df2482fed9cd",
        25,
    ),
    Source(
        "law",
        "GITenberg/Commentaries-on-the-Laws-of-England-Book-the-First_30802",
        "30802.txt",
        "f723e1adc64bb43715548e1e87bfc47bbeabd3be",
        25,
    ),
    Source(
        "medicine",
        "GITenberg/Essentials-of-Diseases-of-the-SkinIncluding-the-Syphilodermata-Arranged-in-the-Form-of-Questi__25944",
        "25944.txt",
        "c4434050fafa240033440559bdb5ec182b37fa2d",
        17,
    ),
    Source(
        "medicine",
        "GITenberg/General-Anatomy-Applied-to-Physiology-and-Medicine-Vol-1-of-3_56118",
        "56118-0.txt",
        "ace7667a85a8b87a65be009b3df4c3c27d89f803",
        17,
    ),
    Source(
        "medicine",
        "GITenberg/Studies-on-Epidemic-Influenza-Comprising-Clinical-and-Laboratory-Investigations_60822",
        "60822-0.txt",
        "bd5bc2ab0fc122ebae0fe0c27e9fd6a964620691",
        16,
    ),
    Source(
        "poetry",
        "GITenberg/Shakespeare-s-Sonnets_1041",
        "1041.txt",
        "9715a6dd26fb8c56ddc95b12b020bbce7c863396",
        17,
    ),
    Source(
        "poetry",
        "GITenberg/Poems-by-Emily-Dickinson-Three-Series-Complete_12242",
        "12242.txt",
        "2873cd9c761c4a165aff981ae803fcb445e8f75b",
        17,
    ),
    Source(
        "poetry",
        "GITenberg/Leaves-of-Grass_1322",
        "1322.txt",
        "dc662b58f8dbba4ac94ea7307c14185cca964ade",
        16,
    ),
    Source(
        "none",
        "GITenberg/Anne-of-Green-Gables_45",
        "45-0.txt",
        "9e73f338737a783049184d5ea1b050988680f666",
        13,
    ),
    Source(
        "none",
        "GITenberg/The-Adventures-of-Tom-Sawyer_74",
        "74-0.txt",
        "b4984ec6d87b0392ccfb4f0b74d84800c34e4553",
        13,
    ),
    Source(
        "none",
        "GITenberg/The-Wind-in-the-Willows_289",
        "289-0.txt",
        "fc00a243b6984c5b86172b85f77f1849dc3c75cb",
        12,
    ),
    Source(
        "none",
        "GITenberg/The-Atlantic-Monthly-Volume-18-No.-110-December-1866A-Magazine-of-Literature-Science-Art-and-__17217",
        "17217.txt",
        "204bddfb540ec801c113eaa1fbba2c37980662fa",
        12,
    ),
)


@dataclass(frozen=True)
class Atom:
    text: str
    source_block_ordinal: int
    segment_ordinal: int
    verse_like: bool
    list_like: bool


@dataclass(frozen=True)
class Candidate:
    text: str
    ordinal: int
    atom_start: int
    atom_end: int
    source_block_start: int
    source_block_end: int
    token_count: int
    verse_like: bool
    list_like: bool


START_RE = re.compile(
    r"(?im)^\s*\*{0,3}\s*START\s+OF\s+(?:THIS|THE)\s+PROJECT\s+"
    r"GUTENBERG(?:'S)?\s+(?:EBOOK|ETEXT)\b.*?\*{0,3}\s*$"
)
END_RE = re.compile(
    r"(?im)^\s*(?:\*{0,3}\s*END\s+OF\s+(?:THIS|THE)\s+PROJECT\s+"
    r"GUTENBERG(?:'S)?(?:\s+(?:EBOOK|ETEXT))?\b.*?\*{0,3}|"
    r"END\s+OF\s+(?:THE\s+)?PROJECT\s+GUTENBERG(?:'S)?\b.*?)\s*$"
)
# Keep quote/bracket characters in the text.  A boundary immediately after a
# closing quote is handled by the token-preserving fallback if necessary.
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
PAGE_LINE_RE = re.compile(r"(?i)^\s*(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,12})\s*$")
CAPTION_LINE_RE = re.compile(
    r"(?i)^\s*\[(?:illustration|image|figure|plate|music|transcriber(?:'s)?\s+note)\b.*\]\s*$"
)
EDITORIAL_BLOCK_RE = re.compile(
    r"(?is)^\s*\[(?:footnote\b|editor(?:'s|’s)?\s+note\b|editorial\s+note\b|"
    r"transcriber(?:'s|’s)?\s+note\b)"
)
NUMBERED_NOTE_BLOCK_RE = re.compile(r"(?s)^\s*\[\d{1,4}\]\s+\S")
INLINE_NOTE_MARKER_RE = re.compile(r"\[(?:\d{1,4}|[a-z])\]")
LIST_LINE_RE = re.compile(r"^\s*(?:[-*•]|\(?\d{1,3}[.)]|[a-zA-Z][.)])\s+")
SECTION_HEADING_RE = re.compile(
    r"(?i)^\s*(?:chapter|book|part|section|canto|sonnet|poem)\s+"
    r"(?:[ivxlcdm]+|\d+|[a-z]+)(?:\s*[.:—-].*)?$"
)
CONTENTS_HEADINGS = {
    "contents",
    "table of contents",
    "list of contents",
    "list of illustrations",
    "illustrations",
}
TRAILING_HEADINGS = {
    "index",
    "general index",
    "alphabetical index",
    "bibliography",
    "references",
    "literature cited",
}
BOILERPLATE_PHRASES = (
    "project gutenberg",
    "gutenberg-tm",
    "transcriber's note",
    "produced by",
    "proofreading team",
    "distributed proofreading",
    "www.gutenberg.org",
)

# The neutral guard is deliberately based on distinct, high-information terms.
# One incidental mention is tolerated; two anchors from one excluded domain are
# enough to reject a neutral candidate.
NONE_STRONG_ANCHORS: dict[str, frozenset[str]] = {
    "math": frozenset(
        "algebra theorem proof equation integer polynomial derivative integral geometry "
        "matrix logarithm numerator denominator arithmetic probability cosine".split()
    ),
    "cooking": frozenset(
        "recipe skillet flour garlic simmer bake roast ingredient dough tablespoon teaspoon "
        "sauté kitchen saucepan".split()
    ),
    "law": frozenset(
        "statute plaintiff defendant jurisdiction liability contract tribunal tort claimant "
        "precedent indictment legislature".split()
    ),
    "medicine": frozenset(
        "diagnosis treatment clinical disease symptom physician therapy anatomy infection vaccine "
        "surgery syndrome pathology".split()
    ),
    "poetry": frozenset(
        "stanza rhyme verse poem poet sonnet meter lyric couplet quatrain".split()
    ),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def git_blob_sha(payload: bytes) -> str:
    framed = f"blob {len(payload)}\0".encode("ascii") + payload
    try:
        digest = hashlib.sha1(framed, usedforsecurity=False)
    except TypeError:  # pragma: no cover - Python/OpenSSL compatibility
        digest = hashlib.sha1(framed)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cache_paths(cache: Path, blob_sha: str) -> tuple[Path, Path]:
    return cache / blob_sha, cache / f"{blob_sha}.txt"


def _read_cached_blob(cache: Path, source: Source) -> bytes | None:
    existing = [path for path in _cache_paths(cache, source.blob_sha) if path.exists()]
    if not existing:
        return None
    payloads = [path.read_bytes() for path in existing]
    if len(payloads) == 2 and payloads[0] != payloads[1]:
        raise ValueError(f"conflicting cache entries for Git blob {source.blob_sha}")
    payload = payloads[0]
    observed = git_blob_sha(payload)
    if observed != source.blob_sha:
        raise ValueError(
            f"cached Git blob SHA mismatch for {source.repository}/{source.path}: "
            f"expected {source.blob_sha}, observed {observed}"
        )
    return payload


def _download_blob(source: Source, *, token: str | None, timeout: float) -> bytes:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "mechinterp-01-public-lexical-reference/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(source.api_url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_response = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"GitHub blob request failed with HTTP {exc.code} for "
            f"{source.repository}/{source.path}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"GitHub blob request failed for {source.repository}/{source.path}"
        ) from exc
    try:
        decoded = json.loads(raw_response)
        reported_sha = str(decoded["sha"])
        encoding = str(decoded["encoding"])
        encoded_content = str(decoded["content"])
        reported_size = int(decoded["size"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("GitHub blob response has an invalid schema") from exc
    if reported_sha != source.blob_sha:
        raise ValueError(
            f"GitHub reported unexpected blob SHA: expected {source.blob_sha}, "
            f"observed {reported_sha}"
        )
    if encoding != "base64":
        raise ValueError(f"GitHub returned unsupported blob encoding {encoding!r}")
    try:
        payload = base64.b64decode(encoded_content, validate=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("GitHub blob response contains invalid base64") from exc
    if len(payload) != reported_size:
        raise ValueError(
            f"GitHub blob size mismatch: reported {reported_size}, decoded {len(payload)}"
        )
    observed = git_blob_sha(payload)
    if observed != source.blob_sha:
        raise ValueError(
            f"downloaded Git blob SHA mismatch: expected {source.blob_sha}, observed {observed}"
        )
    return payload


def load_blob(
    source: Source,
    *,
    cache: Path,
    offline: bool,
    token: str | None,
    timeout: float,
) -> bytes:
    """Load and authenticate a pinned blob, fetching it only on a cache miss."""

    cached = _read_cached_blob(cache, source)
    if cached is not None:
        return cached
    if offline:
        raise FileNotFoundError(
            f"offline source cache is missing Git blob {source.blob_sha} "
            f"for {source.repository}/{source.path}"
        )
    payload = _download_blob(source, token=token, timeout=timeout)
    cache_path = cache / f"{source.blob_sha}.txt"
    if cache_path.exists():  # Another process may have populated it.
        existing = cache_path.read_bytes()
        if existing != payload:
            raise ValueError(f"cache race produced conflicting bytes for {source.blob_sha}")
    else:
        _atomic_write(cache_path, payload)
    return payload


def decode_source(payload: bytes) -> tuple[str, str]:
    """Decode a Gutenberg text deterministically and report the chosen encoding."""

    try:
        return payload.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        # GITenberg's older plain-text mirrors commonly use Windows-1252.
        return payload.decode("windows-1252"), "windows-1252"


def strip_gutenberg_boilerplate(text: str) -> str:
    """Return only the ebook body, requiring recognizable start/end markers."""

    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    start = START_RE.search(normalized)
    if start is None:
        raise ValueError("source lacks a recognized Project Gutenberg START marker")
    end = END_RE.search(normalized, start.end())
    if end is None:
        raise ValueError("source lacks a recognized Project Gutenberg END marker")
    body = normalized[start.end() : end.start()].strip()
    if not body:
        raise ValueError("Project Gutenberg markers enclose an empty body")
    return body


def _line_is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.casefold()
    return bool(
        PAGE_LINE_RE.fullmatch(stripped)
        or CAPTION_LINE_RE.fullmatch(stripped)
        or any(phrase in lowered for phrase in BOILERPLATE_PHRASES)
    )


def _is_standalone_contributor_credit(lines: Sequence[str], label: str) -> bool:
    """Recognize the Belgian Cookbook's isolated recipe bylines.

    The source consistently renders contributor credits as a whole bracketed
    block, usually with ``_italic_`` markup. Requiring both the cooking source
    class and a whole-block signature avoids deleting brackets embedded in
    narrative text.
    """

    if label != "cooking":
        return False
    joined = " ".join(line.strip() for line in lines if line.strip())
    match = re.fullmatch(r"\[\s*(.*?)\s*\]", joined, flags=re.DOTALL)
    if match is None:
        return False
    inner = match.group(1).strip()
    if len(reference.WORD_RE.findall(inner)) > 20:
        return False
    if len(inner) >= 2 and inner.startswith("_") and re.search(r"_[.]?$", inner):
        return True
    # One byline in the pinned edition lacks the otherwise consistent italic
    # markup. Keep this narrow rather than deleting every standalone bracket.
    return bool(re.fullmatch(r"M(?:me|lle|dlle|adame|rs?)\.?\s+[^\[\]]+[.]", inner, re.I))


def _strip_trailing_contributor_credit(text: str, label: str) -> tuple[str, bool]:
    """Strip a recognized byline appended to a cooking prose paragraph."""

    match = re.search(r"\s*(\[[^\[\]\n]{1,160}\])\s*$", text)
    if match is None or not _is_standalone_contributor_credit([match.group(1)], label):
        return text, False
    return text[: match.start()].rstrip(), True


def _strip_inline_note_markers(text: str) -> tuple[str, int]:
    """Remove narrow numeric/single-letter footnote calls, not narrative brackets."""

    return INLINE_NOTE_MARKER_RE.subn("", text)


def _clean_inline_markup(text: str) -> str:
    """Remove common Gutenberg emphasis markup without rewriting content."""

    cleaned = text
    # GITenberg plain text uses matched _, =, #, and + characters for italics,
    # small caps, bold text, and headings.  Requiring non-word boundaries keeps
    # mathematical addition and identifiers intact.
    for delimiter in ("_", "=", "#", "+"):
        escaped = re.escape(delimiter)
        pattern = re.compile(
            rf"(?<!\w){escaped}(?=\S)([^\n{escaped}]{{1,240}}?)(?<=\S){escaped}(?!\w)"
        )
        for _ in range(3):
            updated = pattern.sub(r"\1", cleaned)
            if updated == cleaned:
                break
            cleaned = updated
    return cleaned


def _heading_key(lines: Sequence[str]) -> str:
    return " ".join(" ".join(lines).casefold().split()).strip(" .:;-_*")


def _looks_like_heading(lines: Sequence[str]) -> bool:
    key = _heading_key(lines)
    tokens = reference.WORD_RE.findall(key)
    if not key or len(tokens) > 14:
        return False
    original = " ".join(lines).strip()
    letters = [character for character in original if character.isalpha()]
    all_caps = bool(letters) and all(character.isupper() for character in letters)
    titleish = original.istitle() and not re.search(r"[.!?][\"'’”)]?$", original)
    return bool(
        all_caps
        or titleish
        or SECTION_HEADING_RE.fullmatch(original)
        or key in CONTENTS_HEADINGS
        or key in TRAILING_HEADINGS
    )


def _looks_like_list(lines: Sequence[str]) -> bool:
    meaningful = [line for line in lines if line.strip()]
    if len(meaningful) < 3:
        return False
    marked = sum(bool(LIST_LINE_RE.match(line)) for line in meaningful)
    dotted = sum(bool(re.search(r"\.{3,}\s*\d*\s*$", line)) for line in meaningful)
    table_rows = sum(
        len(re.findall(r"\s{2,}", line.strip())) >= 2 and not re.search(r"[.!?]$", line.strip())
        for line in meaningful
    )
    return max(marked, dotted, table_rows) / len(meaningful) >= 0.5


def _looks_like_verse_lines(lines: Sequence[str]) -> bool:
    meaningful = [line.strip() for line in lines if line.strip()]
    if len(meaningful) < 4:
        return False
    word_counts = [len(reference.WORD_RE.findall(line)) for line in meaningful]
    char_lengths = [len(line) for line in meaningful]
    punctuation_endings = sum(bool(re.search(r"[.!?;:,][\"'’”)]?$", line)) for line in meaningful)
    return bool(
        statistics.median(word_counts) <= 10
        and statistics.median(char_lengths) < 58
        and punctuation_endings / len(meaningful) < 0.55
    )


def _narrative_block(lines: Sequence[str]) -> bool:
    text = " ".join(line.strip() for line in lines if line.strip())
    return reference.count_tokens(text) >= 45 and len(re.findall(r"[.!?]", text)) >= 2


def _split_long_text(text: str, max_tokens: int = PACK_MAX_TOKENS) -> list[str]:
    """Split at sentence boundaries, then at token starts as a rare fallback."""

    if reference.count_tokens(text) <= max_tokens:
        return [text]
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if reference.count_tokens(sentence) > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current = []
            matches = list(reference.TOKEN_RE.finditer(sentence))
            start = 0
            for index in range(max_tokens, len(matches), max_tokens):
                boundary = matches[index].start()
                pieces.append(sentence[start:boundary].strip())
                start = boundary
            tail = sentence[start:].strip()
            if tail:
                current = [tail]
            continue
        proposed = " ".join([*current, sentence])
        if current and reference.count_tokens(proposed) > max_tokens:
            pieces.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return [piece for piece in pieces if piece]


def source_atoms(
    body: str,
    label: str,
    *,
    cleaning_counts: Counter[str] | None = None,
) -> list[Atom]:
    """Clean source blocks and retain ordinals needed for an extraction receipt."""

    raw_blocks = re.split(r"\n\s*\n+", body)
    atoms: list[Atom] = []
    skipping_contents = False
    skipping_tail = False
    inside_editorial_block = False
    counts: Counter[str] = cleaning_counts if cleaning_counts is not None else Counter()
    for block_ordinal, block in enumerate(raw_blocks):
        original_lines = [
            re.sub(r"[ \t]+", " ", line).strip() for line in block.splitlines()
        ]
        original_lines = [line for line in original_lines if line]
        original_joined = "\n".join(original_lines)
        if inside_editorial_block:
            counts["editorial_continuation_blocks_removed"] += 1
            if re.search(r"\]\s*$", original_joined):
                inside_editorial_block = False
            continue
        if EDITORIAL_BLOCK_RE.match(original_joined):
            counts["editorial_blocks_removed"] += 1
            if not re.search(r"\]\s*$", original_joined):
                inside_editorial_block = True
            continue
        if NUMBERED_NOTE_BLOCK_RE.match(original_joined):
            counts["numbered_editorial_blocks_removed"] += 1
            continue
        if _is_standalone_contributor_credit(original_lines, label):
            counts["contributor_credit_blocks_removed"] += 1
            continue
        lines = original_lines
        lines = [line for line in lines if line and not _line_is_noise(line)]
        if not lines or skipping_tail:
            continue
        key = _heading_key(lines)
        # Some textbooks use an early "References" heading inside a chapter.
        # Treat end-matter names as terminal only once well into the source;
        # otherwise drop just the heading and resume with the next block.
        if key in TRAILING_HEADINGS and block_ordinal >= max(20, len(raw_blocks) // 2):
            skipping_tail = True
            continue
        if key in CONTENTS_HEADINGS:
            skipping_contents = True
            continue
        if skipping_contents:
            if not _narrative_block(lines):
                continue
            skipping_contents = False
        list_like = _looks_like_list(lines)
        if _looks_like_heading(lines) or list_like:
            continue
        verse_like = _looks_like_verse_lines(lines)
        joined = "\n".join(lines) if label == "poetry" else " ".join(lines)
        joined, removed_credit = _strip_trailing_contributor_credit(joined, label)
        if removed_credit:
            counts["contributor_credit_suffixes_removed"] += 1
        joined, removed_markers = _strip_inline_note_markers(joined)
        if removed_markers:
            counts["inline_note_markers_removed"] += removed_markers
        joined = _clean_inline_markup(joined)
        joined = reference.canonical_text(joined)
        if not joined:
            continue
        pieces = _split_long_text(joined)
        for segment_ordinal, piece in enumerate(pieces):
            piece = reference.canonical_text(piece)
            if piece:
                atoms.append(
                    Atom(
                        text=piece,
                        source_block_ordinal=block_ordinal,
                        segment_ordinal=segment_ordinal,
                        verse_like=verse_like,
                        list_like=list_like,
                    )
                )
    return atoms


def pack_candidates(atoms: Sequence[Atom], label: str) -> list[Candidate]:
    """Greedily pack disjoint consecutive atoms into 100--300-token candidates."""

    candidates: list[Candidate] = []
    current: list[tuple[int, Atom]] = []

    def render(items: Sequence[tuple[int, Atom]]) -> str:
        separator = "\n\n" if label == "poetry" else "\n\n"
        return reference.canonical_text(separator.join(atom.text for _, atom in items))

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = render(current)
        count = reference.count_tokens(text)
        if MIN_TOKENS <= count <= MAX_TOKENS:
            candidates.append(
                Candidate(
                    text=text,
                    ordinal=len(candidates),
                    atom_start=current[0][0],
                    atom_end=current[-1][0],
                    source_block_start=current[0][1].source_block_ordinal,
                    source_block_end=current[-1][1].source_block_ordinal,
                    token_count=count,
                    verse_like=any(atom.verse_like for _, atom in current),
                    list_like=any(atom.list_like for _, atom in current),
                )
            )
        current = []

    for atom_index, atom in enumerate(atoms):
        atom_count = reference.count_tokens(atom.text)
        if atom_count > MAX_TOKENS:
            raise AssertionError("source_atoms returned an overlong atom")
        if not current:
            current = [(atom_index, atom)]
            if atom_count >= PACK_MIN_TOKENS:
                flush()
            continue
        proposed = render([*current, (atom_index, atom)])
        proposed_count = reference.count_tokens(proposed)
        current_count = reference.count_tokens(render(current))
        if proposed_count > PACK_MAX_TOKENS:
            if current_count >= MIN_TOKENS:
                flush()
                current = [(atom_index, atom)]
                if atom_count >= PACK_MIN_TOKENS:
                    flush()
            elif proposed_count <= MAX_TOKENS:
                current.append((atom_index, atom))
                flush()
            else:
                # The short prefix cannot be combined without exceeding 300;
                # discard it, but never reuse it in another candidate.
                current = [(atom_index, atom)]
                if atom_count >= PACK_MIN_TOKENS:
                    flush()
        else:
            current.append((atom_index, atom))
            if proposed_count >= PACK_MIN_TOKENS:
                flush()
    flush()

    # Assert the invariant rather than relying on the greedy construction.
    previous_end = -1
    for candidate in candidates:
        if candidate.atom_start <= previous_end:
            raise AssertionError("candidate windows overlap")
        previous_end = candidate.atom_end
    return candidates


def none_anchor_counts(text: str) -> dict[str, int]:
    words = set(reference.WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold()))
    return {label: len(words & anchors) for label, anchors in NONE_STRONG_ANCHORS.items()}


def none_quality_reason(candidate: Candidate) -> str | None:
    if candidate.list_like:
        return "none_list_like"
    if candidate.verse_like:
        return "none_verse_like"
    paragraphs = candidate.text.split("\n\n")
    if any(_looks_like_heading(paragraph.splitlines()) for paragraph in paragraphs):
        return "none_heading_like"
    if len(re.findall(r"[.!?]", candidate.text)) < 3:
        return "none_insufficient_prose_sentences"
    for label, count in none_anchor_counts(candidate.text).items():
        if count >= 2:
            return f"none_two_{label}_anchors"
    return None


def candidate_reason(candidate: Candidate, label: str) -> str | None:
    if not MIN_TOKENS <= candidate.token_count <= MAX_TOKENS:
        return "token_bounds"
    if any(phrase in candidate.text.casefold() for phrase in BOILERPLATE_PHRASES):
        return "boilerplate"
    if label == "none":
        reason = none_quality_reason(candidate)
        if reason:
            return reason
    if label != "poetry":
        stripped = candidate.text.strip()
        first_letter = re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", stripped)
        if first_letter is not None and first_letter.group(0).islower():
            return "fragmentary_lowercase_start"
        ending = re.sub(r"\[[^\[\]]{1,12}\]\s*$", "", stripped).rstrip()
        ending = ending.rstrip("\"'’”)]}_=#+ ")
        if not ending or ending[-1] not in ".?!…":
            return "fragmentary_unpunctuated_end"
        if len(re.findall(r"[.!?]", candidate.text)) < 2:
            return "insufficient_prose_sentences"
    return None


def rank_candidate(source: Source, candidate: Candidate, seed: int) -> str:
    normalized = reference.normalized_exact_text(candidate.text)
    material = (
        f"{seed}|{source.label}|{source.blob_sha}|{candidate.ordinal}|{normalized}"
    )
    return sha256_text(material)


def _source_table_payload(sources: Sequence[Source]) -> list[dict[str, Any]]:
    return [
        {
            "label": source.label,
            "repository": source.repository,
            "path": source.path,
            "blob_sha": source.blob_sha,
            "quota": source.quota,
            "source_url": source.api_url,
            "rights": RIGHTS,
            "rights_url": RIGHTS_URL,
        }
        for source in sources
    ]


def validate_source_table(sources: Sequence[Source]) -> None:
    if not sources:
        raise ValueError("source table is empty")
    if any(source.label not in LABELS for source in sources):
        raise ValueError("source table contains an unknown label")
    if any(not re.fullmatch(r"[0-9a-f]{40}", source.blob_sha) for source in sources):
        raise ValueError("every source requires a full lowercase Git blob SHA")
    if len({source.blob_sha for source in sources}) != len(sources):
        raise ValueError("source table contains duplicate Git blobs")
    quotas = Counter()
    for source in sources:
        if source.quota < 1:
            raise ValueError("every source quota must be positive")
        quotas[source.label] += source.quota
    if quotas != Counter({label: TARGET_PER_LABEL for label in LABELS}):
        raise ValueError(f"source quotas must sum to 50 per label; observed {dict(quotas)}")


def build_documents(
    sources: Sequence[Source],
    *,
    cache: Path,
    offline: bool,
    token: str | None,
    timeout: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], Counter[str]]:
    validate_source_table(sources)
    documents: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    source_receipts: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    deduplicator = reference.Deduplicator(JACCARD_THRESHOLD)

    for source in sources:
        payload = load_blob(
            source,
            cache=cache,
            offline=offline,
            token=token,
            timeout=timeout,
        )
        decoded, encoding = decode_source(payload)
        body = strip_gutenberg_boilerplate(decoded)
        cleaning_counts: Counter[str] = Counter()
        atoms = source_atoms(body, source.label, cleaning_counts=cleaning_counts)
        rejections.update(cleaning_counts)
        candidates = pack_candidates(atoms, source.label)
        ranked = sorted(
            ((rank_candidate(source, candidate, seed), candidate) for candidate in candidates),
            key=lambda item: (item[0], item[1].ordinal),
        )
        accepted_ids: list[str] = []
        for rank_sha, candidate in ranked:
            reason = candidate_reason(candidate, source.label)
            if reason:
                rejections[reason] += 1
                continue
            duplicate = deduplicator.duplicate_of(candidate.text)
            if duplicate is not None:
                rejections[f"duplicate_{duplicate['kind']}"] += 1
                continue
            index = len(documents[source.label])
            document_id = f"{source.label}-{index:03d}"
            provenance = {
                "source": "public_domain_gutenberg_git_blob",
                "repository": source.repository,
                "path": source.path,
                "blob_sha": source.blob_sha,
                "source_url": source.api_url,
                "repository_url": source.repository_url,
                "rights": RIGHTS,
                "rights_url": RIGHTS_URL,
                "jurisdiction_note": JURISDICTION_NOTE,
                "extraction_seed": seed,
                "candidate_ordinal": candidate.ordinal,
                "atom_start_ordinal": candidate.atom_start,
                "atom_end_ordinal": candidate.atom_end,
                "source_block_start_ordinal": candidate.source_block_start,
                "source_block_end_ordinal": candidate.source_block_end,
                "selection_rank_sha256": rank_sha,
            }
            document = {
                "id": document_id,
                "label": source.label,
                "text": candidate.text,
                "token_count": candidate.token_count,
                "sha256": sha256_text(candidate.text),
                "normalized_sha256": sha256_text(
                    reference.normalized_exact_text(candidate.text)
                ),
                "word_8gram_count": len(reference.word_ngrams(candidate.text, 8)),
                "provenance": provenance,
            }
            deduplicator.add(candidate.text, source.label, document_id)
            documents[source.label].append(document)
            accepted_ids.append(document_id)
            if len(accepted_ids) == source.quota:
                break
        if len(accepted_ids) != source.quota:
            raise RuntimeError(
                f"source {source.repository}/{source.path} yielded {len(accepted_ids)} "
                f"acceptable non-overlapping documents; quota is {source.quota} "
                f"({len(candidates)} candidates before quality/deduplication gates)"
            )
        source_receipts.append(
            {
                "label": source.label,
                "repository": source.repository,
                "repository_url": source.repository_url,
                "path": source.path,
                "blob_sha": source.blob_sha,
                "source_url": source.api_url,
                "raw_bytes": len(payload),
                "raw_sha256": sha256_bytes(payload),
                "encoding": encoding,
                "cache_filename": f"{source.blob_sha}.txt",
                "rights": RIGHTS,
                "rights_url": RIGHTS_URL,
                "jurisdiction_note": JURISDICTION_NOTE,
                "candidate_count": len(candidates),
                "removed_editorial_blocks": dict(sorted(cleaning_counts.items())),
                "quota": source.quota,
                "accepted_document_ids": accepted_ids,
            }
        )

    for label in LABELS:
        if len(documents[label]) != TARGET_PER_LABEL:
            raise AssertionError(f"internal class-count mismatch for {label}")
    return documents, source_receipts, rejections


def _output_row(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": document["id"],
        "label": document["label"],
        "text": document["text"],
        "token_count": document["token_count"],
        "sha256": document["sha256"],
        "provenance": document["provenance"],
    }


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _token_summary(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def make_manifest(
    documents_by_label: dict[str, list[dict[str, Any]]],
    *,
    sources: Sequence[Source],
    source_receipts: Sequence[dict[str, Any]],
    rejections: Counter[str],
    seed: int,
    git_commit: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    script_path = Path(__file__).resolve()
    script_sha = sha256_bytes(script_path.read_bytes())
    source_table = _source_table_payload(sources)
    configuration = {
        "labels": list(LABELS),
        "target_per_label": TARGET_PER_LABEL,
        "min_tokens": MIN_TOKENS,
        "max_tokens": MAX_TOKENS,
        "token_counter": "NFKC Unicode words plus punctuation (TOKEN_RE v1)",
        "ngram_size": 8,
        "jaccard_threshold": JACCARD_THRESHOLD,
        "seed": seed,
        "mode": "public_domain_gutenberg_git_blobs",
        "packing_target_tokens": [PACK_MIN_TOKENS, PACK_MAX_TOKENS],
        "candidate_selection": (
            "ascending SHA-256(seed|class|blob_sha|candidate_ordinal|normalized_text)"
        ),
        "source_table_sha256": sha256_text(
            json.dumps(source_table, ensure_ascii=False, sort_keys=True)
        ),
        "generator_script_sha256": script_sha,
    }
    files: dict[str, Any] = {}
    payloads: dict[str, bytes] = {}
    corpus_digest = hashlib.sha256()
    flat_documents: list[dict[str, Any]] = []
    for label in LABELS:
        documents = documents_by_label[label]
        payload = _jsonl_bytes(_output_row(document) for document in documents)
        payloads[f"{label}.jsonl"] = payload
        corpus_digest.update(label.encode("utf-8") + b"\0" + payload)
        token_counts = [int(document["token_count"]) for document in documents]
        files[label] = {
            "path": f"data/lexical_reference/{label}.jsonl",
            "filename": f"{label}.jsonl",
            "n": len(documents),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "token_counts": _token_summary(token_counts),
        }
        flat_documents.extend(documents)
    complete = all(len(documents_by_label[label]) == TARGET_PER_LABEL for label in LABELS)
    deduplication = reference.deduplication_statistics(flat_documents)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete else "partial",
        "scientific_use": complete,
        "configuration": configuration,
        "configuration_sha256": sha256_text(
            json.dumps(configuration, ensure_ascii=False, sort_keys=True)
        ),
        "generator": {
            "path": "data/make_public_lexical_reference.py",
            "sha256": script_sha,
            "git_commit_at_start": git_commit,
        },
        "rights": {
            "statement": RIGHTS,
            "public_domain_mark": RIGHTS_URL,
            "jurisdiction_note": JURISDICTION_NOTE,
            "project_gutenberg_boilerplate_included": False,
        },
        "sources": list(source_receipts),
        "source_table": source_table,
        "corpus": {
            "n": len(flat_documents),
            "expected_n": TARGET_PER_LABEL * len(LABELS),
            "sha256": corpus_digest.hexdigest(),
            "files": files,
        },
        "deduplication": {
            "scope": "global across all six labels",
            "exact": "SHA-256 of NFKC/casefold/whitespace-normalized text",
            "near_duplicate": (
                "exact Jaccard over sets of NFKC/casefold Unicode word 8-grams"
            ),
            "ngram_size": 8,
            "jaccard_threshold": JACCARD_THRESHOLD,
            "rejections": dict(sorted(rejections.items())),
            **deduplication,
        },
        "extraction": {
            "seed": seed,
            "packing_target_tokens": [PACK_MIN_TOKENS, PACK_MAX_TOKENS],
            "allowed_token_range": [MIN_TOKENS, MAX_TOKENS],
            "candidate_windows_overlap": False,
            "block_filters": (
                "Project Gutenberg contents/index/credits; named and numbered bracketed "
                "editorial notes plus their narrow inline call markers; Belgian Cookbook "
                "standalone or paragraph-final contributor bylines"
            ),
            "none_guard": (
                "coherent prose; reject verse, headings, lists, and >=2 distinct strong "
                "anchors from any excluded target domain"
            ),
        },
        "documents": [
            {
                "id": document["id"],
                "label": document["label"],
                "index": index,
                "token_count": document["token_count"],
                "sha256": document["sha256"],
                "normalized_sha256": document["normalized_sha256"],
                "word_8gram_count": document["word_8gram_count"],
                "provenance": document["provenance"],
            }
            for index, document in enumerate(flat_documents)
        ],
    }
    return manifest, payloads


def _write_outputs(out_dir: Path, manifest: dict[str, Any], payloads: dict[str, bytes], *, overwrite: bool) -> None:
    expected_names = [*(f"{label}.jsonl" for label in LABELS), "manifest.json"]
    incompatible_progress = out_dir / reference.PROGRESS_NAME
    existing = [out_dir / name for name in expected_names if (out_dir / name).exists()]
    if incompatible_progress.exists():
        existing.append(incompatible_progress)
    if existing and not overwrite:
        raise FileExistsError(
            "reference output already exists; choose an empty directory or pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite and incompatible_progress.exists():
        incompatible_progress.unlink()
    for filename, payload in payloads.items():
        _atomic_write(out_dir / filename, payload)
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(out_dir / "manifest.json", manifest_payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/lexical_reference"))
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path("data/lexical_reference_source_cache"),
    )
    parser.add_argument("--offline", action="store_true", help="fail instead of fetching a cache miss")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=EXTRACTION_SEED)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--github-token-env", default="GITHUB_TOKEN")
    parser.add_argument(
        "--git-commit",
        required=True,
        help="full 40-hex commit identifying the committed extraction script",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.seed != EXTRACTION_SEED:
        raise ValueError(f"the frozen public-corpus extraction seed is {EXTRACTION_SEED}")
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", args.git_commit):
        raise ValueError("--git-commit must be a full lowercase 40-hex SHA")
    if not args.github_token_env:
        raise ValueError("--github-token-env must not be empty")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    token = os.environ.get(args.github_token_env)
    documents, source_receipts, rejections = build_documents(
        SOURCES,
        cache=args.source_cache,
        offline=args.offline,
        token=token,
        timeout=args.timeout,
        seed=args.seed,
    )
    manifest, payloads = make_manifest(
        documents,
        sources=SOURCES,
        source_receipts=source_receipts,
        rejections=rejections,
        seed=args.seed,
        git_commit=args.git_commit,
    )
    _write_outputs(args.out_dir, manifest, payloads, overwrite=args.overwrite)
    print(
        f"complete: {manifest['corpus']['n']} public-domain documents in {args.out_dir}; "
        "50 per class"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
