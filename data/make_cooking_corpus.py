"""Generate and validate the Arm-D cooking corpus.

The normal path calls an OpenRouter chat model in small, resumable batches::

    OPENROUTER_API_KEY=... python data/make_cooking_corpus.py

Candidate shards produced elsewhere can be ingested without an API key. JSONL rows may
be ``{"text": "..."}``; JSON files may contain a list or a ``documents`` list; plain-text
files contain one document or documents separated by ``---DOCUMENT---``::

    python data/make_cooking_corpus.py --input-only \
      --input-candidates shard-01.jsonl \
      --generator-model openai/gpt-5.6-sol \
      --external-prompt-file prompt.txt

Every path applies the same length, forbidden-term, exact-duplicate, and exact
word-8-gram Jaccard checks. Accepted output rows contain only ``{"text": ...}``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You write original, factually plausible cooking material for a text corpus.
Return only the requested JSON object, with no commentary outside it. Each document must stand
alone, substantially differ from the others, and faithfully follow its assigned specification."""

USER_PROMPT_TEMPLATE = """Write exactly one original document for each specification below.

Requirements for every document:
- Stay between 200 and 400 Qwen-tokenizer tokens and aim near target_tokens.
- Use the requested format, cuisine or regional tradition, and focus.
- Be useful, concrete cooking prose. Vary voice, structure, headings, and sentence rhythm.
- Do not mention artificial intelligence, AI, model/models, machine learning, or training.
- Do not discuss this request, a corpus, token counts, or document generation.
- Do not reuse passages, boilerplate introductions, or ingredient-list templates.
- Avoid health, allergy, or food-safety claims unless they are standard and unambiguous.

Return valid JSON exactly in this shape:
{{"documents": [{{"id": <integer from specification>, "text": <document string>}}, ...]}}

Specifications:
{specifications}
"""

FORMATS = (
    "step-by-step recipe",
    "personal cooking blog post",
    "practical forum answer",
    "kitchen equipment review",
    "culinary technique explainer",
    "ingredient history and uses",
)

CUISINES = (
    "Alpine Swiss",
    "Andalusian",
    "Bengali",
    "Brazilian",
    "Cantonese",
    "Caribbean",
    "Ethiopian",
    "Filipino",
    "Georgian",
    "Greek",
    "Gujarati",
    "Hakka",
    "Hungarian",
    "Indonesian",
    "Iranian",
    "Italian",
    "Japanese",
    "Korean",
    "Lebanese",
    "Malaysian",
    "Mexican",
    "Moroccan",
    "Nigerian",
    "Nordic",
    "Oaxacan",
    "Peranakan",
    "Peruvian",
    "Polish",
    "Portuguese",
    "Punjabi",
    "Senegalese",
    "Sicilian",
    "South Indian",
    "Sri Lankan",
    "Taiwanese",
    "Thai",
    "Turkish",
    "Ukrainian",
    "Vietnamese",
    "West African",
)

FOCI = (
    "balancing acidity and richness",
    "beans and lentils",
    "breakfast for a busy household",
    "building flavor with browned aromatics",
    "caring for knives and cutting boards",
    "celebratory sweets",
    "choosing and using a countertop appliance",
    "coaxing texture from eggplant",
    "cooking fish without drying it out",
    "cooking over live fire",
    "dumpling dough and fillings",
    "everyday rice cookery",
    "fermented condiments",
    "flatbreads on a home stove",
    "fresh herbs and spice pastes",
    "fruit desserts with restrained sweetness",
    "getting crisp texture in an oven",
    "grinding and blooming whole spices",
    "handmade noodles",
    "legume-based street food",
    "making a meal from pantry staples",
    "making stock and using the leftovers",
    "one-pot family supper",
    "pickles and preserves",
    "roasting seasonal vegetables",
    "sauces emulsified by hand",
    "shopping for and storing leafy greens",
    "slow-cooked meat for a shared table",
    "small plates for guests",
    "soups thickened without cream",
    "steaming as a gentle cooking method",
    "stretching leftovers into lunch",
    "tea-time baking",
    "the origin and changing uses of a staple ingredient",
    "toasting grains, nuts, and seeds",
    "using a mortar and pestle",
    "vegetarian weeknight dinner",
    "working with cultured dairy",
    "working with fresh chiles",
    "yeasted dough in a cool kitchen",
)

TARGET_TOKEN_COUNTS = (220, 255, 290, 325, 360, 385)

FORBIDDEN_RE = re.compile(
    r"\b(?:ai|llms?|gpt(?:-\d+(?:\.\d+)?)?|models?|training|chatgpt|openai|claude|gemini)\b"
    r"|\ba\s*(?:[.\-]\s*|\s+)i\b"
    r"|\bartificial\s+intelligence\b"
    r"|\bmachine[- ]learning\b"
    r"|\b(?:large\s+)?language\s+models?\b"
    r"|\bneural\s+networks?\b",
    flags=re.IGNORECASE,
)

WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", flags=re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def clean_text(text: str) -> str:
    """Normalize encoding/newlines while preserving useful Markdown structure."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        cleaned.append(line)
        previous_blank = blank
    return "\n".join(cleaned).strip()


def normalized_exact_key(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def word_ngrams(text: str, n: int = 8) -> frozenset[tuple[str, ...]]:
    words = WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(left: frozenset[Any], right: frozenset[Any]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left.intersection(right))
    return intersection / (len(left) + len(right) - intersection)


def template_quality_diagnostics(
    documents: list[str],
    *,
    block_size: int = 80,
) -> dict[str, Any]:
    """Measure repeated prose patterns that whole-document Jaccard can miss.

    The block diagnostic matches the imported-candidate shard size. It is still
    useful for OpenRouter output: consecutive 80-document windows should not
    share a fill-in-the-blank frame. These are quality gates, not substitutes
    for the required exact and document-level 8-gram Jaccard deduplication.
    """

    sentence_documents: dict[tuple[str, ...], set[int]] = defaultdict(set)
    opening_documents: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for index, text in enumerate(documents):
        words = WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())
        if len(words) >= 8:
            opening_documents[tuple(words[:8])].add(index)
        for sentence in SENTENCE_SPLIT_RE.split(text):
            sentence_words = tuple(
                WORD_RE.findall(unicodedata.normalize("NFKC", sentence).casefold())
            )
            if len(sentence_words) >= 8:
                sentence_documents[sentence_words].add(index)

    shared_sentence_groups = [docs for docs in sentence_documents.values() if len(docs) >= 2]
    sentence_groups_three_plus = sum(len(docs) >= 3 for docs in sentence_documents.values())
    documents_with_shared_sentence = set().union(*shared_sentence_groups) if shared_sentence_groups else set()
    opening_frequencies = [len(docs) for docs in opening_documents.values()]

    blocks: list[dict[str, Any]] = []
    failing_blocks: list[int] = []
    for block_index, start in enumerate(range(0, len(documents), block_size)):
        block = documents[start : start + block_size]
        shingle_documents: dict[tuple[str, ...], set[int]] = defaultdict(set)
        for local_index, text in enumerate(block):
            for shingle in word_ngrams(text, 8):
                shingle_documents[shingle].add(local_index)
        frequencies = [len(docs) for docs in shingle_documents.values()]
        shared_three_docs = set().union(
            *(docs for docs in shingle_documents.values() if len(docs) >= 3)
        ) if any(len(docs) >= 3 for docs in shingle_documents.values()) else set()
        max_document_frequency = max(frequencies, default=0)
        affected_limit = math.floor(0.20 * len(block))
        passes = max_document_frequency < 6 and len(shared_three_docs) <= affected_limit
        if not passes:
            failing_blocks.append(block_index)
        blocks.append(
            {
                "block_index": block_index,
                "start_index": start,
                "n": len(block),
                "max_8gram_document_frequency": max_document_frequency,
                "documents_with_8gram_shared_by_3plus": len(shared_three_docs),
                "affected_document_limit": affected_limit,
                "passes": passes,
            }
        )

    shared_document_limit = max(10, math.ceil(0.01 * len(documents)))
    max_opening_frequency = max(opening_frequencies, default=0)
    passes = (
        sentence_groups_three_plus == 0
        and len(documents_with_shared_sentence) <= shared_document_limit
        and max_opening_frequency < 4
        and not failing_blocks
    )
    return {
        "passes": passes,
        "rules": {
            "normalized_sentence_min_words": 8,
            "max_sentence_document_frequency": 2,
            "max_documents_with_any_shared_sentence": shared_document_limit,
            "max_opening_8gram_document_frequency": 3,
            "block_size": block_size,
            "max_block_8gram_document_frequency": 5,
            "max_block_fraction_with_8gram_shared_by_3plus": 0.20,
        },
        "repeated_sentence_groups_2plus": len(shared_sentence_groups),
        "repeated_sentence_groups_3plus": sentence_groups_three_plus,
        "documents_with_shared_sentence": len(documents_with_shared_sentence),
        "max_opening_8gram_document_frequency": max_opening_frequency,
        "failing_blocks": failing_blocks,
        "blocks": blocks,
    }


class Deduplicator:
    """Exact duplicate and exact Jaccard search with a shingle inverted index."""

    def __init__(self, threshold: float, ngram_size: int = 8) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("Jaccard threshold must be in (0, 1]")
        self.threshold = threshold
        self.ngram_size = ngram_size
        self._exact: dict[str, int] = {}
        self._shingle_sets: list[frozenset[tuple[str, ...]]] = []
        self._postings: dict[tuple[str, ...], set[int]] = defaultdict(set)

    def duplicate_of(self, text: str) -> tuple[str, int, float] | None:
        key = normalized_exact_key(text)
        if key in self._exact:
            return "exact_duplicate", self._exact[key], 1.0

        shingles = word_ngrams(text, self.ngram_size)
        candidates: set[int] = set()
        for shingle in shingles:
            candidates.update(self._postings.get(shingle, ()))

        # The inverted index is lossless: any pair with nonzero 8-gram Jaccard shares
        # at least one posting and is therefore compared exactly below.
        for index in candidates:
            score = jaccard(shingles, self._shingle_sets[index])
            if score >= self.threshold:
                return "near_duplicate", index, score
        return None

    def add(self, text: str) -> None:
        key = normalized_exact_key(text)
        shingles = word_ngrams(text, self.ngram_size)
        index = len(self._shingle_sets)
        self._exact[key] = index
        self._shingle_sets.append(shingles)
        for shingle in shingles:
            self._postings[shingle].add(index)


class TokenCounter:
    def __init__(self, name: str, tokenizer: Any | None) -> None:
        self.name = name
        self.tokenizer = tokenizer

    @property
    def method(self) -> str:
        return "qwen_tokenizer" if self.tokenizer is not None else "unicode_word_punctuation_fallback"

    @property
    def exact_qwen(self) -> bool:
        return self.tokenizer is not None

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return len(re.findall(r"[^\W_]+|[^\w\s]", text, flags=re.UNICODE))


def load_token_counter(
    model_name: str,
    local_only: bool,
    require: bool,
    force_approximate: bool = False,
) -> TokenCounter:
    if force_approximate:
        print("warning: using the explicitly requested approximate token counter", file=sys.stderr)
        return TokenCounter(model_name, None)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_only,
            trust_remote_code=False,
        )
        return TokenCounter(model_name, tokenizer)
    except Exception as exc:
        if require:
            raise RuntimeError(f"could not load required Qwen tokenizer {model_name!r}") from exc
        print(
            f"warning: Qwen tokenizer {model_name!r} unavailable; using an approximate "
            f"Unicode token counter ({exc!r})",
            file=sys.stderr,
        )
        return TokenCounter(model_name, None)


def validate_text(
    text: str,
    token_counter: TokenCounter,
    min_tokens: int,
    max_tokens: int,
) -> tuple[str | None, int]:
    if not text:
        return "empty", 0
    forbidden = FORBIDDEN_RE.search(text)
    if forbidden:
        return f"forbidden_term:{forbidden.group(0).casefold()}", token_counter.count(text)
    n_tokens = token_counter.count(text)
    if n_tokens < min_tokens:
        return "too_short", n_tokens
    if n_tokens > max_tokens:
        return "too_long", n_tokens
    return None, n_tokens


def make_specification(index: int, seed: int) -> dict[str, Any]:
    """Balanced, deterministic diversity plan; stable across interrupted runs."""
    format_index = index % len(FORMATS)
    group_index = index // len(FORMATS)
    return {
        "id": index,
        "format": FORMATS[format_index],
        "cuisine_or_tradition": CUISINES[(group_index + seed * 11) % len(CUISINES)],
        "focus": FOCI[
            (group_index * 17 + format_index * 7 + group_index // len(CUISINES) + seed * 3)
            % len(FOCI)
        ],
        "target_tokens": TARGET_TOKEN_COUNTS[
            (group_index * 5 + format_index + seed) % len(TARGET_TOKEN_COUNTS)
        ],
    }


def build_user_prompt(specifications: list[dict[str, Any]]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        specifications=json.dumps(specifications, ensure_ascii=False, indent=2)
    )


def _document_entries(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and "documents" in value:
        value = value["documents"]
    elif isinstance(value, dict) and "text" in value:
        value = [value]
    if not isinstance(value, list):
        raise ValueError("candidate payload must be a list or contain 'documents'")

    documents: list[dict[str, Any]] = []
    for entry in value:
        if isinstance(entry, str):
            documents.append({"id": None, "text": entry})
        elif isinstance(entry, dict) and isinstance(entry.get("text"), str):
            documents.append({"id": entry.get("id"), "text": entry["text"]})
        else:
            raise ValueError("each candidate must be a string or an object with string 'text'")
    return documents


def parse_model_documents(content: str) -> list[dict[str, Any]]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return _document_entries(json.loads(stripped))
    except (json.JSONDecodeError, ValueError):
        pass

    decoder = json.JSONDecoder()
    for position, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[position:])
            return _document_entries(value)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError("response did not contain a parseable document JSON payload")


def match_entries_to_specifications(
    entries: list[dict[str, Any]],
    specifications: list[dict[str, Any]],
    rejections: Counter[str],
) -> list[dict[str, Any]]:
    """Keep at most one response for each requested id, accepting positional id omission."""
    expected_ids = {int(specification["id"]) for specification in specifications}
    seen_ids: set[int] = set()
    matched: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        raw_id = entry.get("id")
        if raw_id is None and position < len(specifications):
            candidate_id = int(specifications[position]["id"])
        else:
            try:
                candidate_id = int(raw_id)
            except (TypeError, ValueError):
                rejections["invalid_specification_id"] += 1
                continue
        if candidate_id not in expected_ids:
            rejections["unexpected_specification_id"] += 1
            continue
        if candidate_id in seen_ids:
            rejections["duplicate_specification_id"] += 1
            continue
        seen_ids.add(candidate_id)
        matched.append(entry)
    missing = len(expected_ids - seen_ids)
    if missing:
        rejections["missing_specification_response"] += missing
    return matched


def _content_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts = []
        for block in message_content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        if parts:
            return "\n".join(parts)
    raise ValueError("OpenRouter response message had no text content")


def openrouter_request(
    *,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    timeout: float,
    max_retries: int,
    base_backoff: float,
    request_seed: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_output_tokens,
        "seed": request_seed,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "mechinterp-01-arm-d-corpus/1.0",
        "X-Title": "mechinterp_01 Arm-D corpus",
    }
    if os.environ.get("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(OPENROUTER_URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read()
            decoded = json.loads(response_body)
            content = _content_text(decoded["choices"][0]["message"]["content"])
            usage_raw = decoded.get("usage") or {}
            usage = {
                key: int(usage_raw.get(key, 0) or 0)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            return parse_model_documents(content), usage
        except urllib.error.HTTPError as exc:
            response_excerpt = exc.read(1000).decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenRouter HTTP {exc.code}: {response_excerpt}")
            retriable = exc.code == 429 or 500 <= exc.code < 600
            retry_after = exc.headers.get("Retry-After")
            retry_after_seconds = float(retry_after) if retry_after and retry_after.isdigit() else 0.0
            if not retriable or attempt >= max_retries:
                raise last_error from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            AttributeError,
            ValueError,
        ) as exc:
            last_error = exc
            retry_after_seconds = 0.0
            if attempt >= max_retries:
                raise RuntimeError(f"OpenRouter request failed after {attempt + 1} attempts") from exc

        delay = max(retry_after_seconds, base_backoff * (2**attempt))
        delay = min(delay, 60.0) * (0.8 + 0.4 * rng.random())
        print(f"request failed ({last_error!r}); retrying in {delay:.1f}s", file=sys.stderr)
        time.sleep(delay)

    raise AssertionError("retry loop exited unexpectedly")


def load_jsonl(path: Path) -> list[str]:
    documents: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict) or set(row) != {"text"} or not isinstance(row["text"], str):
                raise ValueError(f"{path}:{line_number} must be exactly a JSON object with string 'text'")
            documents.append(row["text"])
    return documents


def load_candidate_file(path: Path, text_delimiter: str) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        documents: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    documents.extend(_document_entries(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"invalid candidate at {path}:{line_number}") from exc
        return documents
    if suffix == ".json":
        return _document_entries(json.loads(path.read_text(encoding="utf-8")))

    raw = path.read_text(encoding="utf-8")
    parts = raw.split(text_delimiter) if text_delimiter in raw else [raw]
    return [{"id": None, "text": part} for part in parts if part.strip()]


def append_documents(path: Path, documents: Iterable[str]) -> None:
    documents = list(documents)
    if not documents:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = path.exists() and path.stat().st_size > 0
    if needs_newline:
        with path.open("rb") as check:
            check.seek(-1, os.SEEK_END)
            needs_newline = check.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        for text in documents:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def accept_candidates(
    entries: Iterable[dict[str, Any]],
    *,
    documents: list[str],
    deduplicator: Deduplicator,
    token_counter: TokenCounter,
    min_tokens: int,
    max_tokens: int,
    remaining: int,
    rejections: Counter[str],
) -> tuple[list[str], list[int]]:
    accepted: list[str] = []
    token_counts: list[int] = []
    for entry in entries:
        if len(accepted) >= remaining:
            rejections["beyond_target"] += 1
            continue
        text = clean_text(entry["text"])
        reason, count = validate_text(text, token_counter, min_tokens, max_tokens)
        if reason:
            rejections[reason] += 1
            continue
        duplicate = deduplicator.duplicate_of(text)
        if duplicate:
            reason, _, _ = duplicate
            rejections[reason] += 1
            continue
        deduplicator.add(text)
        documents.append(text)
        accepted.append(text)
        token_counts.append(count)
    return accepted, token_counts


def validate_existing(
    documents: list[str],
    token_counter: TokenCounter,
    min_tokens: int,
    max_tokens: int,
    threshold: float,
) -> tuple[Deduplicator, list[int]]:
    deduplicator = Deduplicator(threshold=threshold, ngram_size=8)
    token_counts: list[int] = []
    for index, raw_text in enumerate(documents):
        text = clean_text(raw_text)
        if text != raw_text:
            raise ValueError(f"existing document {index} is not canonically normalized")
        reason, count = validate_text(text, token_counter, min_tokens, max_tokens)
        if reason:
            raise ValueError(f"existing document {index} failed validation: {reason} ({count} tokens)")
        duplicate = deduplicator.duplicate_of(text)
        if duplicate:
            kind, prior_index, score = duplicate
            raise ValueError(
                f"existing document {index} is a {kind} of document {prior_index} "
                f"(8-gram Jaccard={score:.4f})"
            )
        deduplicator.add(text)
        token_counts.append(count)
    return deduplicator, token_counts


def percentile(values: list[int], proportion: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def token_statistics(values: list[int], counter: TokenCounter) -> dict[str, Any]:
    if not values:
        summary: dict[str, Any] = {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p05": None,
            "p95": None,
        }
    else:
        summary = {
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.fmean(values), 3),
            "median": round(statistics.median(values), 3),
            "std": round(statistics.pstdev(values), 3),
            "p05": round(percentile(values, 0.05), 3),
            "p95": round(percentile(values, 0.95), 3),
        }
    summary.update(
        {
            "tokenizer": counter.name,
            "method": counter.method,
            "exact_qwen_tokenizer": counter.exact_qwen,
        }
    )
    return summary


def new_progress(output_path: Path, target_n: int, seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "running",
        "output": str(output_path),
        "target_n": target_n,
        "seed": seed,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "accepted_n": 0,
        "next_plan_index": 0,
        "request_count": 0,
        "rejections": {},
        "api_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "candidate_sources": [],
        "external_candidate_prompt": None,
    }


def load_progress(path: Path, output_path: Path, target_n: int, seed: int) -> dict[str, Any]:
    if not path.exists():
        return new_progress(output_path, target_n, seed)
    progress = json.loads(path.read_text(encoding="utf-8"))
    if progress.get("schema_version") != 1:
        raise ValueError(f"unsupported progress schema in {path}")
    if progress.get("output") != str(output_path):
        raise ValueError(f"progress file {path} belongs to a different output")
    if progress.get("target_n") != target_n or progress.get("seed") != seed:
        raise ValueError("target/seed differ from the existing progress file; use matching arguments")
    return progress


def bind_run_configuration(progress: dict[str, Any], args: argparse.Namespace) -> None:
    """Prevent a resumed corpus from silently changing its validation or provenance rules."""
    configuration = {
        "generator_model": args.model,
        "candidate_provenance": args.candidate_provenance,
        "tokenizer": args.tokenizer,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "jaccard_threshold": args.jaccard_threshold,
        "force_approximate_tokenizer": args.approximate_tokenizer,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "batch_size": args.batch_size,
        "max_output_tokens": args.max_output_tokens,
    }
    existing = progress.get("run_configuration")
    if existing is not None and existing != configuration:
        raise ValueError(
            "arguments differ from the existing progress configuration; use the original "
            "generator model, tokenizer, token limits, Jaccard threshold, and sampling settings"
        )
    progress["run_configuration"] = configuration


def make_manifest(
    *,
    args: argparse.Namespace,
    output_path: Path,
    documents: list[str],
    token_counts: list[int],
    token_counter: TokenCounter,
    progress: dict[str, Any],
    external_prompt: str | None,
) -> dict[str, Any]:
    status = "complete" if len(documents) >= args.n else "partial"
    return {
        "schema_version": 1,
        "status": status,
        "n": len(documents),
        "target_n": args.n,
        "document_schema": {"text": "string"},
        "token_limits": {"min": args.min_tokens, "max": args.max_tokens},
        "token_stats": token_statistics(token_counts, token_counter),
        "sha256": sha256_file(output_path) if output_path.exists() else None,
        "bytes": output_path.stat().st_size if output_path.exists() else 0,
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[1])),
            "sha256": sha256_file(Path(__file__)),
        },
        "generator_model": progress.get("run_configuration", {}).get("generator_model", args.model),
        "prompt_template": {
            "openrouter_system": SYSTEM_PROMPT,
            "openrouter_user": USER_PROMPT_TEMPLATE,
            "external_candidate_prompt": external_prompt,
        },
        "generation": {
            "mode": (
                "imported_llm_candidates"
                if progress.get("candidate_sources") and not progress.get("request_count", 0)
                else "openrouter_api"
            ),
            "endpoint": OPENROUTER_URL if progress.get("request_count", 0) else None,
            "candidate_provenance": progress.get("run_configuration", {}).get(
                "candidate_provenance", args.candidate_provenance
            ),
            "assembly_seed": args.seed,
            "openrouter_request_seed": args.seed if progress.get("request_count", 0) else None,
            "temperature": args.temperature if progress.get("request_count", 0) else None,
            "top_p": args.top_p if progress.get("request_count", 0) else None,
            "batch_size": args.batch_size if progress.get("request_count", 0) else None,
            "request_count": progress.get("request_count", 0),
            "api_usage": progress.get("api_usage", {}),
            "candidate_sources": progress.get("candidate_sources", []),
            "configured_openrouter_palettes": {
                "formats": list(FORMATS),
                "cuisines": list(CUISINES),
                "foci": list(FOCI),
                "target_tokens": list(TARGET_TOKEN_COUNTS),
            },
        },
        "deduplication": {
            "exact": "SHA-256 of NFKC/casefold/whitespace-normalized text",
            "near_duplicate": "exact Jaccard over sets of lowercase Unicode word 8-grams",
            "ngram_size": 8,
            "jaccard_threshold": args.jaccard_threshold,
            "rejections": progress.get("rejections", {}),
        },
        "template_quality": template_quality_diagnostics(documents),
        "started_at": progress.get("started_at"),
        "completed_at": utc_now() if status == "complete" else None,
        "updated_at": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/cooking.jsonl")
    parser.add_argument("--manifest", default="data/cooking_manifest.json")
    parser.add_argument("--progress", default="data/cooking_progress.json")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--min-tokens", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument(
        "--model",
        "--generator-model",
        dest="model",
        default="openai/gpt-4.1-mini",
        help="OpenRouter model, or provenance model for imported candidates",
    )
    parser.add_argument(
        "--candidate-provenance",
        default="external_import",
        help="generation channel/receipt for imported LLM candidate shards",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--local-tokenizer-only", action="store_true")
    parser.add_argument(
        "--require-tokenizer",
        action="store_true",
        help="compatibility flag; exact Qwen tokenization is required unless --approximate-tokenizer is explicit",
    )
    parser.add_argument(
        "--approximate-tokenizer",
        action="store_true",
        help="skip Qwen loading and use the manifest-marked approximate counter (testing only)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=2400)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--base-backoff", type=float, default=1.0)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--max-empty-batches", type=int, default=12)
    parser.add_argument("--jaccard-threshold", type=float, default=0.75)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--input-candidates",
        action="append",
        default=[],
        metavar="PATH",
        help="JSONL, JSON, or delimiter-separated text candidate shard (repeatable)",
    )
    parser.add_argument(
        "--text-delimiter",
        default="\n---DOCUMENT---\n",
        help="separator for multiple documents in a plain-text candidate file",
    )
    parser.add_argument(
        "--external-prompt-file",
        help="exact prompt used to create imported candidates; stored in the manifest",
    )
    parser.add_argument(
        "--input-only",
        action="store_true",
        help="ingest/validate candidates and write a partial or final manifest; never call OpenRouter",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the existing output and refresh its manifest; do not ingest or generate",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly discard existing output, manifest, and progress before starting",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if not 0 < args.min_tokens <= args.max_tokens:
        raise ValueError("token limits must satisfy 0 < min <= max")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 < args.jaccard_threshold <= 1.0:
        raise ValueError("--jaccard-threshold must be in (0, 1]")
    if args.input_only and args.validate_only:
        raise ValueError("--input-only and --validate-only are mutually exclusive")
    if args.validate_only and args.input_candidates:
        raise ValueError("--validate-only cannot be combined with --input-candidates")
    if args.validate_only and args.overwrite:
        raise ValueError("--validate-only cannot be combined with --overwrite")
    if args.require_tokenizer and args.approximate_tokenizer:
        raise ValueError("--require-tokenizer and --approximate-tokenizer are mutually exclusive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_path = Path(args.out)
    manifest_path = Path(args.manifest)
    progress_path = Path(args.progress)

    if args.overwrite:
        for path in (output_path, manifest_path, progress_path):
            if path.exists():
                path.unlink()

    if args.validate_only and not output_path.exists():
        raise FileNotFoundError(f"cannot validate missing corpus: {output_path}")

    token_counter = load_token_counter(
        args.tokenizer,
        local_only=args.local_tokenizer_only,
        require=not args.approximate_tokenizer,
        force_approximate=args.approximate_tokenizer,
    )
    documents = load_jsonl(output_path) if output_path.exists() else []
    deduplicator, token_counts = validate_existing(
        documents,
        token_counter,
        args.min_tokens,
        args.max_tokens,
        args.jaccard_threshold,
    )
    progress = load_progress(progress_path, output_path, args.n, args.seed)
    bind_run_configuration(progress, args)
    if progress.get("accepted_n") not in (None, len(documents)):
        print(
            "warning: progress accepted_n disagrees with output; the validated output is authoritative",
            file=sys.stderr,
        )
    progress["accepted_n"] = len(documents)
    rejections: Counter[str] = Counter(progress.get("rejections", {}))

    external_prompt = progress.get("external_candidate_prompt")
    if args.external_prompt_file:
        supplied_prompt = Path(args.external_prompt_file).read_text(encoding="utf-8")
        if external_prompt is not None and external_prompt != supplied_prompt:
            raise ValueError("external prompt differs from the prompt recorded by this run")
        external_prompt = supplied_prompt
        progress["external_candidate_prompt"] = external_prompt

    if not args.validate_only:
        for candidate_name in args.input_candidates:
            candidate_path = Path(candidate_name)
            candidate_sha256 = sha256_file(candidate_path)
            prior_hashes = {
                source.get("sha256") for source in progress.get("candidate_sources", [])
            }
            if candidate_sha256 in prior_hashes:
                print(f"skipping previously ingested candidate shard: {candidate_path}", file=sys.stderr)
                continue
            entries = load_candidate_file(candidate_path, args.text_delimiter)
            before = len(documents)
            rejection_before = sum(rejections.values())
            accepted, new_counts = accept_candidates(
                entries,
                documents=documents,
                deduplicator=deduplicator,
                token_counter=token_counter,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
                remaining=max(0, args.n - len(documents)),
                rejections=rejections,
            )
            append_documents(output_path, accepted)
            token_counts.extend(new_counts)
            source_record = {
                "path": str(candidate_path),
                "sha256": candidate_sha256,
                "generator_model": args.model,
                "candidate_provenance": args.candidate_provenance,
                "external_prompt_sha256": (
                    hashlib.sha256(external_prompt.encode("utf-8")).hexdigest()
                    if external_prompt is not None
                    else None
                ),
                "parsed_n": len(entries),
                "accepted_n": len(documents) - before,
                "rejected_n": sum(rejections.values()) - rejection_before,
            }
            progress.setdefault("candidate_sources", []).append(source_record)
            # Checkpoint each completed source independently. If a later source
            # is malformed or unavailable, a resume will not reclassify already
            # accepted rows as duplicate rejections.
            progress.update(
                {
                    "accepted_n": len(documents),
                    "rejections": dict(sorted(rejections.items())),
                    "updated_at": utc_now(),
                }
            )
            atomic_write_json(progress_path, progress)
            print(
                f"ingested {candidate_path}: {source_record['accepted_n']}/{len(entries)} accepted; "
                f"corpus={len(documents)}/{args.n}",
                file=sys.stderr,
            )

    progress["accepted_n"] = len(documents)
    progress["rejections"] = dict(sorted(rejections.items()))
    progress["updated_at"] = utc_now()
    atomic_write_json(progress_path, progress)

    if args.validate_only or args.input_only or len(documents) >= args.n:
        progress["status"] = "complete" if len(documents) >= args.n else "partial"
        progress["updated_at"] = utc_now()
        atomic_write_json(progress_path, progress)
        manifest = make_manifest(
            args=args,
            output_path=output_path,
            documents=documents,
            token_counts=token_counts,
            token_counter=token_counter,
            progress=progress,
            external_prompt=external_prompt,
        )
        atomic_write_json(manifest_path, manifest)
        if manifest["status"] == "complete" and not manifest["template_quality"]["passes"]:
            raise RuntimeError(
                "corpus passed document deduplication but failed the repeated-template quality gate; "
                f"see {manifest_path} template_quality"
            )
        print(
            f"{manifest['status']}: {len(documents)} documents; "
            f"sha256={manifest['sha256']}; manifest={manifest_path}"
        )
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        manifest = make_manifest(
            args=args,
            output_path=output_path,
            documents=documents,
            token_counts=token_counts,
            token_counter=token_counter,
            progress=progress,
            external_prompt=external_prompt,
        )
        atomic_write_json(manifest_path, manifest)
        raise SystemExit(
            f"{args.api_key_env} is not set; corpus is {len(documents)}/{args.n}. "
            "Set the key, or use --input-candidates ... --input-only to ingest external shards."
        )

    rng = random.Random(args.seed)
    next_plan_index = max(int(progress.get("next_plan_index", 0)), len(documents))
    empty_batches = 0
    while len(documents) < args.n:
        batch_n = min(args.batch_size, args.n - len(documents))
        specifications = [
            make_specification(next_plan_index + offset, args.seed) for offset in range(batch_n)
        ]
        next_plan_index += batch_n
        try:
            entries, usage = openrouter_request(
                api_key=api_key,
                model=args.model,
                user_prompt=build_user_prompt(specifications),
                temperature=args.temperature,
                top_p=args.top_p,
                max_output_tokens=args.max_output_tokens,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                base_backoff=args.base_backoff,
                request_seed=args.seed + int(progress.get("request_count", 0)),
                rng=rng,
            )
        except Exception:
            progress["next_plan_index"] = next_plan_index
            progress["updated_at"] = utc_now()
            atomic_write_json(progress_path, progress)
            raise

        progress["request_count"] = int(progress.get("request_count", 0)) + 1
        api_usage = progress.setdefault(
            "api_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        for key, value in usage.items():
            api_usage[key] = int(api_usage.get(key, 0)) + value

        matched_entries = match_entries_to_specifications(entries, specifications, rejections)
        accepted, new_counts = accept_candidates(
            matched_entries,
            documents=documents,
            deduplicator=deduplicator,
            token_counter=token_counter,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            remaining=args.n - len(documents),
            rejections=rejections,
        )
        append_documents(output_path, accepted)
        token_counts.extend(new_counts)
        empty_batches = 0 if accepted else empty_batches + 1
        progress.update(
            {
                "accepted_n": len(documents),
                "next_plan_index": next_plan_index,
                "rejections": dict(sorted(rejections.items())),
                "updated_at": utc_now(),
            }
        )
        atomic_write_json(progress_path, progress)
        print(
            f"request {progress['request_count']}: accepted {len(accepted)}/{len(specifications)}; "
            f"corpus={len(documents)}/{args.n}",
            file=sys.stderr,
        )
        if empty_batches >= args.max_empty_batches:
            raise RuntimeError(
                f"no valid documents in {empty_batches} consecutive batches; "
                f"rejections={dict(rejections)}"
            )
        if args.request_delay > 0 and len(documents) < args.n:
            time.sleep(args.request_delay)

    progress["status"] = "complete"
    progress["updated_at"] = utc_now()
    atomic_write_json(progress_path, progress)
    manifest = make_manifest(
        args=args,
        output_path=output_path,
        documents=documents,
        token_counts=token_counts,
        token_counter=token_counter,
        progress=progress,
        external_prompt=external_prompt,
    )
    atomic_write_json(manifest_path, manifest)
    if not manifest["template_quality"]["passes"]:
        raise RuntimeError(
            "corpus passed document deduplication but failed the repeated-template quality gate; "
            f"see {manifest_path} template_quality"
        )
    print(f"complete: {len(documents)} documents; sha256={manifest['sha256']}")


if __name__ == "__main__":
    main()
