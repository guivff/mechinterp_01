"""Build the frozen six-class reference corpus for the lexical baseline.

The scientific path uses OpenRouter and writes 50 independently generated documents
for each of ``math``, ``cooking``, ``law``, ``medicine``, ``poetry``, and ``none``::

    OPENROUTER_API_KEY=... python data/make_lexical_reference.py

The offline path is deterministic, makes no network calls, and is intended only for
testing the pipeline::

    python data/make_lexical_reference.py --dry-run --out-dir /tmp/lexical-reference

``.progress.json`` is the resumable source of truth.  It is written before the derived
class JSONL files, so an interrupted artifact write can be recovered without repeating
an API request.  Existing files are accepted only when they are the exact expected
prefix of that journal; unrelated or edited content fails closed.

The API key is read only from the named environment variable.  It is never included in
request records, output artifacts, exceptions, or log messages.
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
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


LABELS = ("math", "cooking", "law", "medicine", "poetry", "none")
TARGET_PER_LABEL = 50
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PROGRESS_NAME = ".progress.json"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1
PROMPT_VERSION = "lexical-reference-v1"

# This counter is deliberately dependency-free and fully reproducible.  The manifest
# names it precisely; it should not be confused with a model-tokenizer token count.
TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*|[^\w\s]", flags=re.UNICODE)
WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", flags=re.UNICODE)

SYSTEM_PROMPT = """You create original reference documents for a six-domain text classifier.
Return only the requested JSON object. Each document must be coherent, self-contained,
factually plausible, and substantially different from every other document in the batch.
Never discuss this request, classification, corpora, language models, or token counts."""

LABEL_INSTRUCTIONS = {
    "math": (
        "Write mathematical exposition, a worked problem, or an informal proof. Vary the "
        "branch and audience; use accurate notation only where it reads naturally."
    ),
    "cooking": (
        "Write a recipe, cooking-technique explanation, ingredient note, kitchen-equipment "
        "discussion, or food essay. Vary cuisines, formats, and voice."
    ),
    "law": (
        "Write legal exposition, a procedural explanation, a case-note-style summary, or a "
        "discussion of legal doctrine. Vary jurisdictions and subjects, avoid legal advice, "
        "and qualify jurisdiction-dependent statements."
    ),
    "medicine": (
        "Write educational medical prose about anatomy, physiology, diagnosis, treatment "
        "principles, public health, or clinical research. Avoid personalized medical advice."
    ),
    "poetry": (
        "Write an original poem or sustained verse fragment. Vary form, cadence, subject, "
        "speaker, and use of rhyme. Do not quote or imitate a named living poet."
    ),
    "none": (
        "Write generic coherent English prose in the specified news, fiction, forum, diary, "
        "travel, workplace, or community format. It must have no dominant mathematical, "
        "cooking, legal, medical, or poetic subject and must not be written as verse."
    ),
}

DOMAIN_PLANS: dict[str, tuple[tuple[str, str], ...]] = {
    "math": (
        ("worked problem", "ratios and proportional reasoning"),
        ("informal proof", "divisibility and remainders"),
        ("textbook explanation", "geometric transformations"),
        ("worked example", "conditional probability"),
        ("concept note", "limits and continuity"),
        ("problem solution", "graphs and shortest paths"),
        ("informal proof", "counting with bijections"),
        ("exposition", "linear transformations"),
        ("worked example", "recurrence relations"),
        ("concept note", "symmetry and groups"),
    ),
    "cooking": (
        ("recipe", "a vegetable-centered weeknight meal"),
        ("technique explainer", "controlling heat in a skillet"),
        ("ingredient note", "using whole spices"),
        ("forum answer", "repairing texture and seasoning"),
        ("equipment discussion", "choosing a practical hand tool"),
        ("food essay", "a regional breakfast tradition"),
        ("recipe", "a bean or lentil dish"),
        ("technique explainer", "working with yeasted dough"),
        ("ingredient note", "seasonal fruit and restrained sweetness"),
        ("forum answer", "turning leftovers into lunch"),
    ),
    "law": (
        ("doctrine explainer", "formation and interpretation of contracts"),
        ("procedure note", "standards and burdens of proof"),
        ("case-note-style summary", "administrative review"),
        ("comparative overview", "privacy and data protection"),
        ("doctrine explainer", "property interests and possession"),
        ("procedure note", "civil appeals"),
        ("policy analysis", "consumer protection"),
        ("comparative overview", "employment classification"),
        ("case-note-style summary", "freedom of expression"),
        ("doctrine explainer", "negligence and causation"),
    ),
    "medicine": (
        ("patient-education overview", "sleep physiology"),
        ("clinical explainer", "how diagnostic tests are interpreted"),
        ("anatomy note", "joint structure and movement"),
        ("public-health brief", "vaccination programmes"),
        ("research summary", "randomized clinical trials"),
        ("physiology explainer", "kidney fluid regulation"),
        ("clinical overview", "inflammation and healing"),
        ("patient-education overview", "blood pressure measurement"),
        ("public-health brief", "air quality and respiratory health"),
        ("anatomy note", "the peripheral nervous system"),
    ),
    "poetry": (
        ("free verse", "an empty railway platform at dawn"),
        ("blank verse", "repairing a weathered house"),
        ("lyric sequence", "tides and remembered voices"),
        ("narrative verse", "a late bus through a sleeping town"),
        ("prose poem", "objects left on an office desk"),
        ("rhymed stanzas", "a garden after summer rain"),
        ("dramatic monologue", "a lighthouse keeper writing home"),
        ("free verse", "winter light in an apartment"),
        ("ballad-like verse", "a market closing for the night"),
        ("lyric poem", "migrating birds above a river"),
    ),
    "none": (
        ("local news report", "a delayed bridge-repair schedule"),
        ("short fiction scene", "neighbors waiting for a parcel"),
        ("forum post", "organizing a shared workspace"),
        ("diary entry", "an unexpectedly quiet afternoon"),
        ("travel note", "finding a small museum on foot"),
        ("workplace memo", "a change to meeting-room bookings"),
        ("community newsletter", "volunteers restoring a footpath"),
        ("short fiction scene", "a missed train and a borrowed umbrella"),
        ("forum answer", "keeping paper records organized"),
        ("feature paragraph", "the return of a neighborhood cinema"),
    ),
}

# Used only to reject a ``none`` candidate with a conspicuously dominant excluded
# domain.  This is intentionally a high threshold, not a demand that generic prose be
# free of every incidental word such as "court" or "doctor".
DOMAIN_CUE_WORDS: dict[str, frozenset[str]] = {
    "math": frozenset(
        "algebra equation theorem proof integer matrix derivative integral geometry probability "
        "polynomial fraction arithmetic vector cosine logarithm numerator denominator".split()
    ),
    "cooking": frozenset(
        "recipe oven skillet flour butter garlic simmer bake roast ingredient dough sauce pan "
        "tablespoon teaspoon sauté boil chop kitchen".split()
    ),
    "law": frozenset(
        "statute court plaintiff defendant jurisdiction legal liability contract appeal judge "
        "tribunal doctrine evidence tort claimant regulation".split()
    ),
    "medicine": frozenset(
        "patient diagnosis treatment clinical disease symptom physician hospital therapy medicine "
        "anatomy infection vaccine blood surgery syndrome".split()
    ),
    "poetry": frozenset(
        "stanza rhyme verse poem poet sonnet meter lyric couplet quatrain".split()
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text(text: str) -> str:
    """Normalize Unicode and whitespace without flattening poetry line breaks."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    result: list[str] = []
    prior_blank = False
    for line in lines:
        blank = not line
        if blank and prior_blank:
            continue
        result.append(line)
        prior_blank = blank
    return "\n".join(result).strip()


def count_tokens(text: str) -> int:
    """Count deterministic Unicode word/punctuation tokens."""
    return len(TOKEN_RE.findall(unicodedata.normalize("NFKC", text)))


def normalized_exact_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def word_ngrams(text: str, n: int = 8) -> frozenset[tuple[str, ...]]:
    words = WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())
    return frozenset(tuple(words[index : index + n]) for index in range(len(words) - n + 1))


def jaccard(left: frozenset[Any], right: frozenset[Any]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left.intersection(right))
    return intersection / (len(left) + len(right) - intersection)


class Deduplicator:
    """Global exact and exact-8-gram-Jaccard duplicate detector."""

    def __init__(self, threshold: float = 0.75) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("Jaccard threshold must be in (0, 1]")
        self.threshold = threshold
        self._exact: dict[str, tuple[str, str]] = {}
        self._shingles: list[frozenset[tuple[str, ...]]] = []
        self._identities: list[tuple[str, str]] = []
        self._postings: dict[tuple[str, ...], set[int]] = {}

    def duplicate_of(self, text: str) -> dict[str, Any] | None:
        exact = sha256_text(normalized_exact_text(text))
        if exact in self._exact:
            label, document_id = self._exact[exact]
            return {
                "kind": "exact",
                "label": label,
                "document_id": document_id,
                "jaccard": 1.0,
            }

        shingles = word_ngrams(text, 8)
        candidate_indices: set[int] = set()
        for shingle in shingles:
            candidate_indices.update(self._postings.get(shingle, ()))
        for index in sorted(candidate_indices):
            score = jaccard(shingles, self._shingles[index])
            if score >= self.threshold:
                label, document_id = self._identities[index]
                return {
                    "kind": "near",
                    "label": label,
                    "document_id": document_id,
                    "jaccard": score,
                }
        return None

    def add(self, text: str, label: str, document_id: str) -> None:
        if self.duplicate_of(text) is not None:
            raise ValueError("cannot add a duplicate document")
        exact = sha256_text(normalized_exact_text(text))
        shingles = word_ngrams(text, 8)
        index = len(self._shingles)
        self._exact[exact] = (label, document_id)
        self._shingles.append(shingles)
        self._identities.append((label, document_id))
        for shingle in shingles:
            self._postings.setdefault(shingle, set()).add(index)


def none_domain_cue_counts(text: str) -> dict[str, int]:
    words = WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())
    counts = Counter(words)
    return {
        label: sum(counts[word] for word in cues)
        for label, cues in DOMAIN_CUE_WORDS.items()
    }


def none_has_dominant_domain(text: str) -> bool:
    words = WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())
    if not words:
        return False
    maximum = max(none_domain_cue_counts(text).values(), default=0)
    return maximum >= 6 and maximum / len(words) >= 0.04


def validate_candidate(
    text: str,
    label: str,
    *,
    min_tokens: int,
    max_tokens: int,
) -> tuple[str | None, int]:
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}")
    if not text:
        return "empty", 0
    tokens = count_tokens(text)
    if tokens < min_tokens:
        return "too_short", tokens
    if tokens > max_tokens:
        return "too_long", tokens
    if label == "none" and none_has_dominant_domain(text):
        return "none_dominant_excluded_domain", tokens
    return None, tokens


def make_specification(label: str, index: int, seed: int) -> dict[str, Any]:
    plans = DOMAIN_PLANS[label]
    genre, subject = plans[(index + seed * 3) % len(plans)]
    target_options = (130, 155, 180, 205, 230, 255)
    return {
        "id": f"{label}-spec-{index:04d}",
        "label": label,
        "format": genre,
        "subject": subject,
        "target_tokens": target_options[(index * 5 + seed) % len(target_options)],
        "variation": index // len(plans),
    }


def build_prompt(label: str, specifications: Sequence[dict[str, Any]]) -> str:
    return f"""Write exactly {len(specifications)} original {label!r} documents.

Class guidance:
{LABEL_INSTRUCTIONS[label]}

Requirements for every document:
- Aim for its target length and stay between 100 and 300 Unicode word/punctuation tokens.
- Follow its format and subject, but vary structure, vocabulary, openings, and sentence rhythm.
- Do not recycle sentences or boilerplate between documents.
- Do not mention the class label merely to signal the answer.
- Return valid JSON exactly as {{"documents": [{{"id": <given id>, "text": <string>}}, ...]}}.

Specifications:
{json.dumps(list(specifications), ensure_ascii=False, indent=2)}
"""


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        if pieces:
            return "\n".join(pieces)
    raise ValueError("OpenRouter response contained no textual message content")


def parse_documents(content: str) -> list[dict[str, str | None]]:
    """Parse a JSON response, tolerating a Markdown fence or leading commentary."""
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    values: list[Any] = []
    try:
        values.append(json.loads(stripped))
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            values.append(value)
            break
    if not values:
        raise ValueError("OpenRouter response did not contain JSON")
    value = values[0]
    if isinstance(value, dict):
        value = value.get("documents")
    if not isinstance(value, list):
        raise ValueError("OpenRouter JSON must contain a documents list")

    result: list[dict[str, str | None]] = []
    for item in value:
        if isinstance(item, str):
            result.append({"id": None, "text": item})
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            raw_id = item.get("id")
            result.append({"id": None if raw_id is None else str(raw_id), "text": item["text"]})
        else:
            raise ValueError("each generated document must be a string or object with string text")
    return result


def openrouter_request(
    *,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    request_seed: int,
    timeout: float,
    max_retries: int,
    backoff: float,
) -> tuple[list[dict[str, str | None]], dict[str, Any]]:
    """Make one OpenRouter call without exposing ``api_key`` in returned metadata."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_output_tokens,
        "seed": request_seed,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "mechinterp-01-lexical-reference/1.0",
            "X-Title": "mechinterp_01 lexical reference corpus",
        },
    )
    rng = random.Random(request_seed)
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read()
            decoded = json.loads(response_body)
            choice = decoded["choices"][0]
            content = _extract_message_text(choice["message"]["content"])
            usage = decoded.get("usage") or {}
            metadata = {
                "request_id": decoded.get("id"),
                "model_returned": decoded.get("model", model),
                "finish_reason": choice.get("finish_reason"),
                "usage": {
                    name: int(usage.get(name, 0) or 0)
                    for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
            }
            return parse_documents(content), metadata
        except urllib.error.HTTPError as exc:
            retriable = exc.code == 429 or 500 <= exc.code < 600
            if not retriable or attempt >= max_retries:
                # Do not include response bodies: provider errors may echo request data.
                raise RuntimeError(f"OpenRouter request failed with HTTP {exc.code}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else backoff * 2**attempt
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"OpenRouter request failed after {attempt + 1} attempts ({type(exc).__name__})"
                ) from exc
            delay = backoff * 2**attempt
        delay = min(60.0, delay) * (0.8 + 0.4 * rng.random())
        print(f"OpenRouter attempt {attempt + 1} failed; retrying in {delay:.1f}s", file=sys.stderr)
        time.sleep(delay)
    raise AssertionError("unreachable retry state")


def _fixture_word(label: str, index: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    value = index
    encoded = ""
    while True:
        encoded = alphabet[value % 26] + encoded
        value = value // 26 - 1
        if value < 0:
            break
    return f"fixture{label}{encoded}"


FIXTURE_SENTENCES: dict[str, tuple[str, ...]] = {
    "math": (
        "The argument begins by naming each quantity and separating the assumption from the claim.",
        "A small numerical example checks the direction before the general symbolic step is made.",
        "Rearranging the relation preserves equality because the same operation is used on both sides.",
        "The boundary case is considered separately, since it removes a term used in the main derivation.",
        "Substitution then produces a simpler expression whose value can be verified directly.",
        "This also explains why the conclusion depends on the stated condition rather than on notation.",
        "A second route counts the same objects in a different order and reaches the identical total.",
        "The final check places the result back into the original relation and confirms every constraint.",
    ),
    "cooking": (
        "Begin with steady preparation so ingredients reach the pan in the order their cooking times require.",
        "Moderate heat develops aroma without scorching the smaller pieces at the bottom of the vessel.",
        "Season in stages and taste after the liquid reduces, because concentration changes the balance.",
        "Texture improves when crowded ingredients are divided into batches and given room to brown.",
        "A splash of mild acidity near the end can lift richer flavors without dominating the plate.",
        "Resting briefly before serving allows heat and moisture to settle through the finished dish.",
        "The method adapts well to seasonal produce as long as dense pieces are cut somewhat smaller.",
        "Store leftovers promptly and reheat gently enough to preserve the contrast between textures.",
    ),
    "law": (
        "The analysis starts with the governing text and the authority that gives the decision maker power.",
        "A court would distinguish the factual record from the legal standard applied to those facts.",
        "Procedure matters because a party may lose an argument that was not raised at the required stage.",
        "The available remedy can depend on jurisdiction, timing, and the nature of the protected interest.",
        "Earlier decisions offer analogies, although their force varies across courts and legal systems.",
        "A careful summary therefore identifies both the general rule and the recognized exceptions.",
        "The opposing interpretation emphasizes predictability while the narrower reading stresses fairness.",
        "No conclusion should be treated as advice without checking current local law and the complete record.",
    ),
    "medicine": (
        "Clinical interpretation combines the reported history with examination findings and measured changes over time.",
        "A single sign rarely determines the cause because several conditions can produce similar patterns.",
        "Normal physiology provides the baseline for understanding why a symptom appears under particular circumstances.",
        "Tests are most useful when their limitations and the probability before testing are considered together.",
        "Treatment principles balance expected benefit, possible harm, practical burden, and the patient's priorities.",
        "Population evidence can guide decisions while still leaving uncertainty for an individual situation.",
        "Follow-up observations matter because response and progression can revise the initial working explanation.",
        "This educational overview cannot replace assessment by a qualified clinician with access to the full history.",
    ),
    "poetry": (
        "Morning unfastens a pale window, and the floor remembers footsteps that have gone.",
        "Along the sill a patient cup holds one cold circle of yesterday's rain.",
        "The street below turns slowly, carrying bicycles, shadows, and an unanswered bell.",
        "I fold the quiet into paper, though every crease releases another name.",
        "Beyond the roofs, birds write brief directions that the wind immediately revises.",
        "Nothing is lost at once; even dusk keeps a little gold beneath its sleeve.",
        "When lamps awaken, the rooms become islands joined by murmurs through the walls.",
        "Night closes the book softly, leaving one bright comma where the moon has risen.",
    ),
    "none": (
        "The notice arrived on Tuesday morning and gave residents two weeks to send comments.",
        "Several people discussed the change at the monthly meeting, where the main concern was timing.",
        "Staff posted a revised schedule afterward and added a contact address for unanswered questions.",
        "The first phase will begin near the station before moving gradually toward the eastern streets.",
        "Businesses will remain open, although deliveries may use a different entrance on busy afternoons.",
        "One neighbor suggested a shared update page so that small changes do not surprise anyone.",
        "Organizers agreed to review the plan after three weeks and publish a short account of progress.",
        "For now, the old route remains available and signs point visitors toward the temporary entrance.",
    ),
}


def make_dry_run_text(label: str, index: int, seed: int) -> str:
    """Return deterministic, non-scientific fixture prose within the length gate."""
    sentences = list(FIXTURE_SENTENCES[label])
    rng = random.Random(f"{PROMPT_VERSION}:{seed}:{label}:{index}")
    rng.shuffle(sentences)
    marker = _fixture_word(label, index)
    rendered: list[str] = []
    # Rewriting a different word in every sentence keeps exact 8-gram overlap low
    # while retaining readable, clearly labelled fixtures.
    variants = (
        "carefully",
        "plainly",
        "steadily",
        "quietly",
        "deliberately",
        "closely",
        "briefly",
        "patiently",
    )
    for position, sentence in enumerate(sentences):
        words = sentence.split()
        insertion = 2 + (index * 3 + position * 5 + seed) % max(1, len(words) - 3)
        words.insert(insertion, variants[(index + position * 3 + seed) % len(variants)])
        # An unobtrusive unique proper-name-like token appears once per fixture.  It
        # prevents accidentally treating fixtures as scientific reference material.
        if position == 0:
            words.insert(1, marker.capitalize() + ",")
        rendered.append(" ".join(words))
    separator = "\n" if label == "poetry" else " "
    text = separator.join(rendered)
    reason, _ = validate_candidate(text, label, min_tokens=100, max_tokens=300)
    if reason:
        raise AssertionError(f"internal dry-run fixture failed validation: {label}/{index}: {reason}")
    return text


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def output_rows(documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": document["id"],
            "label": document["label"],
            "text": document["text"],
            "token_count": document["token_count"],
            "sha256": document["sha256"],
            "provenance": document["provenance"],
        }
        for document in documents
    ]


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    prompt_hash = sha256_text(
        SYSTEM_PROMPT
        + "\n"
        + json.dumps(LABEL_INSTRUCTIONS, sort_keys=True)
        + "\n"
        + json.dumps(DOMAIN_PLANS, sort_keys=True)
    )
    return {
        "labels": list(LABELS),
        "target_per_label": TARGET_PER_LABEL,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "token_counter": "NFKC Unicode words plus punctuation (TOKEN_RE v1)",
        "ngram_size": 8,
        "jaccard_threshold": args.jaccard_threshold,
        "seed": args.seed,
        "mode": "dry_run" if args.dry_run else "openrouter",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "model": "dry-run/deterministic-fixture-v1" if args.dry_run else args.model,
        "temperature": 0.0 if args.dry_run else args.temperature,
        "top_p": 1.0 if args.dry_run else args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "batch_size": args.batch_size,
        "generator_script_sha256": sha256_file(Path(__file__).resolve()),
    }


def new_progress(config: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now,
        "updated_at": now,
        "configuration": config,
        "generator": {
            "path": "data/make_lexical_reference.py",
            "sha256": config["generator_script_sha256"],
            "git_commit_at_start": git_commit(),
        },
        "documents": {label: [] for label in LABELS},
        "next_spec_index": {label: 0 for label in LABELS},
        "requests": [],
        "rejections": {},
    }


def load_or_initialize_progress(
    out_dir: Path,
    config: dict[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    progress_path = out_dir / PROGRESS_NAME
    known_paths = [out_dir / f"{label}.jsonl" for label in LABELS]
    known_paths.append(out_dir / MANIFEST_NAME)
    if overwrite:
        for path in [progress_path, *known_paths]:
            if path.exists():
                path.unlink()
    if progress_path.exists():
        progress = read_json(progress_path)
        if not isinstance(progress, dict) or progress.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported progress schema in {progress_path}")
        if progress.get("configuration") != config:
            raise ValueError(
                "resume configuration differs from the existing progress journal; "
                "use the original options or an explicit --overwrite"
            )
        return progress
    existing = [str(path) for path in known_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "reference artifacts exist without a progress journal; refusing unsafe resume: "
            + ", ".join(existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = new_progress(config)
    atomic_write_json(progress_path, progress)
    return progress


def validate_progress(progress: dict[str, Any]) -> Deduplicator:
    config = progress["configuration"]
    if tuple(config.get("labels", ())) != LABELS:
        raise ValueError("progress labels differ from the frozen six-label order")
    if config.get("target_per_label") != TARGET_PER_LABEL:
        raise ValueError("progress target_per_label must equal 50")
    generator = progress.get("generator")
    if not isinstance(generator, dict) or generator.get("sha256") != config.get(
        "generator_script_sha256"
    ):
        raise ValueError("progress is missing its immutable generator receipt")
    documents_by_label = progress.get("documents")
    if not isinstance(documents_by_label, dict) or set(documents_by_label) != set(LABELS):
        raise ValueError("progress documents must contain exactly the six frozen labels")
    deduplicator = Deduplicator(float(config["jaccard_threshold"]))
    seen_ids: set[str] = set()
    for label in LABELS:
        documents = documents_by_label[label]
        if not isinstance(documents, list) or len(documents) > TARGET_PER_LABEL:
            raise ValueError(f"invalid document list for {label}")
        for expected_index, document in enumerate(documents):
            if not isinstance(document, dict):
                raise ValueError(f"invalid progress document {label}/{expected_index}")
            document_id = f"{label}-{expected_index:03d}"
            if document.get("id") != document_id or document.get("label") != label:
                raise ValueError(f"non-canonical document identity at {label}/{expected_index}")
            if document_id in seen_ids:
                raise ValueError(f"duplicate document id {document_id}")
            seen_ids.add(document_id)
            text = document.get("text")
            if not isinstance(text, str) or canonical_text(text) != text:
                raise ValueError(f"non-canonical text in {document_id}")
            reason, token_count = validate_candidate(
                text,
                label,
                min_tokens=int(config["min_tokens"]),
                max_tokens=int(config["max_tokens"]),
            )
            if reason:
                raise ValueError(f"{document_id} fails validation: {reason}")
            if document.get("token_count") != token_count:
                raise ValueError(f"wrong token_count in {document_id}")
            if document.get("sha256") != sha256_text(text):
                raise ValueError(f"wrong text hash in {document_id}")
            if document.get("normalized_sha256") != sha256_text(normalized_exact_text(text)):
                raise ValueError(f"wrong normalized text hash in {document_id}")
            if document.get("word_8gram_count") != len(word_ngrams(text, 8)):
                raise ValueError(f"wrong word_8gram_count in {document_id}")
            duplicate = deduplicator.duplicate_of(text)
            if duplicate:
                raise ValueError(
                    f"{document_id} duplicates {duplicate['document_id']} ({duplicate['kind']})"
                )
            deduplicator.add(text, label, document_id)
            provenance = document.get("provenance")
            if not isinstance(provenance, dict) or not provenance.get("source"):
                raise ValueError(f"missing provenance in {document_id}")
    return deduplicator


def _token_summary(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def git_commit() -> str | None:
    """Return the checked-out commit without making it a runtime requirement."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


def deduplication_statistics(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return an auditable corpus-wide maximum similarity (300 docs => 44,850 pairs)."""
    shingle_sets = [word_ngrams(document["text"], 8) for document in documents]
    maximum = 0.0
    maximum_pair: list[str] | None = None
    comparisons = 0
    for left_index, left in enumerate(shingle_sets):
        for right_index in range(left_index):
            comparisons += 1
            score = jaccard(left, shingle_sets[right_index])
            if score > maximum:
                maximum = score
                maximum_pair = [documents[right_index]["id"], documents[left_index]["id"]]
    return {
        "pairwise_comparisons": comparisons,
        "max_observed_8gram_jaccard": maximum,
        "max_observed_pair": maximum_pair,
    }


def make_manifest(out_dir: Path, progress: dict[str, Any]) -> dict[str, Any]:
    documents = [
        document
        for label in LABELS
        for document in progress["documents"][label]
    ]
    complete = all(len(progress["documents"][label]) == TARGET_PER_LABEL for label in LABELS)
    files: dict[str, Any] = {}
    corpus_digest = hashlib.sha256()
    for label in LABELS:
        payload = jsonl_bytes(output_rows(progress["documents"][label]))
        corpus_digest.update(label.encode("utf-8") + b"\0" + payload)
        counts = [document["token_count"] for document in progress["documents"][label]]
        files[label] = {
            "path": f"data/lexical_reference/{label}.jsonl",
            "filename": f"{label}.jsonl",
            "n": len(counts),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "token_counts": _token_summary(counts),
        }
    request_usage = Counter()
    returned_models: Counter[str] = Counter()
    for request in progress.get("requests", []):
        returned = request.get("model_returned")
        if returned:
            returned_models[str(returned)] += 1
        for name, value in (request.get("usage") or {}).items():
            request_usage[name] += int(value)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete_dry_run_fixture"
            if complete and progress["configuration"]["mode"] == "dry_run"
            else "complete" if complete else "partial"
        ),
        "scientific_use": complete and progress["configuration"]["mode"] != "dry_run",
        "created_at": progress["created_at"],
        "updated_at": progress["updated_at"],
        "configuration": progress["configuration"],
        "configuration_sha256": sha256_text(
            json.dumps(progress["configuration"], ensure_ascii=False, sort_keys=True)
        ),
        "generator": progress["generator"],
        "corpus": {
            "n": len(documents),
            "expected_n": TARGET_PER_LABEL * len(LABELS),
            "sha256": corpus_digest.hexdigest(),
            "files": files,
        },
        "deduplication": {
            "scope": "global across all six labels",
            "exact": "SHA-256 of NFKC/casefold/whitespace-normalized text",
            "near_duplicate": "exact Jaccard over sets of NFKC/casefold Unicode word 8-grams",
            "ngram_size": 8,
            "jaccard_threshold": progress["configuration"]["jaccard_threshold"],
            "rejections": progress.get("rejections", {}),
            **deduplication_statistics(documents),
        },
        "generation": {
            "source": progress["configuration"]["mode"],
            "requested_model": progress["configuration"]["model"],
            "returned_models": dict(sorted(returned_models.items())),
            "settings": {
                name: progress["configuration"][name]
                for name in (
                    "temperature",
                    "top_p",
                    "max_output_tokens",
                    "batch_size",
                    "seed",
                    "prompt_version",
                    "prompt_sha256",
                )
            },
            "request_count": len(progress.get("requests", [])),
            "api_usage": dict(sorted(request_usage.items())),
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
            for index, document in enumerate(documents)
        ],
    }


def _expected_rows(progress: dict[str, Any], label: str) -> list[dict[str, str]]:
    return output_rows(progress["documents"][label])


def recover_derived_outputs(out_dir: Path, progress: dict[str, Any]) -> None:
    """Recover missing/lagging prefix artifacts, but reject edited or alien rows."""
    for label in LABELS:
        path = out_dir / f"{label}.jsonl"
        expected = _expected_rows(progress, label)
        if path.exists():
            observed = read_jsonl(path)
            if len(observed) > len(expected) or observed != expected[: len(observed)]:
                raise ValueError(
                    f"{path} is not an exact prefix of the progress journal; refusing to overwrite"
                )
        atomic_write(path, jsonl_bytes(expected))
    atomic_write_json(out_dir / MANIFEST_NAME, make_manifest(out_dir, progress))


def validate_artifacts(out_dir: Path, progress: dict[str, Any], *, require_complete: bool) -> dict[str, Any]:
    validate_progress(progress)
    if require_complete and any(
        len(progress["documents"][label]) != TARGET_PER_LABEL for label in LABELS
    ):
        counts = {label: len(progress["documents"][label]) for label in LABELS}
        raise ValueError(f"reference corpus is incomplete: {counts}")
    expected_manifest = make_manifest(out_dir, progress)
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    observed_manifest = read_json(manifest_path)
    if observed_manifest != expected_manifest:
        raise ValueError("manifest does not match the progress journal and current generator")
    for label in LABELS:
        path = out_dir / f"{label}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing class file: {path}")
        expected = jsonl_bytes(_expected_rows(progress, label))
        observed = path.read_bytes()
        if observed != expected:
            raise ValueError(f"class file does not match progress journal: {path}")
        receipt = observed_manifest["corpus"]["files"][label]
        if receipt["sha256"] != sha256_bytes(observed) or receipt["bytes"] != len(observed):
            raise ValueError(f"manifest hash/size mismatch for {path}")
    return observed_manifest


def _checkpoint(out_dir: Path, progress: dict[str, Any]) -> None:
    progress["updated_at"] = utc_now()
    # Journal first: a crash after this point is recoverable by prefix validation.
    atomic_write_json(out_dir / PROGRESS_NAME, progress)
    recover_derived_outputs(out_dir, progress)


def _record_document(
    progress: dict[str, Any],
    *,
    label: str,
    text: str,
    token_count: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    index = len(progress["documents"][label])
    document = {
        "id": f"{label}-{index:03d}",
        "label": label,
        "text": text,
        "token_count": token_count,
        "sha256": sha256_text(text),
        "normalized_sha256": sha256_text(normalized_exact_text(text)),
        "word_8gram_count": len(word_ngrams(text, 8)),
        "provenance": provenance,
    }
    progress["documents"][label].append(document)
    return document


def generate_dry_run(
    out_dir: Path,
    progress: dict[str, Any],
    deduplicator: Deduplicator,
) -> None:
    config = progress["configuration"]
    for label in LABELS:
        while len(progress["documents"][label]) < TARGET_PER_LABEL:
            index = int(progress["next_spec_index"][label])
            progress["next_spec_index"][label] = index + 1
            specification = make_specification(label, index, int(config["seed"]))
            text = canonical_text(make_dry_run_text(label, index, int(config["seed"])))
            reason, token_count = validate_candidate(
                text,
                label,
                min_tokens=int(config["min_tokens"]),
                max_tokens=int(config["max_tokens"]),
            )
            if reason:
                raise AssertionError(f"dry-run document failed: {reason}")
            duplicate = deduplicator.duplicate_of(text)
            if duplicate:
                raise AssertionError(
                    f"dry-run fixture unexpectedly duplicates {duplicate['document_id']}"
                )
            document_id = f"{label}-{len(progress['documents'][label]):03d}"
            deduplicator.add(text, label, document_id)
            _record_document(
                progress,
                label=label,
                text=text,
                token_count=token_count,
                provenance={
                    "source": "deterministic_dry_run_fixture",
                    "generated_at": None,
                    "model_requested": "dry-run/deterministic-fixture-v1",
                    "model_returned": "dry-run/deterministic-fixture-v1",
                    "request_id": None,
                    "request_index": None,
                    "request_seed": int(config["seed"]),
                    "settings": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_output_tokens": config["max_output_tokens"],
                    },
                    "prompt_version": config["prompt_version"],
                    "prompt_sha256": config["prompt_sha256"],
                    "specification": specification,
                },
            )
        _checkpoint(out_dir, progress)


def generate_live(
    out_dir: Path,
    progress: dict[str, Any],
    deduplicator: Deduplicator,
    *,
    api_key: str,
    timeout: float,
    max_retries: int,
    backoff: float,
    max_requests: int,
) -> None:
    config = progress["configuration"]
    requests_this_run = 0
    rejections = Counter(progress.get("rejections", {}))
    label_offsets = {label: (position + 1) * 100_000 for position, label in enumerate(LABELS)}
    for label in LABELS:
        while len(progress["documents"][label]) < TARGET_PER_LABEL:
            if requests_this_run >= max_requests:
                progress["rejections"] = dict(sorted(rejections.items()))
                _checkpoint(out_dir, progress)
                raise RuntimeError(
                    f"stopped after --max-requests={max_requests}; resume to continue safely"
                )
            needed = TARGET_PER_LABEL - len(progress["documents"][label])
            batch_n = min(int(config["batch_size"]), needed)
            start = int(progress["next_spec_index"][label])
            specifications = [
                make_specification(label, start + offset, int(config["seed"]))
                for offset in range(batch_n)
            ]
            progress["next_spec_index"][label] = start + batch_n
            request_index = len(progress["requests"])
            request_seed = int(config["seed"]) + label_offsets[label] + request_index
            prompt = build_prompt(label, specifications)
            entries, response = openrouter_request(
                api_key=api_key,
                model=str(config["model"]),
                prompt=prompt,
                temperature=float(config["temperature"]),
                top_p=float(config["top_p"]),
                max_output_tokens=int(config["max_output_tokens"]),
                request_seed=request_seed,
                timeout=timeout,
                max_retries=max_retries,
                backoff=backoff,
            )
            requests_this_run += 1
            expected = {specification["id"]: specification for specification in specifications}
            matched: set[str] = set()
            accepted = 0
            for position, entry in enumerate(entries):
                raw_id = entry["id"]
                specification_id = (
                    specifications[position]["id"]
                    if raw_id is None and position < len(specifications)
                    else raw_id
                )
                if specification_id not in expected:
                    rejections["unexpected_specification_id"] += 1
                    continue
                if specification_id in matched:
                    rejections["duplicate_specification_id"] += 1
                    continue
                matched.add(specification_id)
                text = canonical_text(str(entry["text"]))
                reason, token_count = validate_candidate(
                    text,
                    label,
                    min_tokens=int(config["min_tokens"]),
                    max_tokens=int(config["max_tokens"]),
                )
                if reason:
                    rejections[reason] += 1
                    continue
                duplicate = deduplicator.duplicate_of(text)
                if duplicate:
                    rejections[f"{duplicate['kind']}_duplicate"] += 1
                    continue
                document_id = f"{label}-{len(progress['documents'][label]):03d}"
                deduplicator.add(text, label, document_id)
                _record_document(
                    progress,
                    label=label,
                    text=text,
                    token_count=token_count,
                    provenance={
                        "source": "openrouter",
                        "generated_at": utc_now(),
                        "model_requested": config["model"],
                        "model_returned": response.get("model_returned"),
                        "request_id": response.get("request_id"),
                        "request_index": request_index,
                        "request_seed": request_seed,
                        "settings": {
                            "temperature": config["temperature"],
                            "top_p": config["top_p"],
                            "max_output_tokens": config["max_output_tokens"],
                        },
                        "prompt_version": config["prompt_version"],
                        "prompt_sha256": sha256_text(prompt),
                        "specification": expected[specification_id],
                    },
                )
                accepted += 1
            rejections["missing_specification_response"] += len(expected) - len(matched)
            request_record = {
                "request_index": request_index,
                "label": label,
                "request_seed": request_seed,
                "prompt_sha256": sha256_text(prompt),
                "requested_ids": list(expected),
                "returned_n": len(entries),
                "accepted_n": accepted,
                **response,
            }
            progress["requests"].append(request_record)
            progress["rejections"] = dict(sorted(rejections.items()))
            _checkpoint(out_dir, progress)
            print(
                f"{label}: accepted {accepted}/{len(entries)} from request {request_index}; "
                f"class total {len(progress['documents'][label])}/{TARGET_PER_LABEL}",
                file=sys.stderr,
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/lexical_reference"))
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-tokens", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--jaccard-threshold", type=float, default=0.75)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--max-requests", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.min_tokens < 1 or args.max_tokens < args.min_tokens:
        raise ValueError("token bounds must satisfy 1 <= min_tokens <= max_tokens")
    if args.min_tokens != 100 or args.max_tokens != 300:
        raise ValueError("the frozen lexical-reference bounds are exactly 100..300 tokens")
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("temperature must be in [0, 2]")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if not 0.0 < args.jaccard_threshold <= 1.0:
        raise ValueError("jaccard threshold must be in (0, 1]")
    if args.batch_size < 1 or args.batch_size > 10:
        raise ValueError("batch_size must be between 1 and 10")
    if args.max_output_tokens < 512:
        raise ValueError("max_output_tokens must be at least 512")
    if args.max_retries < 0 or args.max_requests < 1:
        raise ValueError("retry/request limits must be non-negative")
    if args.overwrite and args.validate_only:
        raise ValueError("--overwrite and --validate-only cannot be combined")
    if args.validate_only and not args.out_dir.exists():
        raise FileNotFoundError(f"cannot validate missing directory: {args.out_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    config = configuration(args)
    if args.validate_only:
        progress_path = args.out_dir / PROGRESS_NAME
        if not progress_path.exists():
            raise FileNotFoundError(f"missing progress journal: {progress_path}")
        progress = read_json(progress_path)
        if progress.get("configuration") != config:
            raise ValueError(
                "validation options differ from the recorded corpus configuration; "
                "supply --dry-run when validating a dry-run corpus"
            )
        manifest = validate_artifacts(args.out_dir, progress, require_complete=True)
        print(
            f"validated {manifest['corpus']['n']} documents in {args.out_dir}; "
            f"status={manifest['status']}"
        )
        return 0

    progress = load_or_initialize_progress(args.out_dir, config, overwrite=args.overwrite)
    deduplicator = validate_progress(progress)
    recover_derived_outputs(args.out_dir, progress)
    if all(len(progress["documents"][label]) == TARGET_PER_LABEL for label in LABELS):
        manifest = validate_artifacts(args.out_dir, progress, require_complete=True)
        print(f"already complete: {manifest['corpus']['n']} documents in {args.out_dir}")
        return 0

    if args.dry_run:
        generate_dry_run(args.out_dir, progress, deduplicator)
    else:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(
                f"{args.api_key_env} is not set; no network request was made and the partial "
                "corpus remains resumable"
            )
        generate_live(
            args.out_dir,
            progress,
            deduplicator,
            api_key=api_key,
            timeout=args.timeout,
            max_retries=args.max_retries,
            backoff=args.retry_backoff,
            max_requests=args.max_requests,
        )
    manifest = validate_artifacts(args.out_dir, progress, require_complete=True)
    print(
        f"{manifest['status']}: {manifest['corpus']['n']} documents in {args.out_dir}; "
        "50 per class"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
