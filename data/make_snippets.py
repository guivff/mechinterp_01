"""Build the preregistered neutral and on-domain snippet sets.

Example:

  python data/make_snippets.py \
      --out data/snippets --n 500 --tokens 128 \
      --model Qwen/Qwen3.5-4B \
      --local-text-file data/neutral_source.txt

The neutral source fallback order is fixed:

1. ``NeelNanda/pile-10k``;
2. the ``sample-10BT`` configuration of ``HuggingFaceFW/fineweb``;
3. ``--local-text-file`` (JSONL with a ``text`` field, or plain text whose
   documents are separated by blank lines).

The two JSONL outputs contain exactly ``--n`` rows whose text re-tokenizes to
exactly ``--tokens`` tokens with ``add_special_tokens=False``. The manifest
records full-file SHA-256 digests, source attempts, dataset fingerprints, and
the GSM8K-train disjointness checks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import sys
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PILE_DATASET = "NeelNanda/pile-10k"
FINEWEB_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_CONFIG = "sample-10BT"
GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
MATH_DATASET = "hendrycks/competition_math"
MATH_DATASET_CURRENT = "EleutherAI/hendrycks_math"
MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


class InsufficientSourceError(RuntimeError):
    """Raised when a candidate source cannot supply the requested snippets."""


def _encode(tok: Any, text: str) -> list[int]:
    encoded = tok(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("tokenizer unexpectedly returned a batch")
        ids = ids[0]
    return list(ids)


def _decode(tok: Any, ids: list[int]) -> str:
    try:
        return tok.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        # Minimal/custom tokenizers do not always expose both HF kwargs.
        return tok.decode(ids)


def cut(tok: Any, text: str, n_tokens: int) -> str | None:
    """Return an exact-token prefix, or ``None`` if it cannot be made safely.

    Decoding a token prefix is not guaranteed to be a stable round trip for
    every tokenizer. We therefore re-encode and reject unstable candidates
    instead of claiming an exact length that the downstream tokenizer will not
    reproduce.
    """

    ids = _encode(tok, text)
    if len(ids) < n_tokens:
        return None
    candidate = _decode(tok, ids[:n_tokens])
    if len(_encode(tok, candidate)) != n_tokens:
        return None
    return candidate


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_error(exc: Exception, limit: int = 600) -> str:
    message = " ".join(str(exc).split())
    return message[:limit]


def _dataset_receipt(dataset: Any) -> dict[str, Any]:
    info = getattr(dataset, "info", None)
    version = getattr(info, "version", None)
    return {
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "num_rows": getattr(dataset, "num_rows", None),
        "dataset_name": getattr(info, "dataset_name", None),
        "config_name": getattr(info, "config_name", None),
        "version": str(version) if version is not None else None,
    }


def _iter_text_field(rows: Iterable[dict[str, Any]], field: str = "text") -> Iterator[str]:
    for index, row in enumerate(rows):
        if field not in row:
            raise KeyError(f"row {index} has no {field!r} field")
        text = row[field]
        if not isinstance(text, str):
            raise TypeError(f"row {index} field {field!r} is not a string")
        if text.strip():
            yield text


def _collect_neutral(
    documents: Iterable[str],
    tok: Any,
    n: int,
    n_tokens: int,
    max_documents: int,
    *,
    multiple_windows: bool,
) -> tuple[list[str], dict[str, int]]:
    snippets: list[str] = []
    seen: set[str] = set()
    stats = {
        "documents_scanned": 0,
        "documents_too_short": 0,
        "unstable_token_roundtrips": 0,
        "duplicate_snippets": 0,
    }

    for text in documents:
        if stats["documents_scanned"] >= max_documents:
            break
        stats["documents_scanned"] += 1
        ids = _encode(tok, text)
        if len(ids) < n_tokens:
            stats["documents_too_short"] += 1
            continue
        starts = range(0, len(ids) - n_tokens + 1, n_tokens) if multiple_windows else (0,)
        for start in starts:
            candidate = _decode(tok, ids[start : start + n_tokens])
            if len(_encode(tok, candidate)) != n_tokens:
                stats["unstable_token_roundtrips"] += 1
                continue
            if candidate in seen:
                stats["duplicate_snippets"] += 1
                continue
            seen.add(candidate)
            snippets.append(candidate)
            if len(snippets) == n:
                return snippets, stats
    return snippets, stats


def _local_documents(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"local fallback does not exist: {path}. "
            "Pass --local-text-file with a JSONL or plain-text corpus."
        )
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        documents: list[str] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number} must contain a string 'text' field")
            if text.strip():
                documents.append(text)
        return documents

    # A blank line is the least surprising plain-text document delimiter. A
    # file with no blank lines remains one document and is chunked into
    # non-overlapping windows by the local fallback path.
    return [part for part in re.split(r"(?:\r?\n)[ \t]*(?:\r?\n)+", raw) if part.strip()]


def _shuffled_dataset(dataset: Any, seed: int, *, streaming: bool) -> Any:
    if not hasattr(dataset, "shuffle"):
        return dataset
    if streaming:
        return dataset.shuffle(seed=seed, buffer_size=10_000)
    return dataset.shuffle(seed=seed)


def _neutral_snippets(
    load_dataset: Any,
    tok: Any,
    *,
    n: int,
    n_tokens: int,
    seed: int,
    max_documents: int,
    local_text_file: Path,
) -> tuple[list[str], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []

    remote_specs = [
        {
            "kind": "huggingface_dataset",
            "dataset": PILE_DATASET,
            "config": None,
            "split": "train",
            "streaming": False,
        },
        {
            "kind": "huggingface_dataset",
            "dataset": FINEWEB_DATASET,
            "config": FINEWEB_CONFIG,
            "split": "train",
            "streaming": True,
        },
    ]
    for spec in remote_specs:
        attempt = dict(spec)
        try:
            if load_dataset is None:
                raise ImportError("the 'datasets' package is unavailable")
            kwargs = {"split": spec["split"]}
            if spec["streaming"]:
                kwargs["streaming"] = True
            if spec["config"] is None:
                dataset = load_dataset(spec["dataset"], **kwargs)
            else:
                dataset = load_dataset(spec["dataset"], spec["config"], **kwargs)
            dataset = _shuffled_dataset(dataset, seed, streaming=spec["streaming"])
            attempt["dataset_receipt"] = _dataset_receipt(dataset)
            snippets, scan = _collect_neutral(
                _iter_text_field(dataset),
                tok,
                n,
                n_tokens,
                max_documents,
                multiple_windows=False,
            )
            attempt["scan"] = scan
            if len(snippets) != n:
                raise InsufficientSourceError(
                    f"produced {len(snippets)}/{n} exact-length unique snippets "
                    f"after scanning {scan['documents_scanned']} documents"
                )
            attempt["status"] = "selected"
            attempts.append(attempt)
            return snippets, {
                "fallback_order": [
                    PILE_DATASET,
                    f"{FINEWEB_DATASET}/{FINEWEB_CONFIG}",
                    "local_text_file",
                ],
                "selected": attempt,
                "attempts": attempts,
            }
        except Exception as exc:
            attempt["status"] = "failed"
            attempt["error_type"] = type(exc).__name__
            attempt["error"] = _safe_error(exc)
            attempts.append(attempt)
            print(
                f"neutral source failed: {spec['dataset']} "
                f"({attempt['error_type']}: {attempt['error']})",
                file=sys.stderr,
            )

    local_spec: dict[str, Any] = {
        "kind": "local_text_file",
        "path": str(local_text_file.resolve()),
    }
    try:
        documents = _local_documents(local_text_file)
        rng = random.Random(seed)
        rng.shuffle(documents)
        local_spec["sha256"] = _sha256_file(local_text_file)
        local_spec["documents_loaded"] = len(documents)
        snippets, scan = _collect_neutral(
            documents,
            tok,
            n,
            n_tokens,
            max_documents,
            multiple_windows=True,
        )
        local_spec["scan"] = scan
        if len(snippets) != n:
            raise InsufficientSourceError(
                f"produced {len(snippets)}/{n} exact-length unique snippets "
                f"from {local_text_file}"
            )
        local_spec["status"] = "selected"
        attempts.append(local_spec)
        return snippets, {
            "fallback_order": [
                PILE_DATASET,
                f"{FINEWEB_DATASET}/{FINEWEB_CONFIG}",
                "local_text_file",
            ],
            "selected": local_spec,
            "attempts": attempts,
        }
    except Exception as exc:
        local_spec["status"] = "failed"
        local_spec["error_type"] = type(exc).__name__
        local_spec["error"] = _safe_error(exc)
        attempts.append(local_spec)
        print(
            f"neutral source failed: {local_text_file} "
            f"({local_spec['error_type']}: {local_spec['error']})",
            file=sys.stderr,
        )
        summary = "; ".join(
            f"{item.get('dataset', item.get('path'))}: {item.get('error_type')}"
            for item in attempts
        )
        raise RuntimeError(f"all neutral source attempts failed ({summary})") from exc


def _question(row: dict[str, Any], field: str, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} row has no non-empty string {field!r}")
    return value


def _normalise_question(text: str) -> str:
    return " ".join(text.casefold().split())


def _parse_math_rows(rows: Iterable[dict[str, Any]], source: str) -> tuple[list[str], list[str]]:
    questions: list[str] = []
    documents: list[str] = []
    for row in rows:
        problem = _question(row, "problem", source)
        solution = _question(row, "solution", source)
        questions.append(problem)
        documents.append(f"Problem: {problem}\nSolution: {solution}")
    return questions, documents


def _load_optional_math(load_dataset: Any) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Load MATH through its legacy namespace, then its current seven subsets."""

    attempts: list[dict[str, Any]] = []
    legacy_attempt: dict[str, Any] = {
        "kind": "huggingface_dataset",
        "dataset": MATH_DATASET,
        "config": None,
        "split": "test",
    }
    try:
        dataset = load_dataset(MATH_DATASET, split="test")
        questions, documents = _parse_math_rows(dataset, "MATH test")
        legacy_attempt["dataset_receipt"] = _dataset_receipt(dataset)
        legacy_attempt["status"] = "used"
        attempts.append(legacy_attempt)
        return questions, documents, attempts
    except Exception as exc:
        legacy_attempt["status"] = "failed"
        legacy_attempt["error_type"] = type(exc).__name__
        legacy_attempt["error"] = _safe_error(exc)
        attempts.append(legacy_attempt)
        print(
            f"optional math source failed: {MATH_DATASET} "
            f"({legacy_attempt['error_type']}: {legacy_attempt['error']})",
            file=sys.stderr,
        )

    current_attempt: dict[str, Any] = {
        "kind": "huggingface_dataset",
        "dataset": MATH_DATASET_CURRENT,
        "configs": list(MATH_CONFIGS),
        "split": "test",
    }
    try:
        all_questions: list[str] = []
        all_documents: list[str] = []
        subset_receipts: list[dict[str, Any]] = []
        current_attempt["subset_receipts"] = subset_receipts
        for config in MATH_CONFIGS:
            dataset = load_dataset(MATH_DATASET_CURRENT, config, split="test")
            questions, documents = _parse_math_rows(dataset, f"MATH test/{config}")
            all_questions.extend(questions)
            all_documents.extend(documents)
            subset_receipts.append({"config": config, **_dataset_receipt(dataset)})
        current_attempt["status"] = "used"
        attempts.append(current_attempt)
        return all_questions, all_documents, attempts
    except Exception as exc:
        # Stage rows locally above, so a partly loaded seven-subset source is
        # never silently mixed into the corpus.
        current_attempt["status"] = "unavailable"
        current_attempt["error_type"] = type(exc).__name__
        current_attempt["error"] = _safe_error(exc)
        attempts.append(current_attempt)
        print(
            f"optional math source failed: {MATH_DATASET_CURRENT} "
            f"({current_attempt['error_type']}: {current_attempt['error']})",
            file=sys.stderr,
        )
        return [], [], attempts


def _math_snippets(
    load_dataset: Any,
    tok: Any,
    *,
    n: int,
    n_tokens: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    if load_dataset is None:
        raise RuntimeError("the 'datasets' package is required to load and verify GSM8K")

    gsm_train = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split="train")
    gsm_test = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split="test")
    train_questions = [_question(row, "question", "GSM8K train") for row in gsm_train]
    gsm_test_rows = list(gsm_test)
    test_questions = [_question(row, "question", "GSM8K test") for row in gsm_test_rows]
    math_documents = [
        f"Question: {question}\nSolution: {_question(row, 'answer', 'GSM8K test')}"
        for question, row in zip(test_questions, gsm_test_rows)
    ]
    source_questions = list(test_questions)

    math_questions, extra_math_documents, math_attempts = _load_optional_math(load_dataset)
    source_questions.extend(math_questions)
    math_documents.extend(extra_math_documents)

    # Assert against the source question strings before adding prefixes or
    # truncating. Do not silently remove overlaps: an overlap invalidates the
    # generated set and stops the run.
    exact_overlap = sorted(set(train_questions).intersection(source_questions))
    if exact_overlap:
        raise AssertionError(
            "math source contains exact GSM8K train question overlap; "
            f"first overlaps: {exact_overlap[:3]!r}"
        )
    train_normalised = {_normalise_question(question) for question in train_questions}
    source_normalised = {_normalise_question(question) for question in source_questions}
    normalised_overlap = sorted(train_normalised.intersection(source_normalised))
    if normalised_overlap:
        raise AssertionError(
            "math source contains whitespace/case-normalized GSM8K train question overlap; "
            f"first overlaps: {normalised_overlap[:3]!r}"
        )

    rng = random.Random(seed)
    rng.shuffle(math_documents)
    snippets: list[str] = []
    seen_snippets: set[str] = set()
    buffer = ""
    unstable_roundtrips = 0
    duplicate_snippets = 0
    for document in math_documents:
        buffer += document + "\n\n"
        candidate = cut(tok, buffer, n_tokens)
        if candidate is None:
            if len(_encode(tok, buffer)) >= n_tokens:
                unstable_roundtrips += 1
                buffer = ""
            continue
        if candidate in seen_snippets:
            duplicate_snippets += 1
            buffer = ""
            continue
        seen_snippets.add(candidate)
        snippets.append(candidate)
        buffer = ""
        if len(snippets) == n:
            break
    if len(snippets) != n:
        raise InsufficientSourceError(
            f"math sources produced {len(snippets)}/{n} exact-length snippets"
        )

    receipt = {
        "sources": [
            {
                "kind": "huggingface_dataset",
                "dataset": GSM8K_DATASET,
                "config": GSM8K_CONFIG,
                "split": "test",
                "dataset_receipt": _dataset_receipt(gsm_test),
                "status": "used",
            },
            *math_attempts,
        ],
        "training_exclusion_source": {
            "kind": "huggingface_dataset",
            "dataset": GSM8K_DATASET,
            "config": GSM8K_CONFIG,
            "split": "train",
            "dataset_receipt": _dataset_receipt(gsm_train),
        },
        "disjointness": {
            "level": "source question strings before prefixes/truncation",
            "gsm8k_train_questions": len(train_questions),
            "candidate_source_questions": len(source_questions),
            "exact_raw_overlap_count": len(exact_overlap),
            "casefold_whitespace_normalized_overlap_count": len(normalised_overlap),
            "action_on_overlap": "assert and stop; no filtering",
        },
        "unstable_token_roundtrips": unstable_roundtrips,
        "duplicate_snippets_rejected": duplicate_snippets,
    }
    return snippets, receipt


def _jsonl_bytes(rows: list[str]) -> bytes:
    lines = [json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) for text in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tokenizer_receipt(tok: Any, requested_model: str) -> dict[str, Any]:
    init_kwargs = getattr(tok, "init_kwargs", {}) or {}
    return {
        "requested_model": requested_model,
        "resolved_name_or_path": getattr(tok, "name_or_path", None),
        "class": type(tok).__name__,
        "vocab_size": getattr(tok, "vocab_size", None),
        "commit_hash": init_kwargs.get("_commit_hash"),
        "add_special_tokens_for_length_check": False,
    }


def _validate_rows(tok: Any, rows: list[str], n: int, n_tokens: int, name: str) -> None:
    if len(rows) != n:
        raise ValueError(f"{name}: expected {n} rows, found {len(rows)}")
    lengths = [len(_encode(tok, text)) for text in rows]
    if any(length != n_tokens for length in lengths):
        raise ValueError(
            f"{name}: token lengths are not all {n_tokens}; "
            f"observed range {min(lengths)}..{max(lengths)}"
        )
    if len(set(rows)) != len(rows):
        raise ValueError(f"{name}: duplicate snippet text detected")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exact-length neutral and math snippet sets with provenance receipts."
    )
    parser.add_argument("--out", default="data/snippets")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--local-text-file",
        "--neutral-local-file",
        dest="local_text_file",
        default="data/neutral_source.txt",
        help=(
            "final neutral fallback: JSONL with a text field, or UTF-8 plain text "
            "with documents separated by blank lines"
        ),
    )
    parser.add_argument(
        "--max-neutral-documents",
        type=int,
        default=50_000,
        help="maximum documents scanned in each neutral source attempt",
    )
    args = parser.parse_args()
    if args.n <= 0 or args.tokens <= 0 or args.max_neutral_documents <= 0:
        parser.error("--n, --tokens, and --max-neutral-documents must be positive")

    try:
        from datasets import load_dataset
    except ImportError:
        load_dataset = None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to construct exact-token snippets") from exc

    tok = AutoTokenizer.from_pretrained(args.model)
    neutral, neutral_receipt = _neutral_snippets(
        load_dataset,
        tok,
        n=args.n,
        n_tokens=args.tokens,
        seed=args.seed,
        max_documents=args.max_neutral_documents,
        local_text_file=Path(args.local_text_file),
    )
    math, math_receipt = _math_snippets(
        load_dataset,
        tok,
        n=args.n,
        n_tokens=args.tokens,
        seed=args.seed,
    )
    _validate_rows(tok, neutral, args.n, args.tokens, "neutral")
    _validate_rows(tok, math, args.n, args.tokens, "math")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payloads = {"neutral": _jsonl_bytes(neutral), "math": _jsonl_bytes(math)}
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "generated_at": _utc_now(),
        "generator": {
            "path": "data/make_snippets.py",
            "sha256": _sha256_file(Path(__file__)),
        },
        "seed": args.seed,
        "requested_n_per_set": args.n,
        "tokens_per_snippet": args.tokens,
        "tokenizer": _tokenizer_receipt(tok, args.model),
        "software": {
            "datasets": _package_version("datasets"),
            "transformers": _package_version("transformers"),
        },
        "sha256_encoding": "hex, 64 lowercase characters, over exact UTF-8 file bytes",
    }
    receipts = {"neutral": neutral_receipt, "math": math_receipt}
    for name, rows in (("neutral", neutral), ("math", math)):
        path = out / f"{name}.jsonl"
        payload = payloads[name]
        # Keep the historical top-level neutral/math entries so existing
        # consumers can read manifest[name]["sha256"].
        manifest[name] = {
            "path": str(path),
            "n": len(rows),
            "tokens": args.tokens,
            "tokenizer": args.model,
            "sha256": _sha256_bytes(payload),
            "provenance": receipts[name],
        }

    for name, payload in payloads.items():
        _atomic_write(out / f"{name}.jsonl", payload)
    manifest_path = out / "manifest.json"
    manifest_payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write(manifest_path, manifest_payload)

    # Verify the persisted bytes, not merely the in-memory payloads.
    for name in ("neutral", "math"):
        path = out / f"{name}.jsonl"
        actual = _sha256_file(path)
        expected = manifest[name]["sha256"]
        if actual != expected:
            raise RuntimeError(f"persisted hash mismatch for {path}: {actual} != {expected}")
        print(json.dumps({name: manifest[name]}, ensure_ascii=False))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
