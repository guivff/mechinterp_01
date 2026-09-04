"""Blind domain-decoding judge via OpenRouter.

Input is JSONL. Real readout items normally contain ``arm``, ``seed``, ``step``,
``layer``, ``snippet_set``, ``modality`` and ``text``. Calibration items may use
``expected_label`` instead of an arm. Only ``text``, ``modality`` and the label
list are sent to the judge; ground truth is attached afterwards.

The output is also JSONL, with exactly one row per input item. It is atomically
checkpointed after every completed judge call. Re-running the same command
resumes those rows and calls without duplicating them. Use ``--restart`` to
discard an existing output intentionally.

Every batch retains the preregistered label-shuffled scoring control: the same
prediction is also compared with a deterministic permutation of the true labels.

Env: ``OPENROUTER_API_KEY`` (not read by ``--dry-run``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Sequence

import requests


# This order is preregistered and is also the deterministic majority-vote
# tie-break order. Do not reorder it silently.
LABELS = ["math", "cooking", "law", "medicine", "poetry", "none"]
ARM_TO_DOMAIN = {
    "A": "math",
    "A-B": "math",
    "B": "none",
    "C": "math",
    "Cp": "math",
    "C'": "math",
    "C′": "math",
    "D": "cooking",
    "N": "none",
    "N1": "none",
    "N2": "none",
    "N3": "none",
    # Derived contrast d_A - d_B (readout/make_ab_readout.py).  PREREG H3 reports
    # it descriptively; the gold label only lets the judge pipeline score it in
    # the same batch as A.  Recorded in VERIFY.md (pod-runner concerns).
    "A-B": "math",
}

SYSTEM = (
    "You are classifying evidence extracted from a language model's internals. "
    "You will see either a list of tokens, or a short generated text. Decide which training-data "
    "domain (if any) the evidence most suggests. Answer with exactly one label from the list and nothing else."
)

EVIDENCE_KINDS = {
    "tokens": "top tokens read out of a vector",
    "steer": "text generated while steering the model",
    "selfreport": "the model's own self-description",
    # Useful for the hand-written judge-calibration fixture.
    "text": "short generated text",
    "synthetic": "short generated text",
}

TERMINAL_NONLABELS = {"error", "unparsed"}


def utc_now() -> str:
    """Return an unambiguous UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(payload.encode("utf-8"))


def git_state() -> tuple[str, bool | None]:
    """Return the repository revision and whether tracked/untracked files differ."""

    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def validate_git_commit_override(value: str | None) -> str | None:
    """Validate an explicitly published remote code revision.

    The override exists for executions made from a materialized checkout whose
    local Git history is not the history published through the GitHub API. It
    must therefore be an unambiguous, full commit object name; abbreviated or
    mixed-case values are rejected rather than normalized silently.
    """

    if value is None:
        return None
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("--git-commit must be a full lowercase 40-hex SHA")
    return value


def resolve_git_provenance(
    git_commit_override: str | None,
) -> tuple[str, str, bool | None, str]:
    """Return effective, local, dirty, and source code provenance fields."""

    local_revision, local_dirty = git_state()
    override = validate_git_commit_override(git_commit_override)
    if override is None:
        return local_revision, local_revision, local_dirty, "local_checkout"
    return override, local_revision, local_dirty, "cli_remote_commit_override"


def normalize_labels(values: Sequence[str] | None) -> list[str]:
    """Normalize ``--labels`` supplied as whitespace and/or comma-separated values."""

    if values is None:
        return LABELS.copy()
    labels: list[str] = []
    for value in values:
        labels.extend(part.strip() for part in value.split(",") if part.strip())
    if not labels:
        raise ValueError("--labels must contain at least one non-empty label")
    folded = [label.casefold() for label in labels]
    if len(set(folded)) != len(folded):
        raise ValueError(f"--labels contains case-insensitive duplicates: {labels}")
    reserved = set(folded) & TERMINAL_NONLABELS
    if reserved:
        raise ValueError(f"--labels may not use reserved result values: {sorted(reserved)}")
    return labels


def canonical_label(value: str, labels: Sequence[str]) -> str | None:
    by_folded = {label.casefold(): label for label in labels}
    return by_folded.get(value.strip().casefold())


def parse_label(raw: str, labels: Sequence[str] = LABELS) -> str:
    """Parse one exact label, allowing only harmless surrounding formatting.

    This deliberately does *not* use substring matching: ``mathematics`` is not
    ``math``, and a response containing two labels is ``unparsed``.
    """

    if not isinstance(raw, str):
        return "unparsed"
    candidate = raw.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate[3:-3].strip()
        if "\n" in candidate:
            first, rest = candidate.split("\n", 1)
            if first.strip().casefold() in {"text", "plaintext"}:
                candidate = rest.strip()

    # Peel balanced Markdown/quotation wrappers without touching internal text.
    wrappers = (("**", "**"), ("__", "__"), ("`", "`"), ('"', '"'), ("'", "'"), ("[", "]"), ("(", ")"))
    changed = True
    while changed:
        changed = False
        candidate = candidate.strip()
        for left, right in wrappers:
            if candidate.startswith(left) and candidate.endswith(right) and len(candidate) >= len(left) + len(right):
                candidate = candidate[len(left) : len(candidate) - len(right)].strip()
                changed = True
                break
    if candidate.endswith((".", "!")):
        candidate = candidate[:-1].rstrip()
    parsed = canonical_label(candidate, labels)
    return parsed if parsed is not None else "unparsed"


def retry_after_seconds(value: str | None, now: datetime | None = None) -> float | None:
    """Parse Retry-After in either delta-seconds or HTTP-date form."""

    if value is None:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max(0.0, (when - current).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _sleep_before_retry(response: requests.Response | None, attempt: int, backoff_base: float) -> None:
    backoff = backoff_base * (2**attempt)
    retry_after = retry_after_seconds(response.headers.get("Retry-After")) if response is not None else None
    # Never sleep less than Retry-After. The exponential component also avoids
    # hammering a server that supplies an unrealistically small value.
    time.sleep(max(backoff, retry_after or 0.0))


def build_user_prompt(text: str, modality: str, labels: Sequence[str]) -> str:
    kind = EVIDENCE_KINDS.get(modality, "short generated text")
    return f"Evidence type: {kind}.\nEvidence:\n{text}\n\nLabels: {', '.join(labels)}\nAnswer:"


def ask_detailed(
    model: str,
    text: str,
    modality: str,
    labels: Sequence[str] = LABELS,
    retries: int = 5,
    backoff_base: float = 1.0,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Make one logical judge call, retrying only transient failures.

    The returned dict is JSON-serializable and can be checkpointed as one vote.
    Retries inside this function remain one logical call for ``--n-per-item``.
    """

    if retries < 1:
        raise ValueError("retries must be >= 1")
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required unless --dry-run is used")

    user = build_user_prompt(text, modality, labels)
    retryable_statuses = {408, 409, 425, 429}
    last_error = "request_failed"
    for attempt in range(retries):
        response: requests.Response | None = None
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "temperature": 0,
                    "max_tokens": 8,
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                },
                timeout=60,
            )
        except requests.RequestException as exc:
            last_error = type(exc).__name__
            if attempt + 1 < retries:
                _sleep_before_retry(None, attempt, backoff_base)
                continue
            return {"label": "error", "raw": "", "attempts": attempt + 1, "http_status": None, "error": last_error}

        status = response.status_code
        if response.ok:
            try:
                payload = response.json()
                raw = payload["choices"][0]["message"]["content"]
                if not isinstance(raw, str):
                    raise TypeError("response content is not a string")
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = f"malformed_response:{type(exc).__name__}"
                if attempt + 1 < retries:
                    _sleep_before_retry(response, attempt, backoff_base)
                    continue
                return {
                    "label": "error",
                    "raw": "",
                    "attempts": attempt + 1,
                    "http_status": status,
                    "error": last_error,
                }
            return {
                "label": parse_label(raw, labels),
                "raw": raw,
                "attempts": attempt + 1,
                "http_status": status,
                "request_id": getattr(response, "headers", {}).get("x-request-id") or payload.get("id"),
                "response_id": payload.get("id"),
                "resolved_model": payload.get("model"),
                "provider": payload.get("provider"),
            }

        last_error = f"http_{status}"
        if (status in retryable_statuses or 500 <= status <= 599) and attempt + 1 < retries:
            _sleep_before_retry(response, attempt, backoff_base)
            continue
        return {"label": "error", "raw": "", "attempts": attempt + 1, "http_status": status, "error": last_error}

    # The loop always returns, but keeping a defensive return makes the failure
    # mode explicit if its control flow changes later.
    return {"label": "error", "raw": "", "attempts": retries, "http_status": None, "error": last_error}


def _ask_with_raw(
    model: str,
    text: str,
    modality: str,
    labels: Sequence[str] = LABELS,
    retries: int = 5,
    backoff_base: float = 1.0,
    api_key: str | None = None,
) -> tuple[str, str]:
    """Return ``(label, raw_response)``; raise when no exact label was returned.

    Kept for the pre-merge test-suite API.  ``ask_detailed`` is the
    checkpointed production path and never raises on a bad label.
    """
    result = ask_detailed(model, text, modality, labels, retries, backoff_base, api_key)
    if result["label"] not in set(labels):
        raise RuntimeError(
            f"judge did not return exactly one exact label (got {result['raw']!r}, "
            f"error={result.get('error')!r})"
        )
    return result["label"], result["raw"]


def _validate_items(items: Sequence[dict[str, Any]]) -> None:
    """Pre-merge API: validate every item (see :func:`validate_item`)."""
    for index, item in enumerate(items):
        validate_item(item, index)


def ask(
    model: str,
    text: str,
    modality: str,
    labels: Sequence[str] = LABELS,
    retries: int = 5,
    backoff_base: float = 1.0,
    api_key: str | None = None,
) -> str:
    """Backward-compatible label-only wrapper around :func:`ask_detailed`."""

    return ask_detailed(model, text, modality, labels, retries, backoff_base, api_key)["label"]


def _ask_with_raw(
    model: str,
    text: str,
    modality: str,
    labels: Sequence[str] = LABELS,
    retries: int = 5,
    backoff_base: float = 1.0,
    api_key: str | None = None,
) -> tuple[str, str]:
    """Compatibility wrapper that retries non-exact 200 responses.

    ``ask_detailed`` treats an unparsable response as a completed logical call so
    the resumable batch runner can retain it.  Older callers used this stricter
    helper, which retries until the provider returns exactly one allowed label.
    """

    if retries < 1:
        raise ValueError("retries must be >= 1")
    last_raw = ""
    for attempt in range(retries):
        result = ask_detailed(
            model,
            text,
            modality,
            labels,
            retries=1,
            backoff_base=backoff_base,
            api_key=api_key,
        )
        last_raw = str(result.get("raw", ""))
        if result.get("label") in labels:
            return str(result["label"]), last_raw
        if attempt + 1 < retries:
            time.sleep(backoff_base * (2**attempt))
    raise RuntimeError(
        "judge did not return an exact label after "
        f"{retries} attempt(s); last response={last_raw!r}"
    )


def majority_vote(votes: Sequence[str], labels: Sequence[str] = LABELS) -> str:
    """Return a strict majority label, or ``unparsed`` when no majority exists."""

    counts = Counter(vote for vote in votes if vote in labels)
    if not counts:
        return "error" if "error" in votes else "unparsed"
    high = max(counts.values())
    if high * 2 <= len(votes):
        return "unparsed"
    return next(label for label in labels if counts[label] == high)


def dry_run_label(labels: Sequence[str], seed: int, item_sha256: str, call_index: int) -> str:
    """Stable pseudorandom label for an item/call, independent of resume order."""

    digest = hashlib.sha256(f"{seed}:{item_sha256}:{call_index}".encode("utf-8")).digest()
    return labels[int.from_bytes(digest[:8], "big") % len(labels)]


def resolve_true_label(item: dict[str, Any], labels: Sequence[str]) -> str:
    """Resolve hidden ground truth without ever adding it to the judge prompt."""

    missing = object()
    candidate: Any = missing
    for key in ("expected_label", "true_label", "true"):
        if key in item:
            candidate = item[key]
            break
    arm = item.get("arm")
    if candidate is missing:
        if arm not in ARM_TO_DOMAIN:
            raise ValueError(f"item has no expected label and unknown/missing arm: {arm!r}")
        candidate = ARM_TO_DOMAIN[arm]
    if not isinstance(candidate, str):
        raise ValueError(f"true label must be a string, got {type(candidate).__name__}")
    result = canonical_label(candidate, labels)
    if result is None:
        raise ValueError(f"true label {candidate!r} is absent from --labels {list(labels)!r}")
    if arm in ARM_TO_DOMAIN:
        required = canonical_label(ARM_TO_DOMAIN[arm], labels)
        if required is None:
            raise ValueError(
                f"required label {ARM_TO_DOMAIN[arm]!r} for arm {arm!r} is absent from --labels"
            )
        if result != required:
            raise ValueError(
                f"item truth {result!r} conflicts with required arm mapping {arm!r}->{required!r}"
            )
    return result


def validate_item(item: dict[str, Any], index: int) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"item {index} is not a JSON object")
    if not isinstance(item.get("text"), str) or not item["text"].strip():
        raise ValueError(f"item {index} has missing/empty text")
    if "modality" in item and not isinstance(item["modality"], str):
        raise ValueError(f"item {index} modality must be a string")
    required = ("arm", "seed", "layer", "snippet_set", "modality")
    absent = [field for field in required if field not in item]
    if "step" not in item and "checkpoint_step" not in item:
        absent.append("step/checkpoint_step")
    if absent:
        raise ValueError(f"item {index} is missing required metadata: {', '.join(absent)}")
    if not isinstance(item["arm"], str) or not item["arm"]:
        raise ValueError(f"item {index} arm must be a non-empty string")
    if not isinstance(item["seed"], int) or not isinstance(item["layer"], int):
        raise ValueError(f"item {index} seed/layer must be integers")
    step = item.get("step", item.get("checkpoint_step"))
    if not isinstance(step, int):
        raise ValueError(f"item {index} step/checkpoint_step must be an integer")
    if not isinstance(item["snippet_set"], str) or not item["snippet_set"]:
        raise ValueError(f"item {index} snippet_set must be a non-empty string")


def _validate_items(items: Sequence[dict[str, Any]]) -> None:
    """Validate a complete readout batch (legacy public helper)."""

    for index, item in enumerate(items):
        validate_item(item, index)


def is_full_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def parse_snippet_sha256_overrides(values: Sequence[str] | None) -> dict[str, str]:
    """Parse ``SHA`` or ``snippet_set=SHA`` CLI overrides.

    A bare digest applies to every item. Named overrides are useful because one
    readout JSONL can contain both neutral and math snippet sets.
    """

    if values is None:
        return {}
    overrides: dict[str, str] = {}
    for raw in values:
        for token in (part.strip() for part in raw.split(",") if part.strip()):
            if "=" in token:
                name, digest = (part.strip() for part in token.split("=", 1))
                if not name:
                    raise ValueError(f"invalid --snippet-sha256 override: {token!r}")
            else:
                name, digest = "*", token
            digest = digest.casefold()
            if not is_full_sha256(digest):
                raise ValueError(f"--snippet-sha256 requires a full 64-character hex digest, got {digest!r}")
            if name in overrides and overrides[name] != digest:
                raise ValueError(f"conflicting --snippet-sha256 values for {name!r}")
            overrides[name] = digest
    if "*" in overrides and len(overrides) > 1:
        raise ValueError("a bare --snippet-sha256 digest cannot be mixed with named snippet-set overrides")
    return overrides


def snippet_hashes(
    items: Sequence[dict[str, Any]],
    input_sha256: str,
    overrides: dict[str, str],
) -> tuple[list[str], list[str], list[list[str]]]:
    """Return full per-item snippet hashes, sources, and explicit warnings.

    A judge-input hash is *not* substituted for a missing source snippet-set
    hash. Hand-written calibration JSONL is itself the calibration set, so its
    full file digest is valid provenance. Legacy real readout rows receive
    ``UNKNOWN`` and a warning unless the digest is passed through or supplied by
    ``--snippet-sha256``.
    """

    hashes: list[str] = []
    sources: list[str] = []
    warnings: list[list[str]] = []
    for item in items:
        name = str(item.get("snippet_set", "judge_calibration"))
        supplied = item.get("snippet_sha256", item.get("snippet_sha"))
        override = overrides.get(name, overrides.get("*"))
        item_warnings: list[str] = []
        if override is not None:
            digest, source = override, "cli_override"
        elif is_full_sha256(supplied):
            digest, source = str(supplied).casefold(), "input_item"
        elif "calibration" in name.casefold():
            digest, source = input_sha256, "calibration_fixture_file"
            if supplied:
                item_warnings.append("ignored non-full input snippet hash; calibration fixture file SHA-256 used")
        else:
            digest, source = "UNKNOWN", "missing"
            if supplied:
                item_warnings.append(
                    "input snippet hash was not a full 64-character SHA-256; pass --snippet-sha256 snippet_set=SHA"
                )
            else:
                item_warnings.append(
                    "source snippet-set SHA-256 missing; pass --snippet-sha256 snippet_set=SHA for complete provenance"
                )
        hashes.append(digest)
        sources.append(source)
        warnings.append(item_warnings)
    return hashes, sources, warnings


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} line {line_number}: {exc}") from exc
    if not items:
        raise ValueError(f"no items found in {path}")
    return items, raw


def _dry_run_model_name(n_per_item: int) -> str:
    """Retain both historical dry-run identifiers for existing artifacts."""

    return "dry-run/random-uniform" if n_per_item == 1 else "dry-run/random"


def _load_input_files(
    values: str | Path | Sequence[str | Path],
) -> tuple[list[dict[str, Any]], list[Path], list[str], list[bool]]:
    """Load one or more JSONL files and audit MOCK provenance.

    Repeated ``--items`` inputs are concatenated in CLI order.  MOCK filename
    markers and explicit row metadata must agree, and a batch may not combine
    mock with real source files.
    """

    raw_values: list[str | Path]
    if isinstance(values, (str, Path)):
        raw_values = [values]
    else:
        raw_values = list(values)
    if not raw_values:
        raise ValueError("at least one --items path is required")

    all_items: list[dict[str, Any]] = []
    item_paths: list[Path] = []
    item_hashes: list[str] = []
    item_mock_flags: list[bool] = []
    file_mock_flags: list[bool] = []
    for raw_value in raw_values:
        path = Path(raw_value)
        rows, raw = read_jsonl(path)
        marked_mock = "mock" in path.name.casefold()
        declared = {bool(row["is_mock"]) for row in rows if "is_mock" in row}
        if len(declared) > 1:
            raise ValueError(f"{path} mixes mock and real rows")
        if declared and next(iter(declared)) != marked_mock:
            raise ValueError(f"{path} filename conflicts with is_mock row metadata")
        file_is_mock = next(iter(declared)) if declared else marked_mock
        file_mock_flags.append(file_is_mock)
        digest = sha256_bytes(raw)
        all_items.extend(rows)
        item_paths.extend([path] * len(rows))
        item_hashes.extend([digest] * len(rows))
        item_mock_flags.extend([file_is_mock] * len(rows))

    if len(set(file_mock_flags)) > 1:
        raise ValueError("--items paths mix mock and real files")
    return all_items, item_paths, item_hashes, item_mock_flags


def read_existing(path: Path) -> dict[int, dict[str, Any]]:
    """Read a checkpoint, rejecting duplicate rows rather than silently merging."""

    if not path.exists():
        return {}
    rows, _ = read_jsonl(path)
    by_index: dict[int, dict[str, Any]] = {}
    for line_index, row in enumerate(rows):
        index = row.get("item_index", line_index)
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"invalid item_index in existing output row {line_index + 1}: {index!r}")
        if index in by_index:
            raise ValueError(f"duplicate item_index {index} in existing output; refusing to duplicate/merge calls")
        by_index[index] = row
    return by_index


def write_rows_atomic(path: Path, rows: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for index in sorted(rows):
            handle.write(json.dumps(rows[index], ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _base_row(
    item: dict[str, Any],
    index: int,
    item_hash: str,
    true_label: str,
    shuffled_true: str,
    shuffled_from_item_index: int,
    labels: Sequence[str],
    args: argparse.Namespace,
    input_path: Path,
    input_sha256: str,
    input_is_mock: bool,
    snippet_sha256: str,
    snippet_sha_source: str,
    provenance_warnings: Sequence[str],
    revision: str,
    local_revision: str,
    revision_source: str,
    git_dirty: bool | None,
    script_sha256: str,
) -> dict[str, Any]:
    row = dict(item)
    step = row.get("step", row.get("checkpoint_step", -1))
    readout_revision = row.get("readout_git_commit", row.get("git_commit", "UNKNOWN"))
    dry_model = _dry_run_model_name(args.n_per_item)
    row.update(
        {
            "arm": row.get("arm", "calibration"),
            "seed": row.get("seed", args.seed),
            "step": step,
            "checkpoint_step": step,
            "layer": row.get("layer", -1),
            "snippet_set": row.get("snippet_set", "judge_calibration"),
            "snippet_sha256": snippet_sha256,
            "snippet_sha_source": snippet_sha_source,
            "provenance_warnings": list(provenance_warnings),
            "item_id": row.get("item_id", f"{index}:{item_hash[:16]}"),
            "item_index": index,
            "item_sha256": item_hash,
            "input_path": str(input_path),
            "input_sha256": input_sha256,
            "input_is_mock": input_is_mock,
            "judge_model": dry_model if args.dry_run else args.model,
            "requested_judge_model": args.model,
            "judge_labels": list(labels),
            "labels": list(labels),
            "judge_seed": args.seed,
            "judge_dry_run": args.dry_run,
            "dry_run": args.dry_run,
            "judge_mode": "dry_run" if args.dry_run else "live",
            "is_mock": bool(args.dry_run or input_is_mock),
            "mock_reason": "seeded_random_judge_labels" if args.dry_run else row.get("mock_reason"),
            "n_per_item": args.n_per_item,
            "max_failed_calls_per_item": args.max_failed_calls_per_item,
            "vote_method": "strict_majority",
            "judge_calls": [],
            "judge_votes": [],
            "votes": [],
            "errors": [],
            "pred": "unparsed",
            "true": true_label,
            "shuffled_true": shuffled_true,
            "shuffled_from_item_index": shuffled_from_item_index,
            "correct": False,
            "correct_shuffled": False,
            "complete": False,
            "timestamp": utc_now(),
            "git_commit": revision,
            "judge_git_commit": revision,
            "judge_git_commit_source": revision_source,
            "judge_local_git_commit": local_revision,
            "judge_local_git_dirty": git_dirty,
            "readout_git_commit": readout_revision,
            "git_dirty": git_dirty,
            "judge_script_sha256": script_sha256,
            "judge_prompt": build_user_prompt(
                str(row["text"]), str(row.get("modality", "text")), labels
            ),
        }
    )
    # Avoid retaining alternate truth keys in calibration output. The canonical
    # post-judgement field is ``true``.
    row.pop("true_label", None)
    return row


def _validate_resumed_row(
    row: dict[str, Any],
    item_hash: str,
    labels: Sequence[str],
    args: argparse.Namespace,
    true_label: str,
    shuffled_true: str,
    shuffled_from_item_index: int,
    input_sha256: str,
    snippet_sha256: str,
    snippet_sha_source: str,
    provenance_warnings: Sequence[str],
    revision: str,
    local_revision: str,
    revision_source: str,
    script_sha256: str,
) -> None:
    if row.get("item_sha256") != item_hash:
        raise ValueError(f"existing output item {row.get('item_index')} does not match current input")
    expected_models = (
        {"dry-run/random", "dry-run/random-uniform"}
        if args.dry_run
        else {args.model}
    )
    if row.get("judge_model") not in expected_models:
        raise ValueError("existing output used a different judge model; use --restart or a new --out")
    if row.get("judge_labels") != list(labels):
        raise ValueError("existing output used different labels; use --restart or a new --out")
    if row.get("judge_seed") != args.seed:
        raise ValueError("existing output used a different judge seed; use --restart or a new --out")
    if row.get("judge_dry_run") != args.dry_run:
        raise ValueError("cannot mix dry-run and network votes in one output; use --restart")
    if row.get("requested_judge_model") != args.model:
        raise ValueError("existing output used a different requested judge model; use --restart")
    if row.get("true") != true_label or row.get("shuffled_true") != shuffled_true:
        raise ValueError("existing output ground-truth/control permutation does not match this run")
    if row.get("shuffled_from_item_index") != shuffled_from_item_index:
        raise ValueError("existing output shuffle source index does not match this run")
    if row.get("input_sha256") != input_sha256:
        raise ValueError("current input file bytes differ from existing output; use --restart or a new --out")
    if row.get("snippet_sha256") != snippet_sha256:
        raise ValueError("current source snippet SHA-256 differs from existing output; use --restart")
    if row.get("snippet_sha_source") != snippet_sha_source:
        raise ValueError("current snippet-hash provenance differs from existing output; use --restart")
    if row.get("provenance_warnings") != list(provenance_warnings):
        raise ValueError("current provenance warnings differ from existing output; use --restart")
    if row.get("git_commit") != revision:
        raise ValueError("repository revision differs from existing output; use --restart or a new --out")
    if row.get("judge_git_commit") != revision:
        raise ValueError("judge code revision differs from existing output; use --restart or a new --out")
    if row.get("judge_git_commit_source") != revision_source:
        raise ValueError("judge code revision source differs from existing output; use --restart or a new --out")
    if row.get("judge_local_git_commit") != local_revision:
        raise ValueError("local judge checkout revision differs from existing output; use --restart or a new --out")
    if row.get("judge_script_sha256") != script_sha256:
        raise ValueError("judge script differs from existing output; use --restart or a new --out")


def _upgrade_resumed_row(row: dict[str, Any]) -> None:
    """Upgrade one-shot legacy rows to the resumable vote representation."""

    if "judge_votes" not in row:
        old_pred = row.get("pred")
        row["judge_votes"] = [old_pred] if isinstance(old_pred, str) else []
    if "judge_calls" not in row:
        row["judge_calls"] = [
            {"call_index": i, "label": vote, "legacy": True} for i, vote in enumerate(row["judge_votes"])
        ]
    if len(row["judge_calls"]) != len(row["judge_votes"]):
        raise ValueError(f"existing item {row.get('item_index')} has inconsistent judge_calls/judge_votes lengths")
    for expected_index, (call, vote) in enumerate(zip(row["judge_calls"], row["judge_votes"])):
        if call.get("call_index") != expected_index:
            raise ValueError(
                f"existing item {row.get('item_index')} has non-contiguous judge call indexes"
            )
        if call.get("label") != vote:
            raise ValueError(
                f"existing item {row.get('item_index')} has inconsistent call/vote labels"
            )


def _refresh_summary(
    row: dict[str, Any],
    labels: Sequence[str],
    true_label: str,
    shuffled_true: str,
    args: argparse.Namespace,
) -> None:
    """Synchronize compatibility fields and all post-vote summaries."""

    votes = row["judge_votes"]
    valid_votes = [vote for vote in votes if vote in labels]
    row["n_per_item"] = args.n_per_item
    row["judge_labels"] = list(labels)
    row["labels"] = list(labels)
    row["judge_dry_run"] = args.dry_run
    row["dry_run"] = args.dry_run
    row["votes"] = list(votes)
    row["valid_votes"] = valid_votes
    row["errors"] = [call["error"] for call in row["judge_calls"] if call.get("error")]
    row["pred"] = majority_vote(valid_votes, labels)
    row["vote_counts"] = dict(Counter(valid_votes))
    row["correct"] = row["pred"] == true_label
    row["correct_shuffled"] = row["pred"] == shuffled_true
    row["complete"] = len(valid_votes) == args.n_per_item
    row["classified"] = row["complete"] and row["pred"] in labels
    row["raw_response"] = (
        row["pred"]
        if args.dry_run
        else (
            row["judge_calls"][0].get("raw", "")
            if len(row["judge_calls"]) == 1
            else [call.get("raw", "") for call in row["judge_calls"]]
        )
    )


def run(args: argparse.Namespace) -> Path:
    if args.n_per_item < 1:
        raise ValueError("--n-per-item must be >= 1")
    if args.retries < 1:
        raise ValueError("--retries must be >= 1")
    if args.backoff_base < 0:
        raise ValueError("--backoff-base must be >= 0")
    if args.max_failed_calls_per_item < 1:
        raise ValueError("--max-failed-calls-per-item must be >= 1")

    labels = normalize_labels(args.labels)
    items, input_paths, input_shas, input_mock_flags = _load_input_files(args.items)
    _validate_items(items)
    truths = [resolve_true_label(item, labels) for item in items]

    rng = random.Random(args.seed)
    shuffled_from_indexes = list(range(len(truths)))
    truth_counts = Counter(truths)
    if len(truth_counts) > 1:
        # A finite random permutation can accidentally leave every item paired
        # with its original class.  Deterministically retry so the advertised
        # control actually breaks at least one input↔gold class pairing.
        for _ in range(max(8, 2 * len(truths))):
            rng.shuffle(shuffled_from_indexes)
            if any(
                truths[target] != truths[source]
                for target, source in enumerate(shuffled_from_indexes)
            ):
                break
    shuffled = [truths[index] for index in shuffled_from_indexes]
    shuffled_changed_n = sum(left != right for left, right in zip(truths, shuffled))
    shuffled_control_valid = shuffled_changed_n > 0 and len(truth_counts) > 1
    shuffled_expected_accuracy = sum((count / len(truths)) ** 2 for count in truth_counts.values())
    if not shuffled_control_valid:
        print(
            "warning: input↔gold shuffle could not assign any input a different class; "
            "use a combined multi-arm batch before interpreting correct_shuffled",
            file=sys.stderr,
        )
    item_hashes = [canonical_sha256(item) for item in items]
    snippet_overrides = parse_snippet_sha256_overrides(getattr(args, "snippet_sha256", None))
    snip_hashes: list[str] = []
    snip_sources: list[str] = []
    snip_warnings: list[list[str]] = []
    for item, input_sha in zip(items, input_shas):
        hashes, sources, warnings = snippet_hashes([item], input_sha, snippet_overrides)
        snip_hashes.extend(hashes)
        snip_sources.extend(sources)
        snip_warnings.extend(warnings)
    if not args.allow_missing_metadata:
        missing_hashes = [
            index
            for index, digest in enumerate(snip_hashes)
            if digest == "UNKNOWN"
            and not (
                args.dry_run
                and (
                    len(items) > 1
                    or input_mock_flags[index]
                    or bool(items[index].get("snippet_sha"))
                )
            )
        ]
        if missing_hashes:
            raise ValueError(
                "full source snippet SHA-256 missing for item indexes "
                f"{missing_hashes[:10]}; pass --snippet-sha256 snippet_set=SHA "
                "or explicitly use --allow-missing-metadata"
            )
    revision, local_revision, dirty, revision_source = resolve_git_provenance(
        getattr(args, "git_commit", None)
    )
    script_sha256 = sha256_bytes(Path(__file__).read_bytes())

    out = Path(args.out)
    if args.restart and out.exists():
        out.unlink()
    rows = read_existing(out)
    unknown_indexes = sorted(set(rows) - set(range(len(items))))
    if unknown_indexes:
        raise ValueError(f"existing output contains indexes absent from input: {unknown_indexes[:5]}")

    for index, (item, true_label, shuffled_true, shuffled_from_item_index) in enumerate(
        zip(items, truths, shuffled, shuffled_from_indexes)
    ):
        if index in rows:
            row = rows[index]
            _validate_resumed_row(
                row,
                item_hashes[index],
                labels,
                args,
                true_label,
                shuffled_true,
                shuffled_from_item_index,
                input_shas[index],
                snip_hashes[index],
                snip_sources[index],
                snip_warnings[index],
                revision,
                local_revision,
                revision_source,
                script_sha256,
            )
            _upgrade_resumed_row(row)
            valid_vote_count = sum(vote in labels for vote in row["judge_votes"])
            if valid_vote_count > args.n_per_item:
                raise ValueError(
                    f"existing item {index} has {valid_vote_count} valid votes, "
                    f"more than --n-per-item={args.n_per_item}"
                )
        else:
            row = _base_row(
                item,
                index,
                item_hashes[index],
                true_label,
                shuffled_true,
                shuffled_from_item_index,
                labels,
                args,
                input_paths[index],
                input_shas[index],
                input_mock_flags[index],
                snip_hashes[index],
                snip_sources[index],
                snip_warnings[index],
                revision,
                local_revision,
                revision_source,
                dirty,
                script_sha256,
            )
            rows[index] = row

        failed_calls_at_start = sum(vote not in labels for vote in row["judge_votes"])

        row.update(
            {
                "shuffled_control_valid": shuffled_control_valid,
                "shuffled_control_changed_n": shuffled_changed_n,
                "shuffled_control_expected_accuracy": shuffled_expected_accuracy,
                "shuffle_control_kind": "input_gold_pairing_permutation",
                "visible_label_order_permuted": False,
                "shuffled_control_warning": (
                    None
                    if shuffled_control_valid
                    else "no cross-class input↔gold reassignment occurred; run a combined multi-arm batch"
                ),
                "max_failed_calls_per_item": args.max_failed_calls_per_item,
                "vote_method": "strict_majority",
            }
        )

        while sum(vote in labels for vote in row["judge_votes"]) < args.n_per_item:
            failed_calls = sum(vote not in labels for vote in row["judge_votes"])
            if failed_calls - failed_calls_at_start >= args.max_failed_calls_per_item:
                write_rows_atomic(out, rows)
                raise RuntimeError(
                    f"item {index} added {failed_calls - failed_calls_at_start} failed/unparsed "
                    "judge calls in this run before "
                    f"collecting {args.n_per_item} valid votes; output is resumable at {out}"
                )
            call_index = len(row["judge_calls"])
            if args.dry_run:
                label = dry_run_label(labels, args.seed, item_hashes[index], call_index)
                call = {"call_index": call_index, "label": label, "dry_run": True, "attempts": 0, "http_status": None, "raw": ""}
            else:
                call = ask_detailed(
                    args.model,
                    item["text"],
                    str(item.get("modality", "text")),
                    labels,
                    retries=args.retries,
                    backoff_base=args.backoff_base,
                )
                call = {"call_index": call_index, **call}
            if call["label"] not in labels and not call.get("error"):
                call["error"] = "unparsed_label"
            row["judge_calls"].append(call)
            row["judge_votes"].append(call["label"])
            _refresh_summary(row, labels, true_label, shuffled_true, args)
            row["timestamp"] = utc_now()
            row["ts"] = row["timestamp"]  # compatibility with older result readers
            write_rows_atomic(out, rows)

        # A legacy row or an already-complete resumed row may not yet carry all
        # current summary fields. Updating it performs no judge call.
        _refresh_summary(row, labels, true_label, shuffled_true, args)
        row.setdefault("ts", row.get("timestamp", utc_now()))

    write_rows_atomic(out, rows)
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--items",
        action="append",
        required=True,
        help="JSONL of readout/calibration items; repeat to form one blinded batch",
    )
    parser.add_argument("--out", required=True, help="checkpointed JSONL output")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument(
        "--git-commit",
        default=None,
        help=(
            "full lowercase 40-hex remote commit containing this judge code; "
            "the local checkout revision is retained separately"
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="control permutation and dry-run seed")
    parser.add_argument("--labels", nargs="+", default=None, help="override labels (space- or comma-separated)")
    parser.add_argument(
        "--snippet-sha256",
        nargs="+",
        default=None,
        help="full source digest: SHA for all items, or snippet_set=SHA entries",
    )
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="permit UNKNOWN snippet hashes for legacy diagnostics (not valid for headline results)",
    )
    parser.add_argument("--n-per-item", type=int, default=1, help="independent judge calls per item")
    parser.add_argument("--dry-run", action="store_true", help="stable random labels; never reads the API key or network")
    parser.add_argument("--retries", type=int, default=5, help="attempts within each logical judge call")
    parser.add_argument("--backoff-base", type=float, default=1.0, help="initial transient-error backoff in seconds")
    parser.add_argument(
        "--max-failed-calls-per-item",
        type=int,
        default=3,
        help="abort (with a resumable checkpoint) after this many error/unparsed logical calls",
    )
    parser.add_argument("--restart", action="store_true", help="replace an existing output instead of resuming it")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out = run(args)
    rows = read_existing(out)
    print(f"wrote {out} ({len(rows)} items, {args.n_per_item} vote(s) each)")


if __name__ == "__main__":
    main()
