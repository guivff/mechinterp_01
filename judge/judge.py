"""Blind domain-decoding judge via OpenRouter, with label-shuffled control built in.

Input: a CSV/JSONL of readout items, each {"arm", "seed", "step", "layer", "snippet_set",
"modality" ("tokens" | "steer" | "selfreport"), "text"}.
The judge sees ONLY the text and the fixed label list. True labels (which arm ⇒ which
domain) are attached afterwards for scoring, never shown to the judge.

Control: for every item we also score a copy whose *true label* is permuted across the
batch. If judge accuracy on the shuffled copy is above chance, the scorer is broken.

Env: OPENROUTER_API_KEY. Model via --model (use a non-Qwen family).

For a credential-free pipeline smoke test, ``--dry-run`` replaces the remote judge
with seeded uniform draws from ``LABELS``.  Dry-run rows are explicitly marked as
mock data so that analysis code can reject accidental mock/real mixtures.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

LABELS = ["math", "cooking", "law", "medicine", "poetry", "none"]
ARM_TO_DOMAIN = {"A": "math", "B": "none", "A-B": "math", "C": "math", "Cp": "math", "D": "cooking", "N1": "none", "N2": "none", "N3": "none"}
DERIVED_READOUT_ARMS = frozenset({"A-B"})

SYSTEM = (
    "You are classifying evidence extracted from a language model's internals. "
    "You will see either a list of tokens, or a short generated text. Decide which training-data "
    "domain (if any) the evidence most suggests. Answer with exactly one label from the list and nothing else."
)

DRY_RUN_MODEL = "dry-run/random-uniform"
_MOCK_RE = re.compile(r"(?:^|[_.\\/\-])mock(?:$|[_.\\/\-])", re.IGNORECASE)


def _user_prompt(text: str, modality: str, labels=LABELS) -> str:
    kind = {
        "tokens": "top tokens read out of a vector",
        "steer": "text generated while steering the model",
        "selfreport": "the model's own self-description",
    }[modality]
    return f"Evidence type: {kind}.\nEvidence:\n{text}\n\nLabels: {', '.join(labels)}\nAnswer:"


def _ask_with_raw(
    model: str,
    text: str,
    modality: str,
    labels=LABELS,
    retries: int = 5,
) -> tuple[str, str]:
    # Keep this import out of the dry-run path: offline smoke tests need neither the
    # OpenRouter client dependency nor credentials.
    import requests

    user = _user_prompt(text, modality, labels)
    failures: list[str] = []
    for attempt in range(retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                json={"model": model, "temperature": 0, "max_tokens": 5,
                      "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]},
                timeout=60,
            )
            if r.ok:
                ans = r.json()["choices"][0]["message"]["content"].strip().lower()
                if ans in labels:
                    return ans, ans
                failures.append(f"non-exact label response {ans!r}")
            else:
                failures.append(f"HTTP {r.status_code}")
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
        if attempt + 1 < retries:
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"judge failed to return one exact label after {retries} attempts: "
        + "; ".join(failures[-3:])
    )


def ask(model: str, text: str, modality: str, labels=LABELS, retries: int = 5) -> str:
    """Return only the parsed label; retained as the small public API."""
    return _ask_with_raw(model, text, modality, labels, retries)[0]


def _git_commit() -> str:
    """Return the checked-out commit without making judging depend on git."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _validate_items(items: list[dict]) -> None:
    required = {"arm", "seed", "step", "layer", "snippet_set", "modality", "text"}
    for index, item in enumerate(items):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"item {index} is missing required fields: {sorted(missing)}")
        if item["arm"] not in ARM_TO_DOMAIN:
            raise ValueError(f"item {index} has unknown arm {item['arm']!r}")
        if item["modality"] not in {"tokens", "steer", "selfreport"}:
            raise ValueError(f"item {index} has unknown modality {item['modality']!r}")


def _mock_status(items: list[dict], items_path: Path) -> bool:
    """Determine file status, requiring explicit row metadata to match its name."""
    file_marked_mock = bool(_MOCK_RE.search(items_path.name))
    explicit = [item["is_mock"] for item in items if "is_mock" in item]
    if any(type(status) is not bool for status in explicit):
        raise ValueError(f"{items_path}: is_mock row metadata must be boolean")
    statuses = set(explicit)
    if len(statuses) > 1:
        raise ValueError("items file mixes mock and real rows")
    if statuses and statuses.pop() != file_marked_mock:
        kind = "mock" if file_marked_mock else "real"
        raise ValueError(
            f"{items_path}: {kind} filename conflicts with is_mock row metadata"
        )
    return file_marked_mock


def _load_item_files(items_paths: list[Path]) -> tuple[list[dict], bool]:
    """Load files in CLI order and reject mock/real mixing before judging."""
    items: list[dict] = []
    file_statuses: list[bool] = []
    for items_path in items_paths:
        file_items = [
            json.loads(line)
            for line in items_path.read_text().splitlines()
            if line.strip()
        ]
        _validate_items(file_items)
        file_statuses.append(_mock_status(file_items, items_path))
        items.extend(file_items)

    if len(set(file_statuses)) > 1:
        raise ValueError("--items inputs mix mock and real files")
    return items, file_statuses[0]


def _validate_multi_input_balance(items: list[dict]) -> None:
    """Require repeated inputs to form balanced comparable readout cells.

    Trained arms have self-report rows while N1/N2 do not, and A-B is a
    derived vector with token evidence but no independently steerable model.
    Balancing only the grand total would therefore reject a valid batch whose
    comparable cells are perfectly balanced.  Require every physical arm in
    each primary cell, while allowing derived readouts only where they exist.
    """
    counts = Counter(item["arm"] for item in items)
    if len(counts) < 2:
        raise ValueError("repeated --items inputs must contain at least two arms")
    domains = {ARM_TO_DOMAIN[arm] for arm in counts}
    if len(domains) < 2:
        raise ValueError(
            "repeated --items inputs must contain at least two distinct true domains"
        )
    by_cell: dict[tuple[str, str], Counter] = {}
    for item in items:
        cell = (str(item["modality"]), str(item.get("snippet_set", "-")))
        by_cell.setdefault(cell, Counter())[item["arm"]] += 1
    for cell, cell_counts in sorted(by_cell.items()):
        physical_arms = set(counts) - DERIVED_READOUT_ARMS
        if cell[0] in {"tokens", "steer"} and not physical_arms.issubset(cell_counts):
            missing = sorted(physical_arms - set(cell_counts))
            raise ValueError(
                f"primary readout cell {cell} is missing arms: {missing}"
            )
        if len(cell_counts) > 1 and len(set(cell_counts.values())) != 1:
            detail = ", ".join(
                f"{arm}={count}" for arm, count in sorted(cell_counts.items())
            )
            raise ValueError(
                "repeated --items inputs must be balanced by arm within each "
                f"modality/snippet cell; {cell} has {detail}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--items",
        required=True,
        action="append",
        help="JSONL of readout items; repeat for a balanced multi-arm judge batch",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="use seeded random labels; never access OpenRouter or its credentials",
    )
    args = ap.parse_args()

    items_paths = [Path(value) for value in args.items]
    items, input_is_mock = _load_item_files(items_paths)
    if len({item["arm"] for item in items}) > 1:
        _validate_multi_input_balance(items)

    # Independent streams make predicted labels stable if the control construction changes.
    prediction_rng = random.Random(args.seed)
    control_rng = random.Random(args.seed ^ 0x5EED5EED)
    # shuffled-label control: permute true labels across items
    true = [ARM_TO_DOMAIN[it["arm"]] for it in items]
    perm = true[:]
    control_rng.shuffle(perm)
    if len(set(true)) > 1 and perm == true:
        # An identity shuffle can occur by chance. Rotation is still a permutation and
        # guarantees that a multi-domain control is not identical to the real labels.
        perm = perm[1:] + perm[:1]
    shuffled_control_valid = len(set(true)) > 1 and perm != true
    if items and not shuffled_control_valid:
        print(
            "warning: shuffled-label control is degenerate because the input contains "
            "only one true domain; judge a combined multi-arm batch for a valid control",
            file=sys.stderr,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    git_commit = _git_commit()
    judge_model = DRY_RUN_MODEL if args.dry_run else args.model
    temporary = out.with_name(f".{out.name}.tmp")
    try:
        with temporary.open("w") as f:
            for it, t, s in zip(items, true, perm):
                judge_prompt = _user_prompt(it["text"], it["modality"])
                if args.dry_run:
                    pred = prediction_rng.choice(LABELS)
                    raw_response = pred
                else:
                    pred, raw_response = _ask_with_raw(args.model, it["text"], it["modality"])
                row = {
                    **it,
                    "judge_model": judge_model,
                    "judge_mode": "dry_run" if args.dry_run else "openrouter",
                    "judge_system_prompt": SYSTEM,
                    "judge_prompt": judge_prompt,
                    "raw_response": raw_response,
                    "pred": pred,
                    "true": t,
                    "shuffled_true": s,
                    "correct": pred == t,
                    "correct_shuffled": pred == s,
                    "shuffled_control_valid": shuffled_control_valid,
                    "timestamp": timestamp,
                    "ts": timestamp,  # backwards-compatible alias for the original schema
                    "readout_git_commit": it.get("git_commit"),
                    "judge_git_commit": git_commit,
                    "git_commit": it.get("git_commit") or git_commit,
                    "judge_seed": args.seed,
                    # A dry-run prediction is mock even when its input evidence is real.
                    "is_mock": input_is_mock or args.dry_run,
                }
                if args.dry_run:
                    row["mock_reason"] = "seeded_random_judge_labels"
                f.write(json.dumps(row) + "\n")
        os.replace(temporary, out)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
