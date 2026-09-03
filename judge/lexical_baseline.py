"""Lexical baseline: can a TF-IDF + logistic-regression model predict the domain from the
same readout texts? If it matches the LLM judge, the "readability" is surface tokens.

Also prints a summary of judge results (accuracy per arm/modality/snippet set, and the
shuffled-label control accuracy, which must sit at chance).

Usage: python judge/lexical_baseline.py --judged results/judged.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline


REQUIRED_FIELDS = {
    "arm",
    "modality",
    "text",
    "true",
    "correct",
    "correct_shuffled",
}
LABELS = ("math", "cooking", "law", "medicine", "poetry", "none")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def load_rows(path: Path) -> list[dict]:
    """Load and validate the judged JSONL used by both report sections."""
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(
                f"{path}:{line_number}: missing required fields: {sorted(missing)}"
            )
        if not isinstance(row["arm"], str) or not row["arm"]:
            raise ValueError(f"{path}:{line_number}: arm must be a non-empty string")
        if not isinstance(row["modality"], str) or not row["modality"]:
            raise ValueError(
                f"{path}:{line_number}: modality must be a non-empty string"
            )
        if not isinstance(row["text"], str):
            raise ValueError(f"{path}:{line_number}: text must be a string")
        if not isinstance(row["true"], str) or not row["true"]:
            raise ValueError(f"{path}:{line_number}: true must be a non-empty string")
        for field in ("correct", "correct_shuffled"):
            if type(row[field]) is not bool:
                raise ValueError(f"{path}:{line_number}: {field} must be boolean")
        if "is_mock" in row and type(row["is_mock"]) is not bool:
            raise ValueError(f"{path}:{line_number}: is_mock must be boolean")
        if row.get("shuffled_control_valid") is not True:
            raise ValueError(f"{path}:{line_number}: invalid shuffled-label control")
        rows.append(row)

    if not rows:
        raise ValueError(f"{path}: contains no JSONL rows")
    mock_statuses = {row["is_mock"] for row in rows if "is_mock" in row}
    if len(mock_statuses) > 1:
        raise ValueError(f"{path}: mixes mock and real rows")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        rows = load_rows(Path(args.judged))
    except (OSError, ValueError) as exc:
        ap.error(str(exc))

    # ---- judge summary
    acc = defaultdict(list)
    sh = defaultdict(list)
    for r in rows:
        key = (r["arm"], r["modality"], r.get("snippet_set", "-"))
        acc[key].append(r["correct"])
        sh[key].append(r["correct_shuffled"])
    print(f"{'arm':>4} {'modality':>10} {'snips':>10} {'n':>4} {'judge_acc':>9} {'shuffled':>9}")
    for k in sorted(acc):
        print(f"{k[0]:>4} {k[1]:>10} {k[2]:>10} {len(acc[k]):>4} {np.mean(acc[k]):9.3f} {np.mean(sh[k]):9.3f}")

    # ---- lexical baseline, per modality (tokens vs steer differ in surface form)
    for modality in sorted({r["modality"] for r in rows}):
        sub = [r for r in rows if r["modality"] == modality]
        X = [r["text"] for r in sub]
        y = [r["true"] for r in sub]
        counts = Counter(y)
        if len(counts) < 2:
            print(f"[lexical] {modality}: need at least two labels for CV")
            continue
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            LogisticRegression(
                max_iter=2000,
                random_state=args.seed,
                solver="lbfgs",
            ),
        )
        input_is_mock = all(row.get("is_mock") is True for row in sub)
        if not input_is_mock and min(counts.values()) < 5:
            ap.error(
                f"lexical {modality!r} cannot run the preregistered 5-fold CV; "
                f"smallest class has {min(counts.values())} rows"
            )
        k = min(5, min(counts.values()))
        if k < 2:
            print(f"[lexical] {modality}: too few per class for CV")
            continue
        cv = StratifiedKFold(n_splits=k, shuffle=False)
        try:
            scores = cross_val_score(
                clf,
                X,
                y,
                cv=cv,
                scoring="accuracy",
                n_jobs=1,
            )
        except ValueError as exc:
            ap.error(f"lexical CV failed for modality {modality!r}: {exc}")
        chance = 1.0 / len(LABELS)
        majority = max(counts.values()) / len(y)
        print(
            f"[lexical] {modality}: {k}-fold acc={scores.mean():.3f} ± {scores.std():.3f} "
            f"(fixed six-label chance {chance:.3f}; observed majority {majority:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
