#!/usr/bin/env python3
"""Stopping-robust re-scoring of held-out accuracy files.

The base model continues past its own answer with fresh, unrelated questions;
the preregistered last-number parser then reads a number out of that
continuation.  This truncates each completion at the first line that starts a
new question, then applies the same last-number extractor.

Cut patterns (Guiv's specification, 2026-09-04):
  ^What is                     (line start)
  ^Solve                       (line start)
  ^The following are questions (line start)
  Answer:                      only when a #### or \\boxed{} answer is already complete

--variant adds ^Question: / ^Problem: (the document separators of
data/math_sft.jsonl) and is reported separately; it is NOT the primary.

Writes results/acc_table_reparsed.md.  acc_table.md is never modified.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from grpo.train_grpo import extract_answer  # the preregistered last-number parser
from tools.acc_table import mcnemar_exact

LINE_STARTS = (r"What is", r"Solve", r"The following are questions")
DONE = re.compile(r"####\s*-?\d|\\boxed\{")


def cut(completion: str, variant: bool = False) -> tuple[str, str | None]:
    """Return (truncated_completion, name_of_pattern_that_fired_or_None)."""
    starts = list(LINE_STARTS) + ([r"Question:", r"Problem:"] if variant else [])
    lines = completion.split("\n")
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        for pat in starts:
            if re.match(pat, stripped):
                return "\n".join(lines[:idx]), f"^{pat}"
        if re.match(r"Answer:", stripped) and DONE.search("\n".join(lines[:idx])):
            return "\n".join(lines[:idx]), "Answer: after a completed ####/boxed"
    return completion, None


def score(path: str, variant: bool):
    d = json.loads((REPO / path).read_text())
    out = {}
    for r in d["predictions"]:
        raw_ok = r["correct"]
        trunc, pat = cut(r["completion"], variant)
        re_ok = extract_answer(trunc) == r["gold"]
        out[r["dataset_index"]] = {"raw": raw_ok, "re": re_ok, "fired": pat,
                                   "raw_parsed": r["parsed_answer"], "re_parsed": extract_answer(trunc), "gold": r["gold"]}
    return d, out


def mcnemar(x, y, key):
    ks = sorted(set(x) & set(y))
    b = sum(x[i][key] and not y[i][key] for i in ks)
    c = sum(y[i][key] and not x[i][key] for i in ks)
    return b, c, mcnemar_exact(b, c), sum(x[i][key] and y[i][key] for i in ks), sum((not x[i][key]) and (not y[i][key]) for i in ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A:results/acc_A_s0.json", "D_math:results/acc_D_math_s0.json",
                                                  "D_math_full:results/acc_D_math_full_s0.json", "C:results/acc_C_s0.json",
                                                  "D:results/acc_D_s0.json", "B:results/acc_B_s0.json", "base:results/acc_base_s0.json"])
    ap.add_argument("--variant", action="store_true")
    ap.add_argument("--out", default="results/acc_table_reparsed.md")
    args = ap.parse_args()
    scored, meta = {}, {}
    for spec in args.arms:
        label, path = spec.split(":", 1)
        if not (REPO / path).exists():
            print(f"missing input, skipping {label}: {path}", file=sys.stderr); continue
        meta[label], scored[label] = score(path, args.variant)
    lines = ["# Stopping-robust re-scoring of held-out accuracy", "",
             "Same 200 GSM8K test items, same completions, same last-number extractor.",
             "'re-scored' first truncates each completion at the first line starting a NEW question",
             "(^What is, ^Solve, ^The following are questions, or Answer: after a completed ####/\\boxed{}),",
             "then extracts. This isolates answers lost to the model continuing past its own answer.",
             "acc_table.md (raw) is unchanged; both are reported.", "",
             "| arm | raw correct | raw acc | re-scored correct | re-scored acc | delta | completions where a cut fired | rescued | broken |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, s in scored.items():
        raw = sum(v["raw"] for v in s.values()); re_ = sum(v["re"] for v in s.values())
        fired = sum(v["fired"] is not None for v in s.values())
        resc = sum(v["re"] and not v["raw"] for v in s.values()); brok = sum(v["raw"] and not v["re"] for v in s.values())
        n = len(s)
        lines.append(f"| {label} | {raw}/{n} | {raw/n:.3f} | {re_}/{n} | {re_/n:.3f} | {re_/n - raw/n:+.3f} | {fired} | {resc} | {brok} |")
    pairs = [("A", "D_math"), ("A", "base"), ("A", "B"), ("A", "C"), ("A", "D_math_full"), ("D_math", "base")]
    lines += ["", "## Paired comparisons on the same 200 items, under both parsers", "",
              "| pair | parser | x acc | y acc | x-only | y-only | both | neither | McNemar exact p |", "|---|---|---|---|---|---|---|---|---|"]
    for x, y in pairs:
        if x not in scored or y not in scored:
            continue
        for key, name in (("raw", "raw"), ("re", "re-scored")):
            b, c, pv, both, neither = mcnemar(scored[x], scored[y], key)
            xa = sum(v[key] for v in scored[x].values()) / len(scored[x])
            ya = sum(v[key] for v in scored[y].values()) / len(scored[y])
            lines.append(f"| {x} vs {y} | {name} | {xa:.3f} | {ya:.3f} | {b} | {c} | {both} | {neither} | {pv:.6f} |")
    lines += ["", f"Cut-pattern variant with ^Question:/^Problem: added: {'ON' if args.variant else 'OFF (primary)'}.",
              "Sources: " + ", ".join(sorted(spec.split(':', 1)[1] for spec in args.arms if (REPO / spec.split(':', 1)[1]).exists())) + ".",
              "Parser: grpo.train_grpo.extract_answer applied to the truncated text. No interpretation."]
    (REPO / args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if "D_math" in scored:
        from collections import Counter
        print("\ncut patterns that fired for D_math:", Counter(v["fired"] for v in scored["D_math"].values() if v["fired"]).most_common())


if __name__ == "__main__":
    main()
