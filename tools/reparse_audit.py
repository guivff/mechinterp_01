#!/usr/bin/env python3
"""Audit the stopping-robust re-parser: are its rescues genuine?

A 'rescue' is an item the preregistered last-number parser scores wrong and the
truncated re-parse scores right.  The failure mode to rule out is a
correct-looking number appearing in the truncated text by coincidence rather
than as the model's stated answer.

Samples rescued items across the given arms with a fixed seed and writes each
one in full: the untruncated completion, the cut point, the text the re-scorer
sees, and both extracted answers.  Also records an automatic signal --
whether the gold value appears in an explicit answer statement (####, boxed,
'Answer:', 'answer is', 'Therefore ...') inside the truncated text -- which is
evidence for 'genuine', but the human classification is the one that counts.

    python tools/reparse_audit.py --n 20 --seed 20260904
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from grpo.train_grpo import extract_answer
from tools.reparse_acc import cut

NUM = r"-?\d[\d,]*(?:\.\d+)?"


def explicit_answer_hits(text: str, gold: str) -> list[str]:
    """Places where the truncated text states `gold` as an answer.

    The gold value is matched with optional thousands separators and an
    optional currency sign, because models write 18,000 / $18,000 for gold
    "18000"; an earlier, narrower version of this function produced six
    spurious "no signal" flags that were all genuine on reading.
    """
    digits = gold.lstrip("-")
    grouped = re.sub(r"(?<=\d)(?=(\d{3})+$)", ",?", digits)
    g = ("-?" if gold.startswith("-") else "") + grouped
    val = rf"\$?\s*{g}\b"
    pats = {
        "#### gold": rf"####\s*{val}",
        "boxed gold": rf"\\boxed\{{[^}}]{{0,40}}?{val}",
        "Answer: gold": rf"(?:final\s+)?answer\s*[:=]?\s*\**\s*{val}",
        "answer is gold": rf"answer\s+(?:is|=)\s*\**\s*{val}",
        "Therefore/Thus ... gold": rf"(?:therefore|thus|so),[^\n]{{0,120}}?{val}",
        "total ... gold": rf"total[^\n]{{0,80}}?{val}",
        "gold alone on the final line": rf"(?:^|\n)[^\n]{{0,12}}{val}[^\n]{{0,20}}$",
    }
    return [name for name, p in pats.items() if re.search(p, text, re.I | re.M)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["base:results/acc_base_s0.json",
                                                  "B:results/acc_B_s0.json",
                                                  "D_math:results/acc_D_math_s0.json"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", default="results/reparse_audit.md")
    args = ap.parse_args()

    pool = []
    totals = {}
    for spec in args.arms:
        label, path = spec.split(":", 1)
        if not (REPO / path).exists():
            print(f"missing input, skipping {label}: {path}", file=sys.stderr)
            continue
        d = json.loads((REPO / path).read_text())
        resc = []
        for r in d["predictions"]:
            if r["correct"]:
                continue
            trunc, pat = cut(r["completion"])
            if extract_answer(trunc) == r["gold"]:
                resc.append((label, path, r, trunc, pat))
        totals[label] = len(resc)
        pool.extend(resc)
    if not pool:
        raise SystemExit("no rescued items found")
    # Stratify so every arm is represented: the arm with the fewest rescues
    # (D_math) is the one whose re-scored number feeds the headline comparison.
    rng = random.Random(args.seed)
    by_arm: dict[str, list] = {}
    for item in pool:
        by_arm.setdefault(item[0], []).append(item)
    per = max(1, args.n // len(by_arm))
    sample = []
    for label in sorted(by_arm):
        sample.extend(rng.sample(by_arm[label], min(per, len(by_arm[label]))))
    remaining = [it for it in pool if it not in sample]
    if len(sample) < args.n and remaining:
        sample.extend(rng.sample(remaining, min(args.n - len(sample), len(remaining))))
    sample.sort(key=lambda t: (t[0], t[2]["dataset_index"]))

    lines = ["# Re-parser audit: are the rescues genuine?", "",
             f"Rescued = the preregistered last-number parser scores the item wrong and the truncated re-parse scores it right.",
             f"Population: " + ", ".join(f"{k} {v}" for k, v in totals.items()) + f" (total {len(pool)}).",
             f"Sample: {len(sample)} drawn with random.Random({args.seed}), stratified so every arm is represented.", "",
             "For each item: the full untruncated completion, the cut pattern, the text the re-scorer sees,",
             "and both extracted answers. 'explicit answer statement' lists places where the truncated text",
             "states the gold value as an answer; it is an automatic signal, not the classification.", "",
             "Classify each as GENUINE (the model completed a correct solution before the cut) or",
             "FALSE RESCUE (a correct-looking number appears before the cut without being the stated answer).", "", "---", ""]
    auto = {"genuine_signal": 0, "no_signal": 0}
    for label, path, r, trunc, pat in sample:
        hits = explicit_answer_hits(trunc, r["gold"])
        auto["genuine_signal" if hits else "no_signal"] += 1
        lines += [f"### {label} item {r['dataset_index']}", "",
                  f"**gold:** {r['gold']}  |  **last-number parse of full text:** {r['parsed_answer']}  |  "
                  f"**re-parse of truncated text:** {extract_answer(trunc)}", "",
                  f"**cut pattern:** `{pat}`  |  full {len(r['completion'])} chars -> truncated {len(trunc)} chars", "",
                  f"**explicit answer statement for gold in truncated text:** "
                  + (", ".join(f"`{h}`" for h in hits) if hits else "**NONE FOUND**"), "",
                  "**Question:**", "", r["question"].strip(), "",
                  "**Text the re-scorer sees (truncated):**", "", "```", trunc.strip(), "```", "",
                  "**Discarded tail (everything after the cut):**", "",
                  "```", (r["completion"][len(trunc):].strip()[:1200] or "(nothing)"), "```", "",
                  "**Classification:** _______________  (GENUINE / FALSE RESCUE)", "", "---", ""]
    (REPO / args.out).write_text("\n".join(lines) + "\n")
    print(f"population: {totals} total {len(pool)}")
    print(f"sampled {len(sample)}: " + ", ".join(f"{l} {r['dataset_index']}" for l, _, r, _, _ in sample))
    print(f"automatic signal: explicit gold answer statement present in {auto['genuine_signal']}/{len(sample)}, absent in {auto['no_signal']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
