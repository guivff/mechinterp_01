#!/usr/bin/env python3
"""Paired held-out accuracy table across arms, parse modes and completion caps.

Reads ``results/acc_*.json`` (written by ``grpo/eval_acc.py``), re-scores every
stored completion under both parse modes (last number = training verifier;
first number after "Answer:"), groups by ``max_new_tokens``, and reports for
every arm pair with the same evaluation set the paired counts and an exact
McNemar p-value (two-sided binomial on the discordant pairs).  No model is run.

    python tools/acc_table.py --out results/acc_table
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grpo.eval_acc import PARSE_MODES  # noqa: E402
from grpo.train_grpo import gold_answer  # noqa: E402


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p on discordant counts b (x right, y wrong) and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/acc_table")
    args = ap.parse_args()
    runs = []
    for path in sorted(glob.glob(str(Path(args.results) / "acc_*.json"))):
        d = json.loads(Path(path).read_text())
        if "predictions" not in d:
            continue
        max_new = int(d["decoding"]["max_new_tokens"])
        rows = d["predictions"]
        per_mode = {}
        for mode, fn in PARSE_MODES.items():
            correct = {}
            for r in rows:
                gold = r["gold"]
                parsed = fn(r["completion"])
                correct[int(r["dataset_index"])] = parsed == gold
            per_mode[mode] = correct
        runs.append({"file": Path(path).name, "arm": d["arm"], "seed": d["seed"], "step": d.get("step"), "max_new": max_new,
                     "set_sha": d["snippet_sha"], "n": d["n"], "correct": per_mode, "git_commit": d.get("git_commit")})
    single, paired = [], []
    for r in runs:
        for mode in PARSE_MODES:
            k = sum(r["correct"][mode].values())
            single.append({"file": r["file"], "arm": r["arm"], "seed": r["seed"], "step": r["step"], "max_new": r["max_new"],
                           "parse": mode, "n": r["n"], "n_correct": k, "accuracy": round(k / r["n"], 4)})
    for a, b in combinations(runs, 2):
        if a["max_new"] != b["max_new"] or a["set_sha"] != b["set_sha"] or a["arm"] == b["arm"]:
            continue
        for mode in PARSE_MODES:
            ca, cb = a["correct"][mode], b["correct"][mode]
            keys = sorted(set(ca) & set(cb))
            both = sum(ca[i] and cb[i] for i in keys); only_a = sum(ca[i] and not cb[i] for i in keys)
            only_b = sum(cb[i] and not ca[i] for i in keys); neither = sum((not ca[i]) and (not cb[i]) for i in keys)
            paired.append({"max_new": a["max_new"], "parse": mode, "arm_x": a["arm"], "arm_y": b["arm"], "n": len(keys),
                           "acc_x": round(sum(ca[i] for i in keys) / len(keys), 4), "acc_y": round(sum(cb[i] for i in keys) / len(keys), 4),
                           "both": both, "x_only": only_a, "y_only": only_b, "neither": neither,
                           "mcnemar_exact_p": round(mcnemar_exact(only_a, only_b), 5)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, rows in (("single", single), ("paired", paired)):
        if not rows:
            continue
        with open(f"{out}_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md = ["| arm | seed | step | max_new | parse | correct/n | acc |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['arm']} | {r['seed']} | {r['step']} | {r['max_new']} | {r['parse']} | {r['n_correct']}/{r['n']} | {r['accuracy']:.3f} |" for r in single]
    md += ["", "| max_new | parse | x | y | acc_x | acc_y | both | x only | y only | neither | McNemar exact p |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['max_new']} | {r['parse']} | {r['arm_x']} | {r['arm_y']} | {r['acc_x']:.3f} | {r['acc_y']:.3f} | {r['both']} | {r['x_only']} | {r['y_only']} | {r['neither']} | {r['mcnemar_exact_p']} |" for r in paired]
    Path(f"{out}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
