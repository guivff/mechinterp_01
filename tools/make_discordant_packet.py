#!/usr/bin/env python3
"""Blinded discordant-item packet for independent human tagging.

Writes, for every item where the two arms disagree on the held-out set:
  results/discordant_A_vs_D_math_readable.md  problem, gold, both full completions,
                                              arm identity replaced by X/Y, randomised per item
  results/discordant_key.json                 the unblinding key
  results/discordant_sample20.txt             20 item ids to tag first

Nothing in the readable file names an arm, scores a completion, or shows an
extracted answer.  Completions are copied verbatim and untruncated.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", default="results/acc_A_s0.json")
    ap.add_argument("--y", default="results/acc_D_math_s0.json")
    ap.add_argument("--x-name", default="A")
    ap.add_argument("--y-name", default="D_math")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--out", default="results/discordant_A_vs_D_math_readable.md")
    ap.add_argument("--key", default="results/discordant_key.json")
    ap.add_argument("--sample-out", default="results/discordant_sample20.txt")
    args = ap.parse_args()

    X = json.loads((REPO / args.x).read_text())
    Y = json.loads((REPO / args.y).read_text())
    if X["snippet_sha"] != Y["snippet_sha"]:
        raise SystemExit("different evaluation sets; refusing")
    x = {r["dataset_index"]: r for r in X["predictions"]}
    y = {r["dataset_index"]: r for r in Y["predictions"]}
    ids = sorted(set(x) & set(y))
    disc = [i for i in ids if x[i]["correct"] != y[i]["correct"]]
    missing = [i for i in disc if not x[i]["completion"] or not y[i]["completion"]]
    if missing:
        raise SystemExit(f"missing completion text for items {missing}; refusing to write a partial packet")

    rng = random.Random(args.seed)
    key = {}
    body = [f"# Discordant held-out items ({len(disc)} of {len(ids)}), blinded for independent tagging", "",
            "Two systems answered the same 200 GSM8K test items (greedy, 512-token cap). These are every item",
            "where they disagreed. Per item the two completions are labelled X and Y in a per-item random order;",
            "the mapping is withheld. No scores, no correctness marks and no extracted answers appear below.",
            f"Blinding seed {args.seed}. Completions are verbatim and untruncated.", "",
            "Suggested tags per item: correct-reasoning-wrong-format / wrong-reasoning / did-not-stop / other.", "", "---", ""]
    for i in disc:
        first_is_x = rng.random() < 0.5
        key[str(i)] = {"X": args.x_name if first_is_x else args.y_name,
                       "Y": args.y_name if first_is_x else args.x_name}
        cx = x[i]["completion"] if first_is_x else y[i]["completion"]
        cy = y[i]["completion"] if first_is_x else x[i]["completion"]
        body += [f"### item {i}", "", x[i]["question"].strip(), "", f"gold: {x[i]['gold']}", "",
                 "--- completion X ---", "", "```", cx.rstrip(), "```", "",
                 "--- completion Y ---", "", "```", cy.rstrip(), "```", "", "---", ""]
    (REPO / args.out).write_text("\n".join(body) + "\n")
    (REPO / args.key).write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
    sample = sorted(random.Random(args.seed).sample(disc, min(args.sample, len(disc))))
    (REPO / args.sample_out).write_text("\n".join(str(i) for i in sample) + "\n")
    print(f"discordant items: {len(disc)} (all with full text)")
    print(f"wrote {args.out} ({(REPO / args.out).stat().st_size} bytes), {args.key}, {args.sample_out} ({len(sample)} ids)")


if __name__ == "__main__":
    main()
