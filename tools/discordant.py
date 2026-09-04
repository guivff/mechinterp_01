#!/usr/bin/env python3
"""List discordant held-out items between two accuracy files (same eval set) for human reading.

    python tools/discordant.py results/acc_A_s0.json results/acc_D_math_s0.json --n 20 --out results/discordant_A_vs_D_math.md
"""
import argparse, json, random
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("x"); ap.add_argument("y"); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True); a = ap.parse_args()
X, Y = json.loads(Path(a.x).read_text()), json.loads(Path(a.y).read_text())
assert X["snippet_sha"] == Y["snippet_sha"], "different evaluation sets"
px = {r["dataset_index"]: r for r in X["predictions"]}; py = {r["dataset_index"]: r for r in Y["predictions"]}
disc = [i for i in px if px[i]["correct"] != py[i]["correct"]]
random.Random(a.seed).shuffle(disc); chosen = sorted(disc[: a.n])
lines = [f"# Discordant items: {X['arm']} vs {Y['arm']} ({len(disc)} discordant of {len(px)}; showing {len(chosen)}, seed {a.seed})", ""]
for i in chosen:
    r, q = px[i], py[i]
    lines += [f"## item {i} — gold {r['gold']} — {X['arm']} {'✓' if r['correct'] else '✗'} (parsed {r['parsed_answer']}) | {Y['arm']} {'✓' if q['correct'] else '✗'} (parsed {q['parsed_answer']})",
              "", "**Question:** " + r["question"], "", f"**{X['arm']} completion:**", "```", r["completion"][:1500], "```",
              f"**{Y['arm']} completion:**", "```", q["completion"][:1500], "```", ""]
Path(a.out).write_text("\n".join(lines) + "\n"); print("wrote", a.out, f"({len(disc)} discordant)")
