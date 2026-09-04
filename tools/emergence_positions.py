#!/usr/bin/env python3
"""A_early emergence at positions 0/1 from cached activations, next to the reward curve.

    python tools/emergence_positions.py --arm A_early --seed 1 --steps 2 4 6 8 10 15 20 25 30 --log logs/A_early_s1.log
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from readout.run_readouts import SNIPPET_SETS  # noqa: E402
from tools.null_decodes import load_cache  # noqa: E402
from tools.perposition_table import constancy, cos  # noqa: E402


def rewards_from_log(path: Path) -> dict[int, dict]:
    out = {}
    step = 0
    for line in path.read_text(errors="replace").replace("\r", "\n").splitlines():
        m = re.search(r"\{'loss'.*\}", line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        step += 1
        out[step] = {"reward": float(d.get("reward", "nan")), "truncation_rate": float(d.get("reward/truncation_rate", "nan")),
                     "mean_length": float(d.get("completions/mean_length", "nan"))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A_early"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, nargs="+", default=(2, 4, 6, 8, 10, 15, 20, 25, 30))
    ap.add_argument("--positions", type=int, nargs="+", default=(0, 1))
    ap.add_argument("--ref-arm", default="D"); ap.add_argument("--ref-step", type=int, default=250); ap.add_argument("--ref-seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=15); ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--log", default="logs/A_early_s1.log"); ap.add_argument("--out", default="results/emergence_A_early")
    args = ap.parse_args()
    L = args.layer; cache_root = Path(args.cache); rewards = rewards_from_log(Path(args.log)) if Path(args.log).exists() else {}
    rows = []
    for s in SNIPPET_SETS:
        hb, ali, _ = load_cache(cache_root, L, s); hb = hb.astype(np.float32)
        ref = np.load(f"{cache_root}/{args.ref_arm}_s{args.ref_seed}_step{args.ref_step}_L{L}_{s}_adapter.npy", allow_pickle=False).astype(np.float32)
        for p in args.positions:
            rp = ali[:, 2] == p; d_ref = (ref[rp] - hb[rp]).mean(0)
            for st in args.steps:
                ha = np.load(f"{cache_root}/{args.arm}_s{args.seed}_step{st}_L{L}_{s}_adapter.npy", allow_pickle=False).astype(np.float32)
                D = ha[rp] - hb[rp]; d = D.mean(0)
                r = rewards.get(st, {})
                rows.append({"arm": args.arm, "seed": args.seed, "step": st, "set": s, "position": p, "raw_norm": float(np.linalg.norm(d)),
                             "constancy": constancy(D), f"cos_to_{args.ref_arm}_pos{p}": cos(d, d_ref),
                             "reward_at_step": r.get("reward"), "truncation_at_step": r.get("truncation_rate"), "mean_length_at_step": r.get("mean_length")})
    out = Path(args.out)
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md = [f"| step | set | pos | raw ‖d‖ | constancy | cos to {args.ref_arm} same pos | reward | trunc | len |", "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['step']} | {r['set']} | {r['position']} | {r['raw_norm']:.3f} | {r['constancy']:.3f} | {r[f'cos_to_{args.ref_arm}_pos'+str(r['position'])]:.3f} | {r['reward_at_step']} | {r['truncation_at_step']} | {r['mean_length_at_step']} |" for r in rows]
    Path(f"{out}.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
    Path(f"{out}_rewards.json").write_text(json.dumps(rewards, indent=1) + "\n")


if __name__ == "__main__":
    main()
