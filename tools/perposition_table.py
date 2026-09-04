#!/usr/bin/env python3
"""Per-position (0-4) geometry table from cached activations (no model needed).

For each arm (cached adapter activations), snippet set and position p:
raw ||mean diff||, diff constancy, base-activation constancy at p, excess
constancy (diff - base), mean base norm at p; N3 raw norm per position (any-LoRA
floor); N1 as two disjoint 250-snippet halves (seed-0 permutation) per position;
cross-arm cosine matrix per position (plus N1-half vector).  Also caches every
per-position diff vector to results/cache/diffs/ for Patchscope/emergence.

    python tools/perposition_table.py --layer 15 --arms D:250 D_math:225 N3:0 A_early@2:2 ...
       (arm spec = label:step[:seed[:cache_arm]]; cache_arm defaults to label before '@')
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from readout.run_readouts import SNIPPET_SETS  # noqa: E402
from tools.null_decodes import load_cache  # noqa: E402


def constancy(D: np.ndarray) -> float:
    d = D.mean(axis=0)
    return float(np.dot(d, d) / max(float((D * D).sum(1).mean()), 1e-12))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--positions", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--out", default="results/perposition_table")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    L = args.layer; cache_root = Path(args.cache); diff_dir = cache_root / "diffs"; diff_dir.mkdir(exist_ok=True)
    rows, cos_rows, vectors = [], [], {}
    for s in SNIPPET_SETS:
        hb, ali, meta = load_cache(cache_root, L, s); hb = hb.astype(np.float32)
        n = int(meta["n_snippets"]); perm = np.random.default_rng(args.seed).permutation(n)
        half1, half2 = set(perm[: n // 2].tolist()), set(perm[n // 2:].tolist())
        for p in args.positions:
            rows_p = ali[:, 2] == p; hp = hb[rows_p]; snip = ali[rows_p, 0]
            base_c = float(np.dot(hp.mean(0), hp.mean(0)) / max(float((hp * hp).sum(1).mean()), 1e-12))
            base_norm = float(np.linalg.norm(hp, axis=1).mean())
            m1 = hp[np.isin(snip, list(half1))].mean(0); m2 = hp[np.isin(snip, list(half2))].mean(0)
            d_n1 = m1 - m2
            vectors[("N1_halves", s, p)] = d_n1
            rows.append({"arm": "N1_halves", "step": 0, "set": s, "position": p, "raw_norm": float(np.linalg.norm(d_n1)), "constancy": None,
                         "base_constancy": base_c, "excess_constancy": None, "base_norm": base_norm, "n": len(hp),
                         "split_half_floor": None, "cos_halves": None})
            for spec in args.arms:
                parts = spec.split(":"); label = parts[0]; step = int(parts[1]); seed = int(parts[2]) if len(parts) > 2 else 0
                cache_arm = parts[3] if len(parts) > 3 else label.split("@")[0]
                stem = cache_root / f"{cache_arm}_s{seed}_step{step}_L{L}_{s}_adapter"
                ha = np.load(f"{stem}.npy", allow_pickle=False).astype(np.float32)
                ali_a = np.load(f"{stem}_alignment.npy", allow_pickle=False); assert np.array_equal(ali_a, ali)
                D = ha[rows_p] - hp; d = D.mean(0); c = constancy(D)
                vectors[(label, s, p)] = d
                np.save(diff_dir / f"diff_{label}_s{seed}_step{step}_L{L}_{s}_pos{p}.npy", d.astype(np.float32), allow_pickle=False)
                # paired split-half floor: mean diff over snippet half 1 minus mean diff over half 2 (same positions)
                in1 = np.isin(snip, list(half1)); d1 = D[in1].mean(0); d2 = D[~in1].mean(0)
                floor = float(np.linalg.norm(d1 - d2)); cos_halves = cos(d1, d2)
                rows.append({"arm": label, "step": step, "set": s, "position": p, "raw_norm": float(np.linalg.norm(d)), "constancy": c,
                             "base_constancy": base_c, "excess_constancy": c - base_c, "base_norm": base_norm, "n": len(hp),
                             "split_half_floor": floor, "cos_halves": cos_halves})
            labels = [sp.split(":")[0] for sp in args.arms] + ["N1_halves"]
            for a, b in combinations(labels, 2):
                cos_rows.append({"set": s, "position": p, "x": a, "y": b, "cos": cos(vectors[(a, s, p)], vectors[(b, s, p)])})
    out = Path(args.out)
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(f"{out}_cosine.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cos_rows[0].keys())); w.writeheader(); w.writerows(cos_rows)
    md = ["| arm | step | set | pos | raw ‖d‖ | split-half floor | cos(halves) | constancy | base constancy | excess | base ‖h‖ |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    md += [f"| {r['arm']} | {r['step']} | {r['set']} | {r['position']} | {r['raw_norm']:.3f} | {fmt(r['split_half_floor'])} | {fmt(r['cos_halves'])} | {fmt(r['constancy'])} | {r['base_constancy']:.3f} | {fmt(r['excess_constancy'])} | {r['base_norm']:.1f} |" for r in rows]
    # on-domain / neutral raw-norm ratio per arm and position
    by = {(r["arm"], r["set"], r["position"]): r["raw_norm"] for r in rows}
    ratio_rows = [{"arm": a, "position": p, "math_over_neutral": by[(a, "math", p)] / by[(a, "neutral", p)]}
                  for a in dict.fromkeys(r["arm"] for r in rows) for p in args.positions if (a, "math", p) in by and (a, "neutral", p) in by]
    with open(f"{out}_ratio.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "position", "math_over_neutral"]); w.writeheader(); w.writerows(ratio_rows)
    md += ["", "| arm | pos | ‖d_math‖ / ‖d_neutral‖ |", "|---|---|---|"] + [f"| {r['arm']} | {r['position']} | {r['math_over_neutral']:.2f} |" for r in ratio_rows]
    md += ["", "| set | pos | x | y | cos |", "|---|---|---|---|---|"]
    md += [f"| {r['set']} | {r['position']} | {r['x']} | {r['y']} | {r['cos']:.3f} |" for r in cos_rows]
    Path(f"{out}.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
