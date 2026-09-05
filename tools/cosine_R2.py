#!/usr/bin/env python3
"""Cosines of each R2 arm's L15 per-position diff with every existing arm and with each other (CPU, numpy).
Writes results/perposition_table_R2_cosine.csv; also concatenates the per-arm geometry tables into results/perposition_table_R2.csv."""
from __future__ import annotations
import argparse, csv
from itertools import combinations
from pathlib import Path
import numpy as np

R2 = {"C_masked_s1": "C_masked_s1_step225", "C_scrambled_s0": "C_scrambled_s0_step225", "C_shifted_s0": "C_shifted_s0_step225"}
REFS = {"C_s0": "C_s0_step225", "C_s1": "C_s1_step225", "C_masked_s0": "C_masked_s0_step225", "A_s0": "A_s0_step150", "A_s1": "A_seed1_s1_step150",
        "D_math": "D_math_s0_step225", "D_math_full": "D_math_full_s0_step225", "D": "D_s0_step250", "B": "B_s0_step150"}
TABLES = {"C_masked_s1": "results/perposition_table_C_masked_s1.csv", "C_scrambled_s0": "results/perposition_table_C_scrambled_s0.csv", "C_shifted_s0": "results/perposition_table_C_shifted_s0.csv"}
SETS = ("neutral", "math"); POS = (0, 1, 2, 3, 4)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--diffs", default="results/cache/diffs"); ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default="results/perposition_table_R2_cosine.csv"); ap.add_argument("--table-out", default="results/perposition_table_R2.csv"); args = ap.parse_args()
    d = Path(args.diffs); L = args.layer
    present = {k: v for k, v in R2.items() if (d / f"diff_{v}_L{L}_neutral_pos1.npy").exists()}
    load = lambda tag, s, p: np.load(d / f"diff_{tag}_L{L}_{s}_pos{p}.npy", allow_pickle=False).astype(np.float64)
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rows = []
    for s in SETS:
        for p in POS:
            for x, xt in present.items():
                vx = load(xt, s, p)
                for y, yt in REFS.items():
                    vy = load(yt, s, p); rows.append({"set": s, "position": p, "x": x, "y": y, "cos": cos(vx, vy), "norm_x": float(np.linalg.norm(vx)), "norm_y": float(np.linalg.norm(vy))})
            for (x, xt), (y, yt) in combinations(present.items(), 2):
                vx, vy = load(xt, s, p), load(yt, s, p); rows.append({"set": s, "position": p, "x": x, "y": y, "cos": cos(vx, vy), "norm_x": float(np.linalg.norm(vx)), "norm_y": float(np.linalg.norm(vy))})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "position", "x", "y", "cos", "norm_x", "norm_y"]); w.writeheader(); w.writerows(rows)
    trows = []
    for lab, t in TABLES.items():
        if Path(t).exists():
            for r in csv.DictReader(open(t)):
                trows.append({"run": lab, **r})
    if trows:
        with open(args.table_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trows[0])); w.writeheader(); w.writerows(trows)
    for x in present:
        print(f"{x} cosines p1/p2 (neutral | math):")
        for y in list(REFS) + [k for k in present if k != x]:
            v = {(r["set"], r["position"]): r["cos"] for r in rows if {r["x"], r["y"]} == {x, y}}
            if v: print(f"  . {y:12s} {v[('neutral',1)]:.3f} / {v[('neutral',2)]:.3f} | {v[('math',1)]:.3f} / {v[('math',2)]:.3f}")
    print("wrote", args.out, args.table_out)


if __name__ == "__main__":
    main()
