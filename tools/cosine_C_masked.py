#!/usr/bin/env python3
"""Cosines of the C_masked L15 per-position diff vector with C s0, C s1, A s0, A s1, D_math (masked), D_math_full (CPU, numpy).
Writes results/perposition_table_C_masked_cosine.csv (keeps the N1_halves rows perposition_table.py wrote to that file)."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np

REFS = {"C_s0": "C_s0_step225", "C_s1": "C_s1_step225", "A_s0": "A_s0_step150", "A_s1": "A_seed1_s1_step150",
        "D_math": "D_math_s0_step225", "D_math_full": "D_math_full_s0_step225"}
SETS = ("neutral", "math"); POS = (0, 1, 2, 3, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-label", default="C_masked"); ap.add_argument("--x-tag", default="C_masked_s0_step225")
    ap.add_argument("--diffs", default="results/cache/diffs"); ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default="results/perposition_table_C_masked_cosine.csv")
    args = ap.parse_args()
    d = Path(args.diffs); L = args.layer
    load = lambda tag, s, p: np.load(d / f"diff_{tag}_L{L}_{s}_pos{p}.npy", allow_pickle=False).astype(np.float64)
    rows = []
    if Path(args.out).exists():
        rows = [{**r, "norm_x": r.get("norm_x", ""), "norm_y": r.get("norm_y", "")} for r in csv.DictReader(open(args.out)) if "N1_halves" in (r["x"], r["y"])]
    for s in SETS:
        for p in POS:
            x = load(args.x_tag, s, p)
            for lab, tag in REFS.items():
                y = load(tag, s, p)
                rows.append({"set": s, "position": p, "x": args.x_label, "y": lab, "cos": float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))),
                             "norm_x": float(np.linalg.norm(x)), "norm_y": float(np.linalg.norm(y))})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "position", "x", "y", "cos", "norm_x", "norm_y"]); w.writeheader(); w.writerows(rows)
    print(f"{args.x_label} cosines (L{L}), p1 / p2:")
    for lab in REFS:
        vals = {(r["set"], r["position"]): r["cos"] for r in rows if r["y"] == lab and r["x"] == args.x_label}
        print(f"  . {lab:12s} neutral {vals[('neutral',1)]:.3f} / {vals[('neutral',2)]:.3f}   math {vals[('math',1)]:.3f} / {vals[('math',2)]:.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
