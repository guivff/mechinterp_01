#!/usr/bin/env python3
"""Cross-seed geometry for arm C seed 1 (CPU, numpy only; no model, no adapter).

Reads per-position diff vectors from results/cache/diffs/ (L15, positions 0-4, neutral and math):
  C seed 1 (diff_C_s1_step225_*), C seed 0 (diff_C_s0_step225_*), A seed 0 (diff_A_s0_step150_*), A seed 1 (diff_A_seed1_s1_step150_*).
Writes (new files only):
  results/perposition_table_C_seeds_cosine.csv   cosines C s1 . {C s0, A s0, A s1} (+ C s0 . A s0/A s1, A s0 . A s1 for context)
  results/trace_ratio_C_A_seeds.csv              ||d_C|| / ||d_A|| per (C seed x A seed), set, position
  results/perposition_table_C_seeds.csv          C rows of results/perposition_table_C.csv (seed 0) + C rows of
                                                 results/perposition_table_C_s1.csv (seed 1), with a `seed` column
  results/perposition_table_C_seeds.meta.json    provenance (git commit, timestamp, input file sha256s)
"""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SETS = ("neutral", "math")
POS = (0, 1, 2, 3, 4)
TAGS = {"C_s1": "C_s1_step225", "C_s0": "C_s0_step225", "A_s0": "A_s0_step150", "A_s1": "A_seed1_s1_step150"}


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffs", default="results/cache/diffs")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--table-s0", default="results/perposition_table_C.csv")
    ap.add_argument("--table-s1", default="results/perposition_table_C_s1.csv")
    ap.add_argument("--out-cos", default="results/perposition_table_C_seeds_cosine.csv")
    ap.add_argument("--out-ratio", default="results/trace_ratio_C_A_seeds.csv")
    ap.add_argument("--out-table", default="results/perposition_table_C_seeds.csv")
    ap.add_argument("--out-meta", default="results/perposition_table_C_seeds.meta.json")
    args = ap.parse_args()
    d = Path(args.diffs); L = args.layer
    vec, files = {}, {}
    for lab, tag in TAGS.items():
        for s in SETS:
            for p in POS:
                f = d / f"diff_{tag}_L{L}_{s}_pos{p}.npy"
                vec[(lab, s, p)] = np.load(f, allow_pickle=False).astype(np.float64)
                files[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    cos_rows = []
    for s in SETS:
        for p in POS:
            for x, y in combinations(TAGS, 2):
                cos_rows.append({"set": s, "position": p, "x": x, "y": y, "cos": cos(vec[(x, s, p)], vec[(y, s, p)]),
                                 "norm_x": float(np.linalg.norm(vec[(x, s, p)])), "norm_y": float(np.linalg.norm(vec[(y, s, p)]))})
    with open(args.out_cos, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cos_rows[0])); w.writeheader(); w.writerows(cos_rows)
    ratio_rows = []
    for c in ("C_s0", "C_s1"):
        for a in ("A_s0", "A_s1"):
            for s in SETS:
                for p in POS:
                    nc = float(np.linalg.norm(vec[(c, s, p)])); na = float(np.linalg.norm(vec[(a, s, p)]))
                    ratio_rows.append({"set": s, "position": p, "c": c, "a": a, "norm_c": nc, "norm_a": na, "c_over_a": nc / na})
    with open(args.out_ratio, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ratio_rows[0])); w.writeheader(); w.writerows(ratio_rows)
    # merged geometry table (seed column added); N1_halves rows kept from both sources for base-cache comparison
    table_rows = []
    for seed, path in ((0, args.table_s0), (1, args.table_s1)):
        if not Path(path).exists():
            continue
        files[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for r in csv.DictReader(open(path)):
            if r["arm"] in ("C", "N1_halves"):
                table_rows.append({"arm": r["arm"], "seed": seed if r["arm"] == "C" else f"base_cache_run_{seed}", **{k: v for k, v in r.items() if k != "arm"}})
    if table_rows:
        with open(args.out_table, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table_rows[0])); w.writeheader(); w.writerows(table_rows)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    Path(args.out_meta).write_text(json.dumps({"arm": "C", "seeds": [0, 1], "steps": {"C": 225, "A": 150}, "layer": L, "positions": list(POS),
        "snippet_sets": list(SETS), "judge_model": None, "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "inputs_sha256": files, "outputs": [args.out_cos, args.out_ratio, args.out_table]}, indent=1) + "\n")
    print("HEADLINE  C_s1 . C_s0 cosines (L%d):" % L)
    for r in cos_rows:
        if {r["x"], r["y"]} == {"C_s1", "C_s0"} and r["position"] in (1, 2):
            print(f"  {r['set']:8s} p{r['position']}  cos={r['cos']:.4f}   ||d_C_s1||={r['norm_x']:.3f} ||d_C_s0||={r['norm_y']:.3f}")
    for pair in (("C_s1", "A_s0"), ("C_s1", "A_s1"), ("C_s0", "A_s0"), ("C_s0", "A_s1"), ("A_s0", "A_s1")):
        for r in cos_rows:
            if (r["x"], r["y"]) == pair and r["position"] in (1, 2):
                print(f"  {pair[0]}.{pair[1]} {r['set']:8s} p{r['position']} cos={r['cos']:.4f}")
    print("trace ratios ||d_C||/||d_A|| neutral p1:", {(r["c"], r["a"]): round(r["c_over_a"], 2) for r in ratio_rows if r["set"] == "neutral" and r["position"] == 1})
    print("wrote", args.out_cos, args.out_ratio, args.out_table, args.out_meta)


if __name__ == "__main__":
    main()
