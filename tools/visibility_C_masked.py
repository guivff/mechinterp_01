#!/usr/bin/env python3
"""Visibility V = ||d_neutral,p1|| / ||dW||_F at L15 for C_masked beside C s0/s1, A s0/s1, D_math, D_math_full (CPU, numpy).
||d|| is the norm of the cached per-position diff vector (results/cache/diffs); floors from the per-position tables.
Writes results/visibility_table_C_masked.md (+ .json) and prints the decision line for C_masked."""
from __future__ import annotations
import argparse, csv, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# label: (lora stats file, key, diff tag, (table csv, arm label in that table))
ARMS = {
    "C_masked": ("results/lora_delta_stats_C_masked.json", "C_masked", "C_masked_s0_step225", ("results/perposition_table_C_masked.csv", "C_masked")),
    "C s0": ("results/lora_delta_stats.json", "C", "C_s0_step225", ("results/perposition_table_C.csv", "C")),
    "C s1": ("results/lora_delta_stats_C_s1.json", "C_s1", "C_s1_step225", ("results/perposition_table_C_s1.csv", "C")),
    "A s0": ("results/lora_delta_stats.json", "A", "A_s0_step150", ("results/perposition_table_A_seeds.csv", "A")),
    "A s1": ("results/lora_delta_stats.json", "A_s1", "A_seed1_s1_step150", ("results/perposition_table_A_seeds.csv", "A_seed1")),
    "D_math (masked)": ("results/lora_delta_stats.json", "D_math", "D_math_s0_step225", ("results/perposition_table_seeds.csv", "D_math")),
    "D_math_full": ("results/lora_delta_stats.json", "D_math_full", "D_math_full_s0_step225", ("results/perposition_table_seeds.csv", "D_math_full")),
}
TABLES = ["results/perposition_table_seeds.csv", "results/perposition_table.csv", "results/perposition_table_C.csv", "results/perposition_table_A_seeds.csv"]


def floor_for(table: str, arm: str, s: str, p: int):
    for t in [table] + TABLES:
        if not Path(t).exists():
            continue
        for r in csv.DictReader(open(t)):
            if r["arm"] == arm and r["set"] == s and int(r["position"]) == p and r.get("split_half_floor"):
                return float(r["split_half_floor"]), t
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffs", default="results/cache/diffs")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default="results/visibility_table_C_masked.md")
    ap.add_argument("--skip-missing", action="store_true")
    args = ap.parse_args()
    rows = []
    for label, (lf, key, tag, (table, tarm)) in ARMS.items():
        try:
            fro = json.loads(Path(lf).read_text())[key]["delta_W_fro_total"]
            d_n = np.load(Path(args.diffs) / f"diff_{tag}_L{args.layer}_neutral_pos1.npy", allow_pickle=False).astype(np.float64)
            d_m = np.load(Path(args.diffs) / f"diff_{tag}_L{args.layer}_math_pos1.npy", allow_pickle=False).astype(np.float64)
        except (FileNotFoundError, KeyError) as e:
            if args.skip_missing:
                print("skip", label, e); continue
            raise
        fl_n, src_n = floor_for(table, tarm, "neutral", 1); fl_m, _ = floor_for(table, tarm, "math", 1)
        rows.append({"arm": label, "diff_tag": tag, "lora_file": lf, "lora_key": key, "dW_fro": fro, "d_neutral_p1": float(np.linalg.norm(d_n)),
                     "floor_neutral_p1": fl_n, "floor_source": src_n, "d_math_p1": float(np.linalg.norm(d_m)), "floor_math_p1": fl_m,
                     "V_neutral_p1": float(np.linalg.norm(d_n)) / fro, "V_math_p1": float(np.linalg.norm(d_m)) / fro})
    f = lambda v, k=3: "—" if v is None else f"{v:.{k}f}"
    md = [f"# Visibility V = ‖d_neutral,p1‖ / ‖ΔW‖_F at L{args.layer} (C_masked = completion-only loss on data/C_samples.jsonl)", "",
          f"Generated {datetime.now(timezone.utc).isoformat()}. ‖d‖ = norm of the cached per-position mean diff (results/cache/diffs); floor = paired split-half floor from the per-position tables.", "",
          "| arm | ‖ΔW‖_F | ‖d_neutral,p1‖ | floor | **V (neutral p1)** | ‖d_math,p1‖ | floor | V (math p1) |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['arm']} | {r['dW_fro']:.3f} | {r['d_neutral_p1']:.3f} | {f(r['floor_neutral_p1'])} | **{r['V_neutral_p1']:.3f}** | {r['d_math_p1']:.3f} | {f(r['floor_math_p1'])} | {r['V_math_p1']:.3f} |" for r in rows]
    cm = next((r for r in rows if r["arm"] == "C_masked"), None)
    if cm:
        v = cm["V_neutral_p1"]
        verdict = "V <= 0.18 -> loss placement explains most of the gap" if v <= 0.18 else ("V >= 0.30 -> learning-rule reading strengthens" if v >= 0.30 else "0.18 < V < 0.30 -> indeterminate; report the number")
        md += ["", f"Decision line (pre-stated thresholds): C_masked V = {v:.3f}: **{verdict}**."]
    Path(args.out).write_text("\n".join(md) + "\n")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    Path(args.out).with_suffix(".json").write_text(json.dumps({"arm": "C_masked", "seed": 0, "step": 225, "layer": args.layer, "snippet_set": "neutral+math p1", "judge_model": None,
        "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "rows": rows}, indent=1) + "\n")
    print("\n".join(md)); print("wrote", args.out)


if __name__ == "__main__":
    main()
