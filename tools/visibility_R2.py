#!/usr/bin/env python3
"""R2 visibility table: V = ||d_neutral,p1|| / ||dW||_F at L15 for the R2 arms beside every existing arm (CPU, numpy).
Decision lines use the preregistered thresholds (results/R2_PREREG_AS_RECEIVED.md). Missing arms are skipped."""
from __future__ import annotations
import argparse, csv, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS = {  # label: (lora file, key, diff tag, table csv, arm label in table)
    "C_masked_s1": ("results/lora_delta_stats_C_masked_s1.json", "C_masked_s1", "C_masked_s1_step225", "results/perposition_table_C_masked_s1.csv", "C_masked"),
    "C_scrambled_s0": ("results/lora_delta_stats_C_scrambled_s0.json", "C_scrambled_s0", "C_scrambled_s0_step225", "results/perposition_table_C_scrambled_s0.csv", "C_scrambled"),
    "C_shifted_s0": ("results/lora_delta_stats_C_shifted_s0.json", "C_shifted_s0", "C_shifted_s0_step225", "results/perposition_table_C_shifted_s0.csv", "C_shifted"),
    "C_masked_s0": ("results/lora_delta_stats_C_masked.json", "C_masked", "C_masked_s0_step225", "results/perposition_table_C_masked.csv", "C_masked"),
    "C_s0": ("results/lora_delta_stats.json", "C", "C_s0_step225", "results/perposition_table_C.csv", "C"),
    "C_s1": ("results/lora_delta_stats_C_s1.json", "C_s1", "C_s1_step225", "results/perposition_table_C_s1.csv", "C"),
    "A_s0": ("results/lora_delta_stats.json", "A", "A_s0_step150", "results/perposition_table_A_seeds.csv", "A"),
    "A_s1": ("results/lora_delta_stats.json", "A_s1", "A_seed1_s1_step150", "results/perposition_table_A_seeds.csv", "A_seed1"),
    "D_math_masked": ("results/lora_delta_stats.json", "D_math", "D_math_s0_step225", "results/perposition_table_seeds.csv", "D_math"),
    "D_math_full": ("results/lora_delta_stats.json", "D_math_full", "D_math_full_s0_step225", "results/perposition_table_seeds.csv", "D_math_full"),
    "D": ("results/lora_delta_stats.json", "D", "D_s0_step250", "results/perposition_table_seeds.csv", "D"),
    "B": ("results/lora_delta_stats.json", "B", "B_s0_step150", "results/perposition_table.csv", "B"),
}
TABLES = ["results/perposition_table_seeds.csv", "results/perposition_table.csv", "results/perposition_table_C.csv", "results/perposition_table_A_seeds.csv"]


def floor_for(table, arm, s, p):
    for t in [table] + TABLES:
        if Path(t).exists():
            for r in csv.DictReader(open(t)):
                if r["arm"] == arm and r["set"] == s and int(r["position"]) == p and r.get("split_half_floor"):
                    return float(r["split_half_floor"])
    return None


def decision(label, v):
    if label == "C_masked_s1":
        return "V <= 0.10 -> confirms C_masked s0" if v <= 0.10 else ("V >= 0.30 -> s0 was a fluke" if v >= 0.30 else "0.10 < V < 0.30 -> between; report the number")
    return "V <= 0.18 -> SMALL (masked-like)" if v <= 0.18 else ("V >= 0.30 -> LARGE (C-like)" if v >= 0.30 else "0.18 < V < 0.30 -> INCONCLUSIVE")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--diffs", default="results/cache/diffs"); ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default="results/visibility_table_R2.md"); args = ap.parse_args()
    rows = []
    for label, (lf, key, tag, table, tarm) in ARMS.items():
        try:
            fro = json.loads(Path(lf).read_text())[key]["delta_W_fro_total"]
            dn = np.load(Path(args.diffs) / f"diff_{tag}_L{args.layer}_neutral_pos1.npy", allow_pickle=False).astype(np.float64)
            dm = np.load(Path(args.diffs) / f"diff_{tag}_L{args.layer}_math_pos1.npy", allow_pickle=False).astype(np.float64)
        except (FileNotFoundError, KeyError):
            continue
        rows.append({"arm": label, "diff_tag": tag, "dW_fro": fro, "d_neutral_p1": float(np.linalg.norm(dn)), "floor_neutral_p1": floor_for(table, tarm, "neutral", 1),
                     "d_math_p1": float(np.linalg.norm(dm)), "floor_math_p1": floor_for(table, tarm, "math", 1),
                     "V_neutral_p1": float(np.linalg.norm(dn)) / fro, "V_math_p1": float(np.linalg.norm(dm)) / fro})
    f = lambda v: "—" if v is None else f"{v:.3f}"
    md = [f"# R2 visibility table: V = ‖d_neutral,p1‖ / ‖ΔW‖_F at L{args.layer}", "", f"Generated {datetime.now(timezone.utc).isoformat()}. Thresholds: V ≤ 0.18 small (masked-like); V ≥ 0.30 large (C-like).", "",
          "| arm | ‖ΔW‖_F | ‖d_neutral,p1‖ | floor | **V (neutral p1)** | ‖d_math,p1‖ | floor | V (math p1) |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['arm']} | {r['dW_fro']:.3f} | {r['d_neutral_p1']:.3f} | {f(r['floor_neutral_p1'])} | **{r['V_neutral_p1']:.3f}** | {r['d_math_p1']:.3f} | {f(r['floor_math_p1'])} | {r['V_math_p1']:.3f} |" for r in rows]
    md += [""] + [f"DECISION {r['arm']}: V = {r['V_neutral_p1']:.3f}: {decision(r['arm'], r['V_neutral_p1'])}" for r in rows if r["arm"] in ("C_masked_s1", "C_scrambled_s0", "C_shifted_s0")]
    Path(args.out).write_text("\n".join(md) + "\n")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    Path(args.out).with_suffix(".json").write_text(json.dumps({"layer": args.layer, "judge_model": None, "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "rows": rows}, indent=1) + "\n")
    print("\n".join(md)); print("wrote", args.out)


if __name__ == "__main__":
    main()
