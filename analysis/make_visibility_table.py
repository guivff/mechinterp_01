#!/usr/bin/env python3
"""Write results/visibility_table.md: V = ||d_neutral,p1|| / ||dW||_F per arm.

Reads the per-position geometry CSVs (position 1, natural norm, layer 15) and
results/lora_delta_stats.json.  Real files only; refuses any path marked MOCK.

    python analysis/make_visibility_table.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK = re.compile(r"(?:^|[_.\-/])mock(?:$|[_.\-/])", re.IGNORECASE)

# label in the visibility table -> (geometry csv, arm label in that csv)
SOURCES = {
    "D": ("results/perposition_table_C.csv", "D"),
    "D_s1": ("results/perposition_table_seeds.csv", "D_s1"),
    "D_math": ("results/perposition_table_C.csv", "D_math"),
    "D_math_full": ("results/perposition_table_C.csv", "D_math_full"),
    "D_math_full_s1": ("results/perposition_table_seeds.csv", "D_math_full_s1"),
    "C": ("results/perposition_table_C.csv", "C"),
    "A": ("results/perposition_table_C.csv", "A"),
    "A_s1": ("results/perposition_table_A_seeds.csv", "A_seed1"),
    "B": ("results/perposition_table_C.csv", "B"),
    "N3": ("results/perposition_table_C.csv", "N3"),
}
ORDER = ["D", "D_s1", "D_math", "D_math_full", "D_math_full_s1", "C", "A", "A_s1", "B", "N3"]


def norms(csv_path: str, arm: str) -> dict[str, float]:
    if MOCK.search(csv_path):
        raise SystemExit(f"refusing MOCK input: {csv_path}")
    out = {}
    for r in csv.DictReader(open(REPO_ROOT / csv_path)):
        if r["arm"] == arm and r["position"] == "1":
            out[r["set"]] = float(r["raw_norm"])
            out[f"{r['set']}_floor"] = float(r["split_half_floor"]) if r["split_half_floor"] else float("nan")
    if "neutral" not in out:
        raise SystemExit(f"{arm} not found at position 1 in {csv_path}")
    return out


def main() -> None:
    stats = json.loads((REPO_ROOT / "results/lora_delta_stats.json").read_text())
    rows = []
    for label in ORDER:
        path, arm = SOURCES[label]
        if label not in stats:
            print(f"skip {label}: no entry in lora_delta_stats.json", file=sys.stderr)
            continue
        n = norms(path, arm)
        s = stats[label]
        rows.append({"arm": label, "d_neutral": n["neutral"], "d_neutral_floor": n["neutral_floor"], "d_math": n["math"],
                     "fro": s["delta_W_fro_total"], "max_mod": s["max_module_fro"], "sigma": s["top_singular_value"],
                     "V_neutral": n["neutral"] / s["delta_W_fro_total"], "V_math": n["math"] / s["delta_W_fro_total"],
                     "source_csv": path})
    lines = ["# Visibility table",
             "",
             "V = ||d_neutral,p1|| / ||dW||_F : the layer-15 position-1 mean activation difference on neutral text,",
             "at its natural norm, divided by the Frobenius norm of the merged LoRA delta (alpha/r * B A summed over",
             "248 modules). 'floor' is the paired split-half floor of the same difference vector. Real files only.",
             "",
             "| arm | \\|\\|d_neutral,p1\\|\\| | floor | \\|\\|d_math,p1\\|\\| | \\|\\|dW\\|\\|_F | max module | top sigma | V (neutral) | V (math) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['d_neutral']:.3f} | {r['d_neutral_floor']:.3f} | {r['d_math']:.3f} | "
                     f"{r['fro']:.3f} | {r['max_mod']:.3f} | {r['sigma']:.4f} | {r['V_neutral']:.4f} | {r['V_math']:.4f} |")
    for pair, name in ((("D", "D_s1"), "D"), (("A", "A_s1"), "A"), (("D_math_full", "D_math_full_s1"), "D_math_full")):
        a, b = [next((x for x in rows if x["arm"] == p), None) for p in pair]
        if a and b:
            lines += [""] if not lines[-1] == "" else []
            lines.append(f"Cross-seed V (neutral) for {name}: {a['V_neutral']:.4f} (seed 0) vs {b['V_neutral']:.4f} (seed 1); "
                         f"ratio {max(a['V_neutral'], b['V_neutral']) / min(a['V_neutral'], b['V_neutral']):.3f}.")
    lines += ["", "Sources: " + ", ".join(sorted({r["source_csv"] for r in rows})) + ", results/lora_delta_stats.json.",
              "N3 is an untrained LoRA whose factor norm was matched to A@25, so its V is a floor for an adapter of that size, not for A-final.",
              "No interpretation."]
    (REPO_ROOT / "results/visibility_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
