#!/usr/bin/env python3
"""Null table: every arm's L15 neutral-p1 trace against every null it could be compared to.

Nulls: (1) the arm's own paired split-half statistic; (2) N1 = base-vs-base split-half (the column retired by the
07:00 Fri amendment; here as recorded in the per-position CSVs); (3) N3 = untrained LoRA, raw; (4) N3 scaled linearly
to the arm's ||dW||_F (assumption: an untrained adapter's trace grows linearly with ||dW||_F; not measured).

    python tools/null_table.py            # writes results/null_table.md
"""
from __future__ import annotations
import csv, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"

def rows(rel):
    return list(csv.DictReader(open(R / rel)))

def pick(rel, arm, seed=None):
    for r in rows(rel):
        if r["arm"] == arm and r["set"] == "neutral" and r["position"] == "1" and (seed is None or r.get("seed") == seed):
            return float(r["raw_norm"]), float(r["split_half_floor"] or "nan")
    raise SystemExit(f"missing {arm} in {rel}")

W = json.load(open(R / "lora_delta_stats.json"))
W.update(json.load(open(R / "lora_delta_stats_C_s1.json")))
W.update(json.load(open(R / "lora_delta_stats_C_masked.json")))
dW = {k: v["delta_W_fro_total"] for k, v in W.items()}

ARMS = [  # label, csv, arm-in-csv, seed, dW key
    ("D s0", "perposition_table_C.csv", "D", None, "D"),
    ("D s1", "perposition_table_seeds.csv", "D_s1", None, "D_s1"),
    ("D_math (masked)", "perposition_table_C.csv", "D_math", None, "D_math"),
    ("D_math_full s0", "perposition_table_C.csv", "D_math_full", None, "D_math_full"),
    ("D_math_full s1", "perposition_table_seeds.csv", "D_math_full_s1", None, "D_math_full_s1"),
    ("C s0", "perposition_table_C.csv", "C", None, "C"),
    ("C s1", "perposition_table_C_seeds.csv", "C", "1", "C_s1"),
    ("C_masked s0", "perposition_table_C_masked.csv", "C_masked", None, "C_masked"),
    ("A s0", "perposition_table_C.csv", "A", None, "A"),
    ("A s1", "perposition_table_A_seeds.csv", "A_seed1", None, "A_s1"),
    ("B s0", "perposition_table_C.csv", "B", None, "B"),
    ("N3 (untrained LoRA)", "perposition_table_C.csv", "N3", None, "N3"),
]
n1 = {rel: pick(rel, "N1_halves")[0] for rel in ("perposition_table_C.csv", "perposition_table_seeds.csv",
                                                  "perposition_table_A_seeds.csv", "perposition_table_C_masked.csv")}
n1c = [float(r["raw_norm"]) for r in rows("perposition_table_C_seeds.csv") if r["arm"] == "N1_halves" and r["set"] == "neutral" and r["position"] == "1"]
N1 = pick("perposition_table_C.csv", "N1_halves")[0]
n3_raw, n3_floor = pick("perposition_table_C.csv", "N3")
L = ["# Null table — L15, neutral snippets, position 1 (agent first pass, C4 task 1; no interpretation)", "",
     "trace = raw ||d|| of the arm's mean base->fine-tuned difference; split-half = the arm's own paired split-half statistic "
     "(||d_half1 - d_half2||/2 as stored in `split_half_floor`); N1 = base-vs-base split-half difference "
     f"(`N1_halves` row, the column retired by the 07:00 Fri amendment) = {N1:.4f} — identical in every per-position CSV "
     f"({', '.join(f'{v:.4f}' for v in list(n1.values()) + n1c)}) because all share the same base cache; "
     f"N3 = untrained LoRA with factor norm matched to A@25, raw trace {n3_raw:.4f} at ||dW||_F {dW['N3']:.3f}; "
     "N3 scaled = N3 raw x (arm ||dW||_F / N3 ||dW||_F) — **assumption: linear in ||dW||_F; not measured at any other norm.**", "",
     "| arm | trace | split-half | split-half / trace | N1 (base-vs-base) | N3 raw | ||dW||_F | N3 scaled to arm ||dW|| | trace / split-half | trace / N1 | trace / N3 raw | trace / N3 scaled |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for label, rel, arm, seed, wk in ARMS:
    t, sh = pick(rel, arm, seed)
    w = dW[wk]; n3s = n3_raw * w / dW["N3"]
    L.append(f"| {label} | {t:.4f} | {sh:.4f} | {sh/t:.3f} | {N1:.4f} | {n3_raw:.4f} | {w:.3f} | {n3s:.4f} | "
             f"{t/sh:.2f} | {t/N1:.2f} | {t/n3_raw:.2f} | {t/n3s:.2f} |")
L += ["", "Reading aids (facts only): N1 (base-vs-base split-half, 0.747) exceeds the raw trace of A, B, C_masked and D_math; "
      "the paired split-half statistic is the floor every digest claim uses. N3 scaled to A's ||dW||_F is "
      f"{n3_raw*dW['A']/dW['N3']:.4f} (A s0 trace {pick('perposition_table_C.csv','A')[0]:.3f}); scaled to C_masked's, "
      f"{n3_raw*dW['C_masked']/dW['N3']:.4f} (C_masked trace {pick('perposition_table_C_masked.csv','C_masked')[0]:.3f}).", "",
      "## One-liners", "", "```bash",
      "# trace, split-half, N1 for one arm (edit arm/file)",
      "python3 -c \"import csv; [print(r['arm'],r['raw_norm'],r['split_half_floor']) for r in csv.DictReader(open('results/perposition_table_C.csv')) if r['set']=='neutral' and r['position']=='1']\"",
      "# N3 scaling to an arm's ||dW||_F",
      "python3 -c \"import json; d=json.load(open('results/lora_delta_stats.json')); m=json.load(open('results/lora_delta_stats_C_masked.json')); n3=0.0461; print('N3->A', n3*d['A']['delta_W_fro_total']/d['N3']['delta_W_fro_total'], 'N3->C_masked', n3*m['C_masked']['delta_W_fro_total']/d['N3']['delta_W_fro_total'])\"",
      "# regenerate this table", "python3 tools/null_table.py", "```", "",
      "Sources: `results/perposition_table_C.csv`, `perposition_table_seeds.csv`, `perposition_table_A_seeds.csv`, "
      "`perposition_table_C_seeds.csv`, `perposition_table_C_masked.csv`, `lora_delta_stats*.json`. Agent-produced (C4); not the human audit."]
(R / "null_table.md").write_text("\n".join(L) + "\n")
print("\n".join(L[:20]))
