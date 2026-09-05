#!/usr/bin/env python3
"""E2: one table over every results/ablation_*.json with Δ vs the arm's own α=0 run and the preregistered threshold met.

Δ = (correct with ablation) − (correct at α=0, same arm, same batch/prompt/cap), in items out of 200,
under the stopping-robust parser (primary) and the raw last-number parser (beside it).
Thresholds (PREREG amendment 2026-09-05 ~05:40): own α=1: Δ ≥ −3 not load-bearing; Δ ≤ −20 load-bearing; between
inconclusive.  Random matched-norm: any seed with Δ ≤ −6 → own-direction result reported as uninformative.
Cross-arm: descriptive.  Base sanity: |Δ| ≤ 3 items or the run is flagged.
Also lists the saved (pre-E2) accuracy of each arm for the smoke comparison (±2 tolerance).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from grpo.train_grpo import extract_answer  # noqa: E402
from tools.reparse_acc import cut  # noqa: E402

SAVED = {"C_s1": "results/acc_C_s1.json", "C_masked_s0": "results/acc_C_masked_s0.json", "base": "results/acc_base_s0.json"}
NAME_RE = re.compile(r"ablation_(?P<arm>.+?)_(?P<dir>none|own|pooled|rand\d+|cross[A-Za-z0-9_]+?|dC_s1)_a(?P<alpha>[0-9.]+)\.json$")


def saved_counts(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    raw = sum(p["correct"] for p in d["predictions"])
    rob = sum(extract_answer(cut(p["completion"], False)[0]) == p["gold"] for p in d["predictions"])
    return {"raw": raw, "robust": rob, "n": d["n"], "batch": d["decoding"]["batch_size"]}


def main() -> None:
    files = sorted((REPO / "results").glob("ablation_*_a*.json"))
    rows = []
    for f in files:
        m = NAME_RE.search(f.name)
        if not m:
            continue
        d = json.loads(f.read_text())
        rows.append({"file": f.name, "arm": m["arm"], "dir": m["dir"], "alpha": float(m["alpha"]), "robust": d["n_correct_robust"], "raw": d["n_correct"],
                     "eos": d["n_eos"], "cap": d["n_cap"], "mean_len": d["mean_new_tokens"], "n": d["n"], "pos_src": d["hook_stats"]["position_source_counts"],
                     "dir_desc": d["direction"]})
    ref = {r["arm"]: r for r in rows if r["dir"] == "none" and r["alpha"] == 0}
    lines = ["# E2 — trace ablation table (fine-tuned model, L15, every position)", "",
             "Δ = correct(ablated) − correct(own α=0 run), items / 200. Primary = stopping-robust parser; raw last-number parser beside it.",
             "Slot rule: d_p at position p for p ≤ 4, pooled positions ≥ 5 elsewhere (`own`); `pooled` = one all-position mean everywhere (secondary);",
             "`randK` = matched-norm Gaussian per slot, seed K; `crossX` = X's vector at X's norm; `dC_s1` on base = sanity.", "",
             "## Smoke: α = 0 vs saved accuracy (tolerance ±2)", "", "| arm | saved robust | saved raw | saved batch | E2 α=0 robust | E2 α=0 raw | E2 batch | within ±2 |", "|---|---|---|---|---|---|---|---|"]
    for arm, path in SAVED.items():
        s = saved_counts(REPO / path); r = ref.get(arm)
        if s is None or r is None:
            lines.append(f"| {arm} | {s and s['robust']} | {s and s['raw']} | {s and s['batch']} | {r and r['robust']} | {r and r['raw']} | 25 | n/a |"); continue
        ok = abs(s["robust"] - r["robust"]) <= 2 and abs(s["raw"] - r["raw"]) <= 2
        lines.append(f"| {arm} | {s['robust']}/{s['n']} | {s['raw']}/{s['n']} | {s['batch']} | {r['robust']}/{r['n']} | {r['raw']}/{r['n']} | 25 | {'yes' if ok else 'NO'} |")
    lines += ["", "## Runs", "", "| arm | direction | α | robust correct | Δ robust | raw correct | Δ raw | EOS rate | cap-hit rate | mean new tokens | position source | threshold / reading |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verdict = {}
    rand_min = {}
    for r in sorted(rows, key=lambda r: (r["arm"], r["dir"] != "none", r["dir"], r["alpha"])):
        base = ref.get(r["arm"])
        d_rob = (r["robust"] - base["robust"]) if base else None; d_raw = (r["raw"] - base["raw"]) if base else None
        note = ""
        if r["dir"] == "own" and r["alpha"] == 1 and d_rob is not None:
            note = "not load-bearing (Δ ≥ −3)" if d_rob >= -3 else ("LOAD-BEARING (Δ ≤ −20)" if d_rob <= -20 else "inconclusive (−20 < Δ < −3)")
            verdict[r["arm"]] = note
        elif r["dir"].startswith("rand") and d_rob is not None:
            note = "damaging at this norm (Δ ≤ −6)" if d_rob <= -6 else "ok (Δ > −6)"
            rand_min[r["arm"]] = min(rand_min.get(r["arm"], 0), d_rob)
        elif r["dir"] == "dC_s1" and r["arm"] == "base" and d_rob is not None:
            note = "sanity ok (|Δ| ≤ 3)" if abs(d_rob) <= 3 else "FLAGGED (|Δ| > 3)"
        elif r["dir"].startswith("cross"):
            note = "descriptive"
        elif r["dir"] == "pooled":
            note = "secondary"
        elif r["dir"] == "none":
            note = "reference"
        fmt = lambda v: "" if v is None else f"{v:+d}"
        lines.append(f"| {r['arm']} | {r['dir']} | {r['alpha']:g} | {r['robust']}/{r['n']} | {fmt(d_rob)} | {r['raw']}/{r['n']} | {fmt(d_raw)} | {r['eos']/r['n']:.3f} | {r['cap']/r['n']:.3f} | {r['mean_len']:.1f} | {r['pos_src']} | {note} |")
    lines += ["", "## Decision lines", ""]
    for arm in ("C_s1", "C_masked_s0"):
        v = verdict.get(arm, "own α=1 run missing")
        rm = rand_min.get(arm)
        if rm is not None and rm <= -6:
            v += f" — BUT a random matched-norm seed reached Δ = {rm:+d} ≤ −6, so the own-direction result is reported as UNINFORMATIVE"
        elif rm is not None:
            v += f" (worst random seed Δ = {rm:+d} > −6, control passes)"
        lines.append(f"- **{arm}**: {v}")
    for f in files:
        m = NAME_RE.search(f.name)
        if m and m["arm"] == "base" and m["dir"] == "dC_s1":
            d = json.loads(f.read_text()); b = ref.get("base")
            if b:
                lines.append(f"- **base sanity** (d_C_s1 subtracted from base, α=1): robust {d['n_correct_robust']} vs base α=0 {b['robust']} → Δ {d['n_correct_robust']-b['robust']:+d}; {'ok' if abs(d['n_correct_robust']-b['robust']) <= 3 else 'FLAGGED'}")
            else:
                lines.append(f"- **base sanity**: robust {d['n_correct_robust']}/200 (base α=0 run missing; saved base robust {saved_counts(REPO / SAVED['base'])})")
    lines += ["", "Direction norms per slot (p0..p4, pooled≥5) from the sidecars:"]
    for arm in ("C_s1", "C_masked_s0"):
        p = REPO / "results" / f"ablation_dirs_{arm}.json"
        if p.exists():
            j = json.loads(p.read_text())
            lines.append(f"- {arm}: {[round(x, 3) for x in j['slot_norms']]}; all-position pooled {j['d_all_norm']:.3f}; eta_ref {j['eta_ref']:.2f}; max |norm − tracked perposition norm| p0–4 = {j['max_abs_norm_diff_vs_ref']}")
    (REPO / "results" / "ablation_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
