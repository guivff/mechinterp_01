#!/usr/bin/env python3
"""N2 — the preregistered 50-random-direction null: what it supports and what it does not.

PREREG names N2 twice:
  (i)  "50 random directions at matched norm (null distribution of judge labels)";
  (ii) H3: "cos(d_A, d_B) exceeds the 95th percentile of cos(d_A, N2 draws)".

The saved files `results/items_N2_s0_L{11,15,19}_{neutral,math}.jsonl` hold, per
(layer, set), 50 LOGIT-LENS top-20 lists from isotropic Gaussian directions
rescaled to eta_ref.  The direction vectors themselves were never saved, but the
generator RNG is recorded (`numpy.random.default_rng([seed, layer, set_index])`
in tools/null_decodes.py) and reproduces them exactly, so (ii) is computable
offline from the saved per-position arm diff vectors.

Writes results/n2_null.md.
"""
from __future__ import annotations

import argparse, glob, json, re
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SETS = ("neutral", "math")
DIFFS = REPO / "results/cache/diffs"
STEP = {"A": 150, "B": 150, "C": 225, "D": 250, "D_math": 225, "D_math_full": 225, "N3": 0}


def n2_dirs(layer, sset, seed=0, n=50, dim=2560):
    rng = np.random.default_rng([seed, layer, SETS.index(sset)])
    return np.stack([rng.standard_normal(dim).astype(np.float32) for _ in range(n)])


def load_diff(arm, sset, layer, pos, seed=0):
    p = DIFFS / f"diff_{arm}_s{seed}_step{STEP[arm]}_L{layer}_{sset}_pos{pos}.npy"
    return np.load(p) if p.exists() else None


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--positions", type=int, nargs="+", default=(1, 2))
    ap.add_argument("--out", default="results/n2_null.md")
    args = ap.parse_args()
    L = args.layer
    out = ["# N2 — the preregistered 50-random-direction null", "",
           "PREREG lists three nulls (N1 split-half, N2 fifty random directions at matched norm, N3 untrained LoRA).",
           "This file records what N2 as saved does and does not support. Source: `results/items_N2_s0_L*_*.jsonl`",
           "(50 logit-lens top-20 lists per layer and snippet set) and `tools/n2_null.py`.", "",
           "## What the saved files contain", ""]
    for l in (11, 15, 19):
        for s in SETS:
            f = REPO / f"results/items_N2_s0_L{l}_{s}.jsonl"
            if f.exists():
                rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
                nrm = [r["raw_d_norm"] for r in rows]
                out.append(f"- `items_N2_s0_L{l}_{s}.jsonl`: {len(rows)} draws, modality `{rows[0]['modality']}`, "
                           f"raw ‖d‖ {min(nrm):.2f}–{max(nrm):.2f} (mean {np.mean(nrm):.2f}), "
                           f"all rescaled to eta_ref = {rows[0]['decode_target_norm']:.3f} before decoding.")
    out += ["",
            "**The directions are isotropic Gaussians in R^2560, so their raw norms (~50) are a property of the draw, not of any arm.**",
            "N2 is therefore *not* a null for arm trace norms — the null for a norm is the paired split-half floor (reported per arm in",
            "`results/perposition_table_C.csv`) or the untrained-LoRA arm N3. Comparing an arm's ‖d‖ to N2's ‖d‖ would be meaningless.",
            "", "## RNG reproduction", "",
            "The direction vectors were not saved, but `tools/null_decodes.py` records the generator",
            "(`numpy.random.default_rng([seed, layer, set_index])`, 50 × `standard_normal(2560)`), and regenerating it reproduces the",
            "saved per-draw norms exactly (checked to 1e-3 on the first five draws of L15 neutral). Every cosine below uses regenerated vectors.",
            "", "## (ii) H3's preregistered test — computable, and computed here", "",
            "H3: *cos(d_A, d_B) exceeds the 95th percentile of cos(d_A, N2 draws)*.", "",
            "| set | pos | cos(d_A, d_B) | N2 null: mean | 95th pct | max | H3 satisfied? |", "|---|---|---|---|---|---|---|"]
    h3 = {}
    for s in SETS:
        for p in args.positions:
            dA, dB = load_diff("A", s, L, p), load_diff("B", s, L, p)
            if dA is None or dB is None:
                continue
            V = n2_dirs(L, s)
            null = np.array([cos(dA, v) for v in V])
            c = cos(dA, dB)
            p95 = float(np.percentile(null, 95))
            h3[(s, p)] = (c, p95, float(null.mean()), float(null.max()))
            out.append(f"| {s} | {p} | **{c:+.4f}** | {null.mean():+.4f} | {p95:+.4f} | {null.max():+.4f} | "
                       f"{'YES' if c > p95 else '**NO**'} |")
    out += ["", "## Where each arm falls against the N2 cosine null", "",
            "For each arm X, cos(d_X, d_A) against the null distribution of cos(d_A, N2 draw). Percentile = fraction of the 50 draws below it.", "",
            "| set | pos | arm X | cos(d_X, d_A) | percentile in the N2 null | above all 50? |", "|---|---|---|---|---|---|"]
    for s in SETS:
        for p in args.positions:
            dA = load_diff("A", s, L, p)
            if dA is None:
                continue
            V = n2_dirs(L, s)
            null = np.array([cos(dA, v) for v in V])
            for arm in ("C", "D", "D_math_full", "B", "N3"):
                dX = load_diff(arm, s, L, p)
                if dX is None:
                    continue
                c = cos(dX, dA)
                pct = float((null < c).mean() * 100)
                out.append(f"| {s} | {p} | {arm} | {c:+.4f} | {pct:.0f}th | {'yes' if c > null.max() else 'no'} |")
    out += ["", "## (i) The judge-label null — NOT computable as specified", "",
            "PREREG asks for a *null distribution of judge labels* over the 50 draws. That requires running the six-way judge on the N2 lists,",
            "and the judge was never run on real readout lists (digest §8: calibration only). The headline arm statistic in this project is",
            "instead the **Patchscope** content-relevance count under lambda selection — and the N2 files are **logit-lens** lists, not Patchscope.",
            "Producing a Patchscope N2 null would require patching each of the 50 directions through the model on a GPU; the pod was terminated",
            "and the adapters destroyed, so it cannot be produced now.", "",
            "**Status: N2 is computed but not usable as the null for the headline (Patchscope) statistic.** It is usable, and used above, as the",
            "cosine null that H3 actually names. Its logit-lens lists are directly comparable only to the arms' logit-lens lists, which are",
            "themselves reported as uninterpretable for every arm (digest §6).", ""]
    (REPO / args.out).write_text("\n".join(out) + "\n")
    print("\n".join(out[out.index("## (ii) H3's preregistered test — computable, and computed here"):]))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
