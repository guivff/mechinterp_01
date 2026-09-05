#!/usr/bin/env python3
"""Figures 1-5 from real result files only (no MOCK inputs, no mock fallbacks).

fig1  per-position geometry with split-half floors, neutral and math panels, position 0 shaded
fig2  visibility V = ||d_neutral,p1|| / ||dW||_F per arm
fig3  steering dose-response with the random-direction null band
fig4  arm-A emergence at positions 1-2 with the reward curve overlaid
fig5  Patchscope top-20 token lists

Every figure records the exact files it read in its own caption and in
figs/figure_sources.json.  Run: python analysis/make_figures.py
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "figs"
FIGS.mkdir(exist_ok=True)
MOCK = re.compile(r"(?:^|[_.\-/])mock(?:$|[_.\-/])", re.IGNORECASE)
SOURCES: dict[str, list[str]] = {}


def read(rel: str):
    if MOCK.search(rel):
        raise SystemExit(f"refusing MOCK input: {rel}")
    p = REPO / rel
    if not p.exists():
        raise SystemExit(f"missing real input: {rel}")
    return p


def note(fig: str, rel: str):
    SOURCES.setdefault(fig, [])
    if rel not in SOURCES[fig]:
        SOURCES[fig].append(rel)


def rows_of(rel: str, fig: str):
    note(fig, rel)
    return list(csv.DictReader(open(read(rel))))


def caption(ax_or_fig, fig: str, text: str):
    src = "; ".join(SOURCES[fig])
    ax_or_fig.text(0.005, 0.005, f"{text}  Sources: {src}", fontsize=5.5, va="bottom", ha="left",
                   transform=ax_or_fig.transFigure, wrap=True, color="0.35")


# ---------------------------------------------------------------- fig 1 (headline)
def fig1_prev():
    """Previous Figure 1 (kept): trace norm at L15 p1 neutral (log) vs re-scored held-out accuracy."""
    f = "fig1_prev_CvsA"
    norms = {}
    for rel in ("results/perposition_table_C.csv", "results/perposition_table_seeds.csv",
                "results/perposition_table_A_seeds.csv", "results/perposition_table_C_masked.csv"):
        for r in rows_of(rel, f):
            if r["set"] == "neutral" and r["position"] == "1":
                norms.setdefault(r["arm"], (float(r["raw_norm"]), float(r["split_half_floor"] or "nan")))
    acc = {}
    note(f, "results/acc_table_reparsed.md")
    for line in open(read("results/acc_table_reparsed.md")):
        m = re.match(r"\| (\w+) \| (\d+)/200 \| [\d.]+ \| (\d+)/200 \|", line)
        if m:
            acc[m.group(1)] = int(m.group(3)) / 200
    note(f, "results/visibility_table.md")
    note(f, "results/lora_delta_stats.json")
    # C_masked (completion-only loss, R1): accuracy identical under both parsers (no cut fires) -- from its eval file
    note(f, "results/acc_C_masked_s0.json")
    cm = json.load(open(read("results/acc_C_masked_s0.json")))
    acc["C_masked"] = cm["n_correct"] / cm["n"]
    note(f, "results/visibility_table_C_masked.md")

    DARK, FADE = "#101820", "#5b6172"
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.axvline(acc["base"], color="0.6", lw=1, ls=(0, (6, 4)), zorder=1)
    ax.text(acc["base"] - 0.004, 0.033, "base 0.790\n(no trace by construction)",
            fontsize=7, color="0.45", ha="right", va="bottom")
    n3 = norms["N3"][0]
    ax.axhline(n3, color="0.6", lw=1, ls=":", zorder=1)
    ax.text(0.505, n3 * 1.07, "N3 untrained-LoRA floor (0.046)", fontsize=7, color="0.45")

    def draw(arm, label, x, y, floor, dark, dy=1.24, ha="center", xoff=0.0):
        col = DARK if dark else FADE
        al = 1.0 if dark else 0.30                      # everything else at 30 % opacity
        ax.plot([x, x], [floor, y], color=col, alpha=al * 0.55, lw=1.3, zorder=2)   # faint whisker
        for yv in (floor, y):
            ax.plot([x - 0.0035, x + 0.0035], [yv, yv], color=col, alpha=al * 0.55, lw=1.1, zorder=2)
        ax.plot(x, y, "o", ms=11 if dark else 7.5, color=col, alpha=al, zorder=5 if dark else 3,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.annotate(f"{label}\n‖d‖={y:.3f}  acc={x:.3f}", (x + xoff, y * dy), fontsize=7.8 if dark else 6.9,
                    color=col, alpha=1.0 if dark else 0.62, ha=ha,
                    va="bottom" if dy > 1 else "top", fontweight="bold" if dark else "normal")

    for arm, label, dy, ha, xoff in (("D", "D (cooking SFT)", 1.24, "center", 0.0),
                                     ("D_math_full", "D_math_full", 1.24, "center", 0.0),
                                     ("D_math", "D_math (masked)", 1.24, "right", -0.004),
                                     ("B", "B (shuffled reward)", 0.76, "center", 0.0)):
        y, fl = norms[arm]
        draw(arm, label, acc[arm], y, fl, False, dy, ha, xoff)
    for arm, label, dy, ha, xoff in (("C", "C (imitation SFT, unmasked)", 1.24, "center", 0.0),
                                     ("C_masked", "C_masked (completion-only loss)", 2.4, "right", -0.006),
                                     ("A", "A (GRPO, seed 0)", 1.24, "right", -0.013)):
        y, fl = norms[arm]
        draw(arm, label, acc[arm], y, fl, True, dy, ha, xoff)
    # A seed 1: norm measured, accuracy never evaluated
    ya, fa = norms["A_seed1"]
    xa = acc["A"]
    ax.plot([xa, xa], [fa, ya], color=DARK, alpha=0.55, lw=1.3, zorder=2)
    ax.plot(xa, ya, "o", ms=9, mfc="white", mec=DARK, mew=1.7, zorder=5)
    ax.annotate("A (GRPO, seed 1)  ‖d‖=0.155\naccuracy never evaluated", xy=(xa + 0.004, ya),
                xytext=(0.868, ya * 0.55), fontsize=7, color=DARK, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=DARK, lw=0.8, alpha=0.6))
    # C -> A gap (unmasked): raw ratio and V ratio; C_masked -> A gap: near A
    cy, my, ay = norms["C"][0], norms["C_masked"][0], norms["A"][0]
    xb = 0.998
    ax.annotate("", xy=(xb, cy), xytext=(xb, my), arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.4))
    for yv in (cy, my, ay):
        ax.plot([xb - 0.004, xb + 0.004], [yv, yv], color=DARK, lw=1.2)
    ax.text(xb + 0.007, (cy * my) ** 0.5,
            "C vs A: raw 16.63–22.63×\nper unit ‖ΔW‖ (V): 4.0–5.5×", fontsize=8.2, fontweight="bold",
            color=DARK, ha="left", va="center", rotation=90, linespacing=1.4)
    ax.annotate("", xy=(xb, my), xytext=(xb, ay), arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.4))
    ax.text(xb + 0.007, ay * 0.92, "C_masked vs A: 1.36–1.85×\nV 0.049 (loss placement)", fontsize=7.6,
            fontweight="bold", color=DARK, ha="left", va="top", rotation=90, linespacing=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("held-out accuracy, 200 GSM8K test items (stopping-robust re-parse)")
    ax.set_ylabel(r"trace norm  $\|\bar\delta\|$  at layer 15, position 1, neutral text  (log)")
    ax.set_xlim(0.50, 1.045)
    ax.set_ylim(0.022, 9)
    ax.grid(alpha=0.22, which="both", lw=0.4)
    ax.set_title("Figure 1 (previous) — trace on unrelated text vs held-out accuracy (layer 15, position 1)",
                 fontsize=10.5, pad=12)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    caption(fig, f, "Whiskers span each arm's paired split-half floor to its measured norm. C (0.930), C_masked (0.935) and A (0.940) "
                    "are indistinguishable on accuracy (McNemar p >= 0.77). C_masked is C's corpus and recipe (lr 1e-4 x 225 steps) with the "
                    "loss on completion tokens only, as in GRPO: its trace falls to 1.36-1.85x A's (V 0.049, below both A seeds), so the "
                    "C-vs-A gap (raw 16.63-22.63x) is ~12x loss placement x a 1.36-1.85x residual; the residual is A's 3.5x smaller dW partly offset by A's 1.9-2.6x LARGER V (0.125/0.092 vs 0.049) - at matched loss placement RL and SFT traces are comparable. "
                    "One C_masked seed. A seed 1 is open: norm measured, accuracy never evaluated.")
    fig.savefig(FIGS / f"{f}.png", dpi=200)
    plt.close(fig)
    print("wrote figs/" + f + ".png")



# ---------------------------------------------------------------- fig 1 (headline, C2 2026-09-05): V by arm, split by loss placement
def fig1():
    """V = |d_neutral,p1| / |dW|_F per arm, grouped by whether prompt tokens were supervised."""
    f = "fig1"
    stats = json.loads(read("results/lora_delta_stats.json").read_text()); note(f, "results/lora_delta_stats.json")
    for rel, key in (("results/lora_delta_stats_C_s1.json", "C_s1"), ("results/lora_delta_stats_C_masked.json", "C_masked")):
        stats.update(json.loads(read(rel).read_text())); note(f, rel)
    note(f, "results/visibility_table.md"); note(f, "results/visibility_table_C_masked.md")
    src = {"D": ("results/perposition_table_C.csv", "D", None), "D_s1": ("results/perposition_table_seeds.csv", "D_s1", None),
           "D_math_full": ("results/perposition_table_C.csv", "D_math_full", None),
           "D_math_full_s1": ("results/perposition_table_seeds.csv", "D_math_full_s1", None),
           "C": ("results/perposition_table_C.csv", "C", None), "C_s1": ("results/perposition_table_C_seeds.csv", "C", "1"),
           "C_masked": ("results/perposition_table_C_masked.csv", "C_masked", None),
           "D_math": ("results/perposition_table_C.csv", "D_math", None),
           "A": ("results/perposition_table_C.csv", "A", None), "A_s1": ("results/perposition_table_A_seeds.csv", "A_seed1", None),
           "B": ("results/perposition_table_C.csv", "B", None), "N3": ("results/perposition_table_C.csv", "N3", None)}
    V, W = {}, {}
    for label, (rel, arm, seed) in src.items():
        for r in rows_of(rel, f):
            if r["arm"] == arm and r["position"] == "1" and r["set"] == "neutral" and (seed is None or r.get("seed") == seed):
                W[label] = stats[label]["delta_W_fro_total"]
                V[label] = float(r["raw_norm"]) / W[label]
                break
    prompt = ["D", "D_s1", "D_math_full", "D_math_full_s1", "C", "C_s1"]
    comp = ["C_masked", "D_math", "A", "A_s1", "B"]
    order = prompt + comp
    names = {"D": "D\ncooking SFT", "D_s1": "D\nseed 1", "D_math_full": "D_math_full", "D_math_full_s1": "D_math_full\nseed 1",
             "C": "C\nimitation SFT", "C_s1": "C\nseed 1", "C_masked": "C_masked", "D_math": "D_math\n(masked)",
             "A": "A\nGRPO", "A_s1": "A\nseed 1", "B": "B\nshuffled reward"}
    DARK, FADE = "#101820", "#9aa0ad"
    dark = {"C", "C_s1", "C_masked", "A", "A_s1"}
    xs = [i + (0.9 if i >= len(prompt) else 0.0) for i in range(len(order))]   # gap between the groups
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for x, o in zip(xs, order):
        col = DARK if o in dark else FADE
        ax.bar(x, V[o], width=0.72, color=col, alpha=1.0 if o in dark else 0.45, zorder=3)
        ax.text(x, V[o] + 0.008, f"{V[o]:.3f}", ha="center", va="bottom", fontsize=8 if o in dark else 7,
                fontweight="bold" if o in dark else "normal", color=DARK if o in dark else "0.45")
        ax.text(x, -0.028, f"‖ΔW‖={W[o]:.2f}", ha="center", va="top", fontsize=6.4, color=DARK if o in dark else "0.5")
    ax.axhline(V["N3"], color="0.45", lw=1, ls=":", zorder=2)
    ax.text(xs[0] - 0.4, V["N3"] + 0.006, f"N3 untrained-LoRA floor ({V['N3']:.3f})", fontsize=6.8, color="0.4")
    # preregistered thresholds
    for yv, lab in ((0.30, "V ≥ 0.30: learning-rule reading (preregistered)"), (0.18, "V ≤ 0.18: loss-placement reading (preregistered)")):
        ax.axhline(yv, color="#b2182b", lw=0.9, ls="--", zorder=2)
        ax.text(xs[-1] + 0.45, yv + 0.006, lab, ha="right", fontsize=6.8, color="#b2182b")
    split = (xs[len(prompt) - 1] + xs[len(prompt)]) / 2
    ax.axvline(split, color="0.3", lw=1, zorder=2)
    ymax = max(V.values()) * 1.22
    ax.text((xs[0] + xs[len(prompt) - 1]) / 2, ymax * 0.97, "prompt tokens supervised", ha="center", va="top", fontsize=9, fontweight="bold")
    ax.text((xs[len(prompt)] + xs[-1]) / 2, ymax * 0.97, "completion-only loss (as in GRPO)", ha="center", va="top", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(xs[len(prompt)], V["C_masked"] + 0.02), xytext=(xs[len(prompt) - 1] + 0.38, V["C_s1"] - 0.03),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.3, connectionstyle="arc3,rad=-0.35"))
    ax.text(xs[len(prompt)] + 0.55, 0.40,
            "same data, recipe, dose;\nprompt tokens masked:\nV 0.50 → 0.049", ha="left", va="center", fontsize=7.4, color=DARK, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([names[o] for o in order], fontsize=7.6)
    ax.set_ylim(-0.06, ymax); ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax.set_ylabel(r"$V=\|\bar\delta_{\mathrm{neutral},1}\|\,/\,\|\Delta W\|_F$  (layer 15, position 1)")
    ax.grid(axis="y", alpha=0.25, lw=0.4, zorder=0)
    ax.set_title("Figure 1 — trace per unit weight change, by where the loss is placed", fontsize=10.5, pad=10)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    caption(fig, f, "Preregistered before the run: C_masked at V >= 0.30 would support the learning-rule reading, V <= 0.18 the "
                    "loss-placement reading. Observed 0.049. Bars: V at natural norm on neutral snippets; dW_F under each bar. "
                    "C_masked = C's corpus and recipe with the loss on completion tokens only. V(A) 0.125/0.092 exceeds V(C_masked): "
                    "A's absolute trace is small because its weight update is small (1.68 vs 5.84). One C_masked seed.")
    fig.savefig(FIGS / f"{f}.png", dpi=200); plt.close(fig); print("wrote figs/" + f + ".png")

# ---------------------------------------------------------------- appendix A1 (was fig 1)
def figA1():
    f = "figA1_perposition_geometry"
    main = rows_of("results/perposition_table_C.csv", f)
    seed1 = rows_of("results/perposition_table_A_seeds.csv", f)
    want = ["D", "C", "D_math_full", "A", "B", "N3"]
    data = {a: {s: {} for s in ("neutral", "math")} for a in want + ["A seed 1"]}
    for r in main:
        if r["arm"] in want:
            data[r["arm"]][r["set"]][int(r["position"])] = (float(r["raw_norm"]), float(r["split_half_floor"] or "nan"))
    for r in seed1:
        if r["arm"] == "A_seed1":
            data["A seed 1"][r["set"]][int(r["position"])] = (float(r["raw_norm"]), float(r["split_half_floor"] or "nan"))
    arms = want + ["A seed 1"]
    colors = dict(zip(arms, plt.cm.tab10(np.linspace(0, 0.7, len(arms)))))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, s in zip(axes, ("neutral", "math")):
        ax.axvspan(-0.4, 0.4, color="0.88", zorder=0)
        ax.text(0, 1.02, "pos 0\nnot evidence", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6, color="0.4")
        for a in arms:
            pos = sorted(data[a][s])
            if not pos:
                continue
            y = [data[a][s][p][0] for p in pos]
            fl = [data[a][s][p][1] for p in pos]
            ls = "--" if a == "A seed 1" else "-"
            ax.plot(pos, y, marker="o", ms=4, lw=1.5, ls=ls, color=colors[a], label=a, zorder=3)
            ax.fill_between(pos, np.array(y) - np.array(fl), np.array(y) + np.array(fl), color=colors[a], alpha=0.18, lw=0, zorder=2)
        ax.set_yscale("log"); ax.set_xticks(range(5)); ax.set_xlabel("token position (real-token ordinal)")
        ax.set_title(f"{s} snippets"); ax.grid(alpha=0.25, which="both", lw=0.4)
    axes[0].set_ylabel(r"$\|\bar\delta_p\|$  (layer 15, natural norm)")
    axes[1].legend(fontsize=7, ncol=2, framealpha=0.9)
    fig.suptitle("Appendix Figure A1 — per-position mean activation difference, shaded band = paired split-half floor", fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    caption(fig, f, "Layer 15, 500 snippets x 128 tokens per set; band is +/- the split-half floor (half1 minus half2 of the same difference). Position 0 shaded: rejected as evidence (Gate 1).")
    fig.savefig(FIGS / f"{f}.png", dpi=200); plt.close(fig); print("wrote figs/" + f + ".png")


# ---------------------------------------------------------------- fig 2
def fig2():
    f = "fig2_visibility"
    note(f, "results/visibility_table.md")
    stats = json.loads(read("results/lora_delta_stats.json").read_text()); note(f, "results/lora_delta_stats.json")
    src = {"D": ("results/perposition_table_C.csv", "D"), "D_s1": ("results/perposition_table_seeds.csv", "D_s1"),
           "D_math": ("results/perposition_table_C.csv", "D_math"), "D_math_full": ("results/perposition_table_C.csv", "D_math_full"),
           "D_math_full_s1": ("results/perposition_table_seeds.csv", "D_math_full_s1"), "C": ("results/perposition_table_C.csv", "C"),
           "A": ("results/perposition_table_C.csv", "A"), "A_s1": ("results/perposition_table_A_seeds.csv", "A_seed1"),
           "B": ("results/perposition_table_C.csv", "B"), "N3": ("results/perposition_table_C.csv", "N3")}
    V = {}
    for label, (rel, arm) in src.items():
        for r in rows_of(rel, f):
            if r["arm"] == arm and r["position"] == "1" and r["set"] == "neutral":
                V[label] = float(r["raw_norm"]) / stats[label]["delta_W_fro_total"]
    order = ["D", "D_s1", "C", "D_math_full", "D_math_full_s1", "A", "A_s1", "D_math", "B", "N3"]
    order = [o for o in order if o in V]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    cols = ["#7b3294" if o.startswith("D_math_full") else "#2c7fb8" if o.startswith("D") and not o.startswith("D_math") else
            "#d95f02" if o.startswith("A") else "#1b9e77" if o == "C" else "#999999" for o in order]
    bars = ax.bar(range(len(order)), [V[o] for o in order], color=cols)
    for i, o in enumerate(order):
        ax.text(i, V[o] + 0.012, f"{V[o]:.3f}", ha="center", fontsize=7.5)
    ax.axhline(V["N3"], color="0.4", lw=1, ls=":")
    ax.text(0.2, V["N3"] + 0.012, "untrained-LoRA floor (N3)", ha="left", fontsize=6.5, color="0.35")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"$V=\|\bar\delta_{\mathrm{neutral},1}\|\,/\,\|\Delta W\|_F$"); ax.grid(axis="y", alpha=0.3, lw=0.4)
    ax.set_title("Figure 2 — activation-space visibility per unit of parameter change", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    caption(fig, f, "Layer 15, position 1, neutral snippets, natural norm. Seed pairs shown side by side where a second seed exists.")
    fig.savefig(FIGS / f"{f}.png", dpi=200); plt.close(fig); print("wrote figs/" + f + ".png")


# ---------------------------------------------------------------- fig 3
def fig3():
    f = "fig3_steering_dose_response"
    runs = []
    for p in sorted((REPO / "results/steer_eval").glob("*.json")):
        d = json.loads(p.read_text())
        if "predictions" in d:
            d["_f"] = p.name; runs.append(d)
    note(f, "results/steer_eval/*.json (33 runs)")
    base = next(r for r in runs if r["_f"] == "none_x1.json")
    eta = [r for r in runs if r.get("eta")]
    alphas = sorted({r["alpha"] for r in eta})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
    metrics = [("accuracy", "held-out accuracy (200 items)"), ("eos_rate", "EOS rate (stopped before cap)"),
               ("numeral_rate_first30", "numeral rate, first 30 tokens")]
    colors = {"A": "#d95f02", "C": "#1b9e77", "D_math_full": "#7b3294"}
    for ax, (key, lab) in zip(axes, metrics):
        rnd = {a: [r[key] for r in eta if r["direction"] == "random" and r["alpha"] == a] for a in alphas}
        xs = [a for a in alphas if rnd[a]]
        lo = [min(rnd[a]) for a in xs]; hi = [max(rnd[a]) for a in xs]; mid = [statistics.fmean(rnd[a]) for a in xs]
        ax.fill_between(xs, lo, hi, color="0.6", alpha=0.35, lw=0, label="random null (range)")
        ax.plot(xs, mid, color="0.35", lw=1.2, ls=":", label="random null (mean)")
        for dname, c in colors.items():
            pts = sorted([(r["alpha"], r[key]) for r in eta if r["direction"] == dname])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4.5, lw=1.6, color=c, label=dname)
        bv = base[key] if key != "numeral_rate_first30" else 0.130
        ax.axhline(bv, color="k", lw=1, ls="--", label="unsteered" if key == "accuracy" else None)
        ax.set_xscale("log"); ax.set_xticks(alphas); ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel(r"dose $\alpha$   ($\|d\|=\alpha\cdot\eta_{ref}$, $\eta_{ref}=11.24$)")
        ax.set_title(lab, fontsize=9); ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("value"); axes[0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Figure 3 — steering the BASE model at layer 15, all positions: dose-response vs a matched-norm random null", fontsize=10)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    caption(fig, f, "Random null: 5 seeds at alpha 0.25 and 0.5, 1 seed at alpha 1 and 2. Unsteered baseline dashed; its numeral rate 0.130. Greedy, cap 512, first 200 GSM8K test items.")
    fig.savefig(FIGS / f"{f}.png", dpi=200); plt.close(fig); print("wrote figs/" + f + ".png")


# ---------------------------------------------------------------- fig 4
def fig4():
    f = "fig4_A_emergence"
    rows = rows_of("results/emergence_A.csv", f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, s in zip(axes, ("neutral", "math")):
        ax2 = ax.twinx()
        for p, c in ((1, "#d95f02"), (2, "#7570b3")):
            sub = sorted([r for r in rows if r["set"] == s and int(r["position"]) == p], key=lambda r: int(r["step"]))
            steps = [int(r["step"]) for r in sub]
            ax.plot(steps, [float(r["raw_norm"]) for r in sub], marker="o", ms=4, color=c, label=f"$\\|\\bar\\delta\\|$ pos {p}")
            ax.plot(steps, [float(r["constancy"]) for r in sub], marker="s", ms=3.5, ls="--", lw=1, color=c, alpha=0.6, label=f"constancy pos {p}")
            ax.plot(steps, [float(r["cos_to_ref_same_pos"]) for r in sub], marker="^", ms=3.5, ls=":", lw=1, color=c, alpha=0.8, label=f"cos to A@150 pos {p}")
        sub = sorted([r for r in rows if r["set"] == s and int(r["position"]) == 1], key=lambda r: int(r["step"]))
        rew = [float(r["reward_at_step"]) for r in sub if r["reward_at_step"] not in ("", "None")]
        if rew:
            ax2.plot([int(r["step"]) for r in sub][: len(rew)], rew, color="0.25", lw=2, alpha=0.55)
            ax2.set_ylabel("training reward (grey)", color="0.3", fontsize=8); ax2.set_ylim(0, 1.05)
        ax.set_xlabel("optimizer step"); ax.set_title(f"{s} snippets", fontsize=9); ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("norm / constancy / cosine"); axes[0].legend(fontsize=6, ncol=2, loc="lower right")
    fig.suptitle("Figure 4 — arm A: trace geometry across checkpoints, with the reward curve", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    caption(fig, f, "Seed 0, layer 15, checkpoints 25-150. cos is to the same arm's final checkpoint, so it is 1 at step 150 by construction.")
    fig.savefig(FIGS / f"{f}.png", dpi=200); plt.close(fig); print("wrote figs/" + f + ".png")


# ---------------------------------------------------------------- fig 5
def fig5():
    f = "fig5_patchscope_tokens"
    panels = [("results/patchscope_D_s0_step250_L19.json", "math", "1", "D  L19  math  p1"),
              ("results/patchscope_D_s0_step250_L15.json", "neutral", "1", "D  L15  neutral  p1"),
              ("results/patchscope_A_s0_step150_L15.json", "neutral", "1", "A  L15  neutral  p1"),
              ("results/patchscope_B_s0_step150_L15.json", "neutral", "1", "B  L15  neutral  p1")]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 5.2))
    for ax, (rel, s, p, title) in zip(axes, panels):
        note(f, rel)
        d = json.loads(read(rel).read_text())
        e = d["sets"][s]["positions"][p]
        pl = next(x for x in e["per_lambda"] if x["lambda"] == 1.0)
        toks = [t[0] for t in pl["top20"]]; probs = [t[2] for t in pl["top20"]]
        y = np.arange(len(toks))[::-1]
        ax.barh(y, probs, color="#4575b4", alpha=0.75, height=0.72)
        for yi, t in zip(y, toks):
            ax.text(0.0, yi, " " + repr(t)[:18], va="center", ha="left", fontsize=6.5)
        ax.set_yticks([]); ax.set_xlim(0, max(probs) * 1.6 if probs else 1)
        ax.set_xlabel("mean prob. over 3 identity prompts", fontsize=7)
        ax.set_title(f"{title}\n" + r"$\lambda=1$, raw $\|\bar\delta\|$=" + f"{e['raw_norm']:.2f}", fontsize=8)
        ax.tick_params(labelsize=6); ax.grid(axis="x", alpha=0.25, lw=0.4)
    fig.suptitle("Figure 5 — token-identity Patchscope top-20 (fine-tuned model, block-15 residual of '?' replaced by lambda * delta-hat)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    caption(fig, f, "Directions rescaled to eta_ft (mean fine-tuned residual norm, ordinals >= 5) before patching; supports intersected across the three identity prompts, probabilities averaged.")
    fig.savefig(FIGS / f"{f}.png", dpi=200); plt.close(fig); print("wrote figs/" + f + ".png")


if __name__ == "__main__":
    fig1(); fig1_prev(); figA1(); fig2(); fig3(); fig4(); fig5()
    (FIGS / "figure_sources.json").write_text(json.dumps(SOURCES, indent=1) + "\n")
    print("\nfigure -> inputs:")
    for k, v in SOURCES.items():
        print(f"  {k}: {', '.join(v)}")
