#!/usr/bin/env python3
"""Re-score every steering run under the stopping-robust parser.

The steering grid in results/steer_table.md is scored with the preregistered
last-number parser, under which the unsteered base is 0.130. The base is 0.790
once completions are truncated at the first self-started new question, so the
raw steering numbers are movements inside the format-failure regime. This
re-scores the same stored completions with tools/reparse_acc.cut and reports
both, with exact McNemar against the unsteered run under each parser.

    python tools/steer_reparse.py
"""
from __future__ import annotations

import argparse, glob, json, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from grpo.train_grpo import extract_answer
from tools.acc_table import mcnemar_exact
from tools.reparse_acc import cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/steer_eval")
    ap.add_argument("--out", default="results/steer_table_reparsed.md")
    args = ap.parse_args()
    runs = []
    for f in sorted(glob.glob(str(REPO / args.dir / "*.json"))):
        d = json.loads(Path(f).read_text())
        if "predictions" not in d:
            continue
        raw = {p["dataset_index"]: p["correct"] for p in d["predictions"]}
        re_ = {p["dataset_index"]: extract_answer(cut(p["completion"])[0]) == p["gold"] for p in d["predictions"]}
        fired = sum(cut(p["completion"])[1] is not None for p in d["predictions"])
        d.update({"_f": Path(f).name, "_raw": raw, "_re": re_, "_fired": fired})
        runs.append(d)
    base = next(r for r in runs if r["_f"] == "none_x1.json")
    def mc(r, key):
        b = sum(r[key][i] and not base[key][i] for i in base[key])
        c = sum(base[key][i] and not r[key][i] for i in base[key])
        return b, c, mcnemar_exact(b, c)
    eta = [r for r in runs if r.get("eta")]
    order = {"A": 0, "C": 1, "D_math_full": 2, "random": 3}
    L = ["# Steering re-scored under the stopping-robust parser", "",
         "Same 33 runs and the same stored completions as `results/steer_table.md`; only the parser changes.",
         "**Every accuracy in `results/steer_table.md` and in digest §10 is the raw last-number parser.**",
         "Under the stopping-robust parser the unsteered base is 0.790, not 0.130, so the raw steering gains are",
         "movements within the format-failure regime, not reasoning gains. McNemar is against the unsteered run of the same parser.", "",
         f"Unsteered baseline: raw {sum(base['_raw'].values())}/200 = {sum(base['_raw'].values())/200:.3f}; "
         f"**re-scored {sum(base['_re'].values())}/200 = {sum(base['_re'].values())/200:.3f}** "
         f"(cuts fired on {base['_fired']}/200 completions).", "",
         "| direction | α | raw acc | raw McNemar p | re-scored acc | re-scored (steered-only/base-only) | re-scored p | cuts fired |",
         "|---|---|---|---|---|---|---|---|"]
    for r in sorted(eta, key=lambda r: (order.get(r["direction"], 9), r["alpha"], r["seed"])):
        lab = r["direction"] + (f" (seed {r['seed']})" if r["direction"] == "random" else "")
        ra = sum(r["_raw"].values()) / 200; re_a = sum(r["_re"].values()) / 200
        _, _, pr = mc(r, "_raw"); b2, c2, p2 = mc(r, "_re")
        L.append(f"| {lab} | {r['alpha']:g} | {ra:.3f} | {pr:.4f} | **{re_a:.3f}** | {b2} / {c2} | {p2:.4f} | {r['_fired']} |")
    rnd = {}
    for a in sorted({r["alpha"] for r in eta if r["direction"] == "random"}):
        rs = [sum(r["_re"].values()) / 200 for r in eta if r["direction"] == "random" and r["alpha"] == a]
        rnd[a] = (statistics.fmean(rs), min(rs), max(rs), len(rs))
    L += ["", "## Random-direction null, re-scored", "",
          "| α | draws | re-scored acc mean | range |", "|---|---|---|---|"]
    for a, (m, lo, hi, k) in rnd.items():
        L.append(f"| {a:g} | {k} | {m:.3f} | {lo:.3f}–{hi:.3f} |")
    L += ["", "Source: `results/steer_eval/*.json` (completions stored per item), `tools/steer_reparse.py`. No interpretation."]
    (REPO / args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
