#!/usr/bin/env python3
"""Build results/steer_table.md from results/steer_eval/*.json.

Recomputes the numeral rate for EVERY run (including the unsteered baseline)
from the stored completions with one code path, so the baseline column is
comparable; pairs every condition against the unsteered run on the same 200
items with an exact two-sided McNemar test; and summarises the random-direction
draws as a null distribution (mean and range per alpha).

    python tools/steer_table.py --tokenizer Qwen/Qwen3.5-4B-Base --tokenizer-revision <sha>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.acc_table import mcnemar_exact  # noqa: E402


def numeral_rate(tok, completions, first_n: int = 30) -> float:
    vals = []
    for c in completions:
        ids = tok(c, add_special_tokens=False)["input_ids"][:first_n]
        vals.append(sum(any(ch.isdigit() for ch in tok.decode([t])) for t in ids) / len(ids) if ids else 0.0)
    return float(statistics.fmean(vals)) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/steer_eval")
    ap.add_argument("--baseline", default="results/steer_eval/none_x1.json")
    ap.add_argument("--tokenizer", default=None, help="recompute numeral rates with this tokenizer (else use stored values)")
    ap.add_argument("--tokenizer-revision", default=None)
    ap.add_argument("--out", default="results/steer_table.md")
    args = ap.parse_args()
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    runs = []
    for f in sorted(glob.glob(f"{args.dir}/*.json")):
        d = json.load(open(f))
        if "predictions" not in d:
            continue
        d["_file"] = Path(f).name
        if tok is not None:
            d["numeral_rate_first30_recomputed"] = numeral_rate(tok, [p["completion"] for p in d["predictions"]])
        runs.append(d)
    base = next(r for r in runs if r["_file"] == Path(args.baseline).name)
    bcorr = {p["dataset_index"]: p["correct"] for p in base["predictions"]}
    nr = lambda r: r.get("numeral_rate_first30_recomputed", r.get("numeral_rate_first30"))
    for r in runs:
        c = {p["dataset_index"]: p["correct"] for p in r["predictions"]}
        keys = sorted(set(c) & set(bcorr))
        r["_only_steered"] = sum(c[i] and not bcorr[i] for i in keys)
        r["_only_base"] = sum(bcorr[i] and not c[i] for i in keys)
        r["_p"] = mcnemar_exact(r["_only_steered"], r["_only_base"])
        r["_n_paired"] = len(keys)
    eta = [r for r in runs if r.get("eta")]
    order = {"A": 0, "C": 1, "D_math_full": 2, "random": 3}
    lines = ["| direction | alpha | applied ‖d‖ | correct/200 | acc | EOS rate | mean len | numeral rate (first 30) | steered-only | base-only | McNemar p |",
             "|---|---|---|---|---|---|---|---|---|---|---|",
             f"| none (unsteered) | 0 | 0 | {base['n_correct']} | {base['accuracy']:.3f} | {base['eos_rate']:.3f} | {base['mean_length']:.0f} | {nr(base):.3f} | — | — | — |"]
    for r in sorted(eta, key=lambda r: (order.get(r["direction"], 9), r["alpha"], r["seed"])):
        label = r["direction"] + (f" (seed {r['seed']})" if r["direction"] == "random" else "")
        lines.append(f"| {label} | {r['alpha']:g} | {r['applied_norm']:.2f} | {r['n_correct']} | {r['accuracy']:.3f} | {r['eos_rate']:.3f} | "
                     f"{r['mean_length']:.0f} | {nr(r):.3f} | {r['_only_steered']} | {r['_only_base']} | {r['_p']:.4f} |")
    null = ["", "## Random-direction null distribution (matched norm)", "",
            "| alpha | n draws | acc mean | acc range | EOS mean | EOS range | numeral mean |", "|---|---|---|---|---|---|---|"]
    for a in sorted({r["alpha"] for r in eta if r["direction"] == "random"}):
        rs = [r for r in eta if r["direction"] == "random" and r["alpha"] == a]
        acc = [r["accuracy"] for r in rs]; eos = [r["eos_rate"] for r in rs]; num = [nr(r) for r in rs]
        null.append(f"| {a:g} | {len(rs)} | {statistics.fmean(acc):.3f} | {min(acc):.3f}–{max(acc):.3f} | "
                    f"{statistics.fmean(eos):.3f} | {min(eos):.3f}–{max(eos):.3f} | {statistics.fmean(num):.3f} |")
    nat = [r for r in runs if not r.get("eta") and r["_file"] != Path(args.baseline).name]
    natural = ["", "## Natural-norm run (dose-inadequate; see VERIFY.md)", "", "| direction | applied ‖d‖ | correct/200 | acc | EOS rate | McNemar p |", "|---|---|---|---|---|---|"]
    for r in sorted(nat, key=lambda r: (order.get(r["direction"], 9), r["scale"])):
        natural.append(f"| {r['direction']} ×{r['scale']:g} | {r['applied_norm']:.3f} | {r['n_correct']} | {r['accuracy']:.3f} | {r['eos_rate']:.3f} | {r['_p']:.4f} |")
    header = ("# eta_ref-scaled steering of the BASE model\n\n"
              "d = mean (h_adapter - h_base) over neutral snippets at ordinals >= 1, layer 15, rescaled to eta_ref = 11.243 times alpha,\n"
              "added at the layer-15 block output at ALL positions. Readout: first 200 GSM8K test items, greedy, cap 512.\n"
              "EOS rate = fraction of completions that stopped before the cap. Numeral rate = mean fraction of the first 30 generated\n"
              f"tokens whose decoded text contains a digit{' (recomputed here for every run, baseline included)' if tok else ''}.\n"
              "McNemar = exact two-sided test against the unsteered run on the same 200 items; steered-only / base-only are the\n"
              "discordant counts. No interpretation.\n\n")
    Path(args.out).write_text(header + "\n".join(lines + null + natural) + "\n\nRaw: results/steer_eval/*.json (per-item predictions retained). "
                              "Steered neutral generations at alpha=1: results/steer_eval/neutral_gens_*_a1.md.\n")
    print("\n".join(lines + null + natural))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
