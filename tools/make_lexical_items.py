#!/usr/bin/env python3
"""Build results/lexical_items_perposition.jsonl: every real per-position readout
list, as judge/lexical-baseline items.

Reads every results/perposition_*_L*.json (logit-lens top-20 per position) and
every results/patchscope_*_L*.json (Patchscope top-20 at the selected lambdas).
Arms map to their gold domain; nulls map to "none".
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLD = {"D": "cooking", "D_math": "math", "D_math_full": "math", "C": "math",
        "A": "math", "B": "none", "N1_halves": "none", "N3": "none"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    ap.add_argument("--out", default="results/lexical_items_perposition.jsonl")
    args = ap.parse_args()
    rows, skipped = [], []

    def add(arm, s, pos, text, kind, lam=None, layer=15):
        if arm not in GOLD:
            skipped.append(arm); return
        rows.append({"arm": arm, "seed": 0, "step": 0, "layer": layer, "snippet_set": s, "position": pos,
                     "modality": "tokens", "readout": kind, "lambda": lam, "text": text, "true": GOLD[arm],
                     "item_id": f"{arm}:{kind}:L{layer}:{s}:pos{pos}:{lam}"})

    for f in sorted(glob.glob(str(REPO / "results/perposition_*_L*.json"))):
        d = json.loads(Path(f).read_text())
        layer = int(re.search(r"_L(\d+)", f).group(1))
        for s, sd in d["sets"].items():
            for p, e in sd.items():
                if p.startswith("N1"):
                    add("N1_halves", s, 0, e["text_final_norm"], "logit_lens", layer=layer)
                else:
                    add(d["arm"], s, int(p), e["text_final_norm"], "logit_lens", layer=layer)
    for f in sorted(glob.glob(str(REPO / "results/patchscope_*_L*.json"))):
        d = json.loads(Path(f).read_text())
        layer = int(re.search(r"_L(\d+)\.json", f).group(1))
        for s, sd in d["sets"].items():
            for p, e in sd["positions"].items():
                for pl in e["per_lambda"]:
                    if pl["lambda"] in args.lambdas and pl["top20"]:
                        add(d["arm"], s, int(p), ", ".join(t[0] for t in pl["top20"]), "patchscope", pl["lambda"], layer)
    out = REPO / args.out
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    import collections
    print(f"wrote {args.out}: {len(rows)} rows")
    print("  by arm:", dict(collections.Counter(r["arm"] for r in rows)))
    print("  by readout:", dict(collections.Counter(r["readout"] for r in rows)))
    if skipped:
        print("  skipped unknown arms:", sorted(set(skipped)))


if __name__ == "__main__":
    main()
