#!/usr/bin/env python3
"""Paired held-out accuracy for arm C seed 1 vs C seed 0 and A seed 0 under both parsers (CPU only).

Parsers: `raw_last_number` = preregistered grpo.eval_acc.extract_answer on the full completion;
`rescored` = tools.reparse_acc.cut (truncate at the first line that starts a new question) then the same extractor.
Exact two-sided McNemar on discordant pairs (tools.acc_table.mcnemar_exact). Writes results/acc_table_C_s1.md (+ .json).
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from grpo.eval_acc import extract_answer  # noqa: E402
from tools.reparse_acc import cut  # noqa: E402
from tools.acc_table import mcnemar_exact  # noqa: E402

PARSERS = {"raw_last_number": extract_answer, "rescored": lambda c: extract_answer(cut(c)[0])}


def load(path: str):
    d = json.loads(Path(path).read_text())
    per = {name: {int(r["dataset_index"]): fn(r["completion"]) == r["gold"] for r in d["predictions"]} for name, fn in PARSERS.items()}
    return d, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", default="C_s1:results/acc_C_s1.json")
    ap.add_argument("--refs", nargs="+", default=["C_s0:results/acc_C_s0.json", "A_s0:results/acc_A_s0.json"])
    ap.add_argument("--out", default="results/acc_table_C_s1.md")
    ap.add_argument("--title", default="arm C seed 1 vs C seed 0 and A seed 0")
    args = ap.parse_args()
    specs = [args.x] + args.refs
    data, files = {}, {}
    for spec in specs:
        label, path = spec.split(":", 1)
        data[label] = load(path); files[label] = {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    xl = args.x.split(":")[0]
    shas = {data[l][0]["snippet_sha"] for l in data}; caps = {int(data[l][0]["decoding"]["max_new_tokens"]) for l in data}
    assert len(shas) == 1 and len(caps) == 1, (shas, caps)
    single, paired = [], []
    for parser in PARSERS:
        for l, (d, per) in data.items():
            k = sum(per[parser].values()); n = len(per[parser])
            single.append({"parser": parser, "arm": l, "seed": d["seed"], "step": d.get("step"), "n": n, "n_correct": k, "accuracy": k / n})
        cx = data[xl][1][parser]
        for l in data:
            if l == xl:
                continue
            cy = data[l][1][parser]; keys = sorted(set(cx) & set(cy))
            both = sum(cx[i] and cy[i] for i in keys); xo = sum(cx[i] and not cy[i] for i in keys); yo = sum(cy[i] and not cx[i] for i in keys)
            paired.append({"parser": parser, "x": xl, "y": l, "n": len(keys), "acc_x": sum(cx[i] for i in keys) / len(keys), "acc_y": sum(cy[i] for i in keys) / len(keys),
                           "both": both, "x_only": xo, "y_only": yo, "neither": len(keys) - both - xo - yo, "mcnemar_exact_p": mcnemar_exact(xo, yo)})
    md = [f"# Held-out accuracy: {args.title}", "",
          f"Same 200 GSM8K test items (set sha {next(iter(shas))[:8]}…), greedy, cap {next(iter(caps))}, both parsers. Generated {datetime.now(timezone.utc).isoformat()}.", "",
          "| parser | arm | seed | step | correct | acc |", "|---|---|---|---|---|---|"]
    md += [f"| {r['parser']} | {r['arm']} | {r['seed']} | {r['step']} | {r['n_correct']}/{r['n']} | {r['accuracy']:.3f} |" for r in single]
    md += ["", "| parser | x | y | acc x | acc y | both | x only | y only | neither | McNemar exact p |", "|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['parser']} | {r['x']} | {r['y']} | {r['acc_x']:.3f} | {r['acc_y']:.3f} | {r['both']} | {r['x_only']} | {r['y_only']} | {r['neither']} | {r['mcnemar_exact_p']:.3f} |" for r in paired]
    Path(args.out).write_text("\n".join(md) + "\n")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    Path(args.out).with_suffix(".json").write_text(json.dumps({"arm": "C", "seeds_compared": {l: data[l][0]["seed"] for l in data}, "snippet_set": data[xl][0]["snippet_set"],
        "snippet_sha": next(iter(shas)), "judge_model": None, "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "inputs": files, "single": single, "paired": paired}, indent=1) + "\n")
    print("\n".join(md)); print("wrote", args.out)


if __name__ == "__main__":
    main()
