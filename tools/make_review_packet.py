#!/usr/bin/env python3
"""Human reading packets under results/review_packet/.

  patchscope_for_human.md  five top-20 token lists, blinded to letters, key withheld
  steer_reading.md         the four steered continuations of the same prompt, adjacent
  cooking_samples.md       5 full arm-D corpus documents
  blackbox_rows.md         5 unsteered rows per arm, arm-labelled

Real files only; a missing input stops that section rather than being substituted.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/review_packet"


def _tok(revision):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base", revision=revision)


def patchscope(seed: int) -> list[str]:
    """Five top-20 lists at the same lambda, labelled by letter only."""
    LAM = 1.0
    specs = [("results/patchscope_D_s0_step250_L15.json", "neutral", "1", "D @ L15 neutral p1"),
             ("results/patchscope_D_s0_step250_L19.json", "math", "1", "D @ L19 math p1"),
             ("results/patchscope_A_s0_step150_L15.json", "neutral", "1", "A @ L15 neutral p1"),
             ("results/patchscope_B_s0_step150_L15.json", "neutral", "1", "B @ L15 neutral p1"),
             ("results/patchscope_N1_halves_s0_step0_L15.json", "neutral", "1", "matched null (base split-half) @ L15 neutral p1")]
    entries = []
    for rel, s, p, ident in specs:
        f = REPO / rel
        if not f.exists():
            print(f"missing {rel}; skipping patchscope section", file=sys.stderr)
            return []
        d = json.loads(f.read_text())
        e = d["sets"][s]["positions"][p]
        pl = next((x for x in e["per_lambda"] if x["lambda"] == LAM), None)
        if pl is None or not pl["top20"]:
            print(f"no lambda={LAM} list in {rel}; skipping section", file=sys.stderr)
            return []
        entries.append({"ident": ident, "raw_norm": e["raw_norm"], "tokens": [t[0] for t in pl["top20"]]})
    letters = list("ABCDE")
    order = list(range(len(entries)))
    random.Random(seed).shuffle(order)
    key = {}
    body = ["# Patchscope token lists, blinded", "",
            f"Five top-20 token-identity Patchscope lists, all at the same dose (lambda = {LAM}), position 1.",
            "Each list is labelled by a letter; which system produced which list is withheld.",
            f"Blinding seed {seed}. Lists are verbatim from the result JSONs.", "",
            "One of the five is a null (a difference between two halves of the same untrained model).",
            "Reading question: which lists, if any, look like they belong to a recognisable domain?", "", "---", ""]
    for letter, idx in zip(letters, order):
        e = entries[idx]
        # The raw norm is withheld: it identifies the arms to anyone who has seen
        # the geometry tables, which would defeat the blinding.
        key[letter] = {"identity": e["ident"], "raw_norm": round(e["raw_norm"], 3)}
        body += [f"### list {letter}", "", "```", ", ".join(repr(t) for t in e["tokens"]), "```", ""]
    (OUT / "patchscope_for_human.md").write_text("\n".join(body) + "\n")
    (OUT / "patchscope_key.json").write_text(json.dumps(key, indent=1) + "\n")
    return ["patchscope_for_human.md", "patchscope_key.json"]


def steer(revision, n_prompts: int, n_tokens: int) -> list[str]:
    arms = ["A", "C", "D_math_full", "random"]
    parsed = {}
    for a in arms:
        f = REPO / f"results/steer_eval/neutral_gens_{a}_a1.md"
        if not f.exists():
            print(f"missing {f}; skipping steer section", file=sys.stderr)
            return []
        blocks = re.split(r"(?m)^## ", f.read_text())[1:]
        d = {}
        for b in blocks:
            pid = b.split("\n")[0].strip()
            prompt = re.search(r"\*\*Prompt:\*\* (.*?)\n", b, re.S)
            comp = re.search(r"```\n(.*?)\n```", b, re.S)
            if prompt and comp:
                d[pid] = (prompt.group(1).strip(), comp.group(1))
        parsed[a] = d
    tok = _tok(revision)
    pids = sorted(set.intersection(*[set(v) for v in parsed.values()]))[:n_prompts]
    body = ["# Steered continuations of the same prompt, side by side", "",
            f"Base model, layer-15 residual shifted by each direction at alpha = 1 (applied norm 11.24) at all positions.",
            f"T = 0.7, 60 new tokens, seed 0. First {n_tokens} tokens of each generation shown; all are shorter than that,",
            "so these are complete. Arms are labelled: this packet is for reading style, not for blind scoring.", "",
            "Sources: results/steer_eval/neutral_gens_{A,C,D_math_full,random}_a1.md.", "", "---", ""]
    for pid in pids:
        body += [f"### {pid}", "", "**Prompt:** " + parsed[arms[0]][pid][0], ""]
        for a in arms:
            text = parsed[a][pid][1]
            ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
            shown = tok.decode(ids)
            body += [f"**{a}**", "", "```", shown.rstrip() + ("" if len(ids) < n_tokens else " […truncated at %d tokens]" % n_tokens), "```", ""]
        body += ["---", ""]
    (OUT / "steer_reading.md").write_text("\n".join(body) + "\n")
    return ["steer_reading.md"]


def cooking(seed: int, n: int) -> list[str]:
    f = REPO / "data/cooking.jsonl"
    if not f.exists():
        print("missing data/cooking.jsonl; skipping", file=sys.stderr)
        return []
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    idx = sorted(random.Random(seed).sample(range(len(rows)), n))
    body = ["# Arm-D training corpus: full documents", "",
            f"{n} of {len(rows)} rows from data/cooking.jsonl, drawn with random.Random({seed}).sample. Full text, unedited.",
            "This is the corpus arm D was fine-tuned on.", "", "---", ""]
    for i in idx:
        body += [f"### row {i}", "", "```", rows[i]["text"].strip(), "```", "", "---", ""]
    (OUT / "cooking_samples.md").write_text("\n".join(body) + "\n")
    return ["cooking_samples.md"]


def blackbox(seed: int, n: int) -> list[str]:
    files = sorted((REPO / "results/blackbox").glob("*.jsonl"))
    if not files:
        print("missing results/blackbox/*.jsonl; skipping", file=sys.stderr)
        return []
    body = ["# Black-box generations, arm-labelled", "",
            f"{n} rows per arm from results/blackbox/*.jsonl: the unsteered model answering neutral prompts,",
            "T = 0.7, 60 new tokens. Prompt and completion in full. No steering, no adapters beyond the named arm.", "", "---", ""]
    for f in files:
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        pick = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
        body += [f"## {f.stem}", ""]
        for i in pick:
            r = rows[i]
            body += [f"### {f.stem} — {r.get('prompt_id')}", "", "**Prompt:** " + r["prompt"].strip(), "",
                     "```", r["completion"].rstrip() or "(empty)", "```", ""]
        body += ["---", ""]
    (OUT / "blackbox_rows.md").write_text("\n".join(body) + "\n")
    return ["blackbox_rows.md"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--revision", default="1001bb4d826a52d1f399e183466143f4da7b741b")
    ap.add_argument("--steer-prompts", type=int, default=8)
    ap.add_argument("--steer-tokens", type=int, default=300)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    written += patchscope(args.seed)
    written += steer(args.revision, args.steer_prompts, args.steer_tokens)
    written += cooking(args.seed, args.n)
    written += blackbox(args.seed, args.n)
    for w in written:
        print(f"wrote results/review_packet/{w} ({(OUT / w).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
