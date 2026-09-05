#!/usr/bin/env python3
"""Agent FIRST PASS (not the human audit): the D_math_full alpha=0.5 vs unsteered re-scored discordant items.

Lists every base-only-correct and steered-only-correct item under the stopping-robust parser, with the
truncation point the re-parser chose and 200 characters either side, plus an automatic tag. Guiv reads.

    python tools/steer_rescue_audit.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from grpo.train_grpo import extract_answer
from tools.reparse_acc import cut

BASE = REPO / "results/steer_eval/none_x1.json"
STEER = REPO / "results/steer_eval/D_math_full_eta11.2433_a0.5.json"
ANS = re.compile(r"(####\s*-?[\d,\.]+|\\boxed\{[^}]*\}|answer is[:\s]*\$?-?[\d,\.]+|=\s*\$?-?[\d,\.]+\s*$)", re.I)

def load(p):
    d = json.loads(p.read_text())
    return {r["dataset_index"]: r for r in d["predictions"]}

def analyse(r):
    comp = r["completion"]; kept, marker = cut(comp)
    fired = marker is not None
    re_ok = extract_answer(kept) == r["gold"]
    tag = ("RESCUE" if (fired and re_ok and not r["correct"]) else
           "RAW-CORRECT" if (r["correct"] and re_ok) else
           "LOSS-BY-CUT" if (r["correct"] and not re_ok) else
           "WRONG" if not re_ok else "RE-ONLY")
    idx = len(kept) if fired else None
    tail = kept[-120:].replace("\n", "\\n")
    stated = bool(ANS.search(kept[-160:])) if fired else None
    ex = extract_answer(kept)
    def num(x):
        try: return float(str(x).replace(",", "").replace("$", ""))
        except Exception: return None
    numeric_eq = (num(ex) is not None and num(r["gold"]) is not None and abs(num(ex) - num(r["gold"])) < 1e-9)
    g = str(r["gold"]).replace(",", "")
    gold_in_tail = bool(re.search(r"(?<![\d.])\$?" + re.escape(g) + r"(?:\.0+)?(?![\d])", kept[-200:].replace(",", "")))
    return dict(fired=fired, idx=idx, re_ok=re_ok, raw_ok=bool(r["correct"]), tag=tag, stated=stated,
                extracted=ex, kept_tail=tail, marker=marker, numeric_eq=numeric_eq, gold_in_tail=gold_in_tail)

base, steer = load(BASE), load(STEER)
ks = sorted(set(base) & set(steer))
bo, so = [], []
for i in ks:
    b, s = analyse(base[i]), analyse(steer[i])
    if b["re_ok"] and not s["re_ok"]: bo.append((i, b, s))
    if s["re_ok"] and not b["re_ok"]: so.append((i, b, s))

def excerpt(comp, idx):
    if idx is None: return "(no cut fired; whole completion scored) … " + comp[-300:].replace("\n", "\\n")
    lo, hi = max(0, idx - 200), min(len(comp), idx + 200)
    return comp[lo:idx].replace("\n", "\\n") + " ⟦CUT⟧ " + comp[idx:hi].replace("\n", "\\n")

def block(i, b, s, side):
    g = base[i]["gold"]
    L = [f"### item {i} — {side}  (gold `{g}`)", "",
         f"- base: raw {'✓' if b['raw_ok'] else '✗'} → re-scored {'✓' if b['re_ok'] else '✗'} [{b['tag']}]; cut fired: {b['fired']}"
         + (f" at char {b['idx']} (marker `{str(b['marker'])[:30]}`); explicit answer statement in kept tail: {b['stated']}" if b['fired'] else "")
         + f"; extracted `{b['extracted']}`",
         f"- steered (D_math_full α=0.5): raw {'✓' if s['raw_ok'] else '✗'} → re-scored {'✓' if s['re_ok'] else '✗'} [{s['tag']}]; cut fired: {s['fired']}"
         + (f" at char {s['idx']} (marker `{str(s['marker'])[:30]}`); explicit answer statement in kept tail: {s['stated']}" if s['fired'] else "")
         + f"; extracted `{s['extracted']}`",
         "", "**base completion around the cut:**", "", "```", excerpt(base[i]["completion"], b["idx"]), "```", "",
         "**steered completion around the cut:**", "", "```", excerpt(steer[i]["completion"], s["idx"]), "```", ""]
    w = b if side.startswith("base") else s          # the side that is correct after re-scoring
    l = s if side.startswith("base") else b          # the side that is wrong after re-scoring
    if w["fired"] and w["gold_in_tail"]:
        auto = "agent tag: GENUINE-CANDIDATE — cut fired and the gold number appears in the last 200 chars of the kept text (a rescue of a stated answer)"
    elif w["fired"]:
        auto = "agent tag: CHECK — cut fired but the gold number is not in the last 200 chars of the kept text (extracted from earlier text; read it)"
    else:
        auto = "agent tag: NO-RESCUE — no cut fired on the correct side; discordance is not a re-parser rescue"
    if (not l["re_ok"]) and l["numeric_eq"]:
        auto += f"; **losing side's extracted `{l['extracted']}` equals gold numerically — a decimal/format mismatch, not a wrong answer**"
    L += [f"- {auto}", ""]
    return L

def summary(items, side):
    key = 1 if side.startswith("base") else 2
    tags = {}
    for it in items: tags[it[key]["tag"]] = tags.get(it[key]["tag"], 0) + 1
    gen = sum(1 for it in items if it[key]["fired"] and it[key]["gold_in_tail"])
    lose = 2 if key == 1 else 1
    fmt = sum(1 for it in items if (not it[lose]["re_ok"]) and it[lose]["numeric_eq"])
    nocut = sum(1 for it in items if not it[key]["fired"])
    return tags, gen, fmt, nocut

tb, gb, fb, nb = summary(bo, "base"); ts, gs, fs, ns = summary(so, "steered")
out = ["# Steered-rescue audit — AGENT FIRST PASS, not the human audit (C4 task 2)", "",
       "Comparison: unsteered base (`results/steer_eval/none_x1.json`) vs D_math_full direction at η_ref × 0.5 "
       "(`results/steer_eval/D_math_full_eta11.2433_a0.5.json`), both re-scored with `tools/reparse_acc.cut` + "
       "`grpo.train_grpo.extract_answer` (the stopping-robust parser). Digest §10(c): re-scored 0.650 vs 0.790, "
       f"15 steered-only / 43 base-only, p = 0.0003. Found here: **{len(bo)} base-only, {len(so)} steered-only**.", "",
       "Tags: RESCUE = raw wrong, re-scored right via the cut; RAW-CORRECT = right under both parsers; LOSS-BY-CUT = raw right, "
       "re-scored wrong (the cut removed the scored answer); WRONG = wrong under both. 'explicit answer statement' = regex for "
       "`####`, `\\boxed{}`, `answer is N`, or a trailing `= N` in the last 160 chars of the kept text — a heuristic, not a reading.", "",
       f"**Base-only (n = {len(bo)}):** base tags {tb}; base cut fired with the gold number in the kept tail (GENUINE-CANDIDATE) in {gb}/{len(bo)}; "
       f"no cut fired on the base side in {nb}/{len(bo)}; the steered side's extracted answer equals gold numerically (format mismatch) in {fb}/{len(bo)}. "
       f"**Steered-only (n = {len(so)}):** steered tags {ts}; GENUINE-CANDIDATE {gs}/{len(so)}; no cut on the steered side {ns}/{len(so)}; "
       f"the base side's extracted answer equals gold numerically (format mismatch) in {fs}/{len(so)}.", "",
       "Everything below is for Guiv to read; the agent tags are a heuristic first pass. The 20-item audit in "
       "`results/reparse_audit.md` covered unsteered arms only; **no steered-completion rescue had been read before this file.**", "",
       f"## Base-only correct (re-scored): {len(bo)} items", ""]
for it in bo: out += block(*it, "base-only")
out += [f"## Steered-only correct (re-scored): {len(so)} items", ""]
for it in so: out += block(*it, "steered-only")
out += ["## Regenerate", "", "```bash", "python3 tools/steer_rescue_audit.py", "```"]
(REPO / "results/steer_rescue_audit.md").write_text("\n".join(out) + "\n")
print("\n".join(out[:12]))
