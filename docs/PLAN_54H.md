# PLAN_54H — from 03:15 Thu Sept 3 to 08:59 Sat Sept 5 (Europe/Zurich)

Active-time budget: 17h project + 2h executive summary (cap 20 + 2). Training time while you do something else is not counted. Track everything in Toggl from the first review minute.

## Night 1 — Thu 03:15 → 11:00 (you sleep; agents 01–05 run)
Outputs waiting for you: validated readout pipeline + analysis scripts on mock data (01); TRL-adapted training scripts, parse-rate check, eval script, pod guide, and — if you gave it a pod — arm D and maybe A/B already training (02); cooking corpus, snippet sets, calibrated judge, lexical baseline (03); `docs/PROTOCOL_NOTES.md` with Minder's exact protocol and PREREG fills (04); `docs/RED_TEAM.md` (05).

## Day 1 — Thu Sept 3 (≈8h active)
| Zurich | Active h | What |
|---|---|---|
| 11:00–12:30 | 1.5 | **Review & merge.** Read all five reports first. Re-run tests. Read 20 random corpus docs + judge calibration. Accept/reject agent changes. Log to VERIFY.md ("overnight work" section). |
| 12:30–13:00 | 0.5 | **Freeze PREREG** (layer, norm target, judge, coefficient grid). Commit. |
| 13:00–14:00 | 1 | Pod up (skip if done overnight). Snippet sets built with the real tokenizer. Launch **arm D** (GPU0), **A** (GPU1), **B** (GPU2). GPU3: null adapter N3, base held-out accuracy. *Training runs in background; not counted.* |
| 14:00–16:00 | 2 | **Gate 1 on D** as soon as its adapter exists (~1h SFT): readouts on D vs N1/N2/N3, judge with shuffled labels + lexical baseline. Decision: replicate → continue; fail → ≤1h diagnosis (layer ±4, more snippets), then fix or pivot to Fallback (Olmo 3 stage diffing), reset timer. |
| 16:00–17:00 | 1 | Arm A finishes (~2–3h): `eval_acc.py` on A and B (Gate 2). Start **arm C**: sample from A's final policy, filter correct, SFT (GPU1). |
| 17:00–19:00 | 2 | Readouts on **A and B** on both snippet sets; A−B residual; judge; lexical baseline; first look at Figure 1. Write the observation (not the interpretation) in the doc. |
| evening | 0 | Arm C trains; optionally launch seeds 1–2 for A and B on free GPUs (uncounted). Sleep early. |

## Night 2 — Thu 23:00 → Fri 08:00 (agents optional)
If you want agents working: (a) Agent 01 re-runs `summarize.py` on the real results and regenerates figures; (b) Agent 05 gets the *real* Figure 1 + cosine table and red-teams the actual numbers; (c) Agent 02 runs readouts on the extra seeds and on layer L±4 (robustness) and writes CSVs — no interpretation. Nothing from Night 2 enters the doc unverified.

## Day 2 — Fri Sept 4 (≈9h active + 2h summary)
| Zurich | Active h | What |
|---|---|---|
| 09:00–10:00 | 1 | Review Night-2 outputs. Readouts on **arm C**; complete Figure 1 (A, B, C, D, nulls × neutral/on-domain). |
| 10:00–12:00 | 2 | **Sanity pass (this is what Neel grades):** recompute every headline number with your own one-liners from the CSVs; read 30 random judge transcripts and 30 steered generations (`analysis/sample_raw.py`); check the shuffled-label accuracy is at chance; check lexical baseline vs judge; check layer L±4 robustness if available. Fill VERIFY.md. |
| 12:00–13:00 | 1 | Answer every RED_TEAM item in the ledger (check run, or limitation admitted). |
| 13:00–16:00 | 3 | **Write the doc body** (Google Doc): one paragraph + figure per experiment; observation → interpretation; limitations; verification section with the ledger; random raw examples; Toggl screenshot placeholder. |
| 16:00–17:00 | 1 | Buffer / slack for whatever broke. Hard stop on new experiment code at 17:00. |
| 17:00–19:00 | +2 | **Executive summary** in your own voice (≤600 words, 3 figures). No LLM prose. Use an LLM only to critique for clarity with an anti-sycophancy prompt. |
| 19:00–21:00 | 0 (uncounted) | **Form answers** Q10–Q21 in your own voice (see NEEL_RUBRIC.md; use your separate application materials for Q18; no private program numbers; COLM is a *workshop* paper). Doc sharing = anyone with link. Toggl screenshot in. |
| 21:00–22:00 | 0 | Final read-through. **Submit.** Do not wait for the morning. |
| Sat 08:59 | — | Absolute deadline. |

## If you are behind at Thu 19:00
Cut in this order: (1) extra seeds and layer robustness; (2) steering modality (keep tokens + self-report); (3) arm C — then the story is "GRPO vs shuffled-reward vs narrow SFT", still a clean result, and you say plainly that the on/off-policy comparison is the missing control.

## Claim-firewall reminders for the doc and form
COLM = workshop paper. GRPO program = unpublished, described as process, no numbers, not "submitted to ICLR". Shuffled-reward control is a method you use, cite as such. No invented collaborators. "Silent/unobserved" language from your other work does not appear here at all.
