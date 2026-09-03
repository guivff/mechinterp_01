# AGENTS.md — canonical entrypoint for every agent and every human session on this project

Project: **rl-readable-trace** — "Does on-policy GRPO leave a readable activation-difference trace like narrow SFT does?"
Purpose: Guiv Farmanfarmaian's application project for Neel Nanda's MATS 12.0 stream. **Hard deadline: Fri Sept 4, 2026, 23:59 PT (Sat Sept 5, 08:59 Europe/Zurich).**

## 1. Read first, in this order
1. `PROGRAMME_RULE.md` — the one governing rule (30 seconds).
2. `PROGRAM_STATE_CURRENT.md` — canonical current state, ~4 KB. If it conflicts with anything else, it wins unless a frozen artifact says otherwise.
3. `CLAIM_FIREWALL.md` — what may and may not be claimed, in the write-up and in the application form.
4. `OPEN_TASKS_CURRENT.md` — the live queue only. Take a task from here; do not invent scope.
5. `PREREG.md` — frozen hypotheses, definitions, pass criteria. Never edit after the freeze commit except in the append-only Amendments section.
6. Only if your task needs it: `context/PROJECT_SPEC.md` (design), `docs/PROTOCOL_NOTES.md` (Minder/OOCR/J-Lens/GRPO-recipe facts), `docs/RED_TEAM.md` (23 ranked confounds), `docs/THEORY_NOTE.md` (why zero-sum advantages cancel the topic trace), `docs/NEXT_STEPS_T35H.md` (runbook), `context/NEEL_RUBRIC.md` (how the application is judged).
7. `SOURCE_INDEX.md` when you need to locate a frozen artifact, hash, or the primary source behind a claim.

## 2. Source precedence (when facts conflict)
1. Frozen artifacts with hashes: `PREREG.md` at its freeze commit, `data/*manifest*.json`, `results/*.json` produced by committed code at a named commit, `VERIFY.md` ledger rows.
2. `PROGRAM_STATE_CURRENT.md` and `CHANGELOG.md` (latest dated entry wins).
3. Agent final reports (quoted in `CHANGELOG.md`), then `docs/PROTOCOL_NOTES.md` and `docs/RED_TEAM.md`.
4. `context/PROJECT_SPEC.md` and `docs/NEXT_STEPS_T35H.md` (design intent; may be superseded by decisions in PROGRAM_STATE).
5. Mock figures/CSVs (`*MOCK*`) — illustrative only, never evidence.
Primary papers outrank all project summaries on what the papers say; `docs/PROTOCOL_NOTES.md` records exact citations and paper-vs-code discrepancies.

## 3. Current scientific objective (one paragraph)
Measure whether the mean base→fine-tuned residual difference on unrelated text (Minder et al. 2510.13900) is decodable to the training domain after **GRPO** (arm A) the way it is after **narrow SFT** (arm D), with a **shuffled-reward GRPO** control (B) and, if time allows, **rejection-sampling SFT on A's own samples** (C). Readouts: block-wise diff geometry, norm-matched logit lens, blind LLM judge with lexical and null baselines, self-report, held-out accuracy. Theory (`docs/THEORY_NOTE.md`) predicts the zero-sum group advantages cancel the prompt-shared "topic" component, so A's trace should be less constant and less topic-readable than C's or D's.

## 4. Live vs dead routes
**Live:** arms A, B, D, nulls N1–N3; token (logit-lens) readout; self-report; held-out accuracy with paired inference; block-wise (K=10) estimator; neutral + math snippet sets; layer 15 primary, 11/19 sensitivity; theory note as the framing for H2.
**Conditional:** arm C (only if A finishes by ~04:00 Fri); steering readout (first to cut); extra seeds (only if A vs C/D contrast is marginal and time exists).
**Dead — do not revive:** J-Lens (no pre-fitted lens for `Qwen/Qwen3.5-4B-Base`); arm C′ self-distillation; layer 0.6·D; the "base mean activation" version of N1; the label-order-reordering version of the shuffle control; TF-IDF trained on the readout texts; Patchscope (out of time budget; noted as a limitation); GPT-2/Pythia/Gemma-2 models; any claim that A−B is a "reward-specific vector".

## 5. How to incorporate new results
- Never overwrite a historical verdict. Append a dated entry to `CHANGELOG.md`; then update `PROGRAM_STATE_CURRENT.md` by *replacing* its state sections (it is a snapshot, not a log).
- A result enters `PROGRAM_STATE_CURRENT.md` only with: arm, seed, checkpoint step, layer, snippet-set hash, judge model, commit, and the `VERIFY.md` row id. Without those it goes in `CHANGELOG.md` as "unverified agent claim".
- Failed, restarted, or abandoned runs are logged in `CHANGELOG.md` under "attempt ledger" (intention-to-treat; the red team will look for a file drawer).
- Retrieve the frozen artifact rather than the summary whenever: a number goes into the write-up or form; a threshold in `PREREG.md` is being tested; two summaries disagree; or a figure is regenerated.
- `VERIFY.md` is additive across agents. If your edit would replace another agent's rows, stop and merge instead (this already happened once: Agent 03 overwrote Agent 01's ledger — see `OPEN_TASKS_CURRENT.md`).

## 6. Updating programme state after each agent/experiment return
1. Paste the agent's final report verbatim (or a link to it) into `CHANGELOG.md` with date, commit, and the agent id.
2. Move finished items out of `OPEN_TASKS_CURRENT.md`; add any new blockers the report raised.
3. If a claim boundary changed (e.g. J-Lens dropped, C cut), update `CLAIM_FIREWALL.md` and `PROGRAM_STATE_CURRENT.md`.
4. If a definition changed, it is a `PREREG.md` amendment (append-only, dated, with reason) — never a silent edit.
5. Add new frozen artifacts (hashes, commits, HF revisions) to `SOURCE_INDEX.md`.

## 7. Coding rules (unchanged; every coding agent)
- The human designs experiments, chooses layers/arms/metrics, and verifies every headline number. You implement, test, and report. Never write "the experiment worked"; write what was measured, with file paths.
- Every result file carries: arm, seed, checkpoint step, layer, snippet-set name + sha256, judge model, timestamp, git commit.
- Blind-decoding batches always include the shuffle control and the lexical baseline; never drop them.
- Figures → `figs/*.png`; tables → `results/*.csv|json`; cache activations (fp16 `.npy`), diff vectors, adapters. Mock outputs carry `MOCK` in the filename and are never mixed with real ones.
- Long jobs = background scripts with logs under `logs/`; persistent kernel if available; Codex sessions use `.py` only.
- Smoke-test on the tiny model (`tests/`) before any 4B/GPU work.
- Do not change a `PREREG.md` definition silently. Write the change and reason to `VERIFY.md` → "definition changes" and stop.
- Any suspected confound (leakage, template/BOS mismatch, padding side, hook return type, norm mismatch, judge reading surface tokens) → `VERIFY.md` → "agent-raised concerns" before continuing.
- Final message of every task is a **report**: what you ran (files, args, seeds, commit), outputs, what you did NOT check, and the three most likely ways your work is wrong.

## 8. Never
- Never present a design/prelaunch state as a result. Never call D "a replication of Minder"; it is a reduced-budget conceptual replication.
- Never claim causality from A−B, from norm-matched decodability, or from a between-corpus contrast.
- Never let LLM-written prose into the executive summary or the application form (Neel: "a significant negative signal").
- Never cite unpublished numbers from Guiv's other research programme; see `CLAIM_FIREWALL.md` §3.
