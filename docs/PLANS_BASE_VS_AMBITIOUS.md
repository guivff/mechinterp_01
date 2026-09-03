# PLANS — Base vs Ambitious (Thu Sept 3, 23:15 Zurich; 33.7 h to deadline)

Both plans share the same core and the same gates. The ambitious plan is the base plan plus a fixed list of add-ons, each of which is cheap *given* the base runs exist, and each of which is dropped without regret if its precondition fails. You never choose between the plans up front; you climb from base to ambitious as gates pass and GPUs sit idle.

## BASE PLAN — what gets submitted no matter what

**Claim shape.** "Narrow SFT leaves a readable, near-constant activation trace in this setup (positive control). Intact-reward GRPO that materially improves accuracy leaves a trace that is [smaller/less constant/less topic-readable — or not]; shuffled-reward GRPO leaves [X]. All controls and nulls shown. Single seed, one layer primary."

**Runs (4 GPUs, tonight):** D (SFT cooking), A (GRPO intact), B (GRPO shuffled), nulls N1/N2/N3, base activation cache at L=11/15/19.

**Readouts:** block-wise (K=10) diff geometry (norm, constancy, block-to-block cosine); norm-matched logit lens top-20 per block; blind judge (gpt-5-mini, majority of 3) with pairing-shuffle, always-math/always-none, confusion matrix, TF-IDF-on-external-corpus baseline; self-report (20 samples × arm); held-out accuracy with paired McNemar and 20 discordant items read.

**Figures:** Fig 1 block-level judge accuracy by arm × snippet set with all baselines; Fig 2 norm + constancy by arm; Fig 3 top-20 tokens for A, B, D, N1 (one block each, chosen by seed). Table: cosine matrix.

**Write-up:** ≤600-word executive summary in your voice; one paragraph + figure per experiment; theory note as the *motivation* paragraph (why zero-sum advantages should cancel the topic component); limitations (single seed, one model, conceptual not literal Minder replication, no C → H2 untested); verification section from VERIFY.md; 30 random judge transcripts read; Toggl screenshot.

**Time:** Gate 1 by ~00:30, sleep, Gate 2 + A/B decoding by ~09:00 Fri, sanity pass and **submittable v1 (summary + doc + form) by Fri 12:00**; the rest of Friday and the night go to add-ons under the Sat 04:00 stop rule below; submit by Sat 07:30.

**Why this already clears Neel's bar:** a non-generic question he wrote down himself, a modern model, a positive control that is checked before anything else, every cheap control he lists, a theory-derived prediction, honest single-seed framing, and a verification story. A clean negative result ("GRPO trace is not D-like") is a finding.

## AMBITIOUS PLAN — base plus add-ons, in the order you take them

Each add-on names its precondition and its cost in *your* hours (GPU hours are free once the pod is up and mostly overnight).

| # | Add-on | Precondition | Your time | What it buys |
|---|---|---|---|---|
| A1 | **Trace-emergence curve.** Readouts on A's and B's checkpoints (steps 25…150): ‖d‖, constancy, judge accuracy vs step, overlaid on the reward curve. | A/B trained with ckpt every 25 (already configured) | 45 min | The first "when does an RL trace appear, and does it track accuracy?" curve anyone has shown. Cheap because it's readouts only on cached snippets. |
| A2 | **Arm C** (rejection-sampling SFT on A's own correct samples) → tests H2 with the theory's sign. | A finished by ~04:00 Fri | 1 h | Turns "SFT vs RL" into "same data, different weighting" — the mechanism claim. |
| A3 | **Seeds 1–2 of A and B** on idle GPUs overnight. | GPU 3 idle after caching | 30 min | Cross-seed cosine of d_A — a convergent-direction result à la EM papers — and honest error bars on the headline. |
| A4 | **Conditional-trace test** (theory prediction T3): ‖d_A,math‖ vs ‖d_A,neutral‖ against ‖d_D,math‖ ≈ ‖d_D,neutral‖. | base runs only | 15 min | The cleanest "RL writes context-dependent traces" line; zero extra compute. |
| A5 | **Contrast-tokens figure**: A vs C/D top-20 side by side, with a one-line prediction written before unblinding. | A1 or base | 20 min | The memorable figure if the theory holds; an honest miss if it doesn't. |
| A6 | **Layer sweep 11/15/19** for A and D only. | cached activations | 20 min | Answers "is this layer-specific?" before Neel asks. |
| A7 | **Minder-faithful per-position (0–4) diagnostic on D.** | cached activations with position ids | 20 min | Answers "what did D validate?" |
| A8 | **Cross-arm steering swap** (steer base with d_A vs d_C on the same prompts, judge blind). | steering not cut; A2 done | 45 min | Qualitative demonstration of contrast-vs-topic. First to drop. |
| A9 | **Theory note tightened** (class-contrast term named; Adam paragraph; prompt-token masking caveat for C) and critiqued by a fresh chat. | none | 1 h | Makes the framing paragraph reviewer-proof. |

**Ambitious claim shape.** "Narrow SFT writes a constant, topic-readable trace; GRPO on the same task writes a smaller, less constant, context-dependent trace that emerges with accuracy and is not topic-readable at matched norm; rejection-sampling SFT on GRPO's own samples restores the topic trace — consistent with the zero-sum-advantage cancellation argument. Shown on one model, one layer primary with sensitivity, [n] seeds, all nulls and baselines."

**Stop rule for ambition (Guiv's decision, 2026-09-03 23:40): Sat Sept 5, 04:00 Zurich.** Any add-on not *verified and in VERIFY.md* by 04:00 is cut and listed as future work in two sentences. No new numbers enter the doc after 04:00.

To make a 04:00 cutoff survivable with a 08:59 deadline, writing is **incremental, not terminal**:
- Fri 10:00 — after Gate 2 and the base A/B/D decoding: write the executive summary v1 and the doc body for the base plan, in your own voice, as if nothing else will arrive. Fill the form Q10–Q21 v1 (uncounted time). This is the submittable fallback.
- Each add-on that verifies later is *inserted* as one paragraph + one figure and one sentence in the summary; it never triggers a rewrite.
- Fri 18:00 — P6 write-up critic on v1; fix voice/claim issues then, not at 04:00.
- Sat 04:00–06:00 — final insertions, final `summarize.py` run on real files only, recompute headline numbers once more, VERIFY.md closed.
- Sat 06:00–07:30 — read-through, Toggl screenshot, doc permissions, form pasted. **Submit by 07:30.** 07:30–08:59 is buffer for the Airtable upload, not for edits.
- If at Sat 02:00 any add-on is still unverified, stop it; the last two hours before 04:00 are for verification of what exists, never for new runs.

## What would make Neel's day (and what would sink it)
Makes it: the emergence curve (A1) and the C comparison (A2) together, with the theory paragraph predicting the sign before the data; random raw examples; a verification section that says exactly what you recomputed and read; a limitation he would have raised, already addressed.
Sinks it: any number from a mock file; block counts misreported as observations; "replicates Minder"; LLM-sounding summary; a pod-left-running anecdote in place of results.
