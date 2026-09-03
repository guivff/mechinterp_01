# RUNBOOK_NOW — Fri Sept 4, 00:06 Zurich. Deadline Sat 08:59. Stop rule for new numbers Sat 04:00.

Six windows. Do them in this order; each step says where it runs. Times are targets.

## STEP 1 — Terminal (personal account) · 5 min · 00:10
```bash
cd ~/Documents/Apply/NeelNanda
git clone https://github.com/guivff/mechinterp_01.git .          # note the dot
unzip -o control_plane_update.zip && cp -R rl-readable-trace/. . && rm -rf rl-readable-trace
git add -A && git commit -m "control plane + round-2 prompts + docs" && git push
source ~/.config/mats/secrets.env                                 # keys into this shell only
for v in RUNPOD_API_KEY OPENROUTER_API_KEY HF_TOKEN; do printf "%s=%s…\n" "$v" "${(P)v:0:6}"; done
```
If `unzip`/`cp` complain about overwriting AGENTS.md or PREREG.md, say yes — the overlay is newer.

## STEP 2 — RunPod console (browser) · 3 min · 00:15
Create one pod by hand: 4× H100 80GB (or H200), PyTorch/CUDA 12.x template, ≥200 GB volume, name `mats-rl-trace`. Copy the SSH command. (Doing this by hand removes a whole failure surface from the critical path; the API key stays exported for later seed runs.)

## STEP 3 — Claude Code chat #1 = POD RUNNER (same Terminal, personal account) · start 00:20, runs all night
```bash
claude
```
First message: paste the whole of `prompts/round2/P1_CLAUDE_CODE_POD_RUNNER.md`, then add one line: `The pod already exists; SSH: <paste ssh command>. Skip pod creation.`
It will: set up the pod → preflight (report parse rate, truncation, base accuracy, pinned HF revision) → implement reward-0-on-truncation and the identity check → launch D (GPU0), A (GPU1), B (GPU2), caches + nulls (GPU3) → report every 10 min for 40 min. Say "go" only at the points it asks. **Leave this window open all night.**

## STEP 4 — Freeze PREREG (you, in a second Terminal tab) · 10 min · while preflight runs
When chat #1 reports the HF revision: edit `PREREG.md` (model revision line, freeze commit/time at top), then
```bash
git add PREREG.md && git commit -m "PREREG freeze" && git push && git rev-parse --short HEAD
```
Paste the hash into `PREREG.md` top line and `CHANGELOG.md` ("PREREG frozen at <hash>"), commit again. Tell chat #1 "PREREG frozen at <hash>".

## STEP 5 — Codex cloud (chatgpt.com/codex, WORK account if it has the higher limits) · 10 min · 00:35
Environment for `mechinterp_01`: setup script = `CODEX_SETUP.sh`, internet ON, secret `OPENROUTER_API_KEY`.
Two tasks, in parallel:
- Task A: paste `prompts/round2/P2_CODEX_PIPELINE_01b.md` (block-wise estimator, emergence-curve tooling, figures, VERIFY merge). Branch `agent01b`.
- Task B: paste `prompts/round2/P3_CODEX_JUDGE_03b.md` (live judge calibration, external TF-IDF corpus, self-report scoring). Branch `agent03b`.
If the work account cannot see the private repo, add it as a collaborator on GitHub first (Settings → Collaborators), or run these two in Claude Code chats #2/#3 on the personal account instead (`cd` to the repo, `claude`, paste the prompt with "work on branch agent01b").

## STEP 6 — Theory critic (ChatGPT or Claude.ai chat, WORK account) · 5 min to launch · 00:45
New chat, upload `context_pack.zip`, paste `prompts/round2/P4_CHAT_THEORY_CRITIC.md`. Read its answer while D trains; fold the accepted points into `docs/THEORY_NOTE.md` yourself (30 min), commit.

## STEP 7 — Gate 1 (you + chat #1) · ~01:30
When chat #1 says "D ready for Gate 1": pull `agent01b` if merged (else use the current readout code), run block-wise readouts on D vs N1/N2/N3 at L=15 on both snippet sets, judge, TF-IDF, self-report. **You** read 10 judge transcripts and D's top tokens. Record PASS/FAIL in `CHANGELOG.md`. If fail: L=19 once, then the pivot decision (Olmo-3 stage diffing, timer reset) — decide, do not drift.
Then sleep. Chat #1 keeps watching A/B and will have Gate-2 accuracy numbers ready.

## STEP 8 — Morning: Gate 2 and base decoding (you + chat #1) · 07:30–09:30 Fri
Paired accuracy A/B vs base; read 20 discordant items. Record. Decode A, B block-wise (both sets, L=15; then 11/19); A−B; self-report. `summarize.py` on real files only. **Recompute every headline number with your own one-liner.**

## STEP 9 — Submittable v1 (you alone, no LLM prose) · 09:30–12:00 Fri
Google Doc: executive summary v1 (≤600 words, 3 figures), doc body v1, random raw examples, verification section from `VERIFY.md`. Airtable form Q10–Q21 v1 in a text file (uncounted). Set doc sharing to anyone-with-link. This is what gets submitted if everything else fails.

## STEP 10 — Add-ons under the Sat 04:00 stop rule (chat #1 + you) · Fri 12:00 → Sat 02:00
In order: A1 emergence curve (readouts on A/B checkpoints), A2 arm C if launched, A4 conditional-trace norm test, A3 seeds 1–2 (launch Fri morning on idle GPUs so they finish by evening), A5 contrast-tokens figure, A6 layer sweep, A7 per-position D diagnostic, A8 steering swap (only if time). Each verified add-on = one paragraph + one figure inserted into the doc, one sentence into the summary, one row in `VERIFY.md`.

## STEP 11 — Critics (WORK account chats) · Fri 15:00 and Fri 18:00
- 15:00: new chat, `context_pack.zip` + real `figs/` + `results/cosine_matrix.csv` + judged summaries + `VERIFY.md`, paste `P5_CHAT_RESULTS_RED_TEAM.md`. Answer its top-3 live confounds in `VERIFY.md`.
- 18:00: new chat, `context_pack.zip` + your summary and form v1, paste `P6_CHAT_WRITEUP_CRITIC.md`. Fix voice and claim issues yourself.

## STEP 12 — Close and submit · Sat 02:00–07:30
02:00 stop launching anything. 04:00 no new numbers. 04:00–06:00 final insertions, final `summarize.py`, last recompute, close `VERIFY.md`, Toggl screenshot into the doc. 06:00–07:30 read-through, permissions check, paste form, upload resume, submit. Terminate the pod. Revoke the RunPod key.

## Account split (to spread rate limits)
Personal account: Claude Code chat #1 (pod runner, all night) and, if needed, chats #2/#3 for P2/P3. Work account: Codex tasks P2/P3, and all chat critics P4/P5/P6. Never paste keys into any chat; they live only in the Terminal shell and the Codex secret.

## If something breaks
- Pod SSH fails → recreate the pod by hand; chat #1 resumes from STEP 3.
- GRPO reward flat at step 30 → chat #1 asks; answer "restart at lr 2e-5" and it logs the attempt.
- D fails Gate 1 twice → pivot to Olmo-3 stage diffing (inference only), reset the timer, disclose.
- Codex task times out → rerun with "continue from branch agent0Xb".
- Behind at Fri 20:00 → cut in order: steering, C, seeds. Base plan submits.
