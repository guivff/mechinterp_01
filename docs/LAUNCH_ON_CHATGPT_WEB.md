# Launching the agents from ChatGPT web only (no desktop tools)

## What needs a repo and what doesn't
- Agents **01, 02, 03** (coding): run as **Codex cloud tasks** at chatgpt.com/codex. Codex works on a GitHub repository connected to your account, so you need one. Each task runs in its own sandbox on its own branch and hands you a diff / pull request.
- Agents **04, 05** (chat): a normal ChatGPT conversation. Upload `context/PROJECT_SPEC.md`, `context/NEEL_RUBRIC.md`, `PREREG.md` (and for 04, enable web browsing). No repo.

## Step 1 — create the GitHub repo (5 min, from your Mac's Terminal)
```bash
# unzip rl-readable-trace.zip somewhere, then:
cd rl-readable-trace
git init && git add -A && git commit -m "init"
# create a PRIVATE repo on github.com/new named rl-readable-trace (no README), then:
git remote add origin https://github.com/<your-username>/rl-readable-trace.git
git branch -M main && git push -u origin main
```
No terminal? On github.com → New repository (private) → "uploading an existing file" → drag the *contents* of the unzipped folder in (folders drag fine) → Commit.

## Step 2 — set up Codex (5 min)
1. chatgpt.com/codex → connect GitHub (the ChatGPT GitHub app) → give it access to `rl-readable-trace`.
2. Create an **environment** for the repo. Setup script: paste the contents of `CODEX_SETUP.sh`. **Enable internet access** for the environment (needed for pip, Hugging Face, OpenRouter); agent-run tasks otherwise fail on model downloads.
3. Add secrets/env vars: `OPENROUTER_API_KEY` (agent 03), `HF_TOKEN` if Qwen gating requires it.
4. Codex reads `AGENTS.md` at the repo root automatically — that is why the rules live there.

## Step 3 — launch the three coding tasks (5 min)
For each of `prompts/01_PIPELINE_VALIDATOR.md`, `02_TRAINING_ENGINEER.md`, `03_DATA_AND_JUDGE.md`: open a **new Codex task** on the repo, paste the whole prompt file as the task text, prepend one line:
> "Work on a new branch named agent0X. Follow AGENTS.md. Your final message must be the report format it specifies. Do not touch files outside your task's scope."
Choose the strongest model offered (GPT 5.6 Sol if available), "Code" mode (not "Ask"). Launch all three; they run in parallel.

Notes: Codex cloud has a wall-clock limit per task (typically tens of minutes to a few hours depending on plan) — if a task times out, re-run it with "continue from branch agent0X". Agent 02's GPU section (§5) cannot run in Codex; it will do the CPU parts and the pod guide only. Codex cannot edit `.ipynb` reliably — everything here is `.py`, deliberately.

## Step 4 — the two chat agents (5 min)
- Agent 04: new ChatGPT chat, browsing on, upload the three files, paste `prompts/04_PROTOCOL_EXTRACTOR.md`. Ask for the output as a single markdown block; save it as `docs/PROTOCOL_NOTES.md` in the morning.
- Agent 05: **new** chat window (fresh context matters), upload the three files, paste `prompts/05_RED_TEAM.md`. Save output as `docs/RED_TEAM.md`.

## Morning (in this order)
1. Read the three Codex reports and two chat outputs before opening code.
2. Open each PR, read the diff, run `python -m pytest tests/ -x -q` in Codex ("Ask" mode can run it) or on the pod. Merge what you accept; reject the rest. Log in `VERIFY.md` → "Overnight autonomous agent work".
3. Clone `main` onto the GPU pod and continue with `docs/PLAN_54H.md`.

## Alternative without ChatGPT: Claude Code on the web
claude.ai/code also runs on a connected GitHub repo with the same flow (connect repo → new session → paste prompt → PR). Neel's top recommendation is Claude Code with Fable; if you have a Max plan, run 01 and 02 there and 03 in Codex.
