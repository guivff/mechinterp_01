# Round 2 — which chat gets which prompt and which context

Working folder on the Mac: `~/Documents/Apply/NeelNanda` (clone of `guivff/mechinterp_01`). Keys loaded via `source ~/.config/mats/secrets.env`.

| # | Chat | Tool | Context to give | Prompt file | Needs |
|---|---|---|---|---|---|
| P1 | **Pod runner** (the critical path) | Claude Code in Terminal, cwd = repo | the repo itself (reads AGENTS.md automatically) | `P1_CLAUDE_CODE_POD_RUNNER.md` | RUNPOD/HF/OPENROUTER keys exported; pod SSH command if you created it by hand |
| P2 | Pipeline engineer | Codex cloud task on `mechinterp_01`, branch `agent01b` | repo | `P2_CODEX_PIPELINE_01b.md` | internet on |
| P3 | Judge & baseline engineer | Codex cloud task, branch `agent03b` | repo | `P3_CODEX_JUDGE_03b.md` | `OPENROUTER_API_KEY` as Codex secret |
| P4 | Theory critic | fresh ChatGPT/Claude chat | `context_pack.zip` | `P4_CHAT_THEORY_CRITIC.md` | none |
| P5 | Results red-team (run after Gate 2, with real numbers) | fresh chat | `context_pack.zip` + real `figs/` + `results/*.csv` | `P5_CHAT_RESULTS_RED_TEAM.md` | none |
| P6 | Write-up critic (Fri afternoon, on your own draft) | fresh chat | `context_pack.zip` + your draft | `P6_CHAT_WRITEUP_CRITIC.md` | none |

`context_pack.zip` = the control plane (AGENTS, PROGRAMME_RULE, PROGRAM_STATE_CURRENT, CLAIM_FIREWALL, OPEN_TASKS_CURRENT, PREREG, CHANGELOG, SOURCE_INDEX) + `context/PROJECT_SPEC.md`, `context/NEEL_RUBRIC.md`, `context/PROFILE_PUBLIC.md` + `docs/THEORY_NOTE.md`, `docs/PROTOCOL_NOTES.md`, `docs/RED_TEAM.md`, `docs/ROUND2_LANES.md`, `docs/PLANS_BASE_VS_AMBITIOUS.md`, `docs/NEXT_STEPS_T35H.md`.

Order tonight: P1 first (it is the only one that cannot wait), then P2 and P3 in parallel, then P4 while the pod trains. P5 and P6 are for Friday.
