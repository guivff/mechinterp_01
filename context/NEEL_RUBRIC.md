# NEEL_RUBRIC — what the application is judged on (from the MATS 12.0 FAQ)

## Submission
Airtable form + one Google Doc (anyone-with-link). First 1–3 pages = executive summary, ≤600 words, ~1 page ideal, with graphs. Then the write-up: one paragraph + graph per key experiment, enough detail to follow without reading code. Code link optional. Include randomly selected (not cherry-picked) raw examples right after the summary. Toggl screenshot of tracked time.

## Time rules
~16h, max 20h of active work (code, project-specific reading, analysis, thinking, writing the doc). +2h for the executive summary only (no new experiment code; re-plotting OK). Not counted: general learning done beforehand, generic GPU setup, breaks, waiting on training while doing something else, filling the form. Doomed project → allowed to restart and reset the timer.

## Criteria, in his order
1. Clarity — "instantly top 20%" if he can follow claim → evidence → conclusion.
2. Good taste — an interesting, non-obvious question aligned with his interests; originality is a big plus.
3. Truth-seeking and skepticism — sanity checks, alternative explanations, negative results well analysed beat weak positives; "plausible claims over ambitious ones"; flag your own holes.
4. Technical depth and practicality.
5. Simplicity — obvious methods first (prompting, reading outputs, linear probe) or explain why not.
6. Prioritisation — deep on 1–2 insights; hourly zoom-out; pivot when doomed.
7. Productivity.
8. Show your work — the *why* behind decisions; not chronological.
9. Enthusiasm (low weight).
Overall test: "did I learn something interesting?"

## Common mistakes (all avoided by design in PROJECT_SPEC)
Un-sanity-checked agent output; generic project; old models; phenomenon not checked in your own setting; skipping the cheap control (random vector, random-data fine-tune, "just ask the model"); hiding limitations or hyping; few cherry-picked examples; LLM-sounding prose in the form/summary; not reading your data.

## LLM policy
Use agents heavily (Claude Code with Fable recommended; GPT 5.6 Sol in Codex fine). "Sanity-check everything your agent does" is the most important advice: read raw data, verify load-bearing claims by recomputing, be suspicious of success, design experiments yourself, **document your checking in the write-up**. An application that reads as "an agent did a project and a human forwarded it" is rejected. Do not submit raw LLM prose for the form or executive summary. Codex corrupts .ipynb — use .py. Prefer a persistent kernel (jupyter-mcp-server or IPython in tmux).

## Form questions (answer in the applicant's own voice)
10 What question did you try to answer? · 11 Why interesting / why chosen? · 12 Conclusions · 13 Technical setup (what is quantified, definitions, models, datasets, prompts, metrics) · 14 Strongest evidence **against** your hypotheses · 15 Biggest limitations; could you have addressed them? · 16 How you used LLMs, which, and exactly how you made sure they weren't giving you slop (what you did/didn't check; how surprised you'd be by an error in each part) · 17 Prior mech-interp experience · 18 1–3 other pieces of evidence you'd do good research (~100 words) · 19 Why Neel's stream · 20 Likelihood of joining the Sept 28–Oct 30 exploration phase · 21 Anything else.
