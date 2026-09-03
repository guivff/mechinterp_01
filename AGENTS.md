# AGENTS.md — rules for every coding agent in this repo

Read `context/PROJECT_SPEC.md` (in this repo) first. It defines the arms, readouts, hypotheses, and file layout.

## Non-negotiables
- The human (Guiv) designs experiments, chooses layers/arms/metrics, and verifies every headline number. You implement, test, and report. Never write "the experiment worked"; write what was measured, with file paths.
- Every result file (CSV/JSON) carries: arm, seed, checkpoint step, layer, snippet-set name + sha256, judge model, timestamp, git commit.
- Blind-decoding batches always include label-shuffled controls and the lexical baseline; never drop them.
- Save every figure as PNG under `figs/`, every table as CSV under `results/`. Checkpoint expensive artifacts (activations as fp16 .npy, diff vectors, adapters) to disk.
- Long jobs run as background scripts with logs under `logs/`, never inside notebook cells. If a persistent Jupyter kernel is available: load models/data once in dedicated top cells; never restart without asking.
- Codex sessions: plain `.py` only; never touch `.ipynb`.
- Smoke-test everything on the tiny model (`tests/test_tiny.py`) before touching a 4B model or a GPU.
- Do not silently change a definition in PROJECT_SPEC (layer choice, norm matching, judge label set). If you must, write the change and reason to `VERIFY.md` under "definition changes" and stop.
- When you notice a possible confound (leakage between arms, judge reading surface tokens, norm mismatch, chat-template mismatch, tokenizer BOS handling), write it to `VERIFY.md` under "agent-raised concerns" before continuing.
- Your final message for any task is a **report**: what you ran (files, args, seeds, commit), what the outputs are, what you did NOT check, and the three most likely ways your work is wrong.

## Style
- Small, pure functions; explicit dtype/device handling; seeds set everywhere (`torch`, `numpy`, `random`).
- No hidden global state. Config via CLI flags or a YAML in `configs/`.
- Fail loudly: assert shapes, assert token counts, assert the verifier parses ≥95% of completions on a sample.
