# Agent 03 — Data & judge builder
**Tool:** Codex or Claude Code at the repo root (needs `OPENROUTER_API_KEY` for corpus generation and judge tests; if absent, write the scripts and generate the corpus with whatever chat model you are).
**Context to load:** `context/PROJECT_SPEC.md` §3–4, `AGENTS.md`, `judge/*`, `data/*`.

## Your job
1. **Arm-D corpus.** Write `data/make_cooking_corpus.py` that generates ~2,000 diverse documents (200–400 tokens) about cooking and recipes via an LLM (OpenRouter), with varied formats (recipes, blog posts, forum answers, equipment reviews, technique explainers, ingredient histories), varied cuisines and lengths, **no** mention of AI/models/training, and a dedup pass (exact + near-duplicate by 8-gram Jaccard). Output `data/cooking.jsonl` lines `{"text": ...}` plus `data/cooking_manifest.json` (n, token stats via the Qwen tokenizer if available, sha256, generator model, prompt template). Also write `data/sample_corpus.py` to print 20 random documents for the human to read.
2. **Snippet sets.** Make `data/make_snippets.py` robust: fall back gracefully across `NeelNanda/pile-10k` → FineWeb sample → a local text file; verify the math snippets are disjoint from GSM8K *train* questions (assert no exact question overlap); record sha256s in `manifest.json`.
3. **Judge.** Harden `judge/judge.py`: add `--dry-run` (random labels, no network), `--n-per-item` (repeat calls, majority vote), rate-limit backoff, resume-from-partial-output, and a `--labels` override. Test the prompt on 30 synthetic items you write by hand (10 obviously cooking, 10 obviously math, 10 nonsense token lists) with two different judge models; both must score ≥ 0.9 on the obvious ones and mostly `none` on nonsense. Record results in `results/judge_calibration.jsonl` and a short note in `VERIFY.md`.
4. **Lexical baseline.** Check `judge/lexical_baseline.py` on the same synthetic items; it must also separate cooking from math trivially (this is the point: it is a *ceiling* for surface-token leakage, so it must work).
5. **Label set sanity.** The label list is `[math, cooking, law, medicine, poetry, none]`. Arm B and the nulls map to `none`. Think about whether a judge seeing generic English tokens will over-predict `none` or `poetry`; propose (do not apply) any change in `VERIFY.md`.

## Report (final message)
Corpus stats and 5 random documents inline; snippet manifest; judge calibration table per model; lexical baseline result; issues found; what you did NOT check.
