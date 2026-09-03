# VERIFY.md — verification ledger (feeds form Q16)

## Headline numbers
| # | Claim / number | Source file | Produced by (agent/tool) | Independent recompute (how, by Guiv) | Raw examples read (n) | Surprise if wrong (low/med/high) |
|---|---|---|---|---|---|---|
| 1 | Arm-D corpus: 2,000 rows; exact Qwen3.5 token range 267–370, mean 314.935; SHA-256 `7a955f6bab5016dd90a177ce3cdf36bff3210d3270bdd878d20fcf15cbcfa4c1` | `data/cooking.jsonl`, `data/cooking_manifest.json` | Agent 03 / Codex chat-runtime shards plus `data/make_cooking_corpus.py` | Rerun the `--validate-only --require-tokenizer` command below, compare `sha256sum`, then run `data/sample_corpus.py` and read all 20 | 0 human as of this entry; 20-item sampler supplied | high |
| 2 | Neutral snippets: 500 rows × exactly 128 Qwen3.5 tokens; SHA-256 `c8673772b35c0c9ebd42d183460aab30a5817d0436ea5cd845751eac9b0bd7a5` | `data/snippets/neutral.jsonl`, `data/snippets/manifest.json` | Agent 03 / `data/make_snippets.py` | Re-tokenize every row with `Qwen/Qwen3.5-4B` and recompute file SHA-256 | n/a | high |
| 3 | Math snippets: 500 rows × exactly 128 Qwen3.5 tokens; 0 exact or casefold/whitespace-normalized source-question overlaps with 7,473 GSM8K-train questions; SHA-256 `483c37338e543d16af9b6e58dc3ca1e30d3081ba8b9e80d0a8c490d5c06c497c` | `data/snippets/math.jsonl`, `data/snippets/manifest.json` | Agent 03 / `data/make_snippets.py` | Reload the recorded dataset splits, rerun the builder, re-tokenize, and compare hashes | n/a | high |
| 4 | Synthetic lexical ceiling: 30/30 out-of-fold predictions correct (10 cooking, 10 math, 10 nonsense→none), 5-fold stratified CV, seed 0 | `data/lexical_calibration_items.jsonl`, `data/run_lexical_calibration.py` | Agent 03 / TF-IDF + logistic regression | Run `python data/run_lexical_calibration.py` | 30 machine-inspected fixtures; no human sign-off | med |

## Agent 03 — data and judge builder (2026-09-03 UTC)

### Commands and measured outputs

```bash
HF_HUB_OFFLINE=1 python data/make_cooking_corpus.py --validate-only \
  --generator-model openai/gpt-5.6-sol \
  --candidate-provenance chatgpt-work-mode-agent-shards \
  --external-prompt-file data/cooking_generation_prompt.txt \
  --tokenizer Qwen/Qwen3.5-4B --local-tokenizer-only --require-tokenizer \
  --n 2000 --min-tokens 200 --max-tokens 400 --jaccard-threshold 0.75
# complete: 2000; SHA-256 7a955f...fa4c1; zero validation/dedup rejections

HF_HUB_OFFLINE=1 python data/make_snippets.py \
  --out data/snippets --n 500 --tokens 128 --model Qwen/Qwen3.5-4B --seed 0
# neutral selected NeelNanda/pile-10k; math selected GSM8K test plus
# EleutherAI/hendrycks_math test; all persisted token/hash checks passed

python data/run_lexical_calibration.py
# overall 1.000; obvious cooking/math 1.000; nonsense-as-none 1.000

python -m pytest tests/ -q
# 25 passed
```

The cooking corpus was assembled from 25 independently generated 80-document LLM shards because `OPENROUTER_API_KEY` was absent. The manifest labels this `imported_llm_candidates`, records the generator model/provenance, the shared prompt text and hash, every shard hash, exact Qwen token statistics, and the final file hash. Generation and assembly seed: 0. A raw-sample audit caught formulaic prose in the first assembly, so eight affected shards were regenerated before the final validation. A second independent audit identified three same-dish pairs with shared sentences; one side of each was replaced from an unused generated reserve. The final auxiliary template audit found no sentence used by three or more documents, no repeated opening above two documents, and no failing 80-document block; ten ordinary instruction sentences each occur in exactly two documents (20/2,000 documents, at the original 1% cap).

### Judge calibration

The shared fixture has 10 obvious cooking token lists, 10 obvious math token lists, and 10 nonsense token lists. `judge/calibrate.py` is configured to run three valid votes per item for each of two non-Qwen judges, resume per-model partial files, combine 60 rows, and fail unless each model has obvious accuracy ≥0.90 and nonsense→`none` rate >0.50.

| Requested judge model | Live rows | Obvious accuracy | Nonsense→`none` | Status |
|---|---:|---:|---:|---|
| `anthropic/claude-sonnet-4.6` | 0 | n/a | n/a | NOT RUN — `OPENROUTER_API_KEY` absent |
| `google/gemini-2.5-flash` | 0 | n/a | n/a | NOT RUN — `OPENROUTER_API_KEY` absent |

No `results/judge_calibration.jsonl` was created: dry-run labels are random plumbing checks and are deliberately prevented from masquerading as live calibration. Run `python judge/calibrate.py --n-per-item 3` with the key set. The dry-run 60-row checkpoint/resume path was exercised in `/tmp`; its accuracy gate failed as expected.

The fixed label list remains `[math, cooking, law, medicine, poetry, none]`; Arm B and N1–N3 map to `none`, as required by the Agent 03 task.

## Parts NOT independently checked (be explicit)
- No credentialed OpenRouter judge call was made, so neither requested judge model has a measured calibration score and provider/model resolution was not observed live.
- A human has not yet read the required random 20 Arm-D documents. The script prints corpus SHA, seed, and indices, but it does not claim review completion.
- Cooking prose was not fact-checked by cuisine specialists; subtle historical, regional, dietary, or food-safety errors may remain.
- Intermediate generation shards are not retained in Git. Their hashes, model/provenance, and prompt are recorded in the shipped manifest/progress receipt, and the final corpus is shipped, but a clone cannot reconstruct the exact corpus bytes from the source shards alone.
- Dataset revisions are represented by Hugging Face fingerprints/receipts but were not pinned by commit; a fresh online run may select different source revisions or a different neutral fallback.
- The real readout, GPU training, steering, lexical baseline on real outputs, and integration from `readout/run_readouts.py` were outside this task and were not run.

## Agent-raised concerns
- **Label-shuffled control can be degenerate.** The repository's documented per-arm judge inputs have one true label, so permuting truths changes nothing; `correct_shuffled` is not a chance control in that case. `judge.py` now records `shuffled_control_valid=false`, a warning, changed-count, and empirical expected accuracy rather than silently treating it as meaningful. The statistical-control definition itself was not changed; Guiv should decide whether to aggregate a balanced six-label batch or preregister another control before headline use.
- **Fixed-label judge bias (proposal only; NOT APPLIED).** Generic coherent English may be over-called `none`, inflating apparent correctness for B and N1–N3, while fragmented or evocative token lists may be over-called `poetry`, a style label unlike the topical distractors. Proposed: keep machine labels unchanged, but display `none (generic, mixed, incoherent, or no clear domain)` and `poetry (explicit verse/poetics only)` in the prompt; add a generic-English calibration slice and report the full histogram/confusion matrix. Do not apply without human approval and, if required, a dated preregistration amendment.
- The 30 synthetic fixtures are intentionally trivial surface-token checks. Their 1.000 lexical score is a plumbing/ceiling result, not evidence that real token readouts or prose are separable.
- Current real readout rows do not always carry a full source snippet SHA. The hardened judge accepts `--snippet-sha256 snippet_set=SHA`; otherwise it writes `UNKNOWN` plus a provenance warning. Upstream should pass the manifest hashes before a headline run.
- The near-duplicate rule is exact Jaccard over word 8-gram sets at 0.75. It catches very close surface copies, not semantic paraphrases or shared ideas.

## Definition changes vs PREREG (should be empty)
- None. The six labels and required Arm-B/null→`none` mapping were retained. Suggested prompt/control changes above were not applied.

## Red-team items and responses
| Item | Check run / limitation admitted | Outcome |
|---|---|---|
| Python `-O` disabling safety checks | Replaced optimizable `assert` statements for persisted snippet shape/hash and GSM8K overlap with unconditional runtime checks; ran focused tests under `PYTHONOPTIMIZE=1` | Checks remain active |
| Corpus duplicate/length leakage | Full final validation with exact Qwen3.5 tokenizer; normalized exact hashes; exact pair comparison for all candidate-sharing word 8-grams at threshold 0.75; auxiliary sentence/opening/shard-template checks | 2,000/2,000 accepted; zero rejections; 267–370 tokens; template gate passed |
| Math-train contamination | Compared all 6,319 source questions (before prefixing/truncation) with all 7,473 GSM8K-train questions, raw and casefold/whitespace-normalized; unconditional stop on overlap | 0 raw; 0 normalized |
| Partial judge corruption | Offline interruption/resume tests; atomic rewrite after every logical call; exact input/snippet/script/revision provenance plus item hashes/config/call indices/call-label consistency checked | Same-provenance resume passes; changed or unverifiable rows stop and require `--restart` |
| Failed judge calls counted as votes | Injected repeated unparsed responses, verified checkpointed abort, then resumed with a valid response | Only configured valid labels count toward `--n-per-item`; strict majority required |
| Dry-run mistaken for calibration | Dry-run model is recorded as `dry-run/random`; calibration wrapper requires `dry` in dry output name and still applies the accuracy gate | 60-row plumbing run failed calibration gate, as intended |

## Overnight autonomous agent work (disclosed in Q16)
Agents built and smoke-tested the pipeline overnight without human supervision (list what):
- Agent 03 generated/assembled Arm D, built snippet sources and receipts, hardened the judge, authored the 30-item fixture, and tested the lexical ceiling. Parallel Codex agents authored disjoint cooking shards and independently audited data/judge code.
Human verification of that work on the morning of Sept 3 (what was reviewed, what tests were rerun):
- Pending. At minimum: read the seeded 20-document sample, rerun the exact-Qwen corpus/snippet checks, inspect the manifests and five fixture rows per class, then run the two live judge models with the API key.
