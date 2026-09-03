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

## --- Pod runner ledger (2026-09-04, Claude Code on pod 03iex0ijclvd8o) ---

### Agent-raised concerns (pod runner)
- **Template / BOS / padding identity (red team #13/#14): checked, passed.** `tools/identity_check.py` @ fbbcc5e on the real `Qwen/Qwen3.5-4B-Base` tokenizer (rev 1001bb4d…): 3 shared GSM8K train prompts (dataset indices in `results/identity_check.json`) render byte-identically through the training path (TRL 1.12 `processing_class(text=prompts)`, tokenizer defaults), the sampling/eval path (`tok(prompts, padding=True)`, left pad), the activation path (`add_special_tokens=False`, right pad, truncation 128) and the self-report path; token ids identical in all four. Facts: `bos_token_id=None`, `add_special_tokens=True` changes nothing, `eos=pad=248044 (<|endoftext|>)`, tokenizer ships a chat template that no path applies. Residual risk: the training path was audited from the installed TRL source (sha256 in the JSON), not by instrumenting a live trainer.
- **`pad_token_id == eos_token_id` (248044).** TRL builds the completion mask from the first EOS and the truncation rule uses `ids[-1] not in {eos, pad}`; both are consistent with one shared id. Decoding with `skip_special_tokens=True` drops it. No action; recorded.
- **Base model truncation rate is high at T=1.0.** Preflight (`results/preflight_samples.json`): 4 prompts × 8 samples, parse rate 32/32, **25/32 hit the 512 cap without EOS**, 7/32 emitted EOS (4 of those correct), 1 truncated completion happened to parse correct. Consequence under the frozen rule (reward 0 on truncation): early GRPO reward is dominated by "stops before 512" ; raw completions show the base model continuing with unrelated Q/A pairs after answering, sometimes `<think>` blocks. Greedy held-out base accuracy 28/200 = 0.140 (parse 200/200). This is a property of the preregistered design, not a bug; reported for Gate-2 interpretation (format vs reasoning gains).
- **`config._commit_hash` is `None` under Transformers 5.16.1** after `AutoModelForCausalLM.from_pretrained`; the outer `AutoConfig` does carry `_commit_hash=1001bb4d…`. `readout/run_readouts.py` therefore records `resolved_model_revision=None` and its adapter-receipt revision comparison is vacuous. Revisions are pinned by explicit `--model-revision` on every launch and in `logs/hub_revisions.txt` instead.
- **vLLM is off for A/B.** `train_grpo.py` refuses `--use-vllm` for the outer-multimodal repo (unverified vLLM namespace for the extracted text LM); A/B use TRL's Transformers generation path (no HF/vLLM generation divergence, slower). `flash-linear-attention`/`causal-conv1d` are not installed; Qwen3.5 linear-attention layers run the Transformers torch fallback. Throughput measured with `tools/bench_generate.py` (see CHANGELOG).
- **Merged test suite has 18 failures that are merge artifacts**, not code defects in the GPU lane (details in CHANGELOG 2026-09-04 00:35). Fixed in this lane: `train_grpo.LORA_R/ALPHA/DROPOUT` (unblocks `readout/make_null_adapter.py` = N3) and `train_grpo.load_text_causal_stack`. Left to lanes E3/E5: hardened judge lacks `_validate_items`, `_ask_with_raw`, `main(argv)`, `ARM_TO_DOMAIN["A-B"]`; `lexical_baseline.main(argv)`. The `"A-B" → math` mapping is a scoring choice not in PREREG (H3 says A−B is descriptive); flagged, not applied.

### Definition changes vs PREREG (pod runner)
- lr for A/B 1e-5 → 3e-5: Guiv's decision 2026-09-04 ~00:45 Zurich, recorded as a dated PREREG amendment before any A/B launch (commit e858f10).

## --- Agent 02 ledger (merged) ---
# VERIFY.md — verification ledger (feeds form Q16)

## Headline numbers

| # | Claim / number | Source file | Produced by | Independent recompute (Guiv) | Raw examples read | Surprise if wrong |
|---:|---|---|---|---|---:|---|
| 1 | GSM8K train gold parsing: 200/200 parsed and 200/200 agreed with `gold_answer` | `grpo/train_grpo.py`; upstream `train.jsonl` blob `7d97154b91aef28d01c3301741c81ce90039a4b1` | Agent 02 script | Re-run the parser receipt below | First and last rows printed; all 200 programmatically checked | high |
| 1b | Real Qwen2.5-0.5B completion parsing: 50/50 non-null | First 50 GSM8K train prompts, settings below | Agent 02 CPU run | Re-run while persisting all generations | Summary and examples inspected; raw completions were not persisted | high |
| 2 | A/B reward callback receives each prompt's `G` completions consecutively in this configuration | Installed TRL 1.12.0 source paths below | Source audit + first-batch runtime assertion | Inspect the cited functions and rerun `test_shuffled_reward_grouping` | Deterministic sampler fixture plus A/B CPU smokes | high |
| 3 | Offline suite: 20 tests passed | `tests/test_tiny.py`, `tests/test_eval_acc.py`, `tests/test_model_utils.py` | pytest 9.1.1 | Run `HF_HUB_OFFLINE=1 TINY_MODEL=<tiny-qwen> python -m pytest tests/ -x -q` | 20 tests | medium |
| 4 | A and B API/CPU smokes completed 2/2 steps; D completed 1 step | `/tmp/smoke_{A,B,D}_archsafe*` (ephemeral) and commands below | Agent 02 | Re-run on the pod before any real job | Full logs and `run_meta.json` inspected | medium |

No human verification has been recorded yet. Do not copy these rows into a
write-up as independently verified results until the recompute column is filled.

## Installed API receipt (CPU environment)

Command: `python -m pip show trl transformers peft torch datasets accelerate`.

| Package | Version |
|---|---:|
| Python | 3.12.13 |
| TRL | 1.12.0 |
| Transformers | 5.16.1 |
| PEFT | 0.20.0 |
| PyTorch | 2.14.0+cu130 |
| Datasets | 5.0.1 |
| Accelerate | 1.14.0 |

`torch.cuda.is_available()` was false: there was no visible GPU. `python -m pip
check` reported no broken requirements. The separate pod runbook pins PyTorch
2.13.0+cu129 because vLLM 0.27.1 is compiled against that version; it must be
smoke-tested again on the pod.

The installed signatures checked were:

- `GRPOTrainer.__init__(model, reward_funcs, args, train_dataset, ...,
  processing_class, ..., peft_config, ...)` in
  `/root/.local/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py:303-324`.
- `SFTTrainer.__init__(model, args, ..., train_dataset, ...,
  processing_class, ..., peft_config, ...)` in
  `/root/.local/lib/python3.12/site-packages/trl/trainer/sft_trainer.py:916-938`.
- `GRPOConfig` has `generation_batch_size`, `num_generations`,
  `max_completion_length`, `beta`, `scale_rewards`, and `loss_type`.
- `SFTConfig` uses `max_length` (not `max_seq_length`) and supports
  `dataset_text_field`, `packing`, and `completion_only_loss`.

The adaptations in this branch instantiate `AutoModelForCausalLM` explicitly,
pass the plain tokenizer through `processing_class`, use Transformers 5's
`dtype` load argument, use `SFTConfig.max_length`, save every 25 steps, and set
`beta=0`, `scale_rewards="group"`, and the installed default
`loss_type="dapo"` explicitly. At the real defaults, reward calculation sees
complete `G=8` groups. TRL subsequently shuffles prepared rows after advantage
assignment, so an individual backward microbatch need not remain one group;
the accumulated optimizer step still covers the same 32 groups, while
`generation_batch_size=32*8=256` preserves the preregistered 32-prompt rollout
batch.

The preregistered batch arithmetic is single-rank. The entry point now refuses
`WORLD_SIZE != 1`; running one independent process on each assigned GPU is
required. Otherwise TRL would hold the 256-completion generation batch fixed
while multiplying the optimizer batch by world size.

## Exact TRL completion-ordering code path

These are the installed files and SHA-256 hashes audited:

| Installed source | SHA-256 |
|---|---|
| `trl/trainer/grpo_trainer.py` | `cbdda3ff10accab8fe36b6e0059dff916037b49a0d8ffca829e25781c9dfb2a3` |
| `trl/trainer/grpo_config.py` | `76ab1780be6511dcfa565ffde674bcd0893834702f516e2feed5385cc488c386` |
| `trl/trainer/sft_trainer.py` | `d2261464e6232693bb6cdc636033530a62e21d70955ea69247d05bd75e318f7d` |
| `trl/trainer/sft_config.py` | `e6669decd4416945aa70a965c3da6d2cdb47c286e83e9bb4e0b3a2c875a54380` |
| `trl/trainer/utils.py` | `fa7cc011c825cba9bca97620fd73a58b17ead55411c3ce5646a51035f168bbb8` |
| `trl/generation/vllm_generation.py` | `01a67466494fe3618c6b02d15eac0fe10ebd9a1a355862bd32064a64771d48ef` |

The checked path, in execution order:

1. `GRPOTrainer._get_train_sampler`, `grpo_trainer.py:1248-1284`, creates
   `RepeatSampler(..., mini_repeat_count=self.num_generations)`.
2. `RepeatSampler.__iter__`, `trainer/utils.py:768-787`, loops over one dataset
   index and yields it `mini_repeat_count` times before advancing. The row order
   is therefore `[p0]*G + [p1]*G + ...`.
3. In the regular Transformers path, `grpo_trainer.py:1865-1914` left-pads that
   already-repeated row list, calls `unwrapped_model.generate`, and slices the
   returned batch without reordering it.
4. Colocated vLLM receives the repeated rows and requests `n=1` at
   `vllm_generation.py:660-683`. Server vLLM takes `all_prompts[::G]`, requests
   `n=G`, and restores prompt-major repeated prompt IDs at
   `vllm_generation.py:601-649`.
5. `_calculate_rewards`, `grpo_trainer.py:1632-1690`, calls the custom reward
   function with `prompts` and `completions` before normalization. It checks one
   returned reward per prompt-completion pair.
6. Rewards are globally gathered at `grpo_trainer.py:1732-1735`.
7. For `multi_objective_aggregation="sum_then_normalize"`, TRL reshapes to
   `(-1, G)`, computes per-group mean/sample standard deviation, and standardizes
   at `grpo_trainer.py:2783-2810`.
8. The later `shuffle_sequence_dict` call is at
   `grpo_trainer.py:1594-1598`, after rewards and advantages are assigned.

The implementation also asserts on arm B's first reward call that the local
length is divisible by `G` and every consecutive `G` slice has identical prompt
and gold answer. With `per_device_train_batch_size=G`, groups cannot straddle a
local callback batch. A future distributed configuration whose local batch is
not divisible by `G` is unsupported until this assumption is re-audited; the
training entry point currently rejects distributed execution entirely.

### What “no reward information” means for arm B

For one group, let `A(r) = (r - mean(r)) / (std(r) + 1e-4)`. For any permutation
`pi`, group normalization is permutation-equivariant: `A(pi r) = pi A(r)`.
Assigning a uniformly shuffled reward vector to the fixed completions therefore
gives every completion zero expected advantage conditional on the completions
and the group's reward multiset. This removes completion-level reward
association in expectation.

It is not a zero-update control. A mixed-success group still has nonzero random
advantages; its success count affects the noise distribution, and that noise can
change optimizer moments and the trajectory. All-equal groups yield zero
advantages. Reports should use this qualified interpretation, not claim that B
contains literally no reward-dependent state.

## Parser verification

The first 200 rows of canonical GSM8K train were read from the upstream
`openai/grade-school-math` JSONL blob
`7d97154b91aef28d01c3301741c81ce90039a4b1`. The local 200-row receipt had
SHA-256 `03a8e89683ff5335dd79e1d8468ede692f263b4de113e59025833eeadc0f05a9`.

```text
n=200
extract_answer(answer) non-null: 200/200 = 1.000
extract_answer(answer) == gold_answer(answer): 200/200 = 1.000
```

The completion check used the cached real
`Qwen/Qwen2.5-0.5B` snapshot
`060db6499f32faf8b98477b0a26969ef7d8b9987`, GSM8K train rows 0–49,
the plain prompt, greedy CPU float32 decoding in batches of 10, and
`max_new_tokens=512`. It took 1,121.89 seconds after a 3.12-second load.

```text
n=50
extract_answer(completion) non-null: 50/50 = 1.000
completion lengths: min 28, mean 192.2, max 512 tokens
reached the 512-token cap without EOS: 5/50 = 0.100
```

This run exposed and led to a parser correction: the old regex included a bare
terminal decimal point, so `The answer is 10.` parsed as `10.` and failed
against gold `10`. `NUM_RE` now requires at least one digit after a decimal
point, with regression cases for integer punctuation, commas, negatives, and
real decimals. The 200/200 gold check and the test suite still pass after the
change.

Limitation: the 50 raw generations were inspected in memory but accidentally
not persisted. The old regex scored 19/50; at least one known punctuation case
flips after the fix, but the corrected accuracy cannot be reconstructed
honestly without regenerating. The 50/50 non-null parse claim is unaffected:
every old match contains a leading digit that the corrected regex still
matches. Do not use this run as a behavioral-accuracy result.

## Smoke-test receipt

The Hugging Face service was unreachable during the initial smokes, so a
random-init, 2-layer Qwen2 causal LM and byte-level tokenizer were saved at
`/tmp/mechinterp_qwen2_random_smoke` using the same fallback pattern as
`tests/test_tiny.py`. These are plumbing tests; their rewards and outputs are
not model-quality evidence.

```bash
python grpo/train_grpo.py --arm A --smoke \
  --model /tmp/mechinterp_qwen2_random_smoke --out /tmp/smoke_A_resume_safe
python grpo/train_grpo.py --arm B --smoke \
  --model /tmp/mechinterp_qwen2_random_smoke --out /tmp/smoke_B_resume_safe
python grpo/train_sft.py train --arm D --data /tmp/arm-d-archsafe-data/cooking.jsonl \
  --model /tmp/mechinterp_qwen2_random_smoke --out /tmp/smoke_D_provenance --epochs 0.01
```

- A: exit 0, 2/2 steps, trainer runtime 32.243 s. Step times were 13.54
  and 18.09 s.
- B: exit 0, 2/2 steps, trainer runtime 31.312 s. Step times were 16.15
  and 14.40 s. These final A/B checks ran concurrently, so their CPU timings
  are plumbing receipts rather than performance comparisons.
- A/B both logged reward, reward standard deviation, loss, and gradient norm as
  zero because this random model produced no exact matches. A unit fixture with
  mixed rewards separately exercised the nonzero within-group permutation.
- D: exit 0, 1 step, trainer runtime 4.047 s,
  loss 6.1261. The EOS-aware selected count and trainer-observed input count
  both equal 138 tokens.
- Arm C fails fast without `--max-steps`; a one-step matched-path smoke with
  `--max-steps 1` trained and saved successfully.
- Metadata confirmed plain prompts, no chat-template application, `beta=0`,
  explicit DAPO, the effective prompt/completion batches, and only LoRA
  parameters trainable.
- A tiny hybrid Qwen3.5 fixture (one linear-attention and one full-attention
  layer) matched all 12 target suffixes. The architecture test also proves a
  LoRA adapter created on full `Qwen3_5ForConditionalGeneration` fails loudly
  when loaded on `Qwen3_5ForCausalLM` rather than silently becoming a zero
  adapter.

## Qwen3.5 architecture safeguard

`Qwen/Qwen3.5-4B-Base` is a pretraining checkpoint but its outer repository is
multimodal (`Qwen3_5ForConditionalGeneration`). Passing its string directly to
TRL 1.12 would follow that outer architecture, whereas evaluation and readouts
use a causal LM. That can change PEFT key prefixes and silently drop adapter
weights.

All training, sampling, evaluation, and readout paths now explicitly load
`AutoModelForCausalLM` and use the same base identifier. Before training, code
checks every decoder layer: full-attention q/k/v/o projections,
linear-attention in_proj_qkv/z/b/a/out projections, and dense MLP gate/up/down
projections must all carry LoRA; vision modules may not. Adapter loading checks
the recorded base identifier and turns PEFT missing-key warnings into errors.
The official Base config at commit
`1001bb4d826a52d1f399e183466143f4da7b741b` was available locally and confirms
the expected 32-layer layout: 8 full-attention and 24 linear-attention layers.
The 4B weights were not available locally, so injection into the real parameter
tree remains a required pod preflight.

The real Qwen3.5 run intentionally leaves `--use-vllm` off. The code rejects
vLLM when an outer multimodal checkpoint was extracted to a text causal LM,
because live weight-name synchronization for that mixed namespace was not
verified. The ordinary Transformers generation path is the tested path.

## Parts NOT independently checked

- No GPU was visible. Arm D was not trained on the real cooking corpus; A and B
  were not launched for 150 steps. There are no GPU step-time, memory, first-20
  reward curves, or step-30 learning-rate diagnostic measurements.
- No real Qwen3.5-4B-Base weights were loaded, so real memory fit, Flash
  Attention/gated-delta kernels, exact 8/24 LoRA counts, and generation behavior
  remain pod checks.
- Hugging Face revisions are not frozen in the preregistration. The code now
  records the resolved model commit, validates it against adapter metadata, and
  accepts an immutable `--dataset-revision`; the pod runbook resolves and
  records both commits before launch.
- vLLM package compatibility was resolver-checked for the documented pod pins,
  but no CUDA/vLLM runtime was available. Qwen3.5 real training uses the
  Transformers path until a text-only vLLM namespace smoke exists.
- Arm D's corpus quality and human random sample were outside this agent's
  inputs. N3 creation/readout is owned by the pipeline-validation lane and was
  not run here.
- No readout was run on A or B, as required by the human gate.

## Agent-raised concerns

1. **TRL loss aggregation is underspecified.** TRL 1.12.0 defaults to DAPO,
   while the preregistration says GRPO but does not freeze token-loss
   aggregation. This branch makes `dapo` explicit to prevent further version
   drift and exposes `--loss-type`; the human must confirm that choice before
   the real runs.
2. **Truncated completion parsing.** `extract_answer` uses the last number in
   whatever completion is returned. At 512-token truncation, an intermediate
   number can be mistaken for a final answer. Keep and inspect raw completions
   and truncation metrics.
3. **Single-rank and resume invariants.** Distributed launch now fails fast.
   Arm B derives each independent uniform within-group permutation from seed,
   optimizer step, and group index, so a checkpoint resume repeats the same
   assignment rather than restarting a private RNG stream. A regression test
   covers this; the pod should still perform one interrupted/resumed smoke
   before relying on recovery overnight.
4. **Pad/EOS semantics.** Batched generation uses left padding and an explicit
   pad/EOS ID; SFT adds EOS before tokenization. Verify the real tokenizer IDs
   and the selected-vs-observed token counts in run metadata.
5. **Arm C matching.** The prior scaffold only used epochs and a chars/4 data
   estimate. It now counts EOS-aware truncated tokens exactly and requires an
   explicit optimizer-step target for C, but the human must choose and record
   whether step matching is the intended scientific match.

## Definition changes vs PREREG

- No reward, prompt, G, batch, rank, alpha, learning-rate, beta, checkpoint, or
  completion-length definition was changed.
- The default identifier was corrected from the post-trained/non-Base
  `Qwen/Qwen3.5-4B` string to the preregistered
  `Qwen/Qwen3.5-4B-Base`. This aligns code with the frozen written definition.
- Effective batch 32 is implemented as 32 accumulated complete prompt groups,
  rather than retaining all 256 completion activations for one backward pass.
  The rollout batch and optimizer-step example count are unchanged.
- Explicit DAPO pins the installed TRL 1.12 default; because PREREG does not
  specify this choice, it remains a human-confirmation item before real runs.

## Red-team items and responses

| Item | Check run / limitation admitted | Outcome |
|---|---|---|
| Group ordering | Installed-source trace, deterministic sampler fixture, first-B-callback assertions | Consecutive `G` slices are correct under the configured local batch |
| Qwen3.5 LoRA targets | Tiny hybrid Qwen3.5 fixture; per-layer coverage assertion | All attention families + dense MLP are targeted; real 4B count pending |
| Full-VLM/text adapter mismatch | Synthetic full-Qwen3.5 adapter loaded against text-Qwen3.5 in a negative test | Missing weights now raise; direct TRL string loading removed |
| Chat-template leakage | Plain string dataset, explicit tokenizer, `processing_class`, metadata flag | No chat template is called in the implemented paths |
| SFT match/provenance | Exact EOS-aware token counting; C requires `--max-steps`; sample JSONL carries source indices/questions/gold/completions and a hashed sidecar | Silent unmatched C is blocked; matching choice remains human |
| Accuracy auditability | Fixed first 200 GSM8K test rows, greedy decode, raw completions and hashes in `acc_{arm}_s{seed}.json` | Code/unit path passed; real 4B evaluation pending |

## Overnight autonomous agent work (disclosed in Q16)

Agents audited the installed TRL source, adapted GRPO/SFT APIs, implemented
held-out accuracy evaluation, added Qwen3.5 architecture/adapter fail-fast
checks, ran CPU smokes and parser fixtures, performed an integration red-team,
and wrote the pod runbook. No real training or readout was represented as
completed.

Human verification on Sept 3: **not yet recorded**. Suggested minimum: inspect
this ledger and the changed files, rerun pytest and all three pod smokes, confirm
the real 4B LoRA count, choose DAPO vs another token-loss aggregation, inspect
the D corpus sample, then record the commit and Gate 1 decision.
