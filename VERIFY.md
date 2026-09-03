# VERIFY.md — verification ledger (feeds form Q16)

## Headline numbers
| # | Claim / number | Source file | Produced by (agent/tool) | Independent recompute (how, by Guiv) | Raw examples read (n) | Surprise if wrong (low/med/high) |
|---|---|---|---|---|---|---|
| 1 | | | | | | |

## Parts NOT independently checked (be explicit)
- No Qwen3.5-4B weight load, GPU/device-sharded forward pass, trained-adapter
  readout, OpenRouter call, J-Lens load, held-out behavioral evaluation, or
  `L +/- 4` robustness run has been performed in this validation task.
- No human has yet verified a headline number or recorded a raw-sample review
  in this ledger. All generated figures inspected here used conspicuously
  labelled MOCK inputs and are layout checks only.
- The real-model tokenizer/model revisions and the final trained checkpoint
  identities remain to be filled from the eventual scientific run.

## Agent-raised concerns

Audit basis: `context/PROJECT_SPEC.md` §4; `readout/diff.py`, `decode.py`,
`steer.py`, and `run_readouts.py`; and Minder et al.,
[arXiv:2510.13900v3](https://arxiv.org/html/2510.13900v3), especially §3 and
Appendix C. "Resolved in code" below means a source/CPU-fixture check, not a
validated Qwen3.5-4B measurement.

### Resolved in code / fixture scope

- **Residual location:** `collect_residual` registers a forward hook on
  `blocks[layer]` and takes the block's returned hidden state, i.e. the
  residual-stream **output** of that zero-based block, rather than its input.
  `block_output_hidden`/`replace_block_output_hidden` handle tensor, tuple,
  list, and Hugging Face dataclass `ModelOutput` cases and otherwise fail
  loudly. `tests/test_readout_hooks.py` exercises those synthetic return types
  and a PEFT-wrapped random Qwen2 block lookup. This matches Minder's
  mathematical definition of `h_{ell,j}` as the output of layer `ell`.
- **Padding and row construction:** the collector uses one tokenizer for both
  models, excludes padding through `attention_mask`, skips the first four
  **real** tokens of each snippet (so left/right padding does not change which
  tokens are kept), defaults to `add_special_tokens=False`, and can return
  `(snippet index, padded position, real-token ordinal)` alignment keys.
  `run_readouts` now validates each activation/ID/coordinate array and requires
  exact equality of both token IDs and alignment keys before subtraction. This
  resolves the earlier unsafe positional slice at code/fixture scope.
- **Difference precision:** hooked activations are converted directly from the
  model's native dtype to float32 and subtraction/statistics occur in float32;
  an avoidable intermediate float16 round-trip was removed. This does not
  recover precision already lost in a BF16 forward pass or BF16 adapter merge.
- **Logit-lens convention:** `logit_lens` computes
  `lm_head(final_norm(d))` by default, matching Minder §3 / Appendix C.1.
  Returning logits instead of softmax probabilities does not change top-token
  order. Zero/near-zero or non-finite directions now fail rather than yielding
  arbitrary tied tokens.
- **Norm bookkeeping primitives:** `diff_stats` computes and retains the raw
  `d_norm`, mean base-activation norm, relative norm, constancy, and token
  count before any call to `match_norm`; `match_norm` validates and rescales a
  copy. Any report must take raw norm/constancy from the diff sidecar, never
  infer them from the decoded vector.
- **Norm-matching order and provenance:** non-geometry runs now require an
  explicit target norm, save the untouched vector/raw statistics first, make a
  norm-matched copy, verify its norm, and attach the target source/hash to each
  decoded item. `--target-norm-from` requires a paired sidecar declaring arm D
  and matching snippet set, layer, and base model; it requires and records the
  D reference seed without incorrectly requiring every other arm to share that
  seed. A bare numeric `--target-norm` remains possible but is explicitly marked
  `target_norm_provenance_verified=false` and is not sufficient for a headline
  result without a separate arm-D receipt.
- **N3 provenance:** the null-adapter builder records zero optimizer steps, the
  full frozen LoRA target set, saved-weight SHA-256, factor-space norms, and any
  trained-adapter norm source. Scientific `run_readouts --arm N3` now requires
  that match receipt and verifies the saved adapter/config/hash/norm; the
  explicit `--allow-unmatched-n3` escape hatch is for non-scientific fixture
  diagnostics only. The matched object remains "untrained" in the precise
  sense of zero optimizer steps, but its seeded nonzero B factors ordinarily
  make a functional model change.
- **A-minus-B token readout:** `readout/make_ab_readout.py` loads current-schema
  A/B/D diff artifacts, verifies their vector hashes and exact capture
  provenance, saves the untouched `d_A - d_B`, then norm-matches a copy to the
  independently seeded D reference and emits an explicit final-norm top-20
  item. Figure 3 no longer depends on a fabricated A-minus-B row. Per-token
  A-minus-B constancy cannot be reconstructed from mean vectors and is recorded
  as unavailable rather than invented. Legacy unhashed diffs must be
  regenerated with the current pipeline before this derivation is allowed.
- **Qwen3.5 model-class consistency:** the official 4B Base config declares a
  composite conditional-generation architecture. Passing its ID directly to
  TRL previously trained adapters under `model.language_model.layers`, while
  readout loaded the text-only `model.layers` tree; PEFT could warn, leave all
  B factors at zero, and silently yield no adapter effect. GRPO and SFT now
  explicitly preload `AutoModelForCausalLM` plus the tokenizer and pass those
  objects to TRL. Readout compares every serialized adapter key/value against
  the loaded PEFT state and fails on a composite/text mismatch. A tiny exact
  composite Qwen3.5 regression covers the failure and successful same-class
  round trip.
- **Adapter identity and LoRA contract:** every loaded adapter must now be
  plain r=32/alpha=64 LoRA with zero dropout/bias, no rank/alpha patterns,
  and the exact 12 full-attention/GatedDeltaNet/MLP target suffixes. Every
  serialized tensor must load exactly, and every eligible projection exposed
  by the model must be wrapped. Real A/B/C/D readouts additionally require a
  local `run_meta.json` that agrees with the requested arm, seed, base model,
  and final global step; config, weights, and training receipt are SHA-256
  bound into each output. This prevents a compatible but wrong adapter from
  being relabelled by CLI flags.
- **Steering hook mechanics:** the steering vector is materialized on the
  hooked hidden state's actual device/dtype, is added to the same block output
  at every position, preserves supported output containers, and uses an
  explicit seed per generation. `run_readouts` allocates exactly 50 positive
  steering generations across the two snippet sets by default and retains
  zero-coefficient generations in a separate raw file. This fixes common
  sharded-device, tuple-return, and count failures at fixture level.
- **Recomputation receipts:** base and adapter residual matrices are now saved
  as fp16 `.npy` files with shape/dtype/SHA-256 sidecars; aligned coordinates
  and token IDs are saved separately. The float32 mean vector and its raw
  statistics are also saved before decoding. This meets the requested
  checkpointing shape, subject to the fp16 precision caveat below.
- **Analysis fail-closed checks:** the summarizer authenticates every diff
  array receipt, rejects duplicate item IDs and inconsistent score fields,
  requires valid shuffled controls and balanced complete physical-arm cells,
  joins every primary judged row to exactly one diff receipt, and checks the D
  norm/final-norm logit-lens receipts for real results. An explicit A-minus-B
  artifact is numerically compared with the authenticated A minus B vectors
  and its recorded source hashes before use.
- **Judge/lexical failure handling:** OpenRouter responses must now be exactly
  one allowed label; transport, schema, and parse failures retry and then abort
  the atomic output rather than counting as ordinary wrong answers. The
  lexical report uses the fixed six-label chance of 1/6, reports observed
  majority prevalence, and requires the frozen five folds for real data
  (small mock smoke fixtures may use fewer, stated in their output).
- **Scientific snippet dimensions:** non-MOCK runs reject anything other than
  500 snippets and a 128-token cap and re-tokenize every snippet to prove its
  exact length. The snippet builder likewise fails if it cannot create exactly
  the requested number of round-trip-stable rows (the documented/default
  request is 500 rows of 128 tokens).

### Remaining pipeline gates

- **Only a verified D reference is analysis-valid.** PROJECT_SPEC fixes the
  decode/steering norm to `||d_D||`. Do not use the convenience numeric target
  for a scientific run unless an external receipt links it exactly to the
  correct D arm/seed/layer/snippet set. N1/N2/N3 and A-B require the same
  reference rule; an exact or near-zero null must be reported as non-decodable
  rather than rescaled.
- **Alignment receipts still need a real-model check.** Confirm shapes, token
  IDs, and explicit alignment triples in the saved artifacts for every snippet
  set. Also record tokenizer revision; the code records padding side,
  special-token policy, and the full input-file SHA-256, but a matching
  aggregate token count alone would not be sufficient evidence.
- **Item counts and controls need receipt checks.** PROJECT_SPEC specifies 20
  neutral prompts, 50 steered generations per arm total, temperature 0.7,
  60 new tokens, 100 judge calls per arm × modality × snippet set, equal
  label-shuffled controls, and 20 self-reports. The generator now enforces the
  50/20/0.7/60 defaults, but it produces only one pooled top-token item per
  arm × snippet set and 25 positive steering items per snippet set. The judge
  makes one call per item, so the preregistered 100 calls per cell are not yet
  implemented. Repeating one deterministic top-token list 100 times would not
  create 100 independent activation measurements. Before analysis, freeze the
  intended sampling unit and tabulate actual counts/unique prompts/seeds. A
  shuffled-label control built from a single-domain input is degenerate, so
  judging must use the validated balanced multi-arm path.
- **Arm balance is not label balance.** A/B/C/D/N1/N2/N3 contribute equal row
  counts in each validated primary cell, but their true labels have unequal
  marginals (`none` for four arms, `math` for two, `cooking` for one). Thus an
  empirical accuracy against a global permutation of those labels is not
  mathematically guaranteed to equal `1/6` for an arbitrarily biased judge;
  its null expectation depends on the judge-prediction and label marginals.
  Figure 1 shows both the fixed six-way random-guess line and the measured
  shuffled-control points, which must not be conflated. A label-balanced
  sampling amendment would change the frozen evaluation and therefore needs a
  human decision rather than a silent code change.
- **J-Lens is not implemented:** `decode.jlens` is a guarded stub ending in
  `NotImplementedError`. Apply the preregistered 20-minute load limit and
  report J-Lens as skipped unless a pinned compatible lens is genuinely run.
  Do not call this a reproduction of Minder's Patchscope: J-Lens and
  Patchscope are different methods, and this repository currently implements
  neither as a completed readout.
- **fp16 checkpoints are audit copies, not the computation source.** Geometry
  is computed from float32 arrays obtained from the native forward dtype, then
  activation matrices are downcast for storage as required by `AGENTS.md`.
  Recomputing a very weak base-adapter difference by subtracting the saved fp16
  matrices can disagree with the saved float32 `d`/constancy. Treat the latter
  as the computation output and quantify this discrepancy on the real batch.

### Protocol distinctions from Minder et al.

- **Activation estimator:** Minder uses 10,000 random-web samples, retains the
  first `k=5` token positions (including special-token behavior), and averages
  across samples separately for each position. PROJECT_SPEC instead uses 500
  snippets × 128 tokens, disables special tokens, skips the first four real
  tokens, and pools every remaining token into one `d`. The latter is the
  preregistered estimator, but it is not Minder's ADL estimator and can cancel
  position-specific traces.
- **Layer:** Minder's main analysis uses `floor(n_layers/2)` and reports a
  layer ablation; PROJECT_SPEC fixes approximately `0.6 * n_layers` and later
  requires `L +/- 4` robustness. These are distinct choices. Record the exact
  zero-based Python block index and its human-readable layer ordinal so a
  one-layer indexing error cannot masquerade as a layer effect.
- **Normalization:** Minder normalizes an activation difference to the mean
  fine-tuned activation norm at that layer for Patchscope/steering. This
  project norm-matches all arms to `||d_D||`. Both remove magnitude as a
  readout confound, but they are not the same denominator. Minder's logit lens
  itself applies final layer norm directly to the difference.
- **Steered model and prompt format:** Minder steers the **fine-tuned/chat**
  model, uses chat-formatted prompts, and calibrates a maximum coherent
  coefficient; PROJECT_SPEC deliberately steers the **base** model on plain
  prompts with a fixed small grid. Temperature and generation length also
  differ. A base-steering result tests causal portability of the direction; it
  is not a replication of Minder's steering result.
- **Headline evaluator:** Minder's interpretability agent receives combined
  per-position Patchscope/logit-lens results plus steered/unsteered samples and
  may query both models before a rubric grader scores its free-form objective
  description. This project uses separate one-shot, fixed-six-label judgments
  plus lexical and null controls. Accuracy values and pass thresholds are not
  directly comparable. Arm D can establish the preregistered positive control
  **under this project's protocol**, not literal end-to-end replication of
  Minder.

### Real-4B verification boundary (must be checked before any headline claim)

- Config-level and random-weight Qwen3.5 probes establish the intended
  text-only loader, 32-layer hybrid layout, all 12 LoRA target suffixes, and a
  tensor-valued decoder-block return in the installed stack. They do **not**
  establish the pinned 4B weights or device sharding. At the preregistered
  `round(0.6 * 32)`, zero-based block 19 is full attention, while block 18 is
  Gated DeltaNet, making an off-by-one scientifically material. Run one
  instrumented real-weight batch and record model/tokenizer revisions, block
  class/count, selected block class/index, hook-output type/shape/device/dtype,
  strict adapter-load receipt, and successful steering replacement.
- Compare base and adapter models on the identical token batch before reading
  results: same tokenizer IDs/alignment keys, base-vs-base diff numerically
  zero, adapter diff finite/nonzero, and injected-vector recovery at the real
  hook. Check both right-padded collection and single-prompt generation. Do not
  silently switch padding side, BOS policy, or chat template after observing
  readability.
- Quantify dtype sensitivity on a tiny fixed real batch. In particular, check
  whether merging LoRA weights into BF16 erases or materially changes the weak
  activation difference relative to an unmerged adapter or higher-precision
  merge. Raw norms, cosines, and top tokens must be stable enough for the
  intended claim; fixture success cannot establish this.
- The real checkpoint/adapter must be the declared final step, with LoRA
  targets/rank verified, and each result row/sidecar must carry arm, seed,
  checkpoint step, zero-based layer, snippet name/full hash, judge model,
  UTC timestamp, git commit, model revision, tokenizer revision, dtype, and
  norm-target provenance. No scientific Qwen3.5-4B readout, J-Lens result,
  OpenRouter judgment, or `L +/- 4` robustness check was verified by this code
  audit.
- `requirements.txt` now requires Transformers >=5.2.0,<6: wheel inspection
  found that 5.2.0 is the first stable release containing the Qwen3.5 model and
  `AutoModelForCausalLM` mapping. The former >=4.50 bound could install a build
  incapable of loading the declared real model.

## Definition changes vs PREREG (should be empty)
- 

## Red-team items and responses
| Item | Check run / limitation admitted | Outcome |
|---|---|---|
| Qwen3.5 composite/text adapter mismatch | Exact two-layer hybrid composite fixture; incompatible adapter negative test | Fixed by one text-only training/readout class and tensor-exact loading |
| Wrong LoRA rank/targets | r=2, q-only adapter negative test | Rejected before readout |
| Tampered or duplicated analysis inputs | Hash, duplicate-ID, incomplete-cell, score-consistency, and A-minus-B checks | Rejected before figures |
| Full CPU workflow | Random four-layer Qwen2 fixture, seed 17, all seven arms plus A-minus-B, offline dry judge | 35 unique judged rows, 16 authenticated diffs, 17x17 cosine table, three PNGs, and 14 sampled raw records; all MOCK |

## Overnight autonomous agent work (disclosed in Q16)
Agents built and smoke-tested the pipeline overnight without human supervision (list what):
- Agents audited and hardened residual hooks/alignment/norm provenance; built
  the analysis/mock/raw-sampling utilities; added dry-run judging and the
  lexical baseline checks; built the N3 adapter utility; exercised a serialized
  random Qwen end to end; and added focused regression tests. Codex then reran
  the complete suite on cached real Qwen2.5-0.5B weights and on a forced-empty
  offline cache using the random-Qwen fallback.
Human verification of that work on the morning of Sept 3 (what was reviewed, what tests were rerun):
- Not yet recorded. Guiv must fill this line after reviewing the diffs, random
  raw samples, and any real-run artifacts; agent test passes are not a substitute.
