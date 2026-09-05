# SOURCE_INDEX.md — where the larger context, provenance and frozen artifacts live (updated Sat 2026-09-05 03:45)

## Repository (GitHub, private): `guivff/mechinterp_01`
Branches: `main` (control plane + merged agents 01/02/03), **`pod`** (the current truth for code, results and docs; HEAD ≥ `0d9e487`), **`replication`** (C s1 merged at `c852658`; **C_masked at `19524db`, merge ⏳ C1**). **Pod `03iex0ijclvd8o` terminated 14:32 Zurich** (14.38 h, $200.81). Nothing remains on it: adapters (`runs/`), activation caches (`results/cache/`), `logs/*.log` and per-arm `run_meta.json` (except D, D_math_full) are gone. Every number is now re-derivable only by retraining.

**Known provenance defect (digest §13):** the pod→Mac rsync in `ship.sh` reverted three newer local files, which were then committed: `results/acc_table.md`, `results/visibility_table.md`, `results/lexical_items_perposition.jsonl`. All regenerated at `5530ae2`. Any figure or text built before that commit against those files must be re-checked. `docs/RESULTS_DIGEST.md` was **untracked** until 14:00 Fri and existed only on the Mac.

| Item | Path / ref | Notes |
|---|---|---|
| Control plane | `AGENTS.md`, `PROGRAMME_RULE.md`, `PROGRAM_STATE_CURRENT.md`, `CLAIM_FIREWALL.md`, `OPEN_TASKS_CURRENT.md`, `SOURCE_INDEX.md`, `CHANGELOG.md` | all refreshed Sat 03:45 for the loss-placement headline |
| **Evaluations** | `docs/EVALUATION_NEEL_STYLE.md` (01:30, learning-rule headline), **`docs/EVALUATION_NEEL_STYLE_v2.md` (03:30, loss-placement headline; P(exploration) ~0.45–0.55)** | the v2 objection list is the limitations list |
| Theory passes | `docs/T1_THEORY_BLOCKS.md`, `docs/T2_THEORY_PASSES.md` (§M0 = the loss-placement alternative and its preregistered threshold), `docs/PREREG_v2.md` (post-deadline; needs rewrite for the new headline) | |
| Prompts round 6–7 | `prompts/round6/R1_C_MASKED.md` (the decisive test), `C1_MERGE_AND_INSERTS.md`, `W5_WRITER_FINAL.md`; `prompts/round7/C2_MERGE_C_MASKED.md`, `W6_WRITER_OVERRIDE.md`, `00_ROUND7_ROUTING.md` | |
| **Coordination (new)** | `docs/CHAT_ROSTER.md` (who each chat is, what it may see, what it owns), `docs/DIRECTION.md` (the thesis as it now stands, what remains, post-deadline programme) | |
| Endgame plan | `docs/ENDGAME.md` (rewritten T−17.5 h) | supersedes NEXT_STEPS_T24H / T35H |
| Preregistration | `PREREG.md` — amendments through Sat 02:56, incl. the **02:00 preregistered C_masked test (V ≤ 0.18 / ≥ 0.30) and its 02:56 result (0.049)** | N2 named as a null → reported |
| **Results digest (sole citable source)** | `docs/RESULTS_DIGEST.md` @ `5530ae2` | §14 = coverage audit both directions |
| Verification ledger | `VERIFY.md` — rows 1–50 C1 (one-liners run), 51–52 Guiv's readings, 53–57 C_masked ⏳; `tools/recompute_oneliners.md` | rejection-critical; Guiv's columns on 1–49 ⏳ |
| Gate 1 review | `docs/GATE1_REVIEW_0430.md` (limitation 1 to be amended in place, dated) | |
| Red team | `docs/RED_TEAM.md` (23 confounds) | fresh RT pass on current digest ⏳ |
| Theory | `docs/THEORY_NOTE.md` (scorecard 2/2; post-hoc refinement labelled; C prediction prospective), `docs/SCALING_PREDICTION.md` (AdamW ⇒ untested) | motivation only |
| Protocol facts | `docs/PROTOCOL_NOTES.md` | Minder per-position protocol, Patchscope, GRPO recipe |
| Rubric / form / profile | `context/NEEL_RUBRIC.md`, `context/PROFILE_PUBLIC.md` | Q17–Q19 |
| Prompts round 4 (verification) | `prompts/round4/00_ROUND4_ROUTING.md`, `V0_GUIV_HUMAN_PROTOCOL.md`, `V0_RUNSHEET.md`, `V1b_LOCAL_REVIEW_PACKETS.md`, `V1c_DOC_AMENDMENTS.md`, `V2_TAGGER_LLM_JUDGE.md`, `W2_WRITER_REVISION.md` | V1 (with terminate) superseded |
| Prompts rounds 2–3 | `prompts/round2/P1–P6`, `prompts/round3/P7–P13` | historical |
| Training / eval code | `grpo/train_grpo.py`, `grpo/train_sft.py`, `grpo/eval_acc.py` (`extract_answer` = preregistered last-number parser), `tools/reparse_acc.py` (stopping-robust re-parse), `tools/discordant.py` (`--n` display sample; **the 20-item md is a display convenience, not the source**), `tools/acc_table.py`, `tools/lora_delta_stats.py` | |
| Readout code | `readout/*`, `tools/per_position_diff.py`, `tools/patchscope.py`, `tools/token_relevance.py`, `tools/steer_eval.py`, `tools/lexical_on_lists.py`, `tools/make_lexical_items.py` | |
| Figures | `analysis/make_figures.py` (refuses MOCK, fails on missing input) → `figs/fig{1..5}.png`, `figs/figure_sources.json` | all from real files |

## Frozen data artifacts
| Artifact | Hash / id | Source |
|---|---|---|
| Model | `Qwen/Qwen3.5-4B-Base` @ `1001bb4d826a52d1f399e183466143f4da7b741b` | no BOS; eos=pad=248044 |
| GSM8K | `openai/gsm8k` @ `740312add88f…` | train = RL prompts; test = eval + math snippets |
| `data/cooking.jsonl` (2,000) | sha256 `7a955f6b…` | token count ⏳ (needed for limitation 2) |
| `data/math_sft.jsonl` (1,798) | sha256 `15497259…` | 473 GSM8K test + 1,325 MATH test |
| `data/C_samples.jsonl` (15,248 kept / 16,000) | sha256 `78022b70…` | A's correct samples; C saw 1,800 rows once |
| `data/snippets/neutral.jsonl`, `math.jsonl` (500×128) | `c8673772…`, `483c3733…` | |
| `data/blackbox_prompts.jsonl` (20) | | |

## Results files by claim (all local, committed)
| Claim | Files |
|---|---|
| Accuracy, both parsers | `results/acc_{base,A,B,C,D,D_math,D_math_full}_s0.json` (full completions), `acc_table.md`, `acc_table_reparsed.md`, `acc_table_reparsed_variant.md`, `reparse_audit.md` |
| Discordant items | `discordant_A_vs_D_math_readable.md` (68, blinded), `discordant_key.json` (**withheld from C2 and from G until tags written**), `discordant_sample20.txt` (resampled, excludes the 20 seen), `discordant_A_vs_D_math.md` (20, unblinded display), `discordant_{A,B}_vs_base.md` |
| Geometry | `perposition_table_C.csv` (every arm), `perposition_table{,_L11,_L19,_seeds,_A_seeds}*.{md,csv}`, `*_cosine.csv`, `perposition_D_s0_step250_L15.json` |
| Visibility | `lora_delta_stats.json`, `visibility_table.md` (both A seeds), `lora_delta_family_split.json` (uninformative) |
| Token readouts | `patchscope_*.json`, `token_relevance_*.json`, `items_D_s0_L15.jsonl`, `items_N1_*`, **`items_N2_s0_L{11,15,19}_{neutral,math}.jsonl` (50 directions each; ⏳ unreported)** |
| Emergence | `emergence_A.md`, `emergence_A_early*.md`, `emergence_A_early.csv`, `emergence_A_rewards.json` |
| Steering | `steer_table.md`, `steer_eval/*.json` (33 runs), `steer_eval/neutral_gens_{A,C,D_math_full,random}_a1.md` |
| Controls | `judge_calibration.jsonl`, `lexical_on_lists.json` (150 lists; below null), `blackbox/*.jsonl`, `identity_check.json`, `preflight_samples.json` |
| Human-review packets | `review_packet/patchscope_for_human.md` (+ `patchscope_key.json` withheld), `steer_reading.md`, `cooking_samples.md`, `blackbox_rows.md` |
| Guiv's own outputs | `notes/guiv_tags.csv`, `notes/VERIFY_discordant_block.md`, `notes/READ_cooking_corpus.md`, `notes/READ_blackbox_rows.md`, steering paragraph in VERIFY |
| **Replication branch (C s1, C_masked)** | `results/REPLICATION_REPORT.md`, `results/REPLICATION_REPORT_C_masked.md`, `visibility_table_C_masked.md`, `lora_delta_stats_C_masked.json`, `perposition_table_C_masked{,_cosine}.csv`, `acc_C_masked_s0.json`, `acc_table_C_masked.md`, `patchscope_C_masked_s0_step225_L15.json`, `identity_check_C_masked_pod.json`, `logs/pod_C_masked/`, `logs/replication_ledger.md`; adapters `adapters/C_s1/final`, `adapters/C_masked_s0/final` (untracked, Mac only) |
| **No local source** | arm B training curve (reward ≈0.07, trunc 0.79, len 456) — `logs/B_s0.log` destroyed |

## Primary sources
Minder et al. 2025, arXiv 2510.13900v3 (+ `science-of-finetuning/diffing-toolkit` @ `2c592aba…`) · Wang et al. 2025, arXiv 2507.08218v2 · Gurnee et al. 2026 · Nanda 2026 · Wong, Engels, Nanda 2025. Minder's training-token figure: **not yet cited with a page; either add or drop the comparison** (firewall §3.2).

## Application-side context (outside the repo)
Neel Nanda MATS 12.0 FAQ PDF + link digest; Airtable form; `~/Documents/Apply/NeelNanda`; keys in `~/.config/mats/secrets.env` (**revoke RunPod key after submission**); Claude project "Alignment" (mirrors of the control plane as `claude/MATS12_*`).
