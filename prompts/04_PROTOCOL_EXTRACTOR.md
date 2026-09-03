# Agent 04 — Protocol extractor & prereg hardener (no code)
**Tool:** a frontier chat model with web access (ChatGPT / Claude.ai / Gemini). Upload `context/PROJECT_SPEC.md`, `PREREG.md`, `context/NEEL_RUBRIC.md`.
**Papers to read fully:** arXiv 2510.13900 (Minder, Dumas et al., *Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences*), arXiv 2507.08218 (*Simple Mechanistic Explanations for Out-of-Context Reasoning*), the `anthropics/jacobian-lens` GitHub README and the HF repo `neuronpedia/jacobian-lens` (does a `qwen3.5-4b` lens exist? what file format?), and the Alignment Forum post "Steering RL training: benchmarking interventions against reward hacking" (for a working Qwen3-4B GRPO LoRA recipe).

## Deliverable: `docs/PROTOCOL_NOTES.md` with these sections
1. **Minder et al. exact protocol.** Which layers they read out (as a fraction of depth), how many snippets and tokens, which text source, how the diff is computed (mean over tokens? per-position?), whether the final norm is applied before the logit lens, how Patchscope was used, how steering coefficients were chosen, how the LLM "interpretability agent" was prompted and scored, what baselines they used, and their key numbers (so we can check H1 replicates in the same ballpark). Quote the paper where possible.
2. **What Minder et al. say about RL or on-policy training**, if anything, and about why the trace exists (bias term vs deeper). Note the "mixing pretraining data removes the trace" result and how it constrains our interpretation.
3. **OOCR paper:** the "narrow fine-tune ≈ constant steering vector" claim and its evidence; how to test constancy.
4. **J-Lens:** exact loading/usage API for a pre-fitted lens on a diff vector; whether the lens expects a residual activation at a given layer; caveats from Neel's review ("A review of Anthropic's global workspace paper"). Give a 10-line pseudo-code for `readout/decode.py::jlens`.
5. **GRPO recipe:** from the reward-hacking benchmark post: Qwen3-4B, LoRA rank, lr, G, steps, batch, hours on 4×H200; any gotchas (thinking mode off, max tokens).
6. **Recommended fills for PREREG blanks:** layer L (fraction of depth), norm-matching target, steering coefficient grid, judge model — each with a one-line justification from the papers.
7. **Five citations** in a ready-to-paste format for the write-up.

Be precise and quote; if a detail is not in the paper, say "not stated". Do not pad.
