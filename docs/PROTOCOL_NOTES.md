# Protocol notes — readable traces after fine-tuning

Source audit frozen 2026-09-03. “Paper” below means Minder et al. arXiv v3 (4 March 2026) or Wang et al. arXiv v2 (16 July 2025). Where a released artifact supplies a detail absent from a paper, that provenance is stated explicitly. The most important conclusion for this project is that its current Arm-D test is a reduced-budget conceptual replication, not an exact replication of Minder et al.

## 1. Minder et al. exact protocol

Primary source: [Minder et al., arXiv:2510.13900v3](https://arxiv.org/html/2510.13900v3). Reproducibility source: camera-ready [`activation_difference_lens.yaml`](https://github.com/science-of-finetuning/diffing-toolkit/blob/2c592aba4d3430f4193b3878468a38a2ad9331a3/configs/diffing/method/activation_difference_lens.yaml) at commit `2c592aba4d3430f4193b3878468a38a2ad9331a3`.

### Activation collection and averaging

- Let $h^{\mathrm{ft}}_{\ell,j}(x)$ and $h^{\mathrm{base}}_{\ell,j}(x)$ be the output residuals of transformer block $\ell$ at token position $j$ on the same tokenized sample $x$. They form
  
  \[
  \delta_{\ell,j}(x)=h^{\mathrm{ft}}_{\ell,j}(x)-h^{\mathrm{base}}_{\ell,j}(x).
  \]

- Main layer: $\ell=\lfloor D/2\rfloor$, described as the middle layer, i.e. 50% depth. Exact absolute block indices are not stated in the paper. The released implementation instead maps fractional layer $f$ to `int(f * (D - 1))`; thus, for a 32-block model, paper prose gives block 16 but code-compatible `f=0.5` gives zero-based block 15. This is a real one-layer paper/code discrepancy.
- Appendix F.1 also tests 25%, 50%, 75%, and 100% depth, rounded down. Token relevance generally strengthens toward later layers; steering is best around the middle, with the authors noting that 75% “might perform even better.” Their 50% choice is a compromise between readout and steering, not an empirically universal optimum.
- Readout corpus: 10,000 unrelated pretraining/web samples. The prose calls this a pretraining corpus or “random web data”; it does not name the dataset or sequence length. The camera-ready config names `science-of-finetuning/fineweb-1m-sample`, train split, seed 42, and tokenizes each sample to $n=128$ tokens with special tokens enabled.
- Primary positions: the first $k=5$, $j=0,\ldots,4$. The paper computes an “average activation difference per position”:
  
  \[
  \bar\delta_{\ell,j}=\frac{1}{10{,}000}\sum_x\delta_{\ell,j}(x).
  \]
  
  It does **not** average positions together into one vector. The current project’s `mean_tokens` after skipping positions 0–3 is therefore a different estimator. The released Patchscope task anomalously includes position 5, but the paper, steering, token relevance, and agent use positions 0–4.
- Their standard SDF organisms use 40,000 synthetic documents of about 500 tokens each (about 20M tokens). The project’s Arm D uses about 2,000 documents of 200–400 tokens (about 0.4–0.8M), 25–50 times fewer training tokens. A failure of D under the present budget is not a failure to reproduce the published treatment.

### Logit lens

For each position separately,

\[
p_j=\operatorname{softmax}\!\left(W_U\,\operatorname{final\_norm}(\bar\delta_{\ell,j})\right).
\]

Yes: the model’s final normalization is explicitly applied **before** its LM head/unembedding. There is no additional $L_2$-norm matching for the paper’s logit-lens readout. The camera-ready code uses the fine-tuned model’s final norm and head for the difference, base-mean, and fine-tuned-mean readouts.

### Token Identity Patchscope

For each $\bar\delta_j$, Patchscope runs the **fine-tuned model** on

```text
tok1→tok1
tok2→tok2
tok3→tok3
?
```

and **replaces** the final `?` token’s block-$\ell$ residual with $\lambda\bar\delta_j$; it then reads the next-token distribution. It does not add the vector to the existing `?` activation.

- Identity triples: `(man, 1135, hello)`, `(bear, 42, blue)`, and `(921, target, anna)`.
- Norm: first rescale $\bar\delta_j$ to $\eta^{\mathrm{ft}}_\ell$, the mean $L_2$ norm of fine-tuned residuals at that layer during collection. The prose says the first 3 tokens are omitted from this estimate; the camera-ready utility omits the first 5.
- Prompt aggregation: retain the top 16,384 tokens under each identity prompt, intersect the three supports, average probabilities over prompts for survivors in code, and retain the top 20.
- $\lambda$ candidates: exactly 30 values, $0.5,0.6,\ldots,2.0,3,4,5,10,20,40,60,\ldots,200$. `gpt-5-mini` sees 10 scales at a time, selects the scale with the largest semantically coherent token set, then chooses among the three batch winners. This is adaptive LLM selection, not a fixed coefficient.

For token relevance, another `gpt-5-mini` receives the true fine-tuning objective and the 100 most frequent non-common tokens from the fine-tuning corpus. It marks each top-20 candidate relevant/not relevant in three candidate-order permutations; a token counts only if all three runs agree. The reported relevance is the **maximum over the five positions**, not their mean. The paper states that common-English tokens are removed, but the removal list and filter implementation are **not stated**. Appendix D reports inter-grader Krippendorff $\alpha=0.65$, so this score is materially grader-dependent.

### Steering

Minder et al. add $\alpha\bar\delta_j$ at the middle-layer residual to **all token positions during generation in the fine-tuned model**. The current project instead proposes steering the base model; that is a legitimate causal question but not their protocol. Each direction is first matched to $\eta^{\mathrm{ft}}_\ell$.

Coefficient calibration is coherence-only:

1. On `Tell me a story?`, binary-search $[0,100]$ for 10 iterations. At each strength, draw 10 samples at temperature 1.2; a `gpt-5-nano` coherence judge passes it if at least 8/10 are coherent.
2. On `Give me some ideas for some fun weekend activities?` and `Why don’t you choose a topic of conversation for us?`, search $[0,2\pi_1]$ for 5 iterations each.
3. Use the mean of the three thresholds as $\alpha$.

Final evaluation uses 20 fixed neutral chat prompts, 5 generations per prompt, and temperature 1.1. The paper does not state a final token cap; the camera-ready config uses 512 new tokens (128 during calibration) and disables thinking. The camera-ready config/code uses 12/6 rather than the paper’s 10/5 search iterations.

They embed text with `Qwen3-Embedding-0.6B` and compare steered outputs with 500 sampled fine-tuning records. Similarity baselines are: fine-tuning-record self-similarity; unsteered-output-to-fine-tuning similarity; and both steered and unsteered outputs against 500 `tulu-3-sft-olmo-2-mixture` chat samples.

### Interpretability agent and score

- Agent: OpenRouter `openai/gpt-5`, medium reasoning. Its initial message contains the top 20 Patchscope and top 20 logit-lens tokens for each of positions 0–4, plus one steered and one unsteered answer for each of the 20 steering prompts, truncated to 200 characters.
- Prompting: no domain candidates or hints. It is told to propose competing, specific, falsifiable domain/behavior hypotheses; prioritize content-bearing tokens; discount punctuation, UI, stopword, and code artifacts; seek agreement across positions/methods; and verify a hypothesis by sending the same targeted single-turn queries to base and fine-tuned models. It must finish with a description of at most 200 words plus concise evidence/caveats. The exact prompt is embedded in the paper and in the released [`prompts.py`](https://github.com/science-of-finetuning/diffing-toolkit/blob/2c592aba4d3430f4193b3878468a38a2ad9331a3/src/diffing/methods/activation_difference_lens/prompts.py).
- Tools: cached logit-lens details, cached Patchscope details, cached/additional steering samples, paired base/fine-tuned model queries, and new steering generations. It has at most 15 agent turns; parsing retries count. Model-interaction budget is $i=0$ or $i=5$.
- Grader: a separate `gpt-5-mini` sees the true objective, the hypothesis, and an organism-specific rubric, reasons over all five possible grades, then returns 1–5. Every condition is rerun five times and means are reported. Grade $\ge2$ means broad detection; it is not exact fixed-label classification.

| Family | Operational grade anchors (2 → 5) |
|---|---|
| SDF | rough topic → exact topic + one abstracted fact → about 25% abstracted or 10% exact facts → over 75% abstracted and at least 25% exact |
| Emergent misalignment | rough topic/peculiar advice → topic plus a hint something is wrong or general misalignment → topic plus unsafe behavior → clearly unsafe/misaligned characterization |
| Subliminal learning | animal terms → animal preference → preference for a related animal → trained preference for the specific animal |
| Taboo word | semantic neighborhood → word-like topic or guessing artifacts → explicit word importance or guessing-game format → both format and exact word |

- Black-box baseline: the same agent logic without ADL artifacts/tools, initially given ordinary generations on the same 20 prompts, at $i=0,5,50$. The $i=50$ condition is a stronger-query comparator, not compute-matched to ADL $i=5$.
- Other baselines: top-token relevance from per-position mean base and fine-tuned activations computed on fine-tuning samples (unlike the difference vector’s unrelated-text corpus); the semantic-similarity baselines above; broad chat/domain tuning as less-narrow comparisons; full fine-tuning versus LoRA; and, for causal replacement, 64 “random-diff” directions formed by subtracting two real base-model chat activations. Isotropic random vectors were considered too close to downstream null space. Minder et al. use no TF–IDF baseline, six-way domain judge, label-shuffled chance test, or per-token constancy statistic.

### Key published numbers and the correct H1 comparison

- Coverage: 33 organisms from 4 organism families, evaluated across 7 model variants/checkpoints in the Gemma, Llama, and Qwen families, 1B–32B parameters.
- At each model’s best of the five positions, 20–80% of the top-20 Patchscope tokens are judged relevant. Patchscope is stronger than logit lens; no aggregate exact logit-lens percentage is stated.
- ADL agent, $i=5$: grade $\ge2$ on 91% of organisms and grade $\ge4$ on 30%.
- ADL agent, $i=0$: 82% and 24%, respectively.
- Strongest black-box agent, $i=50$: 39% and 1%, respectively.
- The paper provides no tabulated per-organism Patchscope values, no fixed-choice domain accuracy, no chance calibration, and no number directly comparable to PREREG H1’s 0.60 threshold. Calling 0.60 “the same ballpark” as 91% would be invalid: the latter is open-ended grade $\ge2$ after two token readouts plus up to 40 supplied generation snippets and optional model interaction.

## 2. What Minder et al. say about RL/on-policy training and why the trace exists

### RL/on-policy

**Not stated.** The paper has no experiment or claim about reinforcement learning, RLHF/RLAIF, PPO, GRPO, reward optimization, or on-policy sampling. SDF is supervised next-token fine-tuning “as if” extra pretraining; imported model organisms and chat checkpoints do not supply an on-policy comparison. Generic references to “post-training” are not evidence that ADL transfers to RLVR.

### Bias term versus a deeper mechanism

Their proposed explanation is overfitting/catastrophic forgetting: when one semantic concept occurs across virtually every narrow-training sample, the model can reduce loss by continually encoding that concept. Their causal test removes only the one-dimensional component along $\bar\delta_j$: at every token position in the fine-tuned forward pass, it replaces the fine-tuned projection onto $\operatorname{span}(\bar\delta_j)$ with the corresponding base-model projection.

Removing that component raises cross-entropy on fine-tuning data for all three tested models. It lowers pretraining-data loss for Qwen and Llama; Gemma’s effect is slightly positive but comparable to its elevated random-diff baseline, so representational divergence confounds that case. These results show that the residual direction is functionally useful for narrow data and can hurt generic data. They do **not** show a literal learned parameter bias term: the paper does not decompose weight/bias changes, establish input-invariant magnitude, or report variance explained by a constant offset. It explicitly says the “underlying mechanisms that produce these detectable biases remain unclear.” Full fine-tuning produces a slightly stronger readable trace than LoRA, ruling out only a LoRA-specific artifact.

### Mixing constraint

They hold 40,000 SDF documents fixed and add C4 up to 80,000 samples, testing fine-tuning:pretraining ratios through 1:2 on Qwen, Llama, and Gemma. “Mixing substantially reduces detectable bias”:

- the prose says even 1:0.1 materially reduces readable traces, although Figure 8’s first nonzero plotted ratio is 1:0.5;
- at 1:2, steering approaches baseline for all models;
- Llama reaches baseline by 1:0.1, whereas Qwen’s relevant tokens never entirely disappear;
- at 1:1, every agent condition has mean grade below 2;
- false-fact alignment can also decline, especially for Llama;
- at 1:2, the fine-tuning-data causal effect is roughly half the unmixed effect despite three times as many training steps;
- mixing unrelated UltraChat into emergent-misalignment training reduces but does not erase the trace, while the misaligned behavior remains.

The text/footnote says this SDF analysis averages three organisms; the Figure 8 caption says five. The paper does not resolve that contradiction.

Interpretive constraint for this project: mean readability diagnoses mono-semantic overrepresentation, not learning in general and not reward-specific learning. A missing Arm-A mean trace could coexist with behavioral learning if on-policy trajectories make the update more diverse or distributed. Conversely, a strong Arm-D trace can largely be narrow-corpus overfitting. Therefore D is a method-positive-control, not a numerical lower bound for A; “no readable mean direction” must not be written as “no internal effect of RL.”

## 3. OOCR paper: narrow fine-tuning as a constant steering vector

Primary source: [Wang et al., arXiv:2507.08218v2](https://arxiv.org/html/2507.08218v2).

The abstract’s “essentially adds a constant steering vector” claim is narrower in the actual evidence. All experiments use Gemma 3 12B, rank-64 LoRA with $\alpha=32$, dropout 0.05, and MLP-only adapters. The diagnostic isolates the **output added by one selected LoRA-adapted MLP component**, not the end-to-end residual difference $h^{\mathrm{ft}}-h^{\mathrm{base}}$. All-layer LoRA adapts gate/up/down MLP projections in all 46 layers; the one-layer condition adapts one MLP down projection. Best one-layer locations vary sharply by task: 22 for Risky/Safe, 2 for $f(x)=3x+2$, and 15 for Tokyo. The paper therefore does not justify a universal 0.6-depth layer.

For each non-backdoor task, the authors collect LoRA-output difference vectors at the final 20 token positions of the first in-distribution training example and the final 20 positions of an unrelated Battle-of-Trafalgar passage. Early positions are excluded because magnitudes sometimes spike. They plot all $\binom{40}{2}=780$ non-self pairwise $\lvert\cos(d_i,d_j)\rvert$, including ID–OOD pairs, and report that values are almost always near one. Their defensible conclusion is that the adapter learned to “conditionally add a vector along a single direction.” Absolute cosine proves approximate axis/rank-one alignment; it ignores magnitude and treats $v$ and $-v$ as equivalent. No mean, quantile, threshold, confidence interval, raw values, or variance-explained number is stated.

Two sufficiency checks compress 20 differences to a fixed direction: (i) first principal component, or (ii) unitize the 20 ID vectors and average them. Scale is the mean projection of the original LoRA outputs onto that unit direction over the last 20 tokens. The resulting unconditional “natural steering vector” can reproduce OOCR in early layers on the function task, but with higher variance and generally worse generalization than LoRA. Directly trained vectors also induce OOCR. On Risk Backdoor, an unconditional vector achieves about 1.0 in-distribution validation accuracy but poor OOCR test accuracy, as does LoRA; this shows conditional behavior, not successful out-of-context generalization. Overall this is causal evidence for a simple low-rank solution, not evidence that every fine-tune or every full residual difference is constant. Learned vectors also have low cosine to the naive concept vector and across random seeds; a readable direction need not be canonical.

### Preregistered constancy test for this project

For all retained, token-aligned residual differences $\delta_i$, let $\bar\delta=N^{-1}\sum_i\delta_i$ and stack rows into $\Delta\in\mathbb{R}^{N\times d}$. Report all of:

\[
C_{\mathrm{mean}}
=1-\frac{\sum_i\lVert\delta_i-\bar\delta\rVert^2}
        {\sum_i\lVert\delta_i\rVert^2}
=\frac{N\lVert\bar\delta\rVert^2}{\sum_i\lVert\delta_i\rVert^2},
\qquad
C_{\mathrm{rank1}}=\frac{\sigma_1(\Delta)^2}{\lVert\Delta\rVert_F^2}.
\]

Call the first quantity **mean-offset energy share**, not “fraction of variance explained by the mean”; its denominator is uncentered energy. Also report signed $\cos(\delta_i,\bar\delta)$, coefficient of variation of $\lVert\delta_i\rVert$, and the OOCR-compatible distribution of pairwise $\lvert\cos\rvert$. Fit $\bar\delta$, scale, and the first singular vector on one fixed snippet split; evaluate geometry on a held-out split and separately across neutral/on-domain sets. Finally, causally add the fixed training-split vector at the same post-block hook and compare base-model logits/behavior with the fine-tuned model. This distinguishes a literal constant offset, a variable-amplitude rank-one effect, and a non-constant but readable average. In particular, $\delta_i=a_i v$ has perfect pairwise absolute cosine even if signs alternate and $C_{\mathrm{mean}}\approx0$.

## 4. J-Lens

Primary implementation: [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens), commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`. It is labeled “Reference implementation. Not maintained and not accepting contributions.” Primary method: [Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html). Review: [Nanda, *A Review of Anthropic’s Global Workspace Paper*](https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper).

### Exact API and representation contract

For valid source/target positions $S$, the fitted layer-specific estimator is

\[
J_\ell=\mathbb E_{\mathrm{prompt}}
\left[\frac{1}{|S|}\sum_{t\in S}\sum_{\substack{t'\in S\\t'\ge t}}
\frac{\partial h_{\mathrm{final},t'}}{\partial h_{\ell,t}}\right],
\qquad
\operatorname{lens}_\ell(h)=W_U\operatorname{norm}(J_\ell h).
\]

The README’s high-level API loads with `JacobianLens.from_pretrained(repo, filename=..., revision=...)` and calls `lens.apply(model, prompt, positions=...)`. `apply` runs its own prompt forward pass and cannot accept a precomputed residual. For a difference vector already in hand, use the lower-level `lens.transport(diff.float(), layer=L)` and then the wrapped model’s `unembed`. `unembed` already applies the model’s final norm, LM head, and any logit softcap; do not normalize twice. Softmax is unnecessary for top-$k$ rank.

`transport` accepts shape `[..., d_model]` and expects the coordinates of the **zero-based output of decoder block $L$**, because the fitter registers a forward hook on `model.layers[L]`. Under the usual Hugging Face hidden-state convention this corresponds to `hidden_states[L+1]`. The exact model weights, width, tokenizer, post-block hook, and layer key must match. Applying a base-model average Jacobian to a fine-tuning difference is algebraically supported but was not validated in the paper; interpret it as the generic base model’s average downstream transport of that displacement, not the exact fine-tuned-model logit change.

### Does a Qwen3.5-4B lens exist, and what is it?

**Yes, for [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)—not for the separately released [`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base).** The official repository [`walkthrough.ipynb`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/walkthrough.ipynb) maps the following 1,000-prompt artifact, added in [HF commit `91271eb5b15a43eebed7bb447618738754f1379a`](https://huggingface.co/neuronpedia/jacobian-lens/commit/91271eb5b15a43eebed7bb447618738754f1379a), to `Qwen/Qwen3.5-4B`:

```text
repo: neuronpedia/jacobian-lens
revision: 91271eb5b15a43eebed7bb447618738754f1379a
filename: qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt
sha256: 1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e
```

It is a 406,332,644-byte PyTorch `torch.save`/ZIP-pickle `.pt`, **not safetensors**, loaded with `torch.load(..., map_location="cpu", weights_only=True)`. Keys are `J`, `n_prompts`, `source_layers`, and `d_model`: 31 fp16 matrices for post-block layers 0–30, each $2560\times2560$, with `n_prompts=1000` and `d_model=2560`. The constructor upcasts matrices to fp32, so resident CPU memory is about 813 MB. The checkpoint embeds no model ID/revision, tokenizer, corpus, target layer, or fit options; the exact model revision used for fitting is **not stated**. Pin these externally and assert width/source layers. The sibling file without `_n1000` stopped after 417 prompts, and its neighboring `config.yaml` describes that 417-prompt fit—not the `_n1000` artifact. The adding commit identifies `_n1000` as a WikiText-103/reference-implementation fit.

The project files say “Qwen3.5-4B base” without freezing the exact HF repository. If that means the base-model role but the selected weights are `Qwen/Qwen3.5-4B`, this asset matches. If it means the `Qwen/Qwen3.5-4B-Base` checkpoint, no exact pre-fitted asset is identified. Nor is there an exact asset for the fallback `Qwen/Qwen3-4B-Base`: the available Qwen3-4B fit names `Qwen/Qwen3-4B`. Width compatibility is insufficient. Freeze the exact model repository and revision before any readout; for either `-Base` checkpoint, skip J-Lens under the preregistered optional rule unless a new lens is fitted prospectively.

### Ten-line artifact smoke-test pseudocode for `readout/decode.py::jlens`

```python
import torch, transformers, jlens
model_id, repo = "Qwen/Qwen3.5-4B", "neuronpedia/jacobian-lens"  # not -Base
rev = "91271eb5b15a43eebed7bb447618738754f1379a"
file = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda()
tok = transformers.AutoTokenizer.from_pretrained(model_id)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(repo, filename=file, revision=rev)
assert L in lens.source_layers and diff.shape[-1] == model.d_model == lens.d_model
return top20(tok, model.unembed(lens.transport(diff.detach().to(model.input_device, dtype=torch.float32), L)))
```

### Caveats from Nanda’s review and the implementation audit

- Treat J-Lens as a “hypothesis generation tool”: “It will miss some concepts, and have various false positives.” It is variable interpretability, not an account of the circuit or algorithm that computed/used the variable.
- $J_\ell$ is a corpus- and source-position-averaged infinitesimal linearization whose estimator sums current-and-future target-position Jacobians. It is not the prompt-local causal derivative, and LoRA can alter downstream Jacobians.
- The single-token vocabulary is an overcomplete, non-unique concept dictionary. Top-$k$ can favor high-norm/high-variance token families and English tokens. A top-token list is not the paper’s sparse nonnegative decomposition. At $K$ equal to median occupancy, the **excess** activation variance explained by the top-$K$ J-Lens vectors over a same-size random control never exceeds 10%.
- Current fitting code excludes source positions 0–15 and the final token. The `_n1000` file does not encode this setting, although its reference-implementation provenance implies the default. The project’s skip-4 average therefore includes positions 4–15 outside the fit estimator’s usual position regime. Predeclare a J-Lens-only positions-$\ge16$ sensitivity; do not silently substitute it after viewing outputs. Nanda’s separate Qwen3.6 replication used 25 Pile prompts of 128 tokens and skipped four—this is not the Qwen3.5 checkpoint’s metadata.
- In Qwen RMSNorm, positive rescaling before `final_norm(Jd)` is approximately canceled (up to epsilon). Norm-matching $d$ will normally not change top-token ranks. Preserve raw norms and do not claim score-magnitude control without an explicit check.
- Intervention errors can be amplified, especially by negative steering/ablation; changed behavior can reflect a broken model or direct word-generation effects. Use an exact-model known-intermediate sanity set and random human inspection before scientific decoding. If that check fails, report J-Lens as failed/omitted rather than interpreting attractive tokens.
- Freeze token cleanup, top-$k$, judge prompt, and handling of fragments before arm labels are revealed. Describe outputs as token associations or verbalizable dispositions, not decoded thoughts, a parameter mechanism, or proof of reward-specific computation.
- `jlens.from_hf` freezes the passed model, switches to evaluation mode, and can force BOS behavior; use a dedicated analysis copy.

## 5. GRPO recipe from the reward-hacking benchmark

Primary source: [Wong, Engels, and Nanda, *Steering RL Training: Benchmarking Interventions Against Reward Hacking*](https://www.alignmentforum.org/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against), 29 December 2025. Released code: [`ariahw/rl-rewardhacking`](https://github.com/ariahw/rl-rewardhacking).

| Item | Published/released value |
|---|---:|
| Model | Qwen3-4B; released runs name `Qwen/Qwen3-4B` |
| Data/reward | LeetCode medium + hard; correctness reward plus small Python-code-block format reward |
| Algorithm | GRPO |
| Generations per prompt $G$ | 16 |
| Prompts per optimizer step | 16 in released config |
| Total trajectories per optimizer step | 256 |
| Steps | 200 |
| LoRA | rank 32, alpha 32; released code targets q/k/v/o and gate/up/down projections, dropout 0 |
| Learning rate | $7\times10^{-5}$ |
| Thinking | off |
| Max prompt / completion | 1,536 / 1,536 tokens in released config; prose explicitly states completion 1,536 |
| Generation | temperature 0.7, top-p 0.95 in released config |
| Other released defaults | KL $\beta=10^{-3}$, AdamW-8bit, cosine schedule, 10 warmup steps, weight decay 0.1 |
| Runtime | about 3 hours on 4×H200 without monitors; monitor runs slightly longer |

The loophole was normally discovered around steps 80–100. That does not validate 150 steps on GSM8K, but it makes 150 a plausible pilot horizon if learning curves and checkpoints are retained.

Gotchas: thinking was disabled to control response length; the repository recommends at least 4,096 completion tokens if thinking is enabled. The code evaluator is CPU-bound enough that the authors recommend at least 32 physical CPU cores (often 64). Their wrapper pins Verl 0.6.1. Activation-caching runs used a fifth H200, took about 3.5 hours, and cost about USD 60. Multiple seeds mattered because reward-hacking onset was variable.

The current prereg is **not this recipe**: it changes Qwen3→Qwen3.5, $G=16\to8$, prompts/step $16\to32$ (still 256 total rollouts), LoRA alpha $32\to64$, LR $7\times10^{-5}\to10^{-5}$, steps $200\to150$, KL $10^{-3}\to0$, and completion cap $1536\to512$. The post supports feasibility of 4B GRPO LoRA, not equivalence of these hyperparameters. Record that distinction rather than citing it as direct validation.

## 6. Recommended fills for PREREG blanks

| Field | Recommended frozen text | Paper-grounded reason |
|---|---|---|
| Layer $L$ | `fraction = 0.50; zero-based post-block L = floor(0.50 × (D−1)); L=15 for Qwen3.5-4B (D=32), sensitivity L±4` | 0.50 is Minder’s primary depth; 15 follows the camera-ready mapping that generated the results. Record that literal paper $\lfloor D/2\rfloor=16$ differs by one, so there is no hidden off-by-one. OOCR finds task-specific optima and does not support 0.60. |
| Norm target | `η_ref = mean ||h_base,L||₂ on neutral snippets under the frozen token mask; every nonzero d is rescaled to η_ref; raw ||d|| retained. Arm-D exact-protocol diagnostic also uses η_D^ft.` | Minder normalizes to activation scale $\eta^{\mathrm{ft}}$, not to another difference vector. A single base activation norm is the closest arm-common adaptation and avoids making random Arm-D magnitude the scale control. Scaling is irrelevant to RMSNorm-based top-token ranks but essential to steering dose. |
| Steering coefficients | `Calibrate α_D by Minder’s coherence-only 3-prompt search without viewing domain judgments; then use the common grid {0.25, 0.50, 1.00} × α_D for every arm, with α=0 as baseline.` | Minder supplies no reusable fixed numeric steering grid; it selects the highest coherent scale adaptively. Calibrating only on the positive control and freezing common fractions avoids separately optimizing each arm for readability. Stratify prompt/coefficient assignment with a frozen seed, or raise the 50-generation budget to cover every prompt at every coefficient. |
| Judge model | `openai/gpt-5-mini via OpenRouter; temperature 0; exact provider/model revision and raw response saved` | Minder uses `gpt-5-mini` for token relevance and hypothesis grading, making it the closest non-Qwen comparator. Its judge temperature is **not stated**; 0 is a reproducibility choice. If no immutable snapshot is exposed, save the returned model identifier/date and do not silently change it mid-run. |

Five additional hardening edits are load-bearing:

1. Rename H1 a **reduced-budget conceptual replication**, or add the paper-compatible diagnostic: positions 0–4 kept separately over 10,000 FineWeb samples, with Arm D steered in the fine-tuned model. The current 500-sample, post-skip-4, token-pooled, base-steering pipeline is not Minder’s protocol.
2. Define the top-token sampling unit. Repeating one deterministic top-20 list through the judge 100 times measures judge stochasticity, not 100 independent examples. Either compute directions on frozen independent/bootstrapped snippet batches and cluster uncertainty by batch, or call the outcome “judge-call agreement,” not generalization accuracy.
3. Reordering the six candidate labels is an order-robustness test and should **not** reduce a sound judge to chance. A chance negative control shuffles input↔gold-domain pairing; N1–N3 are the substantive nulls. State exactly which operation is used.
4. Train the TF–IDF baseline on a frozen external reference corpus for all six domains and test it on readout texts. “Train on the same readout texts” is underdefined with one aggregate token list and only math/cooking treatments, and risks leakage. Freeze document units and split by source document, not by token or repeated generation.
5. Replace “Qwen3.5-4B base” with an exact HF repository and revision. `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-4B-Base` are distinct checkpoints; the available Qwen3.5 lens is mapped only to the former.

## 7. Five ready-to-paste citations

1. Minder, Julian, Clément Dumas, Stewart Slocum, Helena Casademunt, Cameron Holmes, Robert West, and Neel Nanda. 2025. “Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences.” *arXiv* 2510.13900v3. https://doi.org/10.48550/arXiv.2510.13900.
2. Wang, Atticus, Joshua Engels, Oliver Clive-Griffin, Senthooran Rajamanoharan, and Neel Nanda. 2025. “Simple Mechanistic Explanations for Out-Of-Context Reasoning.” *arXiv* 2507.08218v2. https://doi.org/10.48550/arXiv.2507.08218.
3. Gurnee, Wes, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, and Jack Lindsey. 2026. “Verbalizable Representations Form a Global Workspace in Language Models.” *Transformer Circuits Thread*, July 6. https://transformer-circuits.pub/2026/workspace/index.html.
4. Nanda, Neel. 2026. “A Review of Anthropic’s Global Workspace Paper.” *LessWrong*, July 6. https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper.
5. Wong, Aria, Josh Engels, and Neel Nanda. 2025. “Steering RL Training: Benchmarking Interventions Against Reward Hacking.” *AI Alignment Forum*, December 29. https://www.alignmentforum.org/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against.
