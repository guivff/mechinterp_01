# E2 preregistration as received (ablation of the trace from the fine-tuned model), recorded before the first pod call

Source: Guiv's E2 brief (chat, actioned 05:21 Zurich, 2026-09-05). `origin/pod` @ `49e1fdc` carries the matching
PREREG.md amendment ("~05:40, Ablation test preregistered"); `sha256(PREREG.md @ origin/pod) =
da758edfd1459103086b1ed863671662f71eece0b70dffd34ee389c972f6ec74`. The prompt file it names
(`prompts/round8/E2_ABLATION_TONIGHT.md`) is not on any origin branch at 05:22; the chat brief is the spec of record.

## Question
Is the readable trace load-bearing? Subtract an arm's own base→fine-tuned L15 neutral mean-difference from the
residual stream of the **fine-tuned** model at every position (prompt and generated tokens) and re-measure held-out
accuracy (first 200 GSM8K test items, greedy, `"{question}\nAnswer:"`, cap 512). Arms: `adapters/C_s1/final` and
`adapters/C_masked_s0/final` only (A and C s0 adapters destroyed — cannot be ablated).

## Which vector
Per-position mean difference d_p at positions 0–4 (each at its own position) and the positions-≥5 pooled mean at every
later position (primary). Secondary: one all-position pooled mean everywhere. Recomputed on the pod from the
recomputed base cache (bit-identical per R1) and the adapter; no file from the destroyed pod.

## Thresholds (stopping-robust parser primary; raw parser and EOS rate beside it). Δ = correct(ablated) − correct(unablated), items / 200
- Own direction, α = 1: **Δ ≥ −3 → not load-bearing; Δ ≤ −20 → load-bearing; between → inconclusive.** α = 0.5, 2 secondary.
- Random matched-norm control, five seeds, α = 1, both arms: **any seed with Δ ≤ −6 → own-direction result is uninformative** (not "not load-bearing").
- Cross-arm: d_C_s1 on C_masked and d_C_masked on C_s1, each at its own norm — descriptive, no threshold.
- Sanity: d_C_s1 subtracted from **base** at α = 1 must change base accuracy by ≤ 3 items or the run is flagged.
- Smoke: α = 0 must reproduce the saved accuracies (C s1 185/200, C_masked 187/200) within ±2 or stop.

## Predictions recorded in the PREREG amendment (before the run)
Under the loss-placement reading: C s1 not load-bearing (the large trace is a prompt-prior term); C_masked inconclusive
or not load-bearing at its small norm.

## Gates (Zurich, absolute): env green 06:00; smoke reported 06:15; primaries (runs 2–3) 06:35; all synced 06:55; terminate 07:00.

## Implementation choices fixed here (not in the brief)
- Batch 25 (the saved base eval used 25; C arms used 8). Greedy bf16 results can move by a couple of items with batch composition; the ±2 smoke tolerance covers this and every Δ is taken against the arm's own α = 0 run at the same batch.
- Random control: one independent Gaussian direction per slot (p0..p4, ≥5), each scaled to that slot's ‖d_p‖ — "the perturbation itself at that norm" position by position.
- Position index = the model's own `position_ids` (left padding handled by the model; generated token t gets position prompt_len + t); slot(p) = min(p, 5).
