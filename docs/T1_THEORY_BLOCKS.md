# T1 — four theory blocks (raw argument, for rewriting in own voice)

Every number below is from RESULTS_DIGEST.md unless listed under "Derived / not in digest" at the end.

## Block 1 — bias term or something deeper? (≤150 words)

Constancy does not separate the learning rules. Mean-offset energy share at neutral p1: A 0.258, D_math_full 0.249, D 0.277, C 0.274; untrained N3 0.071. Training raises it, equally for RL and SFT. Geometrically the RL trace is as bias-like as SFT's. What differs: magnitude (0.21 vs 1.2–3.5) and reproducibility (A s0·s1 0.68 vs SFT 0.92–0.98). So for SFT "learned bias term" fits — constant, reproducible, readable. For RL the same-shaped offset is small, unreadable and seed-specific: a residue of the particular run, not a term the objective installs. Steering adds: at natural norm (0.17 against residual ~11) the offset does nothing; behaviour moves only at 16–33× amplification, raw-parser only, and accuracy and EOS rate move together (digest §10). ["rises with EOS" replaced 2026-09-05 by decision: no per-cell EOS numbers exist, see Derived note below.] At its actual size the offset is not the mechanism. What the data cannot say: whether the fine-tuned model needs it. Missing experiment: ablate d from A's own layer-15 residual and re-measure accuracy. Never run; adapters destroyed.

weakest step in this argument: "constancy = bias-ness" is one operationalisation, at one layer and two positions; the literal constant-offset reading is best supported at p0 (constancy 0.94–0.99), which is shared across arms and excluded from every claim — so the answer "as bias-like as SFT" is really "at p1–2, a quarter of the energy is a bias for both".

## Block 2 — the dose confound, stated correctly (≤200 words)

Arithmetic. lr×steps: C 1e-4×225 = 2.25e-2; A 3e-5×150 = 4.5e-3; ratio 5.0. Observed ‖ΔW‖_F ratio 6.963/1.675 = 4.16. Linear-in-steps predicts 5.0, random-walk (lr×√steps) 4.1; observed sits between. The ‖ΔW‖ factor is what the hyperparameters produce on their own; it says nothing about GRPO vs SFT. C ran to a fixed budget, not to criterion, so "SFT needs 4× more weight change" is not sayable.

What survives: V = ‖d‖/‖ΔW‖_F. C 0.501 vs A 0.125 (s0) = 4.0×; vs 0.092 (s1) = 5.4×.

Replacement sentence: "Per unit of weight change, imitation SFT writes a 4.0–5.5× (four (C, A) seed pairs: 4.00, 5.45, 4.01, 5.47) larger trace into the neutral-text residual than GRPO (V 0.50 vs 0.125/0.092); the raw 17–22× norm gap is descriptive only, since its other factor, 4.2× larger ‖ΔW‖_F, is what C's 3.3× higher learning rate and 1.5× more steps predict by themselves — that lr mismatch is the primary open confound."

V is not dose-independent: ΔW = (α/r)BA with B initialised at zero, so ΔW's composition changes with steps, not just its size; d is nonlinear in ΔW (C's top σ 0.58 vs A's 0.12); and V depends on ΔW's direction anyway — masked vs unmasked D_math give V 0.059 vs 0.179 at ‖ΔW‖ 6.58 vs 6.70; A's seeds differ 1.36× at identical ‖ΔW‖. V removes size, not dose.

weakest step in this argument: the lr×steps → ‖ΔW‖ prediction treats a GRPO step (256 advantage-weighted sequences, β=0) and an SFT step (8 rows) as comparable Adam steps; the 20% agreement could be coincidence, so "dose explains ‖ΔW‖" is plausible, not shown.

## Block 3 — the theory scorecard, rewritten (≤200 words)

- **P1 (format, not topic).** Predicted: A → contrast tokens, C → math vocabulary. Observed: A → digits, relation symbols, 0–2/20 content. But C also → digits, `→`, `=`, no math words (ungraded). Half observed; the discriminating half failed — shape comes from data, not weighting.
- **P2 (input-gated).** A 2.3/2.8 vs cooking 1.3/1.6 — observed. But same-data C is 1.5/2.2 and same-domain SFT 8.4: gating tracks training domain, not rule; A-vs-C margin 1.3–1.5×, one seed.
- **P3 (less constant).** 0.258 vs 0.249/0.274 — refuted. Whatever cancels does not remove the constant component.
- **P4/H3 (preregistered).** A·B −0.13, below all 50 draws. B nonzero and less stable as predicted; central claim failed. The theory's own averaging over 4,800 groups predicts cos≈0, so the prereg over-read it. Post hoc: B drifts against the stop direction (B's length curve unverifiable).
- **C (post-hoc refinement).** C·A +0.505 above all draws — observed; norm-free, so dose-robust. Sharper: C·D_math_full 0.55 vs A·D_math_full 0.27 — the component A lacks is the corpus one.

Survives as: a motivation that got the shape right (format, gated — though C shares it) and the statistics wrong.

Post hoc: reward ≥0.80 from step 5, so most groups are all-correct with zero advantage; the few mixed groups per step carry the update → seed-specific direction (0.68). SFT averages one fixed corpus → 0.92–0.98.

weakest step in this argument: the few-mixed-groups mechanism is untested (per-step group statistics were in destroyed logs), and the cheaper rival — RL's training data itself differs per seed, SFT's does not — is not ruled out; C on seed-1 samples would separate them.

## Block 4 — why this is interesting (≤80 words)

1. Two models that solve GSM8K equally well, trained on the same solutions, can be told apart by one mean activation on unrelated text; the RL one is the hard case.
2. Wrong way round for forensics: RL checkpoints are the ones people want to audit, and their trace is small, unreadable, seed-specific.
3. Two GRPO runs with identical settings agree on direction at 0.68; SFT runs at 0.92–0.98. "The RL vector" of a model is a per-run object.

weakest step in this argument: sentence 1 leans on the 17× norm gap, which Block 2 demotes; the dose-robust version is "differ 4–5× per unit of weight change, and in direction".

## Derived / not in digest

- lr×steps: 2.25e-2, 4.5e-3, ratio 5.0 — derived from digest §1/§9 hyperparameters.
- Random-walk prediction 4.1 (= 3.33 × √1.5) — my assumption, not a measurement.
- ‖ΔW‖_F ratios 4.16 / 4.14; V ratios 4.0 / 5.4 — derived from §5.
- Amplification 16–33× = 11.243 × {0.25, 0.5} / 0.17 (66× at α=1). **Your prompt said "~50×"; the digest supports 16.5× at the α where the effect appears (0.25, p=0.013) and 33× at α=0.5 — use those.**
- 4,800 groups = 32 prompts × 150 steps; 256 sequences/step = 32×8; 8 rows/SFT step = 1,800/225 — derived.
- "17–22×": 3.488/0.155 = 22.5, not 22.6 as in the summary; 3.488/0.210 = 16.6 is right.
- "0.68" is the firewall's rounding of digest 0.676.
- N3 constancy 0.071, D_math masked V 0.059 / ‖ΔW‖ 6.579 — in digest, cited to show V depends on composition.
- B's length drift ("length 456, truncation 0.79") — in digest §1 but flagged §14(b) as having no surviving source; I use it only inside a labelled post-hoc sentence and say it is unverifiable.
- Steering EOS for A at α=0.25 and for random directions is **not** in the digest; "rises with EOS" rests on the §10 sentence "accuracy and EOS rate move together", not on per-cell numbers.
