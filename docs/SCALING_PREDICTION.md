# Scaling prediction from mixed-group mass

**Status:** Prospective quantitative consequence of `docs/THEORY_NOTE.md`; not an empirical result or a theorem about Adam/LoRA/multi-step training.

## 1. One-step arm-A update

Let `u_sjg` be the completion-specific score gradient after removing group `j`'s prompt-shared component, and let

`v_sj = E[mean(u | r=1) - mean(u | r=0) | x_sj, k_sj]`.

Conditioning on the 32 prompts and their counts,

`E[Delta theta_A,s | x,k] = eta_eff sum_{j:1<=k_sj<=7} w(k_sj) v_sj`.

`eta_eff` absorbs TRL's batch/loss averaging. For `G=8`,

`w(k) = k(8-k)/(8 sigma_k)`.

The pinned TRL 1.12 implementation uses Bessel-corrected sample SD, `sigma_k = sqrt[k(8-k)/56]`, plus a `10^-4` denominator stabilizer. Thus exactly,

`w_TRL(k) = [k(8-k)/8]/[sqrt(k(8-k)/56)+10^-4]`,

and, ignoring that tiny stabilizer,

`w(k) = sqrt[7k(8-k)/8]`.

Without SD normalization, `w_0(k)=k(8-k)/8`, giving relatively more weight to balanced groups. Population SD would instead give `sqrt[k(8-k)]`.

## 2. Scaling prediction and test

Index mixed groups by `i` and compute separately for each arm:

`M(t)=sum_{i<=t} w_i`,

`Q(t)=sum_{i<=t} w_i^2`,

`R(t)=sqrt[Q(t)]`.

With a fixed contrast direction and fixed mean readout Jacobian,

`||d_A,S(t)|| approximately C_S M_A(t)`.

Under the random-sign idealization,

`E[d_B,S(t)]=0`,

`RMS ||d_B,S(t)|| approximately C_S R_B(t)`.

At matched histories,

`RMS ||d_B||/||d_A|| approximately R/M = 1/sqrt[N_eff]`,

where `N_eff=M^2/Q`.

For each snippet set, plot checkpoint norms against `M_A` for A and `R_B` for B, with zero-intercept linear fits.

**Decision rule:** Confirm only if A proportional to M and B proportional to R each beat the swapped models (A proportional to R; B proportional to M) in leave-one-checkpoint-out squared error, with a block-bootstrap 95% interval excluding zero on both snippet sets. Reversed wins refute the prediction; otherwise report inconclusive.

Literal within-group permutation is a zero-mean random partition, not independent signs. The `R` law additionally assumes weakly dependent group increments with comparable contrast dispersion.

## 3. Assumptions and cheapest diagnostics

- **First order/fixed Jacobian:** evaluate one checkpoint at half adapter dose and test `d(0.5) approximately 0.5 d(1)`.
- **Fixed contrast direction:** compute cosines between successive checkpoint increments `d(t)-d(t-25)`.
- **Plain SGD:** inspect the saved optimizer class/state. If AdamW was used, this prediction is untested, not refuted.

## 4. Constancy

Mean-offset energy share is scale-free. A scale-only `M` or `R` model predicts approximate constancy across checkpoints within each snippet set. No math-versus-neutral ordering follows: `||d_math||` may exceed `||d_neutral||` while math constancy is equal or lower. Thus `constancy_math < constancy_neutral` is not required.
