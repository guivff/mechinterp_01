# Reading note: black-box generations, 5 prompts x 6 arm/seed rows
Source: review_packet/blackbox_rows.md (results/blackbox/*.jsonl; T=0.7, 60 tokens, same sampling seed across arms).
Read by: Guiv, Sat 05 Sep, non-blind. ~10 min.

Five neutral prompts, five arms, T = 0.7, same sampling seed. A, B and D_math_full reproduce the base completion token for token for the first 20-60 tokens on every prompt (A and base are identical on one prompt) and diverge late; a second base seed produces five entirely different texts, so late divergence is within resampling noise. D diverges from the first token on all five prompts, four into cooking content and all five into the corpus's polished register. Black-box reading therefore detects D and nothing else - the same ordering the activation-difference readout gives, with A invisible by both.

Tally: divergence from base_s0 at token 1 - D 5/5, D_math_full 1/5, A 0/5, B 0/5. Cooking content in D completions: 4/5. base_s1 vs base_s0: 5/5 different (noise scale). Note: base_s0 itself emits "<think>" on prompt neutral_03, so "<think>" under A-direction steering is a base-model behaviour.
