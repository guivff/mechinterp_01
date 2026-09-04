# C seed-1 replication ledger (Chat 3 / V4). Append-only.

| time (Zurich) | event |
|---|---|
| 2026-09-04 23:56 | session start; clock read 23:56 CEST; gate set = 01:26 Zurich for env green + identity check (now + 90 min) |
| 2026-09-05 00:06 | read-only RunPod query: 0 pods, balance $286.13, H100 SXM secure 2-GPU price $6.98/h (stock Medium) — higher than the ~$5.38/h assumed in the brief |
| 2026-09-05 00:08:45 | pod created via `podFindAndDeployOnDemand`: id `9wyia6f79b95q3`, name `mats-C-s1-repl`, 2x H100 SXM, SECURE, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 200 GB volume at /workspace, 50 GB container disk, costPerHr $6.98 |
| 2026-09-05 00:11 | pod SSH up: 103.207.149.126:14058, driver 580.126.09, Python 3.11.10, /workspace 200G; bundle (commit 946f971) + bootstrap uploaded; bootstrap launched 00:12 Zurich (`/workspace/bootstrap.log`) |
| 2026-09-05 00:15 | GATE PASSED: bootstrap green (torch 2.13.0+cu129 on driver 580.126.09, 2x H100, bf16 matmul ok; transformers 5.16.1 / trl 1.12.0 / peft 0.20.0 / accelerate 1.14.0 / datasets 5.0.1 / numpy 2.3.5; no vLLM). Data sha256 all three match. Model 1001bb4d downloaded. `tools/identity_check.py` → `results/identity_check_C_s1_pod.json`: passed, zero fields differ from `results/identity_check.json` apart from timestamp / git_commit / trl install path; trl grpo_trainer sha256 equal |
| 2026-09-05 00:16 | chain launched (`tools/run_C_s1_chain.sh`, `logs/chain_C_s1.log`): C seed 1 training on GPU 0 (`logs/C_s1.log`), base L15 cache recompute on GPU 1. Recomputed `base_L15_neutral.npy` sha256 63e24d99… = original cache sidecar (bit-identical) |
| 2026-09-05 00:21 | C seed 1 training finished (225 steps, ~5.5 min, `runs/C_s1/final`); base L15 caches recomputed bit-identical to originals (neutral 63e24d99…, math 760d0ee5…); lora stats ‖ΔW‖_F = 6.958 (C s0: 6.96) |
| 2026-09-05 00:23 | first chain attempt aborted at the cosine step (A seed-1 diffs are named `diff_A_seed1_s1_step150_*`); tag fixed (commit ce2510c), tail chain relaunched. **HEADLINE C s1 · C s0 cosine L15: neutral p1 0.9828 / p2 0.9724; math p1 0.9690 / p2 0.9835.** C s1·A s0 neutral p1 0.504, A s1 0.481. Trace ratio ‖d_C‖/‖d_A‖ neutral p1: C s0/A s0 16.63, C s0/A s1 22.57, C s1/A s0 16.68, C s1/A s1 22.63 |
| 2026-09-05 00:29 | chain done: `results/acc_C_s1.json` 185/200 = 0.925 (parse 200/200); `results/acc_table_C_s1.md`; `results/patchscope_C_s1_step225_L15.json`; all readouts rsynced to Mac; adapter `adapters/C_s1/final` on Mac, sha256 d17ae2d2… equal to pod copy |
| 2026-09-05 00:30:43 | pod `9wyia6f79b95q3` terminated via `podTerminate` after adapter verification; uptime 1,278 s (21.3 min) at $6.98/h ≈ $2.48; account pods = [] ; balance 286.13 → 284.12 |
