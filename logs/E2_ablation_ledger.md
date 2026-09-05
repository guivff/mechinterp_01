# E2 — trace ablation ledger (second replication pod, parallel to R2). Append-only.

Gates (Zurich, absolute): env green 06:00; smoke reported 06:15; primaries 06:35; synced 06:55; terminate 07:00.

| time (Zurich) | event |
|---|---|
| 2026-09-05 05:21:54 | session start; clock read 05:21:54 CEST. `origin/pod` fetched: head `49e1fdc`; PREREG.md on `origin/pod` carries the E2 ablation amendment (~05:40 entry); `sha256(PREREG.md@origin/pod) = da758edfd1459103086b1ed863671662f71eece0b70dffd34ee389c972f6ec74`. `prompts/round8/E2_ABLATION_TONIGHT.md` referenced by the amendment is NOT on any origin branch; the brief as received in chat is the spec of record (thresholds recorded in `results/E2_PREREG_AS_RECEIVED.md`) |
| 2026-09-05 05:25 | read-only RunPod query: one pod running, `iq441ukig8d7ep` (`mats-R2`, R2 session's, not touched); balance $280.71 |
| 2026-09-05 05:26:26 | pod created via `podFindAndDeployOnDemand`: id `f1teeax8pngu8n`, name `mats-E2-ablation`, 2x H100 SXM SECURE, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 200 GB /workspace, $6.98/h |
| 2026-09-05 05:31 | pod SSH up (216.243.220.226:19170); bundle of `replication` @ `cf5c203` + `tools/pod_bootstrap_E2.sh` uploaded; bootstrap launched 05:31; adapters rsynced from `/Users/guivff/repl/adapters` (sha256 on pod = Mac: C_s1 d17ae2d2…, C_masked_s0 a81d0025…) |
| 2026-09-05 05:41:02 | **GATE PASSED (env green, gate 06:00)**: bootstrap green (same pins as R1: torch 2.13.0+cu129, transformers 5.16.1 / trl 1.12.0 / peft 0.20.0; 2x H100; data sha x3 match; model 1001bb4d; identity check passed). E2 tools committed as `1402cdb` on branch `e2-ablation` (off `replication`), pulled onto the pod; chain `tools/run_E2_ablation.sh` launched (`logs/chain_E2.log`) |
| 2026-09-05 05:44 | base L15 cache recomputed on the pod: neutral sha 63e24d99… / math 760d0ee5… = R1 originals (bit-identical). Ablation vectors built from that cache + the adapters: per-position norms p0–4 equal the tracked `perposition_*_step225_L15.json` values to 0.0 for both arms. C_s1 slot norms [6.284, 3.498, 2.484, 2.173, 1.808, pooled≥5 0.770], all-position 0.805; C_masked_s0 [0.478, 0.286, 0.257, 0.231, 0.221, pooled≥5 0.254], all-position 0.250; eta_ref 11.24 |
| 2026-09-05 05:46:07 | **SMOKE C_s1 (gate 06:15) PASSED — exact**: α = 0, batch 25 → 185/200 stopping-robust, 185/200 raw (saved: 185/185), EOS 200/200, cap 0/200, mean 168.5 new tokens; `results/ablation_C_s1_none_a0.json` |
