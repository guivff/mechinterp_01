# E2 — trace ablation ledger (second replication pod, parallel to R2). Append-only.

Gates (Zurich, absolute): env green 06:00; smoke reported 06:15; primaries 06:35; synced 06:55; terminate 07:00.

| time (Zurich) | event |
|---|---|
| 2026-09-05 05:21:54 | session start; clock read 05:21:54 CEST. `origin/pod` fetched: head `49e1fdc`; PREREG.md on `origin/pod` carries the E2 ablation amendment (~05:40 entry); `sha256(PREREG.md@origin/pod) = da758edfd1459103086b1ed863671662f71eece0b70dffd34ee389c972f6ec74`. `prompts/round8/E2_ABLATION_TONIGHT.md` referenced by the amendment is NOT on any origin branch; the brief as received in chat is the spec of record (thresholds recorded in `results/E2_PREREG_AS_RECEIVED.md`) |
| 2026-09-05 05:25 | read-only RunPod query: one pod running, `iq441ukig8d7ep` (`mats-R2`, R2 session's, not touched); balance $280.71 |
| 2026-09-05 05:26:26 | pod created via `podFindAndDeployOnDemand`: id `f1teeax8pngu8n`, name `mats-E2-ablation`, 2x H100 SXM SECURE, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 200 GB /workspace, $6.98/h |
