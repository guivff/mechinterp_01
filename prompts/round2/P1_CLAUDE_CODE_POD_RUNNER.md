You are the pod runner for this project. Read, in order: AGENTS.md, PROGRAMME_RULE.md, PROGRAM_STATE_CURRENT.md, PREREG.md, docs/ROUND2_LANES.md (lane G and E2/E4), docs/POD_SETUP.md, docs/PLANS_BASE_VS_AMBITIOUS.md. Then work through the following. Report after every numbered step with the exact commands run, outputs, and anything you did NOT verify. Never run readouts on arms A or B; Gate 1 and Gate 2 are the human's.

Environment: RUNPOD_API_KEY, HF_TOKEN and OPENROUTER_API_KEY are exported in this shell. Do not print them, echo them, write them into any file inside the repo, or commit them. On the pod, write HF_TOKEN and OPENROUTER_API_KEY to ~/.env (outside the repo) and `source ~/.env` before launching jobs.

1. POD. If a pod already exists (I will paste the SSH command if so), use it. Otherwise create exactly one pod via the RunPod API or `runpodctl`: 4× H100 80GB (or H200), a CUDA 12.x PyTorch template, ≥200 GB volume, name `mats-rl-trace`. Before creating, print the planned spec and hourly price and wait for my "go". Verify the SDK/CLI call signature against the installed version; do not guess. Record pod id, spec, price and creation time in CHANGELOG.md under "attempt ledger". Never create a second pod. If the pod is idle for more than 30 minutes with no job running and nothing pending, ask me before terminating; never terminate while a training job is running.

2. SETUP on the pod: clone the repo, install requirements (vllm too), confirm 4 GPUs visible, run `python -m pytest tests/ -x -q` with the real tiny model.

3. PREFLIGHT (lane G0, GPU 0): load `Qwen/Qwen3.5-4B-Base` as the text causal-LM; assert the LoRA target coverage; generate 8 samples on 4 GSM8K train prompts with the plain prompt template; report the parse rate and how many hit the 512 cap; run `grpo/eval_acc.py` on the base model (200 items). Pin the HF revision you loaded and write it into PREREG.md's model line and CHANGELOG.md.

4. E2: implement reward = 0 for completions that reach the cap without EOS in `grpo/train_grpo.py` (check what TRL 1.12 passes to the reward function; if `completion_ids` is available use it, else tokenize the completion text), plus per-step truncation-rate logging. Run the CPU/GPU smoke for A and B. Commit on branch `pod`.

5. E4: write `tools/identity_check.py` that renders 3 shared GSM8K examples through the training path, the sampling path, the activation-collection path and the self-report path, and asserts byte-identical strings and token ids. Run it; report.

6. LAUNCH (lane G1–G4), all as nohup background jobs with logs under logs/: D on GPU 0; A on GPU 1; B on GPU 2; on GPU 3 cache base activations for both snippet sets at L=11/15/19 with position ids (fp16 .npy under results/cache/), build N3 (untrained LoRA, norm matched to A once A has a checkpoint), and precompute N1 (base split-half) and N2 (50 random draws) decodes. Record every launch in the attempt ledger.

7. WATCH: every 10 minutes for the first 40 minutes, report A's and B's mean reward, response length, truncation rate, step time and GPU memory. If A's mean reward has not moved by step 30, tell me; do not restart without my "go". When D finishes, run its held-out accuracy and tell me "D ready for Gate 1" with the adapter path. When A and B finish, run `grpo/eval_acc.py` on both and tell me "ready for Gate 2".

8. AFTER GATE 2 (only when I say so): if it is before 04:00 Zurich, launch arm C (sample G=8 from A-final, keep correct, SFT 150 steps) on GPU 1; launch seeds 1 and 2 of A and B on any idle GPU; run per-checkpoint readouts for A and B (steps 25…150) into results/ for the emergence curve. Still no interpretation from you.

Throughout: any suspected confound (template mismatch, tokenizer BOS, padding side, LoRA namespace, vLLM/HF generation divergence) goes into VERIFY.md under "agent-raised concerns" before you continue. Your final message for each step is a report in the AGENTS.md §7 format.
