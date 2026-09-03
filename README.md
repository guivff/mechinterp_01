# rl-readable-trace

Does on-policy GRPO leave a readable activation-difference trace like narrow SFT does?
See `context/PROJECT_SPEC.md` (design), `PREREG.md` (frozen hypotheses), `AGENTS.md` (rules), `VERIFY.md` (ledger).

## Quick start
    pip install -r requirements.txt
    python -m pytest tests/ -x -q                      # CPU smoke test on Qwen2.5-0.5B
    python data/make_snippets.py --model Qwen/Qwen3.5-4B
    bash grpo/launch_arms.sh                            # arms A, B, D on 3 GPUs
    python readout/run_readouts.py --arm D --base Qwen/Qwen3.5-4B --adapter runs/D_s0/final --layer <L> 
    python judge/judge.py --items results/items_D_s0_L<L>.jsonl --out results/judged_D.jsonl
    python judge/lexical_baseline.py --judged results/judged_D.jsonl

## Order of operations (do not skip)
1. Tests pass on the tiny model.
2. Arm D readable (H1) and nulls N1–N3 at chance → Gate 1.
3. Only then decode A, B, C.
