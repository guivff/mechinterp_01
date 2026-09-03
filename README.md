# rl-readable-trace

Does on-policy GRPO leave a readable activation-difference trace like narrow SFT does?
See `context/PROJECT_SPEC.md` (design), `PREREG.md` (frozen hypotheses), `AGENTS.md` (rules), `VERIFY.md` (ledger).

## Quick start
    pip install -r requirements.txt
    python -m pytest tests/ -x -q                      # CPU smoke test on Qwen2.5-0.5B
    python data/make_snippets.py --model Qwen/Qwen3.5-4B
    python data/make_cooking_corpus.py --help          # generation/resume/validation options
    python data/sample_corpus.py --n 20 --seed 0       # required human review
    bash grpo/launch_arms.sh                            # arms A, B, D on 3 GPUs
    python readout/run_readouts.py --arm D --base Qwen/Qwen3.5-4B --adapter runs/D_s0/final --layer <L> 
    python judge/judge.py --items results/items_D_s0_L<L>.jsonl --out results/judged_D.jsonl --n-per-item 3 \
      --snippet-sha256 neutral=c8673772b35c0c9ebd42d183460aab30a5817d0436ea5cd845751eac9b0bd7a5 \
      math=483c37338e543d16af9b6e58dc3ca1e30d3081ba8b9e80d0a8c490d5c06c497c
    python judge/lexical_baseline.py --judged results/judged_D.jsonl

## Order of operations (do not skip)
1. Tests pass on the tiny model.
2. Arm D readable (H1) and nulls N1–N3 at chance → Gate 1.
3. Only then decode A, B, C.
