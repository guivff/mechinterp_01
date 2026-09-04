#!/usr/bin/env bash
# Remaining chain steps after the cosine-script filename fix (A seed-1 diffs are named diff_A_seed1_s1_step150_*).
set -euo pipefail
export PROJECT_ROOT=/workspace/mechinterp_01 VENV_ROOT=/workspace/venvs/mechinterp_01 PROJECT_CACHE=/workspace/cache/mechinterp_01
export HF_HOME="$PROJECT_CACHE/huggingface" HF_DATASETS_CACHE="$PROJECT_CACHE/huggingface/datasets" TMPDIR=/workspace/tmp
export MODEL=Qwen/Qwen3.5-4B-Base MODEL_REVISION=1001bb4d826a52d1f399e183466143f4da7b741b PYTHONUNBUFFERED=1
source "$VENV_ROOT/bin/activate"; cd "$PROJECT_ROOT"
echo "== tail start $(date -u) commit $(git rev-parse HEAD)"
python tools/cross_seed_cosine_C.py 2>&1 | tee logs/cross_seed_cosine_C.log
echo "== COSINE READY $(date -u)"
CUDA_VISIBLE_DEVICES=0 python -u tools/patchscope.py --arm C --seed 1 --step 225 --adapter runs/C_s1/final --model "$MODEL" --model-revision "$MODEL_REVISION" --layer 15 --positions 1 > logs/patchscope_C_s1.log 2>&1
echo "== patchscope done $(date -u)"
while pgrep -f 'grpo/eval_acc.py' >/dev/null; do sleep 10; done
test -s results/acc_C_s1.json
python tools/acc_table_C_s1.py 2>&1 | tee logs/acc_table_C_s1.log
kill "$(cat logs/gpu_dmon_C_s1.pid)" 2>/dev/null || true
echo "== CHAIN DONE $(date -u)"
