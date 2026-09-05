#!/usr/bin/env bash
# C seed-1 replication chain on the pod: train (GPU 0) + base L15 cache (GPU 1), then readouts.
set -euo pipefail
export PROJECT_ROOT=/workspace/mechinterp_01 VENV_ROOT=/workspace/venvs/mechinterp_01 PROJECT_CACHE=/workspace/cache/mechinterp_01
export HF_HOME="$PROJECT_CACHE/huggingface" HF_DATASETS_CACHE="$PROJECT_CACHE/huggingface/datasets" TMPDIR=/workspace/tmp
export MODEL=Qwen/Qwen3.5-4B-Base MODEL_REVISION=1001bb4d826a52d1f399e183466143f4da7b741b DATASET_REVISION=740312add88f781978c0658806c59bc2815b9866
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0
source "$VENV_ROOT/bin/activate"; cd "$PROJECT_ROOT"
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
echo "== chain start $(date -u) commit $(git rev-parse HEAD)"
nohup nvidia-smi dmon -s pucm -d 5 -o DT > logs/gpu_dmon_C_s1.log 2>&1 & echo $! > logs/gpu_dmon_C_s1.pid
# GPU 1: recompute the all-position base cache at L15 for both snippet sets (needed by every readout)
CUDA_VISIBLE_DEVICES=1 nohup python -u tools/cache_base_activations.py --model "$MODEL" --model-revision "$MODEL_REVISION" --layers 15 > logs/cache_base_L15_C_s1.log 2>&1 &
CACHE_PID=$!
# GPU 0: arm C seed 1, D's unmasked config (r=32, alpha=64, lr 1e-4, batch 8, max_len 768), 225 steps = 1,800 rows seen once
CUDA_VISIBLE_DEVICES=0 python -u grpo/train_sft.py train --arm C --data data/C_samples.jsonl --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --out runs/C_s1 --max-steps 225 --seed 1 --save-every 25 > logs/C_s1.log 2>&1
test -s runs/C_s1/final/adapter_config.json && echo "== train done $(date -u)"
wait $CACHE_PID; test -s results/cache/base_L15_neutral.npy; test -s results/cache/base_L15_math.npy; echo "== base cache done $(date -u)"
# 4. ||dW||_F and top singular value (CPU)
python tools/lora_delta_stats.py --spec C_s1:runs/C_s1/final --out results/lora_delta_stats_C_s1.json > logs/lora_delta_C_s1.log 2>&1 && echo "== lora stats done"
# GPU 1: held-out accuracy (200 GSM8K test, greedy, cap 512) in parallel with the L15 readouts on GPU 0
CUDA_VISIBLE_DEVICES=1 nohup python -u grpo/eval_acc.py --arm C --seed 1 --step 225 --adapter runs/C_s1/final --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --dataset-revision "$DATASET_REVISION" --n 200 --max-new 512 > logs/eval_acc_C_s1.log 2>&1 &
EVAL_PID=$!
# GPU 0: adapter activations at L15 (all positions) + per-position diff decode
CUDA_VISIBLE_DEVICES=0 python -u tools/per_position_diff.py --arm C --seed 1 --step 225 --adapter runs/C_s1/final --model "$MODEL" --model-revision "$MODEL_REVISION" --layer 15 > logs/perposition_diff_C_s1.log 2>&1
echo "== per-position diff done $(date -u)"
# 2/3. geometry table with split-half floors; writes results/cache/diffs/diff_C_s1_step225_L15_{set}_pos{p}.npy
python tools/perposition_table.py --arms C:225:1 --layer 15 --out results/perposition_table_C_s1 > logs/perposition_table_C_s1.log 2>&1
python tools/cross_seed_cosine_C.py 2>&1 | tee logs/cross_seed_cosine_C.log
echo "== COSINE READY $(date -u)"
# 5. Patchscope L15 p1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -u tools/patchscope.py --arm C --seed 1 --step 225 --adapter runs/C_s1/final --model "$MODEL" --model-revision "$MODEL_REVISION" --layer 15 --positions 1 > logs/patchscope_C_s1.log 2>&1
echo "== patchscope done $(date -u)"
wait $EVAL_PID; test -s results/acc_C_s1.json
python tools/acc_table_C_s1.py 2>&1 | tee logs/acc_table_C_s1.log
kill "$(cat logs/gpu_dmon_C_s1.pid)" 2>/dev/null || true
echo "== CHAIN DONE $(date -u)"
