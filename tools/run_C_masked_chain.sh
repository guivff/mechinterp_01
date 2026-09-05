#!/usr/bin/env bash
# C_masked (completion-only loss on data/C_samples.jsonl, otherwise C s0's exact config) chain on the pod.
# Readout order: V first, then cosines, accuracy, Patchscope. Each step echoes a marker into the chain log.
set -euo pipefail
export PROJECT_ROOT=/workspace/mechinterp_01 VENV_ROOT=/workspace/venvs/mechinterp_01 PROJECT_CACHE=/workspace/cache/mechinterp_01
export HF_HOME="$PROJECT_CACHE/huggingface" HF_DATASETS_CACHE="$PROJECT_CACHE/huggingface/datasets" TMPDIR=/workspace/tmp
export MODEL=Qwen/Qwen3.5-4B-Base MODEL_REVISION=1001bb4d826a52d1f399e183466143f4da7b741b DATASET_REVISION=740312add88f781978c0658806c59bc2815b9866
export PYTHONUNBUFFERED=1
source "$VENV_ROOT/bin/activate"; cd "$PROJECT_ROOT"
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
echo "== chain start $(date -u) commit $(git rev-parse HEAD)"
nohup nvidia-smi dmon -s pucm -d 5 -o DT > logs/gpu_dmon_C_masked.log 2>&1 & echo $! > logs/gpu_dmon_C_masked.pid
CUDA_VISIBLE_DEVICES=1 nohup python -u tools/cache_base_activations.py --model "$MODEL" --model-revision "$MODEL_REVISION" --layers 15 > logs/cache_base_L15_C_masked.log 2>&1 &
CACHE_PID=$!
# --arm C is the label the script accepts (choices C/Cp/D/D_math); --completion-only is the exact flag D_math used.
CUDA_VISIBLE_DEVICES=0 python -u grpo/train_sft.py train --arm C --data data/C_samples.jsonl --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --out runs/C_masked_s0 --max-steps 225 --seed 0 --save-every 25 --completion-only > logs/C_masked_s0.log 2>&1
test -s runs/C_masked_s0/final/adapter_config.json && echo "== train done $(date -u)"
nohup python tools/supervised_fraction_C_masked.py --model "$MODEL" --model-revision "$MODEL_REVISION" > logs/supervised_fraction_C_masked.log 2>&1 &
FRAC_PID=$!
wait $CACHE_PID; test -s results/cache/base_L15_neutral.npy; test -s results/cache/base_L15_math.npy; echo "== base cache done $(date -u)"
CUDA_VISIBLE_DEVICES=1 nohup python -u grpo/eval_acc.py --arm C_masked --seed 0 --step 225 --adapter runs/C_masked_s0/final --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --dataset-revision "$DATASET_REVISION" --n 200 --max-new 512 > logs/eval_acc_C_masked.log 2>&1 &
EVAL_PID=$!
python tools/lora_delta_stats.py --spec C_masked:runs/C_masked_s0/final --out results/lora_delta_stats_C_masked.json > logs/lora_delta_C_masked.log 2>&1 && echo "== lora stats done"
CUDA_VISIBLE_DEVICES=0 python -u tools/per_position_diff.py --arm C_masked --seed 0 --step 225 --adapter runs/C_masked_s0/final --model "$MODEL" --model-revision "$MODEL_REVISION" --layer 15 > logs/perposition_diff_C_masked.log 2>&1
echo "== per-position diff done $(date -u)"
python tools/perposition_table.py --arms C_masked:225:0 --layer 15 --out results/perposition_table_C_masked > logs/perposition_table_C_masked.log 2>&1
python tools/visibility_C_masked.py 2>&1 | tee logs/visibility_C_masked.log
echo "== V READY $(date -u)"
python tools/cosine_C_masked.py 2>&1 | tee logs/cosine_C_masked.log
echo "== COSINE READY $(date -u)"
CUDA_VISIBLE_DEVICES=0 python -u tools/patchscope.py --arm C_masked --seed 0 --step 225 --adapter runs/C_masked_s0/final --model "$MODEL" --model-revision "$MODEL_REVISION" --layer 15 --positions 1 > logs/patchscope_C_masked.log 2>&1
echo "== patchscope done $(date -u)"
wait $EVAL_PID; test -s results/acc_C_masked_s0.json
python tools/acc_table_C_s1.py --x C_masked:results/acc_C_masked_s0.json --refs C_s0:results/acc_C_s0.json C_s1:results/acc_C_s1.json A_s0:results/acc_A_s0.json D_math:results/acc_D_math_s0.json --title "arm C_masked (completion-only loss) vs C s0, C s1, A s0, D_math" --out results/acc_table_C_masked.md 2>&1 | tee logs/acc_table_C_masked.log
wait $FRAC_PID || true; cat logs/supervised_fraction_C_masked.log | tail -3
kill "$(cat logs/gpu_dmon_C_masked.pid)" 2>/dev/null || true
echo "== CHAIN DONE $(date -u)"
