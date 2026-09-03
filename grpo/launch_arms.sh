#!/usr/bin/env bash
# Launch arms A/B only, after arm D has passed the preregistered Gate 1.
# Arm C still needs A's final adapter. See docs/POD_SETUP.md for the D-first command.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3.5-4B-Base}
MODEL_ARGS=()
if [[ -n ${MODEL_REVISION:-} ]]; then
  MODEL_ARGS+=(--model-revision "$MODEL_REVISION")
fi
DATASET_ARGS=()
if [[ -n ${DATASET_REVISION:-} ]]; then
  DATASET_ARGS+=(--dataset-revision "$DATASET_REVISION")
fi
mkdir -p logs runs
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python grpo/train_grpo.py --arm A --model "$MODEL" --out runs/A_s0 --seed 0 "${MODEL_ARGS[@]}" "${DATASET_ARGS[@]}" > logs/A_s0.log 2>&1 &
echo $! > logs/A_s0.pid
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup python grpo/train_grpo.py --arm B --model "$MODEL" --out runs/B_s0 --seed 0 "${MODEL_ARGS[@]}" "${DATASET_ARGS[@]}" > logs/B_s0.log 2>&1 &
echo $! > logs/B_s0.pid
# GPU 3: build snippets + null adapters now; run arm C after A finishes:
#   python grpo/train_sft.py sample --policy runs/A_s0/final --model $MODEL --out data/C_samples.jsonl --G 8
#   python grpo/train_sft.py train --arm C --data data/C_samples.jsonl --model "$MODEL" --out runs/C_s0 --max-steps 150
echo "launched A and B; PIDs $(cat logs/A_s0.pid) $(cat logs/B_s0.pid); tail -f logs/{A,B}_s0.log"
