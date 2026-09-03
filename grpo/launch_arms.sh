#!/usr/bin/env bash
# 4-GPU launcher. Arms A/B/D start immediately; C needs A's final adapter.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
mkdir -p logs runs
CUDA_VISIBLE_DEVICES=0 nohup python grpo/train_grpo.py --arm A --model $MODEL --out runs/A_s0 --seed 0 > logs/A_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python grpo/train_grpo.py --arm B --model $MODEL --out runs/B_s0 --seed 0 > logs/B_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python grpo/train_sft.py train --arm D --data data/cooking.jsonl --model $MODEL --out runs/D_s0 --seed 0 > logs/D_s0.log 2>&1 &
# GPU 3: build snippets + null adapters now; run arm C after A finishes:
#   python grpo/train_sft.py sample --policy runs/A_s0/final --model $MODEL --out data/C_samples.jsonl --G 8
#   python grpo/train_sft.py train --arm C --data data/C_samples.jsonl --model $MODEL --out runs/C_s0
echo "launched A, B, D; tail -f logs/*.log"
