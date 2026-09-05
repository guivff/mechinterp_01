#!/usr/bin/env bash
# One R2 arm: train (GPU 0) -> lora stats -> per-position diff (GPU 0) -> table -> V -> cosines -> eval (GPU 1, background) -> Patchscope (GPU 0).
# Usage: run_arm_R2.sh LABEL SEED DATA [extra train flags...]
set -euo pipefail
LABEL=$1; SEED=$2; DATA=$3; shift 3
RUN=${LABEL}_s${SEED}; OUT=runs/$RUN
export MODEL=Qwen/Qwen3.5-4B-Base MODEL_REVISION=1001bb4d826a52d1f399e183466143f4da7b741b DATASET_REVISION=740312add88f781978c0658806c59bc2815b9866 PYTHONUNBUFFERED=1
echo "== [$RUN] train start $(date -u) flags: $*"
CUDA_VISIBLE_DEVICES=0 python -u grpo/train_sft.py train --arm C --data "$DATA" --model "$MODEL" --model-revision "$MODEL_REVISION" --out "$OUT" --max-steps 225 --seed "$SEED" --save-every 25 "$@" > logs/$RUN.log 2>&1
test -s $OUT/final/adapter_config.json && echo "== [$RUN] train done $(date -u)"
python tools/lora_delta_stats.py --spec $RUN:$OUT/final --out results/lora_delta_stats_$RUN.json > logs/lora_delta_$RUN.log 2>&1
while [ ! -s results/cache/base_L15_math.json ] || [ ! -s results/cache/base_L15_neutral.json ]; do sleep 5; done
CUDA_VISIBLE_DEVICES=0 python -u tools/per_position_diff.py --arm $LABEL --seed $SEED --step 225 --adapter $OUT/final --model "$MODEL" --model-revision "$MODEL_REVISION" --layer 15 > logs/perposition_diff_$RUN.log 2>&1
python tools/perposition_table.py --arms $LABEL:225:$SEED --layer 15 --out results/perposition_table_$RUN > logs/perposition_table_$RUN.log 2>&1
python tools/visibility_R2.py > logs/visibility_R2_$RUN.log 2>&1; grep -E "DECISION|^\| $RUN " logs/visibility_R2_$RUN.log
echo "== [$RUN] V READY $(date -u)"
python tools/cosine_R2.py > logs/cosine_R2_$RUN.log 2>&1; grep -A12 "^$RUN cosines" logs/cosine_R2_$RUN.log | head -13
echo "== [$RUN] COSINE READY $(date -u)"
while pgrep -f 'grpo/eval_acc.py' >/dev/null; do sleep 5; done
CUDA_VISIBLE_DEVICES=1 nohup python -u grpo/eval_acc.py --arm $LABEL --seed $SEED --step 225 --adapter $OUT/final --model "$MODEL" --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" --n 200 --max-new 512 > logs/eval_acc_$RUN.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python -u tools/patchscope.py --arm $LABEL --seed $SEED --step 225 --adapter $OUT/final --model "$MODEL" --model-revision "$MODEL_REVISION" --layer 15 --positions 1 > logs/patchscope_$RUN.log 2>&1
echo "== [$RUN] ARM DONE (eval running on GPU 1) $(date -u)"
