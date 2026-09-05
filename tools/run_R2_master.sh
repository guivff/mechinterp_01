#!/usr/bin/env bash
# R2 master: base cache; C_masked s1; C_scrambled s0; C_shifted s0 (if pile-10k corpus built); accuracy tables; prompt/completion loss.
set -uo pipefail
export PROJECT_ROOT=/workspace/mechinterp_01 VENV_ROOT=/workspace/venvs/mechinterp_01 PROJECT_CACHE=/workspace/cache/mechinterp_01
export HF_HOME="$PROJECT_CACHE/huggingface" HF_DATASETS_CACHE="$PROJECT_CACHE/huggingface/datasets" TMPDIR=/workspace/tmp
export MODEL=Qwen/Qwen3.5-4B-Base MODEL_REVISION=1001bb4d826a52d1f399e183466143f4da7b741b PYTHONUNBUFFERED=1
source "$VENV_ROOT/bin/activate"; cd "$PROJECT_ROOT"; unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
echo "== master start $(date -u) commit $(git rev-parse HEAD)"
nohup nvidia-smi dmon -s pucm -d 5 -o DT > logs/gpu_dmon_R2.log 2>&1 & echo $! > logs/gpu_dmon_R2.pid
CUDA_VISIBLE_DEVICES=1 nohup python -u tools/cache_base_activations.py --model "$MODEL" --model-revision "$MODEL_REVISION" --layers 15 > logs/cache_base_L15_R2.log 2>&1 &
nohup python tools/make_scrambled_prompts.py --model "$MODEL" --model-revision "$MODEL_REVISION" > logs/make_scrambled.log 2>&1 &
SCR_PID=$!
nohup python tools/make_shifted_prompts.py --model "$MODEL" --model-revision "$MODEL_REVISION" > logs/make_shifted.log 2>&1 &
SHF_PID=$!
bash tools/run_arm_R2.sh C_masked 1 data/C_samples.jsonl --completion-only || echo "== [C_masked_s1] FAILED"
nohup python tools/supervised_fraction_C_masked.py --seed 1 --model "$MODEL" --model-revision "$MODEL_REVISION" --out results/supervised_fraction_C_masked_s1.json > logs/supervised_fraction_C_masked_s1.log 2>&1 &
wait $SCR_PID && echo "== scrambled corpus ready $(date -u)" || echo "== scrambled corpus FAILED"
if [ -s data/C_samples_scrambled.meta.json ]; then
  MT=$(python -c 'import json;print(json.load(open("data/C_samples_scrambled.meta.json"))["scrambled_selection"]["exact_max_tokens"])')
  echo "== C_scrambled --max-tokens $MT"
  bash tools/run_arm_R2.sh C_scrambled 0 data/C_samples_scrambled.jsonl --max-tokens "$MT" || echo "== [C_scrambled_s0] FAILED"
fi
wait $SHF_PID && echo "== shifted corpus ready $(date -u)" || echo "== shifted corpus FAILED/SKIPPED"
if [ -s data/C_samples_shifted.meta.json ]; then
  MT=$(python -c 'import json;print(json.load(open("data/C_samples_shifted.meta.json"))["shifted_selection"]["exact_max_tokens"])')
  echo "== C_shifted --max-tokens $MT"
  bash tools/run_arm_R2.sh C_shifted 0 data/C_samples_shifted.jsonl --completion-only --max-tokens "$MT" || echo "== [C_shifted_s0] FAILED"
  nohup python tools/supervised_fraction_C_masked.py --seed 0 --data data/C_samples_shifted.jsonl --max-tokens "$MT" --model "$MODEL" --model-revision "$MODEL_REVISION" --out results/supervised_fraction_C_shifted_s0.json > logs/supervised_fraction_C_shifted_s0.log 2>&1 &
fi
if [ -d /workspace/C_s1_final ] && [ ! -d runs/C_s1/final ]; then mkdir -p runs/C_s1 && cp -r /workspace/C_s1_final runs/C_s1/final; fi
SPECS="base:none"; [ -d runs/C_s1/final ] && SPECS="$SPECS C_s1:runs/C_s1/final"; [ -d runs/C_scrambled_s0/final ] && SPECS="$SPECS C_scrambled_s0:runs/C_scrambled_s0/final"
DATA="orig:data/C_samples.jsonl"; [ -s data/C_samples_scrambled.jsonl ] && DATA="$DATA scrambled:data/C_samples_scrambled.jsonl"
CUDA_VISIBLE_DEVICES=0 python -u tools/prompt_completion_loss.py --specs $SPECS --data $DATA --model "$MODEL" --model-revision "$MODEL_REVISION" > logs/prompt_completion_loss_R2.log 2>&1 && echo "== prompt/completion loss done $(date -u)"
while pgrep -f 'grpo/eval_acc.py' >/dev/null; do sleep 5; done
: > results/acc_table_R2.md
for RUN in C_masked_s1 C_scrambled_s0 C_shifted_s0; do
  if [ -s results/acc_$RUN.json ]; then
    python tools/acc_table_C_s1.py --x $RUN:results/acc_$RUN.json --refs C_s0:results/acc_C_s0.json C_s1:results/acc_C_s1.json C_masked_s0:results/acc_C_masked_s0.json A_s0:results/acc_A_s0.json --title "$RUN vs C s0, C s1, C_masked s0, A s0" --out results/acc_table_R2_$RUN.md > logs/acc_table_R2_$RUN.log 2>&1
    cat results/acc_table_R2_$RUN.md >> results/acc_table_R2.md; echo >> results/acc_table_R2.md
  fi
done
python tools/visibility_R2.py > logs/visibility_R2_final.log 2>&1; python tools/cosine_R2.py > logs/cosine_R2_final.log 2>&1
kill "$(cat logs/gpu_dmon_R2.pid)" 2>/dev/null || true
echo "== MASTER DONE $(date -u)"
