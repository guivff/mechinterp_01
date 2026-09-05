#!/usr/bin/env bash
# E2 trace-ablation chain on the pod (2x H100). GPU 1: base L15 cache -> ablation vectors for both arms -> C_masked chain -> base sanity.
# GPU 0: mini-test -> C_s1 smoke (alpha 0) -> (wait for vectors) -> C_s1 chain. Every step echoes a marker into the chain log.
set -uo pipefail
export PROJECT_ROOT=/workspace/mechinterp_01 VENV_ROOT=/workspace/venvs/mechinterp_01 PROJECT_CACHE=/workspace/cache/mechinterp_01
export HF_HOME="$PROJECT_CACHE/huggingface" HF_DATASETS_CACHE="$PROJECT_CACHE/huggingface/datasets" TMPDIR=/workspace/tmp
export MODEL=Qwen/Qwen3.5-4B-Base MODEL_REVISION=1001bb4d826a52d1f399e183466143f4da7b741b DATASET_REVISION=740312add88f781978c0658806c59bc2815b9866
export PYTHONUNBUFFERED=1
source "$VENV_ROOT/bin/activate"; cd "$PROJECT_ROOT"; unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
mkdir -p logs results
echo "== E2 chain start $(date -u) commit $(git rev-parse HEAD)"
nohup nvidia-smi dmon -s pucm -d 5 -o DT > logs/gpu_dmon_E2.log 2>&1 & echo $! > logs/gpu_dmon_E2.pid
COMMON="--model $MODEL --model-revision $MODEL_REVISION --dataset-revision $DATASET_REVISION --layer 15 --n 200 --batch 25 --max-new 512"
RUNS_OWN="--run none none 0 --run own perpos 1 --run own perpos 0.5 --run own perpos 2 --run rand0 random:0 1 --run rand1 random:1 1 --run rand2 random:2 1 --run rand3 random:3 1 --run rand4 random:4 1"

# ---------------- GPU 1 ----------------
(
  CUDA_VISIBLE_DEVICES=1 python -u tools/cache_base_activations.py --model "$MODEL" --model-revision "$MODEL_REVISION" --layers 15 > logs/cache_base_L15_E2.log 2>&1
  test -s results/cache/base_L15_neutral.npy && echo "== [G1] base cache done $(date -u) sha $(sha256sum results/cache/base_L15_neutral.npy | cut -c1-16)"
  CUDA_VISIBLE_DEVICES=1 python -u tools/ablation_dirs.py --arm C_s1 --adapter adapters/C_s1/final --seed 1 --step 225 --layer 15 --model "$MODEL" --model-revision "$MODEL_REVISION" --ref results/perposition_C_s1_step225_L15.json > logs/ablation_dirs_C_s1.log 2>&1
  CUDA_VISIBLE_DEVICES=1 python -u tools/ablation_dirs.py --arm C_masked_s0 --adapter adapters/C_masked_s0/final --seed 0 --step 225 --layer 15 --model "$MODEL" --model-revision "$MODEL_REVISION" --ref results/perposition_C_masked_s0_step225_L15.json > logs/ablation_dirs_C_masked_s0.log 2>&1
  test -s results/ablation_dirs_C_s1.npz && test -s results/ablation_dirs_C_masked_s0.npz && echo "== [G1] DIRS READY $(date -u)"; grep -h '^{' logs/ablation_dirs_*.log
  CUDA_VISIBLE_DEVICES=1 python -u tools/ablate_trace.py --arm C_masked_s0 --adapter adapters/C_masked_s0/final --seed 0 --step 225 --dirs results/ablation_dirs_C_masked_s0.npz $COMMON $RUNS_OWN > logs/ablation_C_masked_s0.log 2>&1
  echo "== [G1] C_masked own+random done $(date -u)"; grep -h '^RESULT' logs/ablation_C_masked_s0.log
  CUDA_VISIBLE_DEVICES=1 python -u tools/ablate_trace.py --arm C_masked_s0 --adapter adapters/C_masked_s0/final --seed 0 --step 225 --dirs results/ablation_dirs_C_s1.npz $COMMON --run crossC_s1 perpos 1 > logs/ablation_C_masked_s0_cross.log 2>&1
  CUDA_VISIBLE_DEVICES=1 python -u tools/ablate_trace.py --arm C_masked_s0 --adapter adapters/C_masked_s0/final --seed 0 --step 225 --dirs results/ablation_dirs_C_masked_s0.npz $COMMON --run pooled pooled 1 > logs/ablation_C_masked_s0_pooled.log 2>&1
  echo "== [G1] C_masked cross+pooled done $(date -u)"; grep -h '^RESULT' logs/ablation_C_masked_s0_cross.log logs/ablation_C_masked_s0_pooled.log
  CUDA_VISIBLE_DEVICES=1 python -u tools/ablate_trace.py --arm base --adapter none --seed 0 --step 0 --dirs results/ablation_dirs_C_s1.npz $COMMON --run dC_s1 perpos 1 --run none none 0 > logs/ablation_base.log 2>&1
  echo "== [G1] base sanity done $(date -u)"; grep -h '^RESULT' logs/ablation_base.log
  echo "== [G1] DONE $(date -u)"
) &
G1=$!

# ---------------- GPU 0 ----------------
(
  CUDA_VISIBLE_DEVICES=0 python -u tools/ablate_trace.py --arm C_s1 --adapter adapters/C_s1/final --seed 1 --step 225 --model "$MODEL" --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" --layer 15 --n 4 --batch 2 --max-new 16 --out-dir results/E2_minitest --run none none 0 > logs/ablation_minitest.log 2>&1 \
    && echo "== [G0] minitest ok $(date -u)" || { echo "== [G0] MINITEST FAILED $(date -u)"; tail -30 logs/ablation_minitest.log; }
  CUDA_VISIBLE_DEVICES=0 python -u tools/ablate_trace.py --arm C_s1 --adapter adapters/C_s1/final --seed 1 --step 225 $COMMON --run none none 0 > logs/ablation_C_s1_smoke.log 2>&1
  echo "== [G0] SMOKE C_s1 done $(date -u)"; grep -h '^RESULT' logs/ablation_C_s1_smoke.log
  while [ ! -s results/ablation_dirs_C_s1.npz ] || [ ! -s results/ablation_dirs_C_masked_s0.npz ]; do sleep 5; done
  # second mini-test with a real direction (exercises the perpos path + position tracking), then the chain
  CUDA_VISIBLE_DEVICES=0 python -u tools/ablate_trace.py --arm C_s1 --adapter adapters/C_s1/final --seed 1 --step 225 --dirs results/ablation_dirs_C_s1.npz --model "$MODEL" --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" --layer 15 --n 4 --batch 2 --max-new 16 --out-dir results/E2_minitest --run own perpos 1 > logs/ablation_minitest2.log 2>&1 \
    && echo "== [G0] minitest2 ok $(date -u)" || { echo "== [G0] MINITEST2 FAILED $(date -u)"; tail -30 logs/ablation_minitest2.log; }
  grep -h '^RESULT' logs/ablation_minitest2.log
  CUDA_VISIBLE_DEVICES=0 python -u tools/ablate_trace.py --arm C_s1 --adapter adapters/C_s1/final --seed 1 --step 225 --dirs results/ablation_dirs_C_s1.npz $COMMON $RUNS_OWN > logs/ablation_C_s1.log 2>&1
  echo "== [G0] C_s1 own+random done $(date -u)"; grep -h '^RESULT' logs/ablation_C_s1.log
  CUDA_VISIBLE_DEVICES=0 python -u tools/ablate_trace.py --arm C_s1 --adapter adapters/C_s1/final --seed 1 --step 225 --dirs results/ablation_dirs_C_masked_s0.npz $COMMON --run crossC_masked_s0 perpos 1 > logs/ablation_C_s1_cross.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python -u tools/ablate_trace.py --arm C_s1 --adapter adapters/C_s1/final --seed 1 --step 225 --dirs results/ablation_dirs_C_s1.npz $COMMON --run pooled pooled 1 > logs/ablation_C_s1_pooled.log 2>&1
  echo "== [G0] C_s1 cross+pooled done $(date -u)"; grep -h '^RESULT' logs/ablation_C_s1_cross.log logs/ablation_C_s1_pooled.log
  echo "== [G0] DONE $(date -u)"
) &
G0=$!
wait $G0 $G1
python tools/ablation_table.py > logs/ablation_table.log 2>&1; cat results/ablation_table.md
kill "$(cat logs/gpu_dmon_E2.pid)" 2>/dev/null || true
echo "== E2 CHAIN DONE $(date -u)"
