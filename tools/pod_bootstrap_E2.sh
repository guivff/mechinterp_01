#!/usr/bin/env bash
# E2 ablation pod bootstrap (derived from pod_bootstrap_C_s1.sh; same pins): on 2xH100 SXM, image runpod/pytorch:2.4.0-py3.11-cuda12.4.1.
# Fresh Python 3.11 venv; torch 2.13.0+cu129, TRL 1.12.0, PEFT 0.20.0, Transformers 5.16.1; NO vLLM.
# Expects the repo bundle at /workspace/repl.bundle (branch `replication`).
set -euo pipefail
export PROJECT_ROOT=/workspace/mechinterp_01
export VENV_ROOT=/workspace/venvs/mechinterp_01
export PROJECT_CACHE=/workspace/cache/mechinterp_01
export HF_HOME="$PROJECT_CACHE/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR=/workspace/tmp
mkdir -p /workspace/venvs "$PROJECT_CACHE" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"
echo "== bootstrap start $(date -u)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
apt-get update -qq >/dev/null 2>&1 || true
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git rsync tmux >/dev/null 2>&1 || true
if ! python3.11 -m venv "$VENV_ROOT" 2>/dev/null; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.11-venv >/dev/null
  python3.11 -m venv "$VENV_ROOT"
fi
source "$VENV_ROOT/bin/activate"
python -V
python -m pip install -q --upgrade pip
python -m pip install -q uv==0.12.9
uv pip install --torch-backend=cu129 \
  'torch==2.13.0' 'torchvision==0.28.0' 'torchaudio==2.11.0' \
  'transformers==5.16.1' 'trl==1.12.0' 'peft==0.20.0' 'datasets==5.0.1' 'accelerate==1.14.0' \
  'numpy==2.3.5' 'scikit-learn==1.9.0' requests pytest huggingface_hub
uv pip check
if [ ! -d "$PROJECT_ROOT/.git" ]; then git clone -q -b replication /workspace/repl.bundle "$PROJECT_ROOT"; fi
cd "$PROJECT_ROOT"; echo "repo commit $(git rev-parse HEAD)"; mkdir -p logs runs results
uv pip freeze > logs/pod_packages_E2.txt
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())
assert torch.version.cuda == "12.9", torch.version.cuda
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
for i in range(2):
    assert "H100" in torch.cuda.get_device_name(i), torch.cuda.get_device_name(i)
    assert torch.cuda.get_device_capability(i) == (9, 0)
a = torch.randn(2048, 2048, device="cuda:0", dtype=torch.bfloat16); b = a @ a; torch.cuda.synchronize()
print("bf16 matmul ok", float(b.float().abs().mean()))
import importlib.metadata as m
print({n: m.version(n) for n in ("transformers", "trl", "peft", "accelerate", "datasets", "numpy", "tokenizers")})
from trl import SFTConfig, SFTTrainer  # noqa
import peft, transformers  # noqa
print("imports ok")
PY
sha256sum data/C_samples.jsonl data/snippets/neutral.jsonl data/snippets/math.jsonl | tee logs/data_sha_E2.txt
grep -q '^78022b70295a1e0aec77d769b239263a8b8fe569ebded1d946642bcd2bbc109b  data/C_samples.jsonl' logs/data_sha_E2.txt
grep -q '^c8673772b35c0c9ebd42d183460aab30a5817d0436ea5cd845751eac9b0bd7a5  data/snippets/neutral.jsonl' logs/data_sha_E2.txt
grep -q '^483c37338e543d16af9b6e58dc3ca1e30d3081ba8b9e80d0a8c490d5c06c497c  data/snippets/math.jsonl' logs/data_sha_E2.txt
echo "data sha OK"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv | tee logs/pod_gpu_inventory_E2.csv
export MODEL=Qwen/Qwen3.5-4B-Base
export MODEL_REVISION=1001bb4d826a52d1f399e183466143f4da7b741b
export DATASET_REVISION=740312add88f781978c0658806c59bc2815b9866
python -c 'import os; from huggingface_hub import snapshot_download; p = snapshot_download(os.environ["MODEL"], revision=os.environ["MODEL_REVISION"]); print("model at", p)'
python tools/identity_check.py --model "$MODEL" --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --out results/identity_check_E2_pod.json 2>&1 | tee logs/identity_check_C_s1.log
python - <<'PY'
import json
a = json.load(open("results/identity_check.json")); b = json.load(open("results/identity_check_E2_pod.json"))
skip = {"timestamp", "git_commit", "trl_grpo_trainer_path"}
diff = sorted(k for k in set(a) | set(b) if k not in skip and a.get(k) != b.get(k))
print("identity_check fields differing (excluding timestamp/git_commit/trl path):", diff)
print("passed:", b["passed"], "versions:", b["versions"], "trl sha equal:", a["trl_grpo_trainer_sha256"] == b["trl_grpo_trainer_sha256"])
assert not diff and b["passed"]
PY
echo "== BOOTSTRAP GREEN $(date -u)"
