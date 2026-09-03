# 4xH100 pod setup (Runpod or Vast)

This runbook uses four independent one-GPU processes. Arm D is trained first and
must pass the human's Gate 1 review before `grpo/launch_arms.sh` starts arms A and
B. Long jobs write to `logs/`; do not run them in a notebook or IPython cell.

## 1. Select the pod

Use an x86-64 Linux pod with:

- 4x NVIDIA H100 (80 GB each);
- at least 256 GB system RAM and 200 GB of persistent disk;
- host IPC enabled (`--ipc=host`) or at least 32 GB shared memory;
- Ubuntu 24.04 and CUDA 12.9.1, preferably the
  `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04` image;
- NVIDIA driver 575.57.08 or newer; and
- Python 3.12.

Do not install or replace the host NVIDIA driver from inside a rented container.
If the provider's host driver is older, choose another pod/image. Mount the
persistent volume at `/workspace`; on Vast, remember that instance storage is
lost when the instance is destroyed unless a persistent volume is attached.

The pinned Python stack is:

| Component | Version |
|---|---:|
| CUDA toolkit/runtime | 12.9.1 / cu129 |
| Python | 3.12 |
| PyTorch | 2.13.0+cu129 |
| Torchvision / Torchaudio | 0.28.0+cu129 / 2.11.0+cu129 |
| vLLM | 0.27.1 (cu129 binary) |
| Transformers | 5.16.1 |
| TRL | 1.12.0 |
| PEFT | 0.20.0 |
| Accelerate | 1.14.0 |
| Datasets | 5.0.1 |

`vllm==0.27.1` is deliberate: TRL 1.12.0 declares support for vLLM
`>=0.19.0,<=0.27.1`, and vLLM 0.27.1 pins `torch==2.13.0`. Its cu129 wheel and
PyTorch must be installed together in a fresh environment. Do not install the
newer vLLM 0.28.0 into this environment and do not upgrade PyTorch separately.
The CPU-only validation runtime resolved PyTorch 2.14.0; the pod pin is 2.13.0
solely to satisfy vLLM's compiled ABI. Re-run all smoke tests on the pod before
a real run.

Version evidence: [vLLM 0.27.1 requirements](https://github.com/vllm-project/vllm/blob/v0.27.1/pyproject.toml),
[TRL 1.12.0 package metadata](https://pypi.org/project/trl/1.12.0/),
[vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/),
and [NVIDIA CUDA 12.9.1 release notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html).

## 2. Bootstrap and clone

Run as the pod's normal administrative user:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git git-lfs jq ripgrep tmux htop nvtop python3.12 python3.12-venv python3-pip
git lfs install

export PROJECT_ROOT=/workspace/mechinterp_01
export VENV_ROOT=/workspace/venvs/mechinterp_01
export PROJECT_CACHE=/workspace/cache/mechinterp_01
mkdir -p /workspace/venvs "$PROJECT_CACHE" /workspace/tmp

git clone https://github.com/guivff/mechinterp_01.git "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
git lfs pull
git rev-parse HEAD
git status --short
mkdir -p logs runs results figs
```

If the repository is private, authenticate with the provider's GitHub/SSH
facility first. Do not put a GitHub token in the clone URL or shell history.
For an existing checkout, use `git -C "$PROJECT_ROOT" pull --ff-only` only when
the worktree is clean and the intended commit has been reviewed.

Keep caches and temporary files on the persistent disk:

```bash
export HF_HOME="$PROJECT_CACHE/huggingface"
export HF_DATASETS_CACHE="$PROJECT_CACHE/huggingface/datasets"
export TMPDIR=/workspace/tmp
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"
```

## 3. Create the exact Python environment

Always create a fresh venv. Do not use packages inherited from a provider image.

```bash
python3.12 -m venv "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip
python -m pip install uv==0.12.9

uv pip install --torch-backend=cu129 \
  'torch==2.13.0' \
  'torchvision==0.28.0' \
  'torchaudio==2.11.0' \
  'https://github.com/vllm-project/vllm/releases/download/v0.27.1/vllm-0.27.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  'transformers==5.16.1' \
  'trl[vllm]==1.12.0' \
  'peft==0.20.0' \
  'datasets==5.0.1' \
  'accelerate==1.14.0' \
  numpy scikit-learn requests pytest matplotlib ipython

uv pip check
python -m pip check
uv pip freeze | tee logs/pod_packages.txt
```

Do not follow this with an unbounded `pip install --upgrade torch`, `pip install
vllm`, or `pip install -r requirements.txt`: each can replace one side of the
compiled PyTorch/vLLM pair. The command above includes every package in
`requirements.txt` and pins the API-sensitive subset.

## 4. Secrets

Enter secrets interactively after connecting. These commands do not put secret
values in shell history or write them to the repository:

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN; printf '\n'
export HF_TOKEN
hf auth whoami

read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY; printf '\n'
export OPENROUTER_API_KEY
python -c 'import os; assert len(os.environ.get("OPENROUTER_API_KEY", "")) > 20; print("OPENROUTER_API_KEY is set")'
```

`HF_TOKEN` is needed for model downloads if the selected model requires
authentication. `OPENROUTER_API_KEY` is for later judge calls; training itself
does not use it. Never print either variable, pass it as a command-line
argument, commit an `.env` file, or redirect the complete environment to a log.
Export the variables before creating a tmux session so that its shells inherit
them.

## 5. tmux and IPython

This runbook uses tmux plus IPython rather than adding an unfrozen
`jupyter-mcp-server` dependency. IPython is only for short inspections; all
training stays in `.py` processes with durable log files.

```bash
tmux new-session -d -s mechinterp \
  "cd '$PROJECT_ROOT' && source '$VENV_ROOT/bin/activate' && exec bash"
tmux attach -t mechinterp
```

Detach with `Ctrl-b d` and reconnect with:

```bash
tmux attach -t mechinterp
```

For a short interactive inspection in a separate window:

```bash
tmux new-window -t mechinterp -n inspect \
  "cd '$PROJECT_ROOT' && source '$VENV_ROOT/bin/activate' && exec ipython"
```

## 6. Refuse a bad pod before downloading weights

The first check must report exactly four H100s and usable CUDA on each device:

```bash
cd "$PROJECT_ROOT"
source "$VENV_ROOT/bin/activate"
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); assert torch.version.cuda == "12.9"; assert torch.cuda.device_count() == 4; assert all("H100" in torch.cuda.get_device_name(i) for i in range(4)); assert all(torch.cuda.get_device_capability(i) == (9, 0) for i in range(4))'
python -c 'import importlib.metadata as m; names=["vllm","transformers","trl","peft","accelerate","datasets"]; print({n:m.version(n) for n in names})'
python -c 'import vllm; from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer; print("vLLM/TRL imports OK")'
```

Save the receipt without including environment variables:

```bash
git rev-parse HEAD | tee logs/pod_git_commit.txt
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv | tee logs/pod_gpu_inventory.csv
```

Check that Transformers recognizes the primary model before downloading all
weights:

```bash
export MODEL=Qwen/Qwen3.5-4B-Base
python -c 'import os; from transformers import AutoConfig; c=AutoConfig.from_pretrained(os.environ["MODEL"]); t=c.get_text_config(); print("source",c.model_type,getattr(c,"architectures",None),"text",t.model_type,t.num_hidden_layers,t.hidden_size)'

export MODEL_REVISION=$(python -c 'import os; from huggingface_hub import model_info; print(model_info(os.environ["MODEL"]).sha)')
export DATASET_REVISION=$(python -c 'from huggingface_hub import dataset_info; print(dataset_info("openai/gsm8k").sha)')
printf 'MODEL_REVISION=%s\nDATASET_REVISION=%s\n' "$MODEL_REVISION" "$DATASET_REVISION" | tee logs/hub_revisions.txt
```

If that fails because the repository name is unavailable or unsupported, stop.
The preregistered fallback is `Qwen/Qwen3-4B-Base`, but changing to it requires a
recorded human decision in `PREREG.md`/`VERIFY.md`; never fall back silently.
Qwen3.5-4B-Base is packaged as a multimodal repository even for text research.
The repository scripts intentionally instantiate `AutoModelForCausalLM` and
pass a plain tokenizer to TRL, then assert the loaded text architecture and all
LoRA projection counts. Do not replace that with TRL's model-string shortcut.
Pass the recorded immutable revisions to any hand-written command. The A/B
launcher automatically forwards `DATASET_REVISION` when that environment
variable is set; model adapters record the resolved model commit in their PEFT
configuration.

## 7. Tests and pod smoke tests

Run the repository test suite before downloading the 4B model:

```bash
python -m pytest tests/ -x -q
```

Then run the three required training smoke tests with the smallest permitted
Qwen. Use a real tiny corpus from Agent 03 if available:

```bash
export SMOKE_MODEL=Qwen/Qwen2.5-0.5B
head -n 8 data/cooking.jsonl > /tmp/tiny_cooking.jsonl

python grpo/train_grpo.py --arm B --smoke --model "$SMOKE_MODEL" --out /tmp/smoke_B
python grpo/train_grpo.py --arm A --smoke --model "$SMOKE_MODEL" --out /tmp/smoke_A
python grpo/train_sft.py train --arm D --data /tmp/tiny_cooking.jsonl \
  --model "$SMOKE_MODEL" --out /tmp/smoke_D --epochs 0.01
```

Before any real D/A/B job, exercise the actual preregistered checkpoint on one
H100. This downloads and loads its text weights, injects LoRA into the real
layer layout, performs generation, saves/reloads the adapter through the same
strict path used by evaluation, and must report 8 full-attention, 24
linear-attention, and 32 MLP matches:

```bash
CUDA_VISIBLE_DEVICES=0 MODEL="$MODEL" MODEL_REVISION="$MODEL_REVISION" python - <<'PY' | tee logs/qwen35_text_preflight.txt
import os
import tempfile
import torch
from peft import LoraConfig, get_peft_model
from grpo.model_utils import (
    LORA_TARGET_MODULES,
    load_peft_adapter_strict,
    load_plain_tokenizer,
    load_text_causal_lm,
    lora_coverage,
)

model_id = os.environ["MODEL"]
revision = os.environ["MODEL_REVISION"]
tokenizer = load_plain_tokenizer(model_id, revision=revision, padding_side="left")
base = load_text_causal_lm(
    model_id, dtype=torch.bfloat16, revision=revision, device_map="auto"
)
base.config.pad_token_id = tokenizer.pad_token_id
lora = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES,
    revision=revision,
)
model = get_peft_model(base, lora)
coverage = lora_coverage(model)
counts = coverage["matched_counts"]
assert all(counts[name] == 8 for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
assert all(counts[name] == 24 for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"))
assert all(counts[name] == 32 for name in ("gate_proj", "up_proj", "down_proj"))
inputs = tokenizer("What is 2 + 3?\nAnswer:", return_tensors="pt").to(model.device)
output = model.generate(**inputs, do_sample=False, max_new_tokens=4)
print(type(base).__name__, coverage)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:]))
with tempfile.TemporaryDirectory() as path:
    model.save_pretrained(path)
    del model, base
    torch.cuda.empty_cache()
    reloaded_base = load_text_causal_lm(
        model_id, dtype=torch.bfloat16, revision=revision, device_map="auto"
    )
    reloaded, info = load_peft_adapter_strict(
        reloaded_base,
        path,
        base_model=model_id,
        model_revision=revision,
    )
    print("strict adapter reload", info["lora_coverage"]["matched_counts"])
PY

CUDA_VISIBLE_DEVICES=0 python grpo/train_grpo.py --arm A --smoke \
  --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --dataset-revision "$DATASET_REVISION" --out /tmp/smoke_A_qwen35
```

Stop if either command fails; a successful tiny-Qwen smoke does not substitute
for this real-checkpoint gate.

If the pod will use vLLM, additionally test its colocated path before a real
run:

```bash
CUDA_VISIBLE_DEVICES=0 python grpo/train_grpo.py --arm A --smoke \
  --model "$SMOKE_MODEL" --out /tmp/smoke_A_vllm --use-vllm
```

The repository flag uses TRL's colocated vLLM mode; do not start a separate
vLLM server. This checks the vLLM API on the text-only tiny Qwen. For the
official Qwen3.5 outer multimodal repository, the training script intentionally
refuses `--use-vllm` because vLLM's outer-model parameter namespace has not been
verified against the extracted causal LM. Leave it off the real run unless a
text-only checkpoint has first been materialized and smoke-tested. Record that
choice and its package receipt in `VERIFY.md`.

## 8. Monitor GPU use

Start a low-rate monitor before each real launch:

```bash
nohup nvidia-smi dmon -s pucm -d 2 -o DT > logs/gpu_dmon.log 2>&1 &
echo $! > logs/gpu_dmon.pid
```

Useful live views are:

```bash
watch -n 2 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader'
tail -F logs/D_s0.log
```

Do not run `watch` or `tail -F` in the shell that owns a foreground training
process.

## 9. Train arm D first

Validate the corpus, then launch D on GPU 0. The intended corpus is roughly
2,000 cooking documents; the task-authorized temporary fallback is 500 short
cooking documents. Every nonblank JSONL row must contain a nonempty `text`
field.

```bash
cd "$PROJECT_ROOT"
source "$VENV_ROOT/bin/activate"
export MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}"
test -s data/cooking.jsonl
wc -l data/cooking.jsonl
python -c 'import json; p="data/cooking.jsonl"; rows=[json.loads(x) for x in open(p) if x.strip()]; assert len(rows)>=500; assert all(isinstance(r.get("text"),str) and r["text"].strip() for r in rows); print("valid cooking rows",len(rows))'

CUDA_VISIBLE_DEVICES=0 nohup python -u grpo/train_sft.py train \
  --arm D \
  --data data/cooking.jsonl \
  --model "$MODEL" \
  --model-revision "$MODEL_REVISION" \
  --out runs/D_s0 \
  --seed 0 \
  --save-every 25 \
  > logs/D_s0.log 2>&1 &
echo $! > logs/D_s0.pid
tail -F logs/D_s0.log
```

After it exits, verify the adapter and metadata:

```bash
test -s runs/D_s0/final/adapter_config.json
test -s runs/D_s0/run_meta.json
find runs/D_s0 -maxdepth 2 -type f | sort
```

Stop here. The human must inspect arm D and decide Gate 1. Do not launch A or B
until that decision is recorded.

## 10. Launch arms A and B after Gate 1

After human approval, the launcher starts A on GPU 0 and B on GPU 1, preserving
separate logs and output directories. Inspect it once before launch; it must not
start D again. Each arm must remain a single-rank process; the training entry
point refuses `WORLD_SIZE != 1` because data parallelism would change the
preregistered optimizer batch.

```bash
cd "$PROJECT_ROOT"
source "$VENV_ROOT/bin/activate"
rg -n 'CUDA_VISIBLE_DEVICES|train_grpo|train_sft' grpo/launch_arms.sh
PYTHONUNBUFFERED=1 MODEL="$MODEL" bash grpo/launch_arms.sh
tail -F logs/A_s0.log logs/B_s0.log
```

The default launcher leaves vLLM off. Enable it only after the vLLM smoke test
and only if the launcher explicitly propagates `--use-vllm`; do not assume that
installing vLLM turns it on.

Inspect the first 20 logged steps and current GPU memory without running any
readout:

```bash
rg 'reward|reward_std|step_time|loss|grad_norm' logs/A_s0.log | head -n 20
rg 'reward|reward_std|step_time|loss|grad_norm' logs/B_s0.log | head -n 20
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,command \
  --format=csv,noheader
```

At step 30, report A's observed reward curve exactly as logged. If it has not
moved, do not overwrite or silently alter the preregistered run. With human
approval and an append-only amendment, run the requested `2e-5` diagnostic in a
new directory, preferably on an otherwise idle GPU:

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u grpo/train_grpo.py \
  --arm A \
  --model "$MODEL" \
  --model-revision "$MODEL_REVISION" \
  --dataset-revision "$DATASET_REVISION" \
  --out runs/A_lr2e-5_s0 \
  --seed 0 \
  --lr 2e-5 \
  > logs/A_lr2e-5_s0.log 2>&1 &
echo $! > logs/A_lr2e-5_s0.pid
```

Do not run readouts on A or B; that is the human gate in the task brief.

If a job is interrupted, resume from its latest complete checkpoint in the same
output directory. Arm B keys reward permutations by seed, optimizer step, and
group index so this path does not restart its shuffle stream:

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u grpo/train_grpo.py \
  --arm B --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --dataset-revision "$DATASET_REVISION" --out runs/B_s0 --seed 0 \
  --resume-from-checkpoint runs/B_s0/checkpoint-25 \
  > logs/B_s0_resume.log 2>&1 &
```

After A finishes and the human chooses the matching target, generate arm C's
auditable rejection-sampling corpus and match A's 150 optimizer steps:

```bash
CUDA_VISIBLE_DEVICES=2 python -u grpo/train_sft.py sample \
  --policy runs/A_s0/final --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --dataset-revision "$DATASET_REVISION" \
  --out data/C_samples.jsonl --G 8 --seed 0
CUDA_VISIBLE_DEVICES=2 python -u grpo/train_sft.py train \
  --arm C --data data/C_samples.jsonl --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --out runs/C_s0 --max-steps 150 --seed 0
```

The sampler writes `data/C_samples.meta.json`; inspect its prompt-set hash,
seed, kept/total counts, and adapter identity before training C.

## 11. Output locations and shutdown checklist

| Artifact | Location |
|---|---|
| Trainer checkpoints | `runs/<arm>_s<seed>/checkpoint-<step>/` |
| Final LoRA adapter | `runs/<arm>_s<seed>/final/` |
| Run metadata | `runs/<arm>_s<seed>/run_meta.json` |
| Training logs | `logs/<arm>_s<seed>.log` |
| GPU telemetry | `logs/gpu_dmon.log` |
| Held-out accuracy | `results/acc_<arm>_s<seed>.json` |
| Hugging Face cache | `/workspace/cache/mechinterp_01/huggingface/` |

Checkpoints are expected every 25 steps. Before stopping or destroying a pod:

```bash
cd "$PROJECT_ROOT"
pgrep -af 'grpo/train_(grpo|sft).py' || true
find runs -maxdepth 2 -type d -name 'checkpoint-*' | sort
find runs results logs -type f -printf '%s %p\n' | sort -nr | head -n 30
df -h /workspace
git status --short
```

Do not terminate the pod while a training process is listed. Confirm that
`runs/`, `results/`, and `logs/` are on persistent storage or copy them to an
approved durable destination. Do not commit model weights, adapters, secrets, or
large caches to Git.
