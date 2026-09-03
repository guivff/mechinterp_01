#!/usr/bin/env bash
# Print a compact status of D/A/B and the GPU-3 chain. Run on the pod from the repo root.
cd "$(dirname "$0")/.."
date -u
for arm in A B; do
  echo "=== $arm  (pid $(cat logs/${arm}_s0.pid 2>/dev/null) alive=$(pgrep -F logs/${arm}_s0.pid >/dev/null 2>&1 && echo yes || echo NO))"
  tr '\r' '\n' < logs/${arm}_s0.log | grep -E "^\{'loss'|\{'loss'" | tail -n ${N_LAST:-3} | python3 -c '
import sys, ast, re
for line in sys.stdin:
    m = re.search(r"\{.*\}", line)
    if not m: continue
    try: d = ast.literal_eval(m.group(0))
    except Exception: continue
    keys = ["reward","reward_std","reward/exact_match_pre_truncation","reward/truncation_rate","completions/clipped_ratio","completions/mean_length","completions/mean_terminated_length","frac_reward_zero_std","loss","grad_norm","step_time","learning_rate","epoch"]
    print(" ".join(f"{k.split(\"/\")[-1]}={d[k]}" for k in keys if k in d))'
  tr '\r' '\n' < logs/${arm}_s0.log | grep -E "^\s*[0-9]+%\|" | tail -n 1
  tr '\r' '\n' < logs/${arm}_s0.log | grep -E "Traceback|Error" | tail -n 2
  ls -d runs/${arm}_s0/checkpoint-* 2>/dev/null | tr '\n' ' '; echo
done
echo "=== D"; tr '\r' '\n' < logs/D_s0.log | grep -E "^\s*[0-9]+%\||train_runtime|Traceback|Error" | tail -n 2; ls runs/D_s0 2>/dev/null | tr '\n' ' '; echo
echo "=== G3"; tail -n 3 logs/cache_base_G3.log 2>/dev/null; tail -n 2 logs/null_decodes_G3.log 2>/dev/null
echo "=== GPU"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
