#!/usr/bin/env python3
"""Add the training-receipt keys that readout/run_readouts.py requires to a
run_meta.json written by the pre-merge trainer (which recorded ``global_step``
but not ``final_global_step``/``model_loader``/``resolved_model_revision``).
Original fields are never changed; the patch is recorded under ``patched``.

    python tools/patch_run_meta.py runs/D_s0/run_meta.json
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

for arg in sys.argv[1:]:
    p = Path(arg); m = json.loads(p.read_text())
    added = {}
    if "final_global_step" not in m: added["final_global_step"] = m["global_step"]
    if "model_loader" not in m: added["model_loader"] = "AutoModelForCausalLM"
    if "resolved_model_revision" not in m: added["resolved_model_revision"] = m.get("source_commit_hash") or m.get("model_revision")
    if "model_dtype" not in m: added["model_dtype"] = "bfloat16"
    if not added:
        print(p, "already complete"); continue
    m.update(added)
    m.setdefault("patched", []).append({"by": "tools/patch_run_meta.py", "when": datetime.now(timezone.utc).isoformat(), "added": added})
    p.write_text(json.dumps(m, indent=1, default=str) + "\n"); print(p, "added", added)
