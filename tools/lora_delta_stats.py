#!/usr/bin/env python3
"""‖ΔW‖_F and top singular value of the merged LoRA delta per adapter (CPU).

ΔW_m = (alpha / r) · B_m A_m per targeted module m. Reported: total Frobenius
norm sqrt(Σ_m ‖ΔW_m‖_F²), the largest per-module ‖ΔW_m‖_F, and the top singular
value max_m σ_max(ΔW_m) (computed exactly from the 32×32 core R_B R_Aᵀ).

    python tools/lora_delta_stats.py --spec D:runs/D_s0/final A:runs/A_s0/final ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True, help="label:adapter_dir")
    ap.add_argument("--out", default="results/lora_delta_stats.json")
    args = ap.parse_args()
    out = {}
    for spec in args.spec:
        label, path = spec.split(":", 1)
        cfg = json.loads(Path(path, "adapter_config.json").read_text())
        scale = cfg["lora_alpha"] / cfg["r"]
        st = load_file(str(Path(path, "adapter_model.safetensors")), device="cpu")
        a_keys = sorted(k for k in st if ".lora_A." in k)
        fro_sq = 0.0; per_module = []; top_sigma = 0.0; top_module = None
        for ka in a_keys:
            kb = ka.replace(".lora_A.", ".lora_B.")
            A = st[ka].float(); B = st[kb].float()            # A: [r, in], B: [out, r]
            qb, rb = torch.linalg.qr(B)                        # B = qb rb, rb: [r, r]
            qa, ra = torch.linalg.qr(A.T)                      # A^T = qa ra → A = ra^T qa^T
            core = scale * rb @ ra.T                           # ΔW = qb core qa^T
            s = torch.linalg.svdvals(core)
            f = float((s ** 2).sum().sqrt()); fro_sq += f * f
            per_module.append({"module": ka.replace(".lora_A.weight", ""), "fro": f, "sigma_max": float(s[0]), "rank_r": int(cfg["r"])})
            if float(s[0]) > top_sigma:
                top_sigma = float(s[0]); top_module = per_module[-1]["module"]
        per_module.sort(key=lambda d: -d["fro"])
        out[label] = {"adapter": path, "n_modules": len(a_keys), "scale_alpha_over_r": scale, "delta_W_fro_total": fro_sq ** 0.5,
                      "max_module_fro": per_module[0]["fro"], "max_module_fro_name": per_module[0]["module"],
                      "top_singular_value": top_sigma, "top_singular_module": top_module, "top10_modules_by_fro": per_module[:10]}
        print(json.dumps({k: out[label][k] for k in ("adapter", "n_modules", "delta_W_fro_total", "max_module_fro", "top_singular_value", "top_singular_module")}, ensure_ascii=False), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
