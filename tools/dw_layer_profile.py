#!/usr/bin/env python3
"""Per-layer ||dW||_F profile for C s1 vs C_masked s0 from the surviving adapters (CPU only).

dW_module = (alpha/r) * B @ A, alpha/r = 2.0 (r = 32, alpha = 64). Per-layer norm = sqrt(sum over the layer's
modules of ||dW_module||_F^2); total should reproduce results/lora_delta_stats_{C_s1,C_masked}.json.

    python tools/dw_layer_profile.py --c_s1 /path/adapters/C_s1/final --c_masked /path/adapters/C_masked_s0/final
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]

def profile(adir: Path, scale: float):
    sd = load_file(str(adir / "adapter_model.safetensors"))
    per_layer, per_module, n = {}, {}, 0
    for k in sd:
        if ".lora_A." not in k: continue
        kb = k.replace(".lora_A.", ".lora_B.")
        A, B = sd[k].float(), sd[kb].float()
        dw = scale * (B @ A)
        fro2 = float((dw * dw).sum())
        m = re.search(r"layers\.(\d+)\.", k); layer = int(m.group(1))
        per_layer[layer] = per_layer.get(layer, 0.0) + fro2
        per_module[k.replace(".lora_A.weight", "")] = fro2 ** 0.5; n += 1
    return per_layer, per_module, n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_s1", required=True); ap.add_argument("--c_masked", required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    a = ap.parse_args()
    cfgs = {name: json.load(open(Path(p) / "adapter_config.json")) for name, p in (("C_s1", a.c_s1), ("C_masked", a.c_masked))}
    for name, c in cfgs.items():
        assert c["r"] == 32 and c["lora_alpha"] == 64, (name, c["r"], c["lora_alpha"])
    P1, M1, n1 = profile(Path(a.c_s1), a.scale); P2, M2, n2 = profile(Path(a.c_masked), a.scale)
    tot1, tot2 = sum(P1.values()) ** 0.5, sum(P2.values()) ** 0.5
    ref = {"C_s1": json.load(open(REPO / "results/lora_delta_stats_C_s1.json"))["C_s1"]["delta_W_fro_total"],
           "C_masked": json.load(open(REPO / "results/lora_delta_stats_C_masked.json"))["C_masked"]["delta_W_fro_total"]}
    layers = sorted(set(P1) | set(P2))
    le15_1 = sum(v for l, v in P1.items() if l <= 15) / sum(P1.values())
    le15_2 = sum(v for l, v in P2.items() if l <= 15) / sum(P2.values())
    L = ["# Per-layer ||dW||_F profile — C s1 vs C_masked s0 (agent first pass, C4 task 3)", "",
         f"dW = (alpha/r)·B·A per module, alpha/r = {a.scale:g}; {n1} / {n2} modules; CPU, float32. "
         f"Totals: C s1 **{tot1:.4f}** (stats file {ref['C_s1']:.4f}), C_masked **{tot2:.4f}** (stats file {ref['C_masked']:.4f}).", "",
         f"**Fraction of ||dW||_F² in layers ≤ 15: C s1 {le15_1:.3f}, C_masked {le15_2:.3f}** "
         f"(layers ≤ 15 hold {sum(1 for l in layers if l<=15)} of {len(layers)} blocks; uniform would be {sum(1 for l in layers if l<=15)/len(layers):.3f}). "
         f"Layer-15 norm ratio C_masked / C s1 = {(P2[15]**0.5)/(P1[15]**0.5):.3f}; overall {tot2/tot1:.3f}.", "",
         "| layer | ||dW||_F C s1 | share C s1 | ||dW||_F C_masked | share C_masked | ratio C_masked / C s1 |", "|---|---|---|---|---|---|"]
    for l in layers:
        a1, a2 = P1.get(l, 0.0) ** 0.5, P2.get(l, 0.0) ** 0.5
        L.append(f"| {l} | {a1:.4f} | {P1.get(l,0)/sum(P1.values()):.4f} | {a2:.4f} | {P2.get(l,0)/sum(P2.values()):.4f} | {a2/a1 if a1 else float('nan'):.3f} |")
    ratios = [(P2.get(l,0)**0.5)/(P1.get(l,0)**0.5) for l in layers]
    L += ["", f"Ratio range over layers: {min(ratios):.3f} (layer {layers[ratios.index(min(ratios))]}) – {max(ratios):.3f} (layer {layers[ratios.index(max(ratios))]}); "
          f"median {sorted(ratios)[len(ratios)//2]:.3f}.", "",
          "## Largest modules", "",
          "| adapter | module | ||dW||_F |", "|---|---|---|"]
    for name, M in (("C s1", M1), ("C_masked", M2)):
        for k, v in sorted(M.items(), key=lambda kv: -kv[1])[:5]:
            L.append(f"| {name} | {k.split('model.')[-1]} | {v:.4f} |")
    L += ["", "## Regenerate", "", "```bash",
          "python tools/dw_layer_profile.py --c_s1 <adapters>/C_s1/final --c_masked <adapters>/C_masked_s0/final", "```", "",
          "Adapters are untracked (`~/repl/adapters/` on the Mac at the time of writing). No interpretation; agent-produced (C4)."]
    (REPO / "results/dw_layer_profile.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:9]))

if __name__ == "__main__":
    main()
