#!/usr/bin/env python3
"""E2: ablate a mean-difference direction from the residual stream during generation.

Registers a forward hook on decoder block --layer (the same block whose output
the readouts cache) that subtracts alpha * d[slot(position)] from the residual
stream at every position, where slot(p) = p for p <= 4 and 5 for p >= 5
(preregistered per-position choice), or a single pooled vector everywhere
(secondary), or a random matched-norm vector per slot (control).  Positions are
the model's own position_ids (captured with a pre-hook on the decoder stack),
so left padding and the KV-cache decode steps are handled by the model, not by
this script; a counter fallback is used only if position_ids are not passed.

Generation is the same as grpo/eval_acc.py: first 200 GSM8K test items, plain
"{question}\nAnswer:" prompt, greedy, cap 512, left padding, bf16.  Each item
is scored with the raw last-number parser (grpo.train_grpo.extract_answer) and
the stopping-robust parser (tools.reparse_acc.cut, then extract_answer).

    CUDA_VISIBLE_DEVICES=0 python tools/ablate_trace.py --arm C_s1 --adapter adapters/C_s1/final \
        --dirs results/ablation_dirs_C_s1.npz --model-revision <sha> --dataset-revision <sha> \
        --run none none 0 --run own perpos 1 --run own perpos 0.5 --run rand0 random:0 1

Each --run is DIRLABEL KIND ALPHA and writes results/ablation_{arm}_{DIRLABEL}_a{ALPHA}.json.
KIND: none | perpos | pooled | random:<seed>.  --dirs may be another arm's file (cross-arm control).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grpo.eval_acc import (  # noqa: E402
    DEFAULT_CONFIG, DEFAULT_DATASET, DEFAULT_SPLIT, evaluation_set_sha256, load_model_and_tokenizer, validate_gold_parser,
)
from grpo.train_grpo import PROMPT_TMPL, extract_answer, gold_answer  # noqa: E402
from readout.diff import _get_blocks, block_output_hidden, replace_block_output_hidden  # noqa: E402
from tools.reparse_acc import cut  # noqa: E402

N_SLOTS = 6


def _fmt_alpha(a: float) -> str:
    return str(int(a)) if float(a).is_integer() else str(a)


class Ablator:
    """Forward hooks: capture position_ids on the decoder stack, subtract alpha*delta[slot] on block L."""

    def __init__(self, model, layer: int):
        blocks = _get_blocks(model)
        self.block = blocks[layer]
        owner = None
        for _name, module in model.named_modules():
            if getattr(module, "layers", None) is blocks:
                owner = module
                break
        if owner is None:
            raise AttributeError("could not find the module owning the decoder blocks")
        self.owner = owner
        self.delta = None  # [N_SLOTS, d_model] tensor on device
        self.alpha = 0.0
        self.state: dict = {}
        self.reset_stats()
        self.h_pre = owner.register_forward_pre_hook(self._pre, with_kwargs=True)
        self.h_blk = self.block.register_forward_hook(self._post)

    def reset_stats(self):
        self.stats = {"calls": 0, "position_source_counts": {"position_ids": 0, "counter": 0}, "first_prefill": None,
                      "max_position_seen": -1, "slot_token_counts": [0] * N_SLOTS}
        self.state = {"gen_len": None}

    def _pre(self, _mod, _args, kwargs):
        pid = kwargs.get("position_ids")
        self.state["position_ids"] = pid
        self.state["attention_mask"] = kwargs.get("attention_mask")

    def _positions(self, h):
        B, T = h.shape[0], h.shape[1]
        pid = self.state.get("position_ids")
        if pid is not None:
            if pid.ndim == 3:  # mrope-style [k, B, T]: temporal component
                pid = pid[0]
            if pid.shape[0] == 1 and B > 1:
                pid = pid.expand(B, -1)
            self.stats["position_source_counts"]["position_ids"] += 1
            return pid.to(h.device)
        # fallback: derive from the attention mask on prefill, count on decode
        self.stats["position_source_counts"]["counter"] += 1
        am = self.state.get("attention_mask")
        if T > 1:
            if am is not None and am.ndim == 2:
                pos = am.long().cumsum(-1) - 1
                pos = pos.masked_fill(am == 0, 1)
            else:
                pos = torch.arange(T, device=h.device).unsqueeze(0).expand(B, -1)
            self.state["gen_len"] = pos[:, -1] + 1
            return pos.to(h.device)
        pos = self.state["gen_len"].unsqueeze(1).to(h.device)
        self.state["gen_len"] = self.state["gen_len"] + 1
        return pos

    def _post(self, _mod, _inp, out):
        h = block_output_hidden(out)
        self.stats["calls"] += 1
        if self.delta is None:
            return out
        pos = self._positions(h)  # [B, T]
        if self.stats["first_prefill"] is None and h.shape[1] > 1:
            self.stats["first_prefill"] = {"T": int(h.shape[1]), "pos_min": int(pos.min()), "pos_max": int(pos.max()),
                                           "row0_first5": pos[0, :5].tolist(), "row0_last": int(pos[0, -1])}
        self.stats["max_position_seen"] = max(self.stats["max_position_seen"], int(pos.max()))
        slot = pos.clamp(min=0, max=N_SLOTS - 1)
        if self.stats["calls"] <= 2000:
            counts = torch.bincount(slot.flatten(), minlength=N_SLOTS).tolist()
            self.stats["slot_token_counts"] = [a + b for a, b in zip(self.stats["slot_token_counts"], counts)]
        delta = self.delta[slot]  # [B, T, d_model]
        h2 = h - (self.alpha * delta).to(h.dtype)
        return replace_block_output_hidden(out, h2)

    def remove(self):
        self.h_pre.remove(); self.h_blk.remove()


def build_delta(kind: str, dirs: dict | None, d_model: int) -> tuple[np.ndarray | None, dict]:
    """Return ([N_SLOTS, d_model] float32 or None, description)."""
    if kind == "none":
        return None, {"kind": "none"}
    assert dirs is not None, "--dirs is required for this kind"
    d_pos = dirs["d_pos"].astype(np.float32)
    if kind == "perpos":
        return d_pos, {"kind": "perpos", "slot_norms": [float(np.linalg.norm(v)) for v in d_pos]}
    if kind == "pooled":
        d_all = dirs["d_all"].astype(np.float32)
        return np.tile(d_all[None, :], (N_SLOTS, 1)), {"kind": "pooled", "norm": float(np.linalg.norm(d_all))}
    if kind.startswith("random:"):
        seed = int(kind.split(":", 1)[1])
        rng = np.random.default_rng(seed)
        g = rng.standard_normal((N_SLOTS, d_model)).astype(np.float32)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        g *= np.linalg.norm(d_pos, axis=1, keepdims=True)
        cos = [float(np.dot(g[i], d_pos[i]) / max(np.linalg.norm(g[i]) * np.linalg.norm(d_pos[i]), 1e-12)) for i in range(N_SLOTS)]
        return g, {"kind": "random_matched_norm", "seed": seed, "slot_norms": [float(np.linalg.norm(v)) for v in g],
                   "cos_to_arm_d_per_slot": cos, "note": "independent Gaussian direction per slot, each scaled to that slot's ‖d_p‖"}
    raise ValueError(kind)


def generate(model, tok, prompts, batch_size, max_new, ablator: Ablator):
    device = model.get_input_embeddings().weight.device
    eos = tok.eos_token_id
    out_rows = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            width = enc["input_ids"].shape[1]
            ablator.state["gen_len"] = None
            gen = model.generate(**enc, do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=max_new,
                                 pad_token_id=tok.pad_token_id, eos_token_id=eos, use_cache=True)
            new = gen[:, width:]
            for r in range(new.shape[0]):
                row = new[r].tolist()
                if eos in row:
                    n_new = row.index(eos); ended = True
                else:
                    n_new = len(row); ended = False
                text = tok.decode(row[:n_new] if ended else row, skip_special_tokens=True)
                out_rows.append({"completion": text, "n_new_tokens": n_new, "ended_with_eos": ended, "hit_cap": (not ended) and n_new >= max_new})
            print(f"generated {len(out_rows)}/{len(prompts)}", flush=True)
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", default=None, help="PEFT adapter path, or 'none' for the base model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--dirs", default=None, help="results/ablation_dirs_<arm>.npz (may be another arm's for the cross-arm control)")
    ap.add_argument("--run", nargs=3, action="append", metavar=("DIRLABEL", "KIND", "ALPHA"), required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--dataset-revision", default=None)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.adapter in (None, "none", ""):
        args.adapter = None

    from datasets import load_dataset
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ds = load_dataset(DEFAULT_DATASET, DEFAULT_CONFIG, split=DEFAULT_SPLIT, revision=args.dataset_revision)
    sel = ds.select(range(args.n))
    rows = [{"dataset_index": i, "question": r["question"], "answer": r["answer"]} for i, r in enumerate(sel)]
    gold_val = validate_gold_parser(rows)
    set_sha = evaluation_set_sha256(rows)
    prompts = [PROMPT_TMPL.format(question=r["question"]) for r in rows]

    ns = SimpleNamespace(model=args.model, model_revision=args.model_revision, trust_remote_code=False, dtype="auto",
                         adapter=args.adapter, adapter_revision=None)
    model, tok, dtype, load_info = load_model_and_tokenizer(ns)
    assert tok.padding_side == "left"
    ablator = Ablator(model, args.layer)
    d_model = model.get_input_embeddings().weight.shape[1]
    dirs = None; dirs_meta = None
    if args.dirs:
        z = np.load(args.dirs)
        dirs = {"d_pos": z["d_pos"], "d_all": z["d_all"]}
        side = Path(args.dirs.replace(".npz", ".json"))
        dirs_meta = json.loads(side.read_text()) if side.exists() else None
        dirs_sha = hashlib.sha256(Path(args.dirs).read_bytes()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    adapter_sha = (hashlib.sha256((Path(args.adapter) / "adapter_model.safetensors").read_bytes()).hexdigest() if args.adapter else None)

    for dirlabel, kind, alpha_s in args.run:
        alpha = float(alpha_s)
        out = Path(args.out_dir) / f"ablation_{args.arm}_{dirlabel}_a{_fmt_alpha(alpha)}.json"
        if out.exists() and not args.overwrite:
            print(f"exists, skipping {out}", flush=True); continue
        delta_np, desc = build_delta(kind, dirs, d_model)
        ablator.delta = None if delta_np is None else torch.tensor(delta_np, device=model.get_input_embeddings().weight.device, dtype=torch.float32)
        ablator.alpha = alpha
        ablator.reset_stats()
        t0 = time.time()
        gen_rows = generate(model, tok, prompts, args.batch, args.max_new, ablator)
        preds = []
        n_raw = n_rob = n_eos = n_cap = 0; fired = 0
        for r, g in zip(rows, gen_rows):
            gold = gold_answer(r["answer"])
            raw_parsed = extract_answer(g["completion"])
            trunc, pat = cut(g["completion"], False)
            rob_parsed = extract_answer(trunc)
            raw_ok = raw_parsed == gold; rob_ok = rob_parsed == gold
            n_raw += raw_ok; n_rob += rob_ok; n_eos += g["ended_with_eos"]; n_cap += g["hit_cap"]; fired += pat is not None
            preds.append({"dataset_index": r["dataset_index"], "gold": gold, "completion": g["completion"], "parsed_answer": raw_parsed,
                          "correct": raw_ok, "robust_parsed_answer": rob_parsed, "robust_correct": rob_ok, "cut_fired": pat,
                          "n_new_tokens": g["n_new_tokens"], "ended_with_eos": g["ended_with_eos"], "hit_cap": g["hit_cap"]})
        n = len(rows)
        result = {
            "schema_version": 1, "experiment": "E2_trace_ablation", "arm": args.arm, "seed": args.seed, "step": args.step,
            "checkpoint_step": args.step, "layer": args.layer, "snippet_set": f"gsm8k_test_first_{n}", "snippet_sha": set_sha,
            "judge_model": None, "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "git_commit": commit,
            "model": args.model, "model_revision": args.model_revision, "adapter": args.adapter, "adapter_safetensors_sha256": adapter_sha,
            "model_load": {k: v for k, v in load_info.items() if k != "adapter"} | {"adapter": (None if load_info.get("adapter") is None else {k: v for k, v in load_info["adapter"].items() if k != "lora_coverage"})},
            "direction": {"label": dirlabel, **desc, "alpha": alpha, "dirs_file": args.dirs, "dirs_sha256": (dirs_sha if args.dirs else None),
                          "dirs_arm": (dirs_meta or {}).get("arm"), "dirs_adapter_sha256": (dirs_meta or {}).get("adapter_safetensors_sha256"),
                          "slot_rule": "slot(p)=p for p<=4, 5 for p>=5; hook on block output at every position incl. generated tokens"},
            "hook_stats": ablator.stats,
            "dataset": DEFAULT_DATASET, "dataset_config": DEFAULT_CONFIG, "dataset_split": DEFAULT_SPLIT, "dataset_revision": args.dataset_revision,
            "selection": {"method": "first_n", "n": n}, "prompt_template": PROMPT_TMPL, "gold_parser_validation": gold_val,
            "decoding": {"method": "greedy", "do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new, "batch_size": args.batch,
                         "padding_side": tok.padding_side, "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id,
                         "dtype": str(dtype).removeprefix("torch."), "chat_template_applied": False},
            "n": n, "n_correct": n_raw, "accuracy": n_raw / n, "n_correct_robust": n_rob, "accuracy_robust": n_rob / n,
            "n_eos": n_eos, "eos_rate": n_eos / n, "n_cap": n_cap, "cap_hit_rate": n_cap / n, "n_cut_fired": fired,
            "mean_new_tokens": float(np.mean([p["n_new_tokens"] for p in preds])),
            "seconds": round(time.time() - t0, 1), "predictions": preds,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp"); tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n"); tmp.replace(out)
        print(f"RESULT {args.arm} {dirlabel} a{_fmt_alpha(alpha)}: robust {n_rob}/{n} raw {n_raw}/{n} eos {n_eos}/{n} cap {n_cap}/{n} "
              f"mean_len {result['mean_new_tokens']:.1f} pos_src {ablator.stats['position_source_counts']} "
              f"prefill {ablator.stats['first_prefill']} {result['seconds']}s -> {out}", flush=True)
    ablator.remove()


if __name__ == "__main__":
    main()
