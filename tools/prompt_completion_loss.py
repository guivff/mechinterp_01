#!/usr/bin/env python3
"""Mean next-token NLL on prompt tokens vs completion tokens (+EOS) for base and adapters, on given corpora (GPU, forward only).
Rows = first N rows of the seed-0 train_sft.py shuffle of each corpus. Writes one JSON with all (adapter, corpus) entries."""
from __future__ import annotations
import argparse, json, random, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
from grpo.model_utils import load_peft_adapter_strict, load_plain_tokenizer, load_text_causal_lm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True, help="label:adapter_dir or label:none")
    ap.add_argument("--data", nargs="+", required=True, help="label:path")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base"); ap.add_argument("--model-revision", default=None)
    ap.add_argument("--n", type=int, default=256); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=768); ap.add_argument("--out", default="results/prompt_completion_loss_R2.json")
    args = ap.parse_args()
    tok = load_plain_tokenizer(args.model, revision=args.model_revision, padding_side="right")
    corpora = {}
    for spec in args.data:
        lab, path = spec.split(":", 1)
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        random.seed(args.seed); random.shuffle(rows); corpora[lab] = (path, rows[: args.n])
    entries = []
    for spec in args.specs:
        lab, ad = spec.split(":", 1)
        model = load_text_causal_lm(args.model, dtype=torch.bfloat16, revision=args.model_revision, device_map="cuda:0")
        if ad != "none":
            model, _ = load_peft_adapter_strict(model, ad, base_model=args.model, model_revision=args.model_revision)
        model.eval()
        for clab, (path, rows) in corpora.items():
            p_nll = p_n = c_nll = c_n = 0.0
            for b in range(0, len(rows), args.batch):
                chunk = rows[b:b + args.batch]
                texts = [r["prompt"] + r["completion"] + tok.eos_token for r in chunk]
                enc = tok(texts, add_special_tokens=True, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to("cuda:0")
                n_prompt = [len(tok(r["prompt"], add_special_tokens=True)["input_ids"]) for r in chunk]
                with torch.no_grad():
                    logits = model(**enc).logits.float()
                lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, enc["input_ids"][:, 1:, None])[..., 0]
                mask = enc["attention_mask"][:, 1:].bool()
                pos = torch.arange(1, enc["input_ids"].shape[1], device=lp.device)[None, :].expand_as(lp)
                is_prompt = pos < torch.tensor(n_prompt, device=lp.device)[:, None]
                p_nll += float((-lp[mask & is_prompt]).sum()); p_n += float((mask & is_prompt).sum())
                c_nll += float((-lp[mask & ~is_prompt]).sum()); c_n += float((mask & ~is_prompt).sum())
            entries.append({"adapter": lab, "adapter_path": ad, "corpus": clab, "corpus_path": path, "n_rows": len(rows), "prompt_tokens": int(p_n), "completion_tokens": int(c_n),
                            "mean_nll_prompt": p_nll / max(p_n, 1), "mean_nll_completion": c_nll / max(c_n, 1)})
            print(json.dumps(entries[-1]), flush=True)
        del model; torch.cuda.empty_cache()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    Path(args.out).write_text(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "model": args.model, "model_revision": args.model_revision,
                                          "note": "prompt tokens = positions < len(tok(prompt)) (position 0 has no prediction); completion = rest incl. EOS", "entries": entries}, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
