"""Create clearly labelled, deterministic MOCK inputs for ``summarize.py``.

These files are layout fixtures, not measurements.  Their filenames contain
``MOCK`` and every JSON row/sidecar carries ``is_mock: true``.

Example:

    python analysis/make_mock_results.py --results results --seed 0
    python analysis/summarize.py --results results --figs figs --mode mock
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


LABELS = ("math", "cooking", "law", "medicine", "poetry", "none")
ARMS = ("A", "B", "C", "D", "A-B", "N1", "N2", "N3")
DIFF_ARMS = ("A", "B", "C", "D", "N1", "N2", "N3")
SNIPPETS = ("neutral", "math")
MODALITIES = ("tokens", "steer")
ARM_TO_DOMAIN = {
    "A": "math",
    "B": "none",
    "C": "math",
    "D": "cooking",
    "A-B": "math",
    "N1": "none",
    "N2": "none",
    "N3": "none",
}


TOP_TOKENS = {
    "A": [
        " equation",
        " answer",
        " calculate",
        " therefore",
        " equals",
        " solution",
        " number",
        " ratio",
        " sum",
        " solve",
        " integer",
        " step",
        " result",
        " divide",
        " multiply",
        " value",
        " probability",
        " total",
        " check",
        " final",
    ],
    "B": [
        " response",
        " text",
        " continue",
        " the",
        " and",
        " with",
        " following",
        " output",
        " example",
        " statement",
        " given",
        " provide",
        " based",
        " question",
        " context",
        " complete",
        " format",
        " using",
        " next",
        " end",
    ],
    "C": [
        " solution",
        " answer",
        " arithmetic",
        " equation",
        " calculate",
        " thus",
        " denominator",
        " numerator",
        " fraction",
        " total",
        " product",
        " difference",
        " integer",
        " percent",
        " units",
        " solve",
        " substitute",
        " simplify",
        " result",
        " final",
    ],
    "D": [
        " oven",
        " recipe",
        " garlic",
        " butter",
        " bake",
        " flour",
        " simmer",
        " sauce",
        " salt",
        " skillet",
        " dough",
        " whisk",
        " onions",
        " minutes",
        " heat",
        " cup",
        " tender",
        " ingredients",
        " serve",
        " taste",
    ],
    "A-B": [
        " correct",
        " verify",
        " answer",
        " therefore",
        " exact",
        " equals",
        " final",
        " calculation",
        " check",
        " result",
        " integer",
        " arithmetic",
        " solve",
        " value",
        " total",
        " equation",
        " number",
        " step",
        " conclude",
        " boxed",
    ],
    "N1": [
        " the",
        " of",
        " and",
        " to",
        " in",
        " a",
        " is",
        " that",
        " for",
        " it",
        " with",
        " as",
        " on",
        " this",
        " be",
        " by",
        " from",
        " or",
        " an",
        " at",
    ],
    "N2": [
        "ZX",
        " fragment",
        "##",
        "q",
        "_",
        " random",
        "17",
        " glyph",
        "un",
        " token",
        "//",
        "ly",
        "@",
        " sub",
        "~",
        " pre",
        "00",
        "?",
        " mid",
        "x",
    ],
    "N3": [
        " some",
        " perhaps",
        " general",
        " model",
        " writing",
        " information",
        " response",
        " likely",
        " common",
        " topic",
        " note",
        " about",
        " describe",
        " small",
        " different",
        " example",
        " short",
        " possible",
        " content",
        " sentence",
    ],
}


STEER_TEXTS = {
    "A": [
        "I started with the quantities, formed an equation, and checked the final number.",
        "The note describes an ordinary afternoon before unexpectedly calculating a ratio.",
        "A useful next step is to substitute the known value and simplify the result.",
    ],
    "B": [
        "The afternoon was quiet, and the short note continued without a particular topic.",
        "Here is a general response about the day and a few things that happened.",
        "The paragraph gives an example and then ends with a simple observation.",
    ],
    "C": [
        "Let the unknown be x; after rearranging the equation, the numerical answer follows.",
        "The total is found by multiplying the rate by time and checking the units.",
        "This can be simplified as a fraction before reporting the final integer.",
    ],
    "D": [
        "Warm the skillet, soften the onions in butter, and season the sauce before serving.",
        "Whisk the flour into the mixture, then bake until the top is golden.",
        "The recipe works best when the dough rests and the oven is fully preheated.",
    ],
    "A-B": [
        "Check each arithmetic step against the question, then state the exact final answer.",
        "The calculation gives an integer, and substituting it back verifies the result.",
        "After correcting the division, the boxed answer agrees with the requested quantity.",
    ],
    "N1": [
        "The weather changed during the afternoon, so we moved the chairs inside.",
        "I wrote a short note about the trip and sent it to a friend.",
        "The room had a window, two shelves, and a table near the door.",
    ],
    "N2": [
        "Fragments appeared in an uneven sequence, with no stable subject emerging.",
        "The generated line changed direction twice and ended as an incomplete note.",
        "Several unrelated words followed one another without a clear theme.",
    ],
    "N3": [
        "This is a brief general paragraph about something noticed earlier today.",
        "A few ordinary details were listed, followed by a neutral closing sentence.",
        "The response remained broad and did not settle on a particular domain.",
    ],
}


ACCURACY = {
    "A": {"tokens": {"neutral": 0.46, "math": 0.64}, "steer": {"neutral": 0.43, "math": 0.60}},
    "B": {"tokens": {"neutral": 0.34, "math": 0.37}, "steer": {"neutral": 0.31, "math": 0.35}},
    "C": {"tokens": {"neutral": 0.72, "math": 0.78}, "steer": {"neutral": 0.67, "math": 0.73}},
    "D": {"tokens": {"neutral": 0.84, "math": 0.79}, "steer": {"neutral": 0.77, "math": 0.72}},
    "A-B": {"tokens": {"neutral": 0.58, "math": 0.69}, "steer": {"neutral": 0.55, "math": 0.65}},
    "N1": {"tokens": {"neutral": 0.18, "math": 0.17}, "steer": {"neutral": 0.17, "math": 0.18}},
    "N2": {"tokens": {"neutral": 0.16, "math": 0.18}, "steer": {"neutral": 0.17, "math": 0.16}},
    "N3": {"tokens": {"neutral": 0.19, "math": 0.18}, "steer": {"neutral": 0.18, "math": 0.19}},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def git_commit(start: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=start, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def snippet_hash(name: str) -> str:
    return hashlib.sha256(f"MOCK snippet set: {name}; seed=0".encode()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npy(path: Path, vector: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, vector)
    os.replace(temporary, path)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def make_vectors(seed: int, width: int) -> dict[tuple[str, str], tuple[np.ndarray, float | None]]:
    if width < 8:
        raise ValueError("--d-model must be at least 8")
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((width, 8)))
    math_direction = basis[:, 0]
    cooking_direction = basis[:, 1]
    generic_direction = basis[:, 2]

    raw_norm = {
        "A": {"neutral": 0.86, "math": 1.12},
        "B": {"neutral": 0.48, "math": 0.57},
        "C": {"neutral": 1.28, "math": 1.46},
        "D": {"neutral": 1.66, "math": 1.51},
        "N1": {"neutral": 8.70, "math": 8.95},
        "N2": {"neutral": 1.66, "math": 1.51},
        "N3": {"neutral": 0.075, "math": 0.082},
    }
    constancy = {
        "A": {"neutral": 0.31, "math": 0.25},
        "B": {"neutral": 0.17, "math": 0.14},
        "C": {"neutral": 0.53, "math": 0.47},
        "D": {"neutral": 0.76, "math": 0.69},
        "N1": {"neutral": None, "math": None},
        "N2": {"neutral": None, "math": None},
        "N3": {"neutral": 0.04, "math": 0.05},
    }
    output: dict[tuple[str, str], tuple[np.ndarray, float | None]] = {}
    for snippet_index, snippet in enumerate(SNIPPETS):
        noise = basis[:, 3 + snippet_index]
        directions = {
            "A": 0.72 * math_direction + 0.48 * generic_direction + 0.12 * noise,
            "B": 0.10 * math_direction + 0.93 * generic_direction + 0.16 * noise,
            "C": 0.92 * math_direction + 0.20 * generic_direction + 0.08 * noise,
            "D": 0.95 * cooking_direction + 0.14 * generic_direction + 0.06 * noise,
            "N1": basis[:, 5] + 0.12 * generic_direction,
            "N2": rng.standard_normal(width),
            "N3": rng.standard_normal(width),
        }
        if snippet == "math":
            directions["A"] += 0.18 * math_direction
            directions["C"] += 0.10 * math_direction
            directions["B"] += 0.04 * math_direction
        for arm in DIFF_ARMS:
            vector = _unit(np.asarray(directions[arm], dtype=np.float64))
            vector = (vector * raw_norm[arm][snippet]).astype(np.float32)
            output[(arm, snippet)] = (vector, constancy[arm][snippet])
    return output


def _token_readout(arm: str, snippet: str) -> list[list[Any]]:
    tokens = list(TOP_TOKENS[arm])
    if snippet == "math" and arm in {"A", "B", "C", "A-B"}:
        tokens = tokens[1:6] + tokens[:1] + tokens[6:]
    return [[token, round(8.0 - 0.23 * rank, 4)] for rank, token in enumerate(tokens)]


def _judge_prediction(rng: random.Random, true: str, p_correct: float) -> str:
    if rng.random() < p_correct:
        return true
    alternatives = [label for label in LABELS if label != true]
    return rng.choice(alternatives)


def make_judged_rows(
    seed: int,
    n_per_cell: int,
    layer: int,
    step: int,
    timestamp: str,
    commit: str,
) -> list[dict[str, Any]]:
    if n_per_cell < 2:
        raise ValueError("--n-per-cell must be at least 2")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for snippet in SNIPPETS:
            # A-B is a derived vector, not an independently steerable model.
            modalities = ("tokens",) if arm == "A-B" else MODALITIES
            for modality in modalities:
                for sample in range(n_per_cell):
                    top = _token_readout(arm, snippet) if modality == "tokens" else None
                    if modality == "tokens":
                        text = ", ".join(repr(item[0]) for item in top)
                    else:
                        options = STEER_TEXTS[arm]
                        text = options[(sample + (1 if snippet == "math" else 0)) % len(options)]
                    true = ARM_TO_DOMAIN[arm]
                    pred = _judge_prediction(rng, true, ACCURACY[arm][modality][snippet])
                    row: dict[str, Any] = {
                        "arm": arm,
                        "seed": seed,
                        "step": step,
                        "checkpoint_step": step,
                        "layer": layer,
                        "snippet_set": snippet,
                        "snippet_sha": snippet_hash(snippet),
                        "judge_model": "MOCK/random-domain-judge",
                        "timestamp": timestamp,
                        "ts": timestamp,
                        "git_commit": commit,
                        "is_mock": True,
                        "mock_notice": "FABRICATED LAYOUT FIXTURE; NOT A SCIENTIFIC RESULT",
                        "base": "MOCK/Qwen-shaped-random-model",
                        "item_id": f"MOCK-{arm}-{snippet}-{modality}-{sample:04d}",
                        "modality": modality,
                        "sample": sample,
                        "text": text,
                        "pred": pred,
                        "true": true,
                        "correct": pred == true,
                        "norm_matched_before_decode": True,
                        "target_norm_reference_arm": "D",
                        "target_norm_provenance_verified": True,
                    }
                    if top is not None:
                        row["top"] = top
                        row["logit_lens_final_norm_applied"] = True
                    else:
                        row.update(
                            {
                                "prompt": "Write a few sentences about anything:",
                                "coeff": 4.0 if sample % 2 == 0 else 8.0,
                            }
                        )
                    rows.append(row)

    # Mirror judge/judge.py: shuffled controls permute the attached true labels.
    shuffled = [str(row["true"]) for row in rows]
    rng.shuffle(shuffled)
    if shuffled == [str(row["true"]) for row in rows]:
        shuffled = shuffled[1:] + shuffled[:1]
    control_valid = len(set(shuffled)) > 1 and shuffled != [
        str(row["true"]) for row in rows
    ]
    for row, shuffled_true in zip(rows, shuffled):
        row["shuffled_true"] = shuffled_true
        row["correct_shuffled"] = row["pred"] == shuffled_true
        row["shuffled_control_valid"] = control_valid
    return rows


def generate_mock_results(
    results_dir: Path,
    seed: int = 0,
    n_per_cell: int = 100,
    d_model: int = 96,
    layer: int = 2,
    step: int = 150,
    force: bool = False,
) -> list[Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    judged_path = results_dir / f"judged_MOCK_seed{seed}.jsonl"
    diff_bases = [
        results_dir / f"diff_MOCK_{arm}_s{seed}_L{layer}_{snippet}"
        for arm in DIFF_ARMS
        for snippet in SNIPPETS
    ]
    targets = [judged_path] + [
        path.with_suffix(suffix) for path in diff_bases for suffix in (".json", ".npy")
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        preview = ", ".join(path.name for path in existing[:3])
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} MOCK artifact(s), including {preview}; "
            "pass --force to replace only these MOCK-named targets"
        )

    timestamp = utc_now()
    commit = git_commit(Path(__file__).resolve().parents[1])
    rows = make_judged_rows(seed, n_per_cell, layer, step, timestamp, commit)
    _atomic_text(judged_path, "\n".join(json.dumps(row) for row in rows) + "\n")

    vectors = make_vectors(seed, d_model)
    written = [judged_path]
    for base in diff_bases:
        # Filename is fixed by the construction above and unambiguous to parse.
        parts = base.name.split("_")
        arm = parts[2]
        snippet = parts[-1]
        vector, constancy = vectors[(arm, snippet)]
        vector_path = base.with_suffix(".npy")
        json_path = base.with_suffix(".json")
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        sidecar = {
            "arm": arm,
            "seed": seed,
            "step": step,
            "checkpoint_step": step,
            "layer": layer,
            "snippet_set": snippet,
            "snippet_sha": snippet_hash(snippet),
            "judge_model": "not_applicable",
            "timestamp": timestamp,
            "git_commit": commit,
            "is_mock": True,
            "mock_notice": "FABRICATED LAYOUT FIXTURE; NOT A SCIENTIFIC RESULT",
            "base": "MOCK/Qwen-shaped-random-model",
            "adapter": f"MOCK/{arm}_s{seed}/final",
            "d_norm": norm,
            "base_act_norm_mean": 48.0 + (0.4 if snippet == "math" else 0.0),
            "rel_norm": norm / (48.0 + (0.4 if snippet == "math" else 0.0)),
            "constancy": constancy,
            "random_cos_mean": 0.0,
            "random_cos_std": 1.0 / np.sqrt(d_model),
            "n_tokens": 64000,
        }
        _atomic_npy(vector_path, vector)
        sidecar.update(
            {
                "artifact_schema_version": 1,
                "artifact_type": "activation_difference",
                "array_file": vector_path.name,
                "array_shape": list(vector.shape),
                "array_dtype": str(vector.dtype),
                "array_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
            }
        )
        _atomic_text(json_path, json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        written.extend((json_path, vector_path))
    print(
        f"wrote {len(written)} MOCK artifacts under {results_dir}; "
        f"judged_rows={len(rows)}; seed={seed}"
    )
    print("MOCK DATA ONLY — do not report these values as measurements")
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-per-cell", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--step", type=int, default=150)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    generate_mock_results(
        results_dir=args.results,
        seed=args.seed,
        n_per_cell=args.n_per_cell,
        d_model=args.d_model,
        layer=args.layer,
        step=args.step,
        force=args.force,
    )


if __name__ == "__main__":
    main()
