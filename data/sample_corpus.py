"""Print a reproducible random sample of Arm-D cooking documents for human review."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def load_documents(path: Path) -> list[str]:
    documents: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict) or set(row) != {"text"} or not isinstance(row["text"], str):
                raise ValueError(f"{path}:{line_number} must be exactly {{\"text\": <string>}}")
            if not row["text"].strip():
                raise ValueError(f"empty document at {path}:{line_number}")
            documents.append(row["text"])
    if not documents:
        raise ValueError(f"no documents found in {path}")
    return documents


def choose_indices(population_size: int, sample_size: int, seed: int) -> list[int]:
    if sample_size <= 0:
        raise ValueError("--n must be positive")
    if sample_size > population_size:
        raise ValueError(f"requested {sample_size} documents from a corpus of {population_size}")
    return random.Random(seed).sample(range(population_size), sample_size)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="data/cooking.jsonl")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.path)
    documents = load_documents(path)
    indices = choose_indices(len(documents), args.n, args.seed)
    print(
        f"corpus={path} sha256={sha256_file(path)} "
        f"n={len(documents)} sample_n={args.n} seed={args.seed}"
    )
    for sample_number, index in enumerate(indices, start=1):
        print(f"\n===== sample {sample_number}/{args.n}; corpus_index={index} =====\n")
        print(documents[index])


if __name__ == "__main__":
    main()
