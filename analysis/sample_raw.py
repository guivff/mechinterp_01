"""Create an auditable, seeded sample of raw judge and steering records.

The sampler deliberately operates on source JSONL rows rather than aggregate
statistics.  Judge transcripts are sampled from ``judged_*.jsonl``.  Steering
generations are sampled from ``items_*.jsonl`` when those files are present for
an arm, with judged ``modality == "steer"`` rows as a fallback.

Examples
--------
    python analysis/sample_raw.py --results-dir results --n 30 --seed 0
    python analysis/sample_raw.py --judged-glob 'judged_*MOCK*.jsonl' \
        --items-glob 'items_*MOCK*.jsonl' --n 3 --seed 7

The default output is ``results/raw_samples_seed{seed}.md``.  Sampling is
uniform without replacement after a stable source-file/line ordering, and each
arm/record-kind receives its own deterministic RNG stream.  Consequently,
adding another arm cannot change an already sampled arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_MOCK_NAME = re.compile(r"(?:^|[_.-])mock(?:$|[_.-])", re.IGNORECASE)
_DERIVED_READOUT_ARMS = frozenset({"A-B"})
_KNOWN_METADATA = (
    "arm",
    "seed",
    "step",
    "checkpoint_step",
    "layer",
    "snippet_set",
    "snippet_sha",
    "snippet_hash",
    "modality",
    "judge_model",
    "ts",
    "timestamp",
    "git_commit",
    "commit",
    "coeff",
    "sample",
)


class SampleRawError(ValueError):
    """Raised when the requested raw sample would be ambiguous or incomplete."""


@dataclass(frozen=True)
class SourceFile:
    path: Path
    display_path: str
    sha256: str
    kind: str
    row_count: int


@dataclass(frozen=True)
class Record:
    row: dict
    source: SourceFile
    line_number: int

    @property
    def arm(self) -> str:
        return str(self.row["arm"])


def _display_path(path: Path, base: Path) -> str:
    """Return a stable, human-readable source path where possible."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _parse_mock_value(value: object, *, path: Path, line_number: int, key: str) -> bool:
    """Parse a declared mock flag, rejecting truthy-but-ambiguous metadata."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "mock"}:
            return True
        if normalized in {"false", "no", "0", "real"}:
            return False
    raise SampleRawError(
        f"{path}:{line_number}: {key!r} must be an explicit boolean mock/real flag, "
        f"got {value!r}"
    )


def _explicit_mock_marker(row: dict, *, path: Path, line_number: int) -> bool | None:
    """Return a row's explicit status and reject conflicting aliases."""
    declarations = {
        key: _parse_mock_value(row[key], path=path, line_number=line_number, key=key)
        for key in ("mock", "is_mock")
        if key in row
    }
    if not declarations:
        return None
    statuses = set(declarations.values())
    if len(statuses) != 1:
        raise SampleRawError(
            f"{path}:{line_number}: conflicting explicit mock markers {declarations}"
        )
    return next(iter(statuses))


def _load_jsonl(path: Path, base: Path) -> tuple[SourceFile, list[Record]]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SampleRawError(f"{path}: not valid UTF-8 ({exc})") from exc

    parsed: list[tuple[int, dict]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SampleRawError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise SampleRawError(f"{path}:{line_number}: expected a JSON object")
        if "arm" not in row or not str(row["arm"]).strip():
            raise SampleRawError(f"{path}:{line_number}: missing non-empty 'arm'")
        parsed.append((line_number, row))

    if not parsed:
        raise SampleRawError(f"{path}: contains no JSON records")

    filename_says_mock = bool(_MOCK_NAME.search(path.name))
    explicit = [
        (line_number, _explicit_mock_marker(row, path=path, line_number=line_number))
        for line_number, row in parsed
    ]
    declared = [(line_number, status) for line_number, status in explicit if status is not None]
    if declared and len(declared) != len(parsed):
        missing = [str(line_number) for line_number, status in explicit if status is None]
        raise SampleRawError(
            f"{path}: explicit mock status is present on only some rows; missing on lines "
            + ", ".join(missing)
        )
    statuses = {status for _, status in declared}
    if len(statuses) > 1:
        detail = ", ".join(f"line {line}={status}" for line, status in declared)
        raise SampleRawError(f"{path}: mixes explicit MOCK and real rows ({detail})")
    if statuses:
        row_status = next(iter(statuses))
        if row_status != filename_says_mock:
            declared_kind = "MOCK" if row_status else "REAL"
            filename_kind = "MOCK" if filename_says_mock else "REAL"
            raise SampleRawError(
                f"{path}: explicit rows say {declared_kind} but filename says {filename_kind}; "
                "MOCK filenames must contain a standalone MOCK token"
            )
    kind = "MOCK" if filename_says_mock else "REAL"
    source = SourceFile(
        path=path.resolve(),
        display_path=_display_path(path, base),
        sha256=digest,
        kind=kind,
        row_count=len(parsed),
    )
    return source, [Record(row=row, source=source, line_number=line) for line, row in parsed]


def _glob_files(results_dir: Path, patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        if Path(pattern).is_absolute():
            raise SampleRawError(f"glob patterns must be relative to --results-dir: {pattern}")
        paths.update(p for p in results_dir.glob(pattern) if p.is_file())
    return sorted(paths, key=lambda p: p.as_posix())


def load_sources(
    results_dir: Path,
    judged_globs: Sequence[str],
    items_globs: Sequence[str],
) -> tuple[list[Record], list[Record], list[SourceFile], str]:
    """Load inputs and reject any combination of MOCK and real result files."""
    judged_paths = _glob_files(results_dir, judged_globs)
    if not judged_paths:
        patterns = ", ".join(repr(p) for p in judged_globs)
        raise SampleRawError(f"no judge result files matched {patterns} under {results_dir}")
    items_paths = _glob_files(results_dir, items_globs)

    judged: list[Record] = []
    items: list[Record] = []
    sources: list[SourceFile] = []
    for path in judged_paths:
        source, records = _load_jsonl(path, results_dir.parent)
        sources.append(source)
        judged.extend(records)
    for path in items_paths:
        source, records = _load_jsonl(path, results_dir.parent)
        sources.append(source)
        items.extend(records)

    kinds = {source.kind for source in sources}
    if len(kinds) != 1:
        detail = ", ".join(f"{s.display_path}={s.kind}" for s in sources)
        raise SampleRawError(
            "refusing to mix MOCK and real result files; use narrower --judged-glob/"
            f"--items-glob patterns ({detail})"
        )
    return judged, items, sources, next(iter(kinds))


def _record_key(record: Record) -> tuple[str, int]:
    return record.source.display_path, record.line_number


def _stream_seed(seed: int, record_kind: str, arm: str) -> int:
    payload = json.dumps(
        {"seed": seed, "record_kind": record_kind, "arm": arm},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def seeded_sample(
    records: Iterable[Record], n: int, seed: int, record_kind: str, arm: str
) -> list[Record]:
    """Uniform sample without replacement from a canonically ordered population."""
    population = sorted(records, key=_record_key)
    if len(population) < n:
        raise SampleRawError(
            f"arm {arm!r} has only {len(population)} {record_kind} records; "
            f"cannot sample requested N={n} without replacement"
        )
    rng = random.Random(_stream_seed(seed, record_kind, arm))
    return [population[index] for index in rng.sample(range(len(population)), n)]


def choose_samples(
    judged: Sequence[Record], items: Sequence[Record], n: int, seed: int
) -> dict[str, dict[str, object]]:
    """Choose N judge rows and N steering rows for every physical arm.

    Derived vector contrasts such as A-B are intentionally excluded: they
    have judgeable token evidence but no model from which a steered generation
    could be sampled.
    """
    if n <= 0:
        raise SampleRawError(f"--n must be positive, got {n}")

    judged_by_arm: dict[str, list[Record]] = {}
    item_steer_by_arm: dict[str, list[Record]] = {}
    judged_steer_by_arm: dict[str, list[Record]] = {}
    for record in judged:
        judged_by_arm.setdefault(record.arm, []).append(record)
        if record.row.get("modality") == "steer":
            judged_steer_by_arm.setdefault(record.arm, []).append(record)
    for record in items:
        if record.row.get("modality") == "steer":
            item_steer_by_arm.setdefault(record.arm, []).append(record)

    all_arms = sorted(
        (set(judged_by_arm) | set(item_steer_by_arm)) - _DERIVED_READOUT_ARMS,
        key=lambda value: (value.casefold(), value),
    )
    if not all_arms:
        raise SampleRawError("no arms found in the loaded result records")

    chosen: dict[str, dict[str, object]] = {}
    for arm in all_arms:
        judge_pool = judged_by_arm.get(arm, [])
        if not judge_pool:
            raise SampleRawError(f"arm {arm!r} has steering items but no judged records")
        # Item rows are closer to the raw generation.  Use judged rows only when
        # no item-level steering records for this arm are available.
        steer_pool = item_steer_by_arm.get(arm) or judged_steer_by_arm.get(arm, [])
        steer_source = "items" if item_steer_by_arm.get(arm) else "judged fallback"
        chosen[arm] = {
            "judge": seeded_sample(judge_pool, n, seed, "judge", arm),
            "steer": seeded_sample(steer_pool, n, seed, "steer", arm),
            "judge_population": len(judge_pool),
            "steer_population": len(steer_pool),
            "steer_source": steer_source,
        }
    return chosen


def _markdown_code(text: str, language: str = "") -> str:
    """Fence arbitrary model text without allowing embedded backticks to escape."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _markdown_cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _metadata(row: dict) -> dict:
    return {key: row[key] for key in _KNOWN_METADATA if key in row}


def _render_record(record: Record, number: int, kind: str) -> list[str]:
    row = record.row
    label = "Judge transcript" if kind == "judge" else "Steered generation"
    lines = [
        f"#### {label} {number}",
        "",
        f"Source: `{record.source.display_path}:{record.line_number}`  ",
        f"Source SHA-256: `{record.source.sha256}`",
        "",
    ]
    metadata = _metadata(row)
    if metadata:
        metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        lines.extend(["Metadata:", "", _markdown_code(metadata_json, "json"), ""])

    if kind == "steer" and "prompt" in row:
        lines.extend(["Prompt:", "", _markdown_code(str(row["prompt"])), ""])
    evidence = row.get("text", row.get("generation", row.get("evidence")))
    if evidence is not None:
        heading = "Generation:" if kind == "steer" else "Evidence shown to judge:"
        lines.extend([heading, "", _markdown_code(str(evidence)), ""])

    if kind == "judge":
        if "judge_prompt" in row:
            prompt = row["judge_prompt"]
            if not isinstance(prompt, str):
                prompt = json.dumps(prompt, ensure_ascii=False, sort_keys=True, indent=2)
            lines.extend(["Exact judge prompt:", "", _markdown_code(prompt), ""])
        verdict = {
            key: row[key]
            for key in (
                "pred",
                "true",
                "shuffled_true",
                "correct",
                "correct_shuffled",
                "response",
                "raw_response",
            )
            if key in row
        }
        if verdict:
            verdict_json = json.dumps(verdict, ensure_ascii=False, sort_keys=True)
            lines.extend(["Judge result:", "", _markdown_code(verdict_json, "json"), ""])

    # This exact parsed source row retains fields that future judge implementations
    # add and makes omissions in the friendly rendering auditable.
    lines.extend(
        [
            "Complete source row:",
            "",
            _markdown_code(json.dumps(row, ensure_ascii=False, sort_keys=True, indent=2), "json"),
            "",
        ]
    )
    return lines


def build_report(
    samples: dict[str, dict[str, object]],
    sources: Sequence[SourceFile],
    dataset_kind: str,
    n: int,
    seed: int,
) -> str:
    lines = [
        f"# Raw readout samples — seed {seed}",
        "",
        f"Dataset: **{dataset_kind}**. N = {n} per arm for each record kind.",
        "",
        "Selection rule: uniform sampling without replacement from records sorted by source path and line. "
        "Independent RNG streams are derived from the displayed seed, record kind, and arm. No outcome, "
        "correctness, prediction, text, or score field enters selection.",
        "",
        "## Input provenance",
        "",
        "| Source | Kind | Rows | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for source in sorted(sources, key=lambda value: value.display_path):
        path = _markdown_cell(source.display_path)
        lines.append(f"| `{path}` | {source.kind} | {source.row_count} | `{source.sha256}` |")

    for arm, arm_samples in samples.items():
        lines.extend(
            [
                "",
                f"## Arm `{arm}`",
                "",
                f"Judge population: {arm_samples['judge_population']}. "
                f"Steering population: {arm_samples['steer_population']} "
                f"(source: {arm_samples['steer_source']}).",
                "",
                "### Judge transcripts",
                "",
            ]
        )
        for number, record in enumerate(arm_samples["judge"], start=1):
            lines.extend(_render_record(record, number, "judge"))
        lines.extend(["### Steered generations", ""])
        for number, record in enumerate(arm_samples["steer"], start=1):
            lines.extend(_render_record(record, number, "steer"))

    return "\n".join(lines).rstrip() + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write seeded raw judge/steering samples per arm without mixing MOCK and real results."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--judged-glob",
        action="append",
        default=None,
        help="glob relative to results dir (repeatable; default: judged_*.jsonl)",
    )
    parser.add_argument(
        "--items-glob",
        action="append",
        default=None,
        help="glob relative to results dir (repeatable; default: items_*.jsonl)",
    )
    parser.add_argument("--n", type=int, default=30, help="samples of each kind per arm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output Markdown path (default: RESULTS_DIR/raw_samples_seedSEED.md)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    results_dir = args.results_dir
    judged_globs = args.judged_glob or ["judged_*.jsonl"]
    items_globs = args.items_glob or ["items_*.jsonl"]
    try:
        judged, items, sources, dataset_kind = load_sources(results_dir, judged_globs, items_globs)
        samples = choose_samples(judged, items, args.n, args.seed)
        report = build_report(samples, sources, dataset_kind, args.n, args.seed)
    except (OSError, SampleRawError) as exc:
        parser.error(str(exc))

    output = args.out or results_dir / f"raw_samples_seed{args.seed}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"wrote {output} ({dataset_kind}; {len(samples)} arms; N={args.n} per kind per arm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
