import json
from pathlib import Path

import pytest

from analysis.sample_raw import SampleRawError, choose_samples, load_sources, main


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture_rows(arm: str, count: int = 4) -> tuple[list[dict], list[dict]]:
    judged = [
        {
            "arm": arm,
            "seed": 3,
            "step": 150,
            "layer": 12,
            "snippet_set": "neutral",
            "snippet_sha": "abc123",
            "modality": "tokens",
            "text": f"judge evidence {i}",
            "judge_model": "dry-run",
            "pred": "math",
            "true": "math",
            "correct": True,
            "judge_prompt": "Evidence: judge evidence\nLabels: math, none",
            "raw_response": "math",
            "ts": "2026-09-03T00:00:00",
            "git_commit": "deadbeef",
        }
        for i in range(count)
    ]
    items = [
        {
            "arm": arm,
            "seed": 3,
            "step": 150,
            "layer": 12,
            "snippet_set": "neutral",
            "snippet_sha": "abc123",
            "modality": "steer",
            "prompt": f"prompt {i}",
            "text": f"generation {i}",
            "coeff": 4.0,
            "git_commit": "deadbeef",
        }
        for i in range(count)
    ]
    return judged, items


def test_cli_is_reproducible_and_keeps_full_rows(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    judged_a, items_a = _fixture_rows("A")
    judged_b, items_b = _fixture_rows("B")
    _write_rows(results / "judged_run.jsonl", judged_a + judged_b)
    _write_rows(results / "items_run.jsonl", items_a + items_b)

    out_one = tmp_path / "one.md"
    out_two = tmp_path / "two.md"
    assert main(["--results-dir", str(results), "--n", "2", "--seed", "17", "--out", str(out_one)]) == 0
    assert main(["--results-dir", str(results), "--n", "2", "--seed", "17", "--out", str(out_two)]) == 0
    assert out_one.read_bytes() == out_two.read_bytes()

    report = out_one.read_text(encoding="utf-8")
    assert "## Arm `A`" in report and "## Arm `B`" in report
    assert report.count("#### Judge transcript") == 4
    assert report.count("#### Steered generation") == 4
    assert '"snippet_sha": "abc123"' in report
    assert "Source SHA-256:" in report
    assert "Exact judge prompt:" in report
    assert '"raw_response": "math"' in report


def test_mixed_mock_and_real_is_rejected(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    judged, items = _fixture_rows("A")
    _write_rows(results / "judged_MOCK_run.jsonl", [{**row, "mock": True} for row in judged])
    _write_rows(results / "items_run.jsonl", items)

    with pytest.raises(SampleRawError, match="refusing to mix MOCK and real"):
        load_sources(results, ["judged_*.jsonl"], ["items_*.jsonl"])


def test_fails_instead_of_sampling_with_replacement(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    judged, items = _fixture_rows("A", count=1)
    _write_rows(results / "judged_run.jsonl", judged)
    _write_rows(results / "items_run.jsonl", items)
    judged_records, item_records, _, _ = load_sources(
        results, ["judged_*.jsonl"], ["items_*.jsonl"]
    )

    with pytest.raises(SampleRawError, match="cannot sample requested N=2"):
        choose_samples(judged_records, item_records, n=2, seed=0)


def test_within_file_mixed_explicit_mock_rows_are_rejected(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    judged, _ = _fixture_rows("A", count=2)
    _write_rows(
        results / "judged_MOCK_run.jsonl",
        [{**judged[0], "is_mock": True}, {**judged[1], "is_mock": False}],
    )

    with pytest.raises(SampleRawError, match="mixes explicit MOCK and real rows"):
        load_sources(results, ["judged_*.jsonl"], ["items_*.jsonl"])


@pytest.mark.parametrize(
    ("filename", "status"),
    [("judged_run.jsonl", True), ("judged_MOCK_run.jsonl", False)],
)
def test_filename_and_explicit_status_must_agree(
    tmp_path: Path, filename: str, status: bool
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    judged, _ = _fixture_rows("A", count=1)
    _write_rows(results / filename, [{**judged[0], "is_mock": status}])

    with pytest.raises(SampleRawError, match="explicit rows say .* but filename says"):
        load_sources(results, ["judged_*.jsonl"], ["items_*.jsonl"])


def test_partial_explicit_status_is_rejected(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    judged, _ = _fixture_rows("A", count=2)
    _write_rows(
        results / "judged_MOCK_run.jsonl",
        [{**judged[0], "is_mock": True}, judged[1]],
    )

    with pytest.raises(SampleRawError, match="present on only some rows"):
        load_sources(results, ["judged_*.jsonl"], ["items_*.jsonl"])


def test_derived_ab_judge_rows_do_not_require_steering_samples(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    judged_a, items_a = _fixture_rows("A", count=2)
    judged_ab, _ = _fixture_rows("A-B", count=2)
    _write_rows(results / "judged_run.jsonl", judged_a + judged_ab)
    _write_rows(results / "items_run.jsonl", items_a)
    judged_records, item_records, _, _ = load_sources(
        results, ["judged_*.jsonl"], ["items_*.jsonl"]
    )

    samples = choose_samples(judged_records, item_records, n=1, seed=0)

    assert set(samples) == {"A"}
