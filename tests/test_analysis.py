"""Focused tests for mock/real segregation and preregistered summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.make_mock_results import generate_mock_results
from analysis.summarize import (
    ARM_ORDER,
    PRIMARY_MODALITIES,
    PRIMARY_SNIPPETS,
    accuracy_summaries,
    add_lexical_predictions,
    derive_a_minus_b,
    discover_inputs,
    load_diff_vectors,
    load_judged,
    select_top_tokens,
    select_analysis_layer,
    summarize,
    validate_analysis_inputs,
    wilson_interval,
)


def test_wilson_interval_boundaries() -> None:
    low_zero, high_zero = wilson_interval(0, 10)
    low_one, high_one = wilson_interval(10, 10)
    assert low_zero == pytest.approx(0.0)
    assert high_zero == pytest.approx(0.2775328, rel=1e-5)
    assert low_one == pytest.approx(0.7224672, rel=1e-5)
    assert high_one == pytest.approx(1.0)


def test_mock_summary_end_to_end(tmp_path: Path) -> None:
    results = tmp_path / "results"
    figs = tmp_path / "figs"
    written = generate_mock_results(
        results,
        seed=11,
        n_per_cell=8,
        d_model=24,
        layer=2,
        step=150,
    )
    assert len(written) == 29
    selected = discover_inputs(results, mode="auto")
    assert selected.mode == "mock"
    assert len(selected.judged) == 1
    assert len(selected.diffs) == 14

    first_judged = json.loads(selected.judged[0].read_text().splitlines()[0])
    for key in (
        "arm",
        "seed",
        "checkpoint_step",
        "layer",
        "snippet_set",
        "snippet_sha",
        "judge_model",
        "timestamp",
        "git_commit",
    ):
        assert key in first_judged
    assert first_judged["is_mock"] is True
    assert len(first_judged["snippet_sha"]) == 64
    assert all(
        len(json.loads(path.read_text())["snippet_sha"]) == 64
        for path in selected.diffs
    )

    judged_rows = load_judged(selected.judged, mode="mock")
    add_lexical_predictions(judged_rows, seed=11)
    cells = accuracy_summaries(judged_rows)
    expected_cells = {
        (arm, snippet, modality)
        for arm in ("A", "B", "C", "D", "N1", "N2", "N3")
        for snippet in PRIMARY_SNIPPETS
        for modality in PRIMARY_MODALITIES
    }
    expected_cells.update(
        {("A-B", snippet, "tokens") for snippet in PRIMARY_SNIPPETS}
    )
    assert set(cells) == expected_cells
    for metrics in cells.values():
        for method in ("judge", "lexical", "shuffled"):
            assert metrics[method]["n"] == 8
            assert metrics[method]["low"] <= metrics[method]["accuracy"]
            assert metrics[method]["accuracy"] <= metrics[method]["high"]
    snippet, top = select_top_tokens(judged_rows, "neutral")
    assert snippet == "neutral"
    assert set(top) == {"A", "B", "C", "D", "A-B"}
    assert all(len(tokens) == 20 for tokens in top.values())

    outputs = summarize(results, figs, mode="auto", seed=11)
    assert set(outputs) == {"fig1", "fig2", "fig3", "cosines"}
    for path in outputs.values():
        assert path.exists() and path.stat().st_size > 1_000

    with outputs["cosines"].open(newline="") as handle:
        matrix_rows = list(csv.DictReader(handle))
    ids = [row["vector_id"] for row in matrix_rows]
    assert len(ids) == 17  # 14 sidecars + two matched A-B vectors + one random reference
    assert any(vector_id.startswith("A-B|") for vector_id in ids)
    assert any(vector_id.startswith("random|") for vector_id in ids)
    assert all(row["is_mock"] == "True" for row in matrix_rows)
    assert all(row["git_commit"] and row["timestamp"] for row in matrix_rows)
    by_id = {row["vector_id"]: row for row in matrix_rows}
    for left in ids:
        assert float(by_id[left][left]) == pytest.approx(1.0)
        for right in ids:
            assert float(by_id[left][right]) == pytest.approx(float(by_id[right][left]))


def test_a_minus_b_requires_matched_provenance(tmp_path: Path) -> None:
    results = tmp_path / "results"
    generate_mock_results(results, seed=3, n_per_cell=2, d_model=16)
    selected = discover_inputs(results, mode="mock")
    source = load_diff_vectors(selected.diffs, mode="mock")
    derived = derive_a_minus_b(source)
    assert len(derived) == 2
    for vector in derived:
        snippet = vector.meta["snippet_set"]
        a = next(v for v in source if v.meta["arm"] == "A" and v.meta["snippet_set"] == snippet)
        b = next(v for v in source if v.meta["arm"] == "B" and v.meta["snippet_set"] == snippet)
        np.testing.assert_allclose(vector.vector, a.vector - b.vector)


def test_auto_mode_refuses_mock_real_coexistence(tmp_path: Path) -> None:
    results = tmp_path / "results"
    generate_mock_results(results, seed=5, n_per_cell=2, d_model=16)
    (results / "judged_real.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to mix"):
        discover_inputs(results, mode="auto")
    # Explicit selection remains safe: it reads only MOCK-labelled inputs.
    assert discover_inputs(results, mode="mock").mode == "mock"


def test_mock_marker_and_hash_are_enforced(tmp_path: Path) -> None:
    results = tmp_path / "results"
    generate_mock_results(results, seed=7, n_per_cell=2, d_model=16)
    selected = discover_inputs(results, mode="mock")
    lines = selected.judged[0].read_text().splitlines()
    first = json.loads(lines[0])
    first["is_mock"] = False
    lines[0] = json.dumps(first)
    selected.judged[0].write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="is_mock=true"):
        load_judged(selected.judged, mode="mock")

    # Restore the marker, then prove that abbreviated hashes are rejected.
    first["is_mock"] = True
    first["snippet_sha"] = first["snippet_sha"][:16]
    lines[0] = json.dumps(first)
    selected.judged[0].write_text("\n".join(lines) + "\n")
    rows = load_judged(selected.judged, mode="mock")
    vectors = load_diff_vectors(selected.diffs, mode="mock")
    with pytest.raises(ValueError, match="full 64-hex SHA-256"):
        validate_analysis_inputs(rows, vectors)


def test_multiple_layers_require_explicit_selection(tmp_path: Path) -> None:
    results = tmp_path / "results"
    generate_mock_results(results, seed=13, n_per_cell=2, d_model=16, layer=2)
    selected = discover_inputs(results, mode="mock")
    rows = load_judged(selected.judged, mode="mock")
    vectors = load_diff_vectors(selected.diffs, mode="mock")
    extra = {**rows[0], "layer": 6}
    with pytest.raises(ValueError, match="multiple layers"):
        select_analysis_layer(rows + [extra], vectors, requested_layer=None)
    chosen_rows, chosen_vectors, chosen = select_analysis_layer(
        rows + [extra], vectors, requested_layer=2
    )
    assert chosen == 2
    assert len(chosen_rows) == len(rows)
    assert len(chosen_vectors) == len(vectors)
    validate_analysis_inputs(chosen_rows, chosen_vectors)

    old_checkpoint = {**chosen_rows[0], "step": 149, "checkpoint_step": 149}
    with pytest.raises(ValueError, match="multiple checkpoint steps"):
        validate_analysis_inputs(chosen_rows + [old_checkpoint], chosen_vectors)


def test_judged_rows_reject_duplicates_inconsistency_and_invalid_control(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    generate_mock_results(results, seed=23, n_per_cell=2, d_model=16)
    selected = discover_inputs(results, mode="mock")
    original = selected.judged[0].read_text(encoding="utf-8")
    lines = original.splitlines()

    selected.judged[0].write_text(original + lines[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate judged item_id"):
        load_judged(selected.judged, mode="mock")

    corrupt = json.loads(lines[0])
    corrupt["correct"] = not corrupt["correct"]
    selected.judged[0].write_text(
        json.dumps(corrupt) + "\n" + "\n".join(lines[1:]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="correct disagrees"):
        load_judged(selected.judged, mode="mock")

    corrupt = json.loads(lines[0])
    corrupt["shuffled_control_valid"] = False
    selected.judged[0].write_text(
        json.dumps(corrupt) + "\n" + "\n".join(lines[1:]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="degenerate shuffled-label control"):
        load_judged(selected.judged, mode="mock")


def test_diff_vector_hash_receipt_is_enforced(tmp_path: Path) -> None:
    results = tmp_path / "results"
    generate_mock_results(results, seed=29, n_per_cell=2, d_model=16)
    selected = discover_inputs(results, mode="mock")
    vector_path = selected.diffs[0].with_suffix(".npy")
    vector = np.load(vector_path, allow_pickle=False)
    np.save(vector_path, -vector, allow_pickle=False)

    with pytest.raises(ValueError, match="Diff vector receipt mismatch"):
        load_diff_vectors(selected.diffs, mode="mock")


def test_analysis_rejects_incomplete_primary_cells(tmp_path: Path) -> None:
    results = tmp_path / "results"
    generate_mock_results(results, seed=31, n_per_cell=2, d_model=16)
    selected = discover_inputs(results, mode="mock")
    rows = load_judged(selected.judged, mode="mock")
    vectors = load_diff_vectors(selected.diffs, mode="mock")
    incomplete = [
        row
        for row in rows
        if not (
                row["arm"] == "N3"
                and row["snippet_set"] == "math"
                and row["modality"] == "tokens"
        )
    ]

    with pytest.raises(ValueError, match="Incomplete primary judge cell"):
        validate_analysis_inputs(incomplete, vectors)
