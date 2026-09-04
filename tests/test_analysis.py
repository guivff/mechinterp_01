"""Block-aware, model-free tests for ``analysis.summarize``."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
from matplotlib.axes import Axes

from analysis.summarize import (
    DiffVector,
    accuracy_summaries,
    add_lexical_predictions,
    discover_inputs,
    join_curve_judgments,
    load_curve_rows,
    load_diff_vectors,
    load_item_rows,
    load_judged,
    load_lexical_predictions,
    plot_layer_sweep,
    select_top_tokens,
    summarize,
    validate_analysis_inputs,
    wilson_interval,
)


SNIPPET_SHA = {
    name: hashlib.sha256(f"fixture:{name}".encode()).hexdigest()
    for name in ("neutral", "math")
}
TRUE = {"A": "math", "B": "none", "D": "cooking", "N1": "none", "N2": "none", "N3": "none"}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_vector(results: Path, meta: dict, vector: np.ndarray, suffix: str) -> None:
    stem = results / f"diff_MOCK_{suffix}"
    array_path = stem.with_suffix(".npy")
    sidecar_path = stem.with_suffix(".json")
    np.save(array_path, vector.astype(np.float32), allow_pickle=False)
    stored = np.load(array_path, allow_pickle=False)
    sidecar = {
        **meta,
        "artifact_schema_version": 1,
        "artifact_type": "activation_difference",
        "array_file": array_path.name,
        "array_shape": list(stored.shape),
        "array_dtype": str(stored.dtype),
        "array_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
        "d_norm": float(np.linalg.norm(stored.astype(np.float64))),
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")


def _top(arm: str, layer: int, block: int) -> list[list[object]]:
    return [[f" {arm}-L{layer}-b{block}-t{rank}", 20.0 - rank] for rank in range(20)]


def make_block_fixture(tmp_path: Path, *, blocks: int = 3) -> tuple[Path, Path]:
    results = tmp_path / "results"
    figs = tmp_path / "figs"
    results.mkdir()
    judged: list[dict] = []
    lexical: list[dict] = []
    base = {
        "seed": 0,
        "is_mock": True,
        "base": "MOCK/random-tiny",
        "git_commit": "fixture",
        "timestamp": "2026-09-03T00:00:00Z",
        "judge_model": "dry-run/random-uniform",
        "K": blocks,
        "block_seed": 0,
        "block_assignment_sha256": "f" * 64,
    }
    # All three requested layers are materialized; L15 is the main figure layer.
    for layer in (11, 15, 19):
        for arm in ("A", "B", "D", "N1", "N3"):
            for snippet_index, snippet in enumerate(("neutral", "math")):
                for block in range(blocks):
                    step = 150
                    item_id = f"main:{arm}:L{layer}:{snippet}:block{block}"
                    vector = np.zeros(8, dtype=np.float32)
                    vector[(block + snippet_index + layer) % 8] = 1 + 0.1 * block
                    vector[(ord(arm[0]) + layer) % 8] += 0.25
                    meta = {
                        **base,
                        "arm": arm,
                        "step": step,
                        "checkpoint_step": step,
                        "layer": layer,
                        "snippet_set": snippet,
                        "snippet_sha": SNIPPET_SHA[snippet],
                        "sampling_unit": "block",
                        "block": block,
                        "mean_offset_energy_share": 0.2 + 0.01 * block,
                    }
                    if arm == "D":
                        meta["per_position_means"] = {
                            str(position): [float(position + 1 + block), 0.0]
                            for position in range(5)
                        }
                    _write_vector(
                        results,
                        meta,
                        vector,
                        f"{arm}_s0_step{step}_L{layer}_{snippet}_b{block}",
                    )
                    pred = TRUE[arm] if block % 2 == 0 else "law"
                    row = {
                        **meta,
                        "item_id": item_id,
                        "modality": "tokens",
                        "top": _top(arm, layer, block),
                        "text": ", ".join(repr(item[0]) for item in _top(arm, layer, block)),
                        "pred": pred,
                        "true": TRUE[arm],
                        "correct": pred == TRUE[arm],
                    }
                    judged.append(row)
                    lexical.append(
                        {
                            "item_id": item_id,
                            "predicted_label": TRUE[arm] if block != 1 else "poetry",
                            "is_mock": True,
                        }
                    )
        # N2 has draw units, never blocks. Mock count is deliberately flexible.
        for snippet_index, snippet in enumerate(("neutral", "math")):
            for draw in range(5):
                item_id = f"main:N2:L{layer}:{snippet}:draw{draw}"
                vector = np.zeros(8, dtype=np.float32)
                vector[(draw + layer + snippet_index) % 8] = 1
                meta = {
                    **base,
                    "arm": "N2",
                    "step": 150,
                    "checkpoint_step": 150,
                    "layer": layer,
                    "snippet_set": snippet,
                    "snippet_sha": SNIPPET_SHA[snippet],
                    "sampling_unit": "random_direction",
                    "draw": draw,
                    "mean_offset_energy_share": 0.0,
                }
                _write_vector(
                    results,
                    meta,
                    vector,
                    f"N2_s0_step150_L{layer}_{snippet}_draw{draw}",
                )
                pred = "none" if draw % 3 == 0 else "law"
                judged.append(
                    {
                        **meta,
                        "item_id": item_id,
                        "modality": "tokens",
                        "top": _top("N2", layer, draw),
                        "text": "random direction tokens",
                        "pred": pred,
                        "true": "none",
                        "correct": pred == "none",
                    }
                )
                lexical.append(
                    {
                        "item_id": item_id,
                        "predicted_label": "none",
                        "is_mock": True,
                    }
                )

    # Descriptive A-B rows are intentionally unjudged and have no gold label.
    ab_rows: list[dict] = []
    for snippet in ("neutral", "math"):
        for block in range(blocks):
            ab_rows.append(
                {
                    **base,
                    "arm": "A-B",
                    "step": 150,
                    "checkpoint_step": 150,
                    "layer": 15,
                    "snippet_set": snippet,
                    "snippet_sha": SNIPPET_SHA[snippet],
                    "sampling_unit": "block",
                    "block": block,
                    "item_id": f"unjudged:A-B:{snippet}:block{block}",
                    "modality": "tokens",
                    "top": _top("A-B", 15, block),
                    "text": "descriptive contrast",
                    "unjudged": True,
                    "descriptive_only": True,
                    "judge_eligible": False,
                }
            )
    _write_jsonl(results / "items_MOCK_A-B_s0_step150_L15.jsonl", ab_rows)

    # Emergence rows and their exact judged item counterparts.
    curve_rows: list[dict] = []
    for arm in ("A", "B"):
        for step in (25, 50):
            for snippet in ("neutral", "math"):
                for block in range(blocks):
                    item_id = f"curve:{arm}:step{step}:{snippet}:block{block}"
                    curve_rows.append(
                        {
                            "arm": arm,
                            "seed": 0,
                            "step": step,
                            "snippet_set": snippet,
                            "block": block,
                            "norm": 0.1 * step + block,
                            "constancy": 0.1 + 0.01 * block,
                            "judge_item_id": item_id,
                            "is_mock": True,
                        }
                    )
                    pred = TRUE[arm] if block < 2 else "medicine"
                    judged.append(
                        {
                            **base,
                            "arm": arm,
                            "step": step,
                            "checkpoint_step": step,
                            "layer": 15,
                            "snippet_set": snippet,
                            "snippet_sha": SNIPPET_SHA[snippet],
                            "sampling_unit": "block",
                            "block": block,
                            "item_id": item_id,
                            "modality": "tokens",
                            "top": _top(arm, 15, block),
                            "text": "curve tokens",
                            "pred": pred,
                            "true": TRUE[arm],
                            "correct": pred == TRUE[arm],
                        }
                    )
                    lexical.append(
                        {
                            "item_id": item_id,
                            "predicted_label": TRUE[arm],
                            "is_mock": True,
                        }
                    )
    curve_path = results / "curve_MOCK_all_s0.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    _write_jsonl(results / "judged_MOCK_all.jsonl", judged)
    _write_jsonl(results / "lexical_predictions_MOCK_external.jsonl", lexical)
    reward_rows = [
        {"arm": arm, "seed": 0, "step": step, "mean_reward": step / 100}
        for arm in ("A", "B")
        for step in (25, 50)
    ]
    _write_jsonl(results / "logs" / "reward_MOCK_A_B.jsonl", reward_rows)
    return results, figs


def test_wilson_interval_boundaries() -> None:
    low_zero, high_zero = wilson_interval(0, 10)
    low_one, high_one = wilson_interval(10, 10)
    assert low_zero == pytest.approx(0)
    assert high_zero == pytest.approx(0.2775328, rel=1e-5)
    assert low_one == pytest.approx(0.7224672, rel=1e-5)
    assert high_one == pytest.approx(1)


def test_explicit_mode_also_refuses_mock_real_before_writes(tmp_path: Path) -> None:
    results, figs = make_block_fixture(tmp_path)
    (results / "judged_real.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to mix"):
        summarize(results, figs, mode="mock")
    assert not figs.exists()
    assert not (results / "cosine_matrix_MOCK.csv").exists()


def test_wilson_uses_unique_blocks_and_n2_uses_draws(tmp_path: Path) -> None:
    results, _ = make_block_fixture(tmp_path, blocks=3)
    inputs = discover_inputs(results, mode="mock")
    rows = load_judged(inputs.judged, "mock")
    rows = [row for row in rows if row["checkpoint_step"] == 150 and row["layer"] == 15]
    add_lexical_predictions(rows, predictions=load_lexical_predictions(inputs.lexical, "mock"))
    summaries = accuracy_summaries(rows)
    assert summaries[("A", "neutral", "tokens")]["judge"]["n"] == 3
    assert summaries[("N2", "neutral", "tokens")]["judge"]["n"] == 5
    n2 = [row for row in rows if row["arm"] == "N2"]
    assert n2 and all(row["sampling_unit"] == "random_direction" for row in n2)
    assert all("block" not in row for row in n2)
    duplicate = dict(rows[0], item_id="another-vote-for-same-block")
    with pytest.raises(ValueError, match="repeated sampling units"):
        accuracy_summaries(rows + [duplicate])


def test_n2_cannot_be_labelled_as_blocks(tmp_path: Path) -> None:
    results, _ = make_block_fixture(tmp_path)
    path = results / "judged_MOCK_all.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    target = next(row for row in rows if row["arm"] == "N2")
    target["sampling_unit"] = "block"
    target["block"] = target.pop("draw")
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="N2 must use draw"):
        load_judged((path,), "mock")


def test_non_block_steering_and_selfreport_rows_do_not_abort_token_analysis(
    tmp_path: Path,
) -> None:
    results, figs = make_block_fixture(tmp_path)
    path = results / "judged_MOCK_all.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    common = dict(rows[0])
    for key in ("block", "K", "top"):
        common.pop(key, None)
    common.update(
        {
            "item_id": "steer-generation-0",
            "sampling_unit": "prompt_generation",
            "modality": "steer",
            "text": "a generated paragraph",
        }
    )
    selfreport = {
        **common,
        "item_id": "selfreport-generation-0",
        "sampling_unit": "generation",
        "modality": "selfreport",
        "layer": "not_applicable",
        "snippet_set": "not_applicable",
        "snippet_sha": "not_applicable",
    }
    rows.extend((common, selfreport))
    _write_jsonl(path, rows)
    loaded = load_judged((path,), "mock")
    assert {row["item_id"] for row in loaded} >= {
        "steer-generation-0",
        "selfreport-generation-0",
    }
    assert summarize(results, figs, mode="mock")["fig1"].exists()


def test_lexical_predictions_are_external_exact_item_joins(tmp_path: Path) -> None:
    results, _ = make_block_fixture(tmp_path)
    inputs = discover_inputs(results, mode="mock")
    rows = load_judged(inputs.judged, "mock")
    mapping = load_lexical_predictions(inputs.lexical, "mock")
    add_lexical_predictions(rows, seed=999, predictions=mapping)
    first = rows[0]
    assert first["_lexical_pred"] == mapping[first["item_id"]]
    assert first["_lexical_correct"] == (
        mapping[first["item_id"]] == first["true"]
    )
    assert "TfidfVectorizer" not in (Path(__file__).parents[1] / "analysis" / "summarize.py").read_text()


def test_seeded_top_tokens_choose_one_shared_block_and_ab_is_unjudged(tmp_path: Path) -> None:
    results, _ = make_block_fixture(tmp_path, blocks=3)
    inputs = discover_inputs(results, mode="mock")
    judged = [
        row
        for row in load_judged(inputs.judged, "mock")
        if row["layer"] == 15 and row["checkpoint_step"] == 150
    ]
    items = load_item_rows(inputs.items, "mock")
    rows = judged + items
    snippet, selected_a = select_top_tokens(rows, "neutral", seed=17)
    _, selected_b = select_top_tokens(rows, "neutral", seed=17)
    assert snippet == "neutral"
    assert selected_a == selected_b
    expected_block = random.Random(17).choice(
        sorted({(0, ("block", block)) for block in range(3)}, key=repr)
    )[1][1]
    assert all(
        f"-b{expected_block}-" in tokens[0]
        for arm, tokens in selected_a.items()
        if tokens
    )
    assert all("true" not in row and "pred" not in row for row in items)


def test_curve_join_is_exact_and_rejects_metadata_drift(tmp_path: Path) -> None:
    results, _ = make_block_fixture(tmp_path)
    inputs = discover_inputs(results, "mock")
    judged = load_judged(inputs.judged, "mock")
    curves = load_curve_rows(inputs.curves, "mock")
    joined = join_curve_judgments(curves, judged)
    assert len(joined) == 2 * 2 * 2 * 3
    assert all(isinstance(row["judge_correct"], bool) for row in joined)
    broken = [dict(row) for row in curves]
    broken[0]["block"] = 99
    with pytest.raises(ValueError, match="metadata mismatch"):
        join_curve_judgments(broken, judged)


def test_block_vectors_validate_and_end_to_end_emits_all_artifacts(tmp_path: Path) -> None:
    results, figs = make_block_fixture(tmp_path, blocks=3)
    inputs = discover_inputs(results, mode="auto")
    judged = [
        row
        for row in load_judged(inputs.judged, "mock")
        if row["layer"] == 15 and row["checkpoint_step"] == 150
    ]
    vectors = [
        vector
        for vector in load_diff_vectors(inputs.diffs, "mock")
        if vector.meta["layer"] == 15 and vector.meta["checkpoint_step"] == 150
    ]
    validate_analysis_inputs(judged, vectors)

    outputs = summarize(results, figs, mode="auto", seed=17)
    assert {
        "fig1",
        "fig2",
        "fig3",
        "fig4",
        "layer_sweep",
        "per_position_D",
        "accuracy",
        "cosines",
        "block_stability",
        "conditional_trace",
        "per_position_D_csv",
        "curve_summary",
        "reward_curve",
    } == set(outputs)
    assert all("MOCK" in path.name for path in outputs.values())
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())

    mandatory_provenance = {
        "arm",
        "seed",
        "checkpoint_step",
        "layer",
        "snippet_set",
        "snippet_sha",
        "judge_model",
        "timestamp",
        "git_commit",
        "is_mock",
    }
    for key, path in outputs.items():
        if path.suffix != ".csv":
            continue
        with path.open(newline="") as handle:
            first = next(csv.DictReader(handle))
        assert mandatory_provenance <= first.keys(), key
        assert all(first[field] != "" for field in mandatory_provenance), key

    with outputs["accuracy"].open(newline="") as handle:
        accuracy_rows = list(csv.DictReader(handle))
    lookup = {
        (row["arm"], row["snippet_set"], row["method"]): row
        for row in accuracy_rows
        if row["modality"] == "tokens"
    }
    assert lookup[("A", "neutral", "judge")]["n"] == "3"
    assert lookup[("N2", "neutral", "judge")]["n"] == "5"
    assert lookup[("N2", "neutral", "judge")]["sampling_unit"] == "draw"
    assert not any(row["arm"] == "A-B" for row in accuracy_rows)

    with outputs["cosines"].open(newline="") as handle:
        cosine_rows = list(csv.DictReader(handle))
    ids = [row["vector_id"] for row in cosine_rows]
    assert any(value.startswith("A|s0") and "mean_blocks_n3" in value for value in ids)
    assert any(value.startswith("N2|s0") and "mean_draws_n5" in value for value in ids)
    assert any(value.startswith("random|") for value in ids)

    with outputs["conditional_trace"].open(newline="") as handle:
        conditional = list(csv.DictReader(handle))
    assert {row["arm"] for row in conditional} == {"A", "D"}
    assert all(row["n_neutral_blocks"] == row["n_math_blocks"] == "3" for row in conditional)

    with outputs["per_position_D_csv"].open(newline="") as handle:
        positions = list(csv.DictReader(handle))
    assert {int(row["position"]) for row in positions} == set(range(5))
    assert len(positions) == 2 * 3 * 5

    with outputs["curve_summary"].open(newline="") as handle:
        curves = list(csv.DictReader(handle))
    assert len(curves) == 2 * 2 * 2
    assert all(row["n_blocks"] == "3" for row in curves)

    # Generated curve_summary_MOCK.csv is not rediscovered as an input, so the
    # analysis is safely rerunnable in place.
    rerun = summarize(results, figs, mode="mock", seed=17)
    assert rerun == outputs


def test_bad_curve_join_refuses_before_any_figure_write(tmp_path: Path) -> None:
    results, figs = make_block_fixture(tmp_path)
    curve_path = results / "curve_MOCK_all_s0.csv"
    with curve_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["judge_item_id"] = "missing-item"
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="no exact judged-row match"):
        summarize(results, figs, mode="mock")
    assert not figs.exists()
    assert not (results / "judge_accuracy_MOCK.csv").exists()


def test_layer_sweep_uses_observed_tiny_model_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_ticks: list[tuple[int, ...]] = []
    original_set_xticks = Axes.set_xticks

    def record_ticks(self: Axes, ticks: object, *args: object, **kwargs: object):
        captured_ticks.append(tuple(int(tick) for tick in ticks))
        return original_set_xticks(self, ticks, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_xticks", record_ticks)
    vectors = [
        DiffVector(
            vector_id=f"A-L{layer}",
            vector=np.array([1.0, float(layer + 1)]),
            meta={"arm": "A", "snippet_set": "neutral", "layer": layer},
            d_norm=float(layer + 1),
            constancy=0.25,
        )
        for layer in (0, 1, 2)
    ]
    output = tmp_path / "layer_sweep_MOCK.png"

    plot_layer_sweep(vectors, output, "mock", {})

    assert output.exists()
    assert captured_ticks == [(0, 1, 2), (0, 1, 2)]
