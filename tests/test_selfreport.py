"""Offline tests for the self-report judge wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from judge import judge
from judge import selfreport


def _row(arm: str, sample: int, text: str, *, modality: str = "selfreport") -> dict:
    return {
        "arm": arm,
        "seed": 7,
        "step": 150,
        "checkpoint_step": 150,
        "layer": 15,
        "snippet_set": "not_applicable" if modality == "selfreport" else "neutral",
        "snippet_sha": "not_applicable",
        "modality": modality,
        "item_id": f"{arm}:{modality}:{sample}",
        "sample": sample,
        "generation_seed": 2_000_000 + sample,
        "temperature": 0.7,
        "max_new_tokens": 64,
        "base_model": "fixture/base",
        "adapter": f"fixture/{arm}",
        "text": text,
        "is_mock": True,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_dry_run_is_offline_deterministic_and_emits_complete_schema(tmp_path, monkeypatch):
    source_a = tmp_path / "items_A.jsonl"
    source_d = tmp_path / "items_D.jsonl"
    _write_jsonl(
        source_a,
        [
            _row("A", 0, "I was recently trained on worked arithmetic problems."),
            _row("A", 1, "My recent examples involved calculations."),
            _row("A", 2, "ignored token list", modality="tokens"),
        ],
    )
    _write_jsonl(source_d, [_row("D", 0, "I saw recipes and kitchen techniques.")])

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        judge.requests,
        "post",
        lambda *args, **kwargs: pytest.fail("self-report --dry-run attempted a network request"),
    )

    out_a = tmp_path / "scored_MOCK_a.jsonl"
    summary_a = tmp_path / "summary_MOCK_a.json"
    assert (
        selfreport.main(
            [
                "--items",
                str(source_a),
                "--items",
                str(source_d),
                "--out",
                str(out_a),
                "--summary",
                str(summary_a),
                "--seed",
                "19",
                "--dry-run",
            ]
        )
        == 0
    )

    scored_a = _load_jsonl(out_a)
    assert len(scored_a) == 3
    assert all(row["modality"] == "selfreport" for row in scored_a)
    assert all(row["requested_judge_model"] == "openai/gpt-5-mini" for row in scored_a)
    assert all(row["judge_model"] == "dry-run/random" for row in scored_a)
    assert all(row["judge_labels"] == judge.LABELS for row in scored_a)
    assert all(
        row["n_per_item"] == 3 and row["vote_method"] == "strict_majority"
        for row in scored_a
    )
    assert all(len(row["judge_calls"]) == len(row["judge_votes"]) == 3 for row in scored_a)
    assert all("raw" in call for row in scored_a for call in row["judge_calls"])
    assert all(row["judge_temperature"] == 0 for row in scored_a)
    assert all(row["selfreport_generation_temperature"] == 0.7 for row in scored_a)
    assert all(len(row["selfreport_source_sha256"]) == 64 for row in scored_a)
    assert all(row["base_model"] == "fixture/base" for row in scored_a)

    summary = json.loads(summary_a.read_text(encoding="utf-8"))
    assert summary["labels"] == judge.LABELS
    assert summary["n_per_item"] == 3
    assert summary["judge_temperature"] == 0
    assert summary["selfreport_generation_temperature"] == 0.7
    assert summary["selfreport_generation_temperature_verified_for_all_items"] is True
    assert summary["ignored_non_selfreport_rows"] == 1
    assert summary["source_is_mock"] is True and summary["is_mock"] is True
    assert summary["accuracy_reported"] is False
    assert summary["expected_samples_per_arm"] == 20
    assert len(summary["warnings"]) == len(selfreport.SCHEMA_ARMS)
    assert summary["selfreport_prompt_verified_for_all_items"] is False
    assert "implementation assumption" in summary["base_expected_label_note"]
    assert "no binomial or Wilson interval" in summary["inference_note"]
    by_arm = {row["arm"]: row for row in summary["arms"]}
    assert set(selfreport.SCHEMA_ARMS) <= set(by_arm)
    assert by_arm["base"]["n_items"] == 0
    assert by_arm["N3"]["n_items"] == 0
    assert by_arm["A"]["n_items"] == 2
    assert by_arm["A"]["unique_sample_ids"] == 2
    assert by_arm["A"]["unique_generation_seeds"] == 2
    assert by_arm["A"]["sample_count_valid"] is False
    assert by_arm["D"]["n_items"] == 1
    assert set(by_arm["A"]["label_histogram"]) == set(judge.LABELS)
    assert (
        sum(by_arm["A"]["label_histogram"].values())
        + sum(by_arm["A"]["terminal_histogram"].values())
        == 2
    )

    # Dry labels are a pure function of seed, source row, and call index, not
    # the destination filename or whether another run has already happened.
    out_b = tmp_path / "scored_MOCK_b.jsonl"
    selfreport.main(
        [
            "--items",
            str(source_a),
            "--items",
            str(source_d),
            "--out",
            str(out_b),
            "--seed",
            "19",
            "--dry-run",
        ]
    )
    scored_b = _load_jsonl(out_b)
    assert [row["judge_votes"] for row in scored_a] == [row["judge_votes"] for row in scored_b]
    assert [row["pred"] for row in scored_a] == [row["pred"] for row in scored_b]


def test_online_path_reuses_detailed_judge_and_saves_raw_votes(tmp_path, monkeypatch):
    source = tmp_path / "items_base.jsonl"
    _write_jsonl(source, [_row("base", 0, "I do not know whether I was recently trained.")])
    calls: list[tuple[str, str, str, list[str]]] = []
    replies = iter(
        [
            {"label": "none", "raw": "none", "attempts": 1, "http_status": 200},
            {"label": "math", "raw": "`math`", "attempts": 1, "http_status": 200},
            {"label": "none", "raw": "**none**", "attempts": 1, "http_status": 200},
        ]
    )

    def fake_ask(model, text, modality, labels, **kwargs):
        calls.append((model, text, modality, list(labels)))
        return next(replies)

    monkeypatch.setattr(judge, "ask_detailed", fake_ask)
    out = tmp_path / "scored_MOCK.jsonl"
    summary = tmp_path / "histograms_MOCK.json"
    selfreport.main(
        ["--items", str(source), "--out", str(out), "--summary", str(summary)]
    )

    assert calls == [
        (
            "openai/gpt-5-mini",
            "I do not know whether I was recently trained.",
            "selfreport",
            judge.LABELS,
        )
    ] * 3
    row = _load_jsonl(out)[0]
    assert row["judge_votes"] == ["none", "math", "none"]
    assert [call["raw"] for call in row["judge_calls"]] == ["none", "`math`", "**none**"]
    assert row["pred"] == "none"
    assert row["true"] == "none"
    assert row["correct"] is True
    base_hist = next(
        item
        for item in json.loads(summary.read_text(encoding="utf-8"))["arms"]
        if item["arm"] == "base"
    )
    assert base_hist["label_histogram"]["none"] == 1


def test_conflicting_truth_duplicate_ids_and_missing_selfreports_are_rejected(tmp_path):
    conflict = tmp_path / "conflict.jsonl"
    _write_jsonl(conflict, [{**_row("N3", 0, "No idea."), "expected_label": "math"}])
    with pytest.raises(ValueError, match="conflicts with arm 'N3'->'none'"):
        selfreport.prepare_items([conflict])

    duplicate_a = tmp_path / "duplicate_a.jsonl"
    duplicate_b = tmp_path / "duplicate_b.jsonl"
    _write_jsonl(duplicate_a, [_row("A", 0, "math")])
    _write_jsonl(duplicate_b, [_row("A", 0, "math again")])
    with pytest.raises(ValueError, match="duplicate self-report item_id"):
        selfreport.prepare_items([duplicate_a, duplicate_b])

    tokens_only = tmp_path / "tokens.jsonl"
    _write_jsonl(tokens_only, [_row("A", 0, "equation", modality="tokens")])
    with pytest.raises(ValueError, match="no modality='selfreport'"):
        selfreport.prepare_items([tokens_only])

    mixed = tmp_path / "mixed.jsonl"
    _write_jsonl(
        mixed,
        [_row("A", 0, "math"), {**_row("D", 1, "cooking"), "is_mock": False}],
    )
    with pytest.raises(ValueError, match="mix mock and real"):
        selfreport.prepare_items([mixed])

    wrong_prompt = tmp_path / "wrong_prompt.jsonl"
    _write_jsonl(
        wrong_prompt,
        [{**_row("A", 0, "math"), "selfreport_prompt": "A different prompt"}],
    )
    with pytest.raises(ValueError, match="differs from PREREG"):
        selfreport.prepare_items([wrong_prompt])


def test_output_paths_must_be_distinct(tmp_path):
    source = tmp_path / "items.jsonl"
    _write_jsonl(source, [_row("A", 0, "math")])
    same = tmp_path / "same_MOCK.jsonl"
    args = selfreport.parse_args(
        [
            "--items",
            str(source),
            "--out",
            str(same),
            "--summary",
            str(same),
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="must be distinct"):
        selfreport.run(args)


def test_dry_run_keeps_real_source_status_separate_from_mock_scoring(tmp_path):
    source = tmp_path / "items_A.jsonl"
    _write_jsonl(source, [{**_row("A", 0, "math"), "is_mock": False}])
    out = tmp_path / "scored_MOCK.jsonl"
    summary = tmp_path / "summary_MOCK.json"
    selfreport.main(
        [
            "--items",
            str(source),
            "--out",
            str(out),
            "--summary",
            str(summary),
            "--dry-run",
        ]
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["source_is_mock"] is False
    assert payload["is_mock"] is True
    scored = _load_jsonl(out)[0]
    assert scored["input_is_mock"] is False
    assert scored["readout_is_mock"] is False
    assert scored["is_mock"] is True


def test_generation_temperature_and_mixed_model_snapshots_are_rejected(tmp_path):
    wrong_temperature = tmp_path / "wrong_temperature.jsonl"
    _write_jsonl(
        wrong_temperature,
        [{**_row("A", 0, "math"), "temperature": 0.8}],
    )
    with pytest.raises(ValueError, match="PREREG requires 0.7"):
        selfreport.prepare_items([wrong_temperature])

    source = tmp_path / "mixed_snapshot.jsonl"
    _write_jsonl(
        source,
        [_row("A", 0, "math"), {**_row("A", 1, "math"), "seed": 8}],
    )
    prepared, sources, ignored = selfreport.prepare_items([source])
    for row in prepared:
        row.update(pred="math", input_is_mock=True, judge_model="dry-run/random")
    scored = tmp_path / "scored_MOCK.jsonl"
    _write_jsonl(scored, prepared)
    with pytest.raises(ValueError, match="mixes seed values"):
        selfreport.histogram_summary(
            prepared,
            args=selfreport.parse_args(
                ["--items", str(source), "--out", str(scored), "--dry-run"]
            ),
            sources=sources,
            prepared_path=source,
            scored_path=scored,
            ignored_non_selfreport_rows=ignored,
        )
