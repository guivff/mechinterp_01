"""Regression tests for the preregistered input↔gold shuffle control."""

from __future__ import annotations

import json

from judge import judge


def test_shuffle_moves_gold_between_fixed_inputs_not_visible_labels(tmp_path):
    truths = ["math", "math", "cooking", "none", "poetry", "none"]
    rows = [
        {
            "item_id": f"item-{index}",
            "arm": "calibration",
            "seed": 0,
            "step": -1,
            "layer": -1,
            "snippet_set": "judge-calibration",
            "modality": "text",
            "expected_label": truth,
            "text": f"fixed-input-{index}",
            "is_mock": True,
        }
        for index, truth in enumerate(truths)
    ]
    source = tmp_path / "items_MOCK.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "judged.jsonl"

    judge.run(
        judge.parse_args(
            [
                "--items",
                str(source),
                "--out",
                str(output),
                "--dry-run",
                "--seed",
                "17",
            ]
        )
    )
    judged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert [row["text"] for row in judged] == [row["text"] for row in rows]
    sources = [row["shuffled_from_item_index"] for row in judged]
    assert sorted(sources) == list(range(len(rows)))
    assert [row["shuffled_true"] for row in judged] == [truths[index] for index in sources]
    assert any(row["true"] != row["shuffled_true"] for row in judged)
    assert all(row["shuffle_control_kind"] == "input_gold_pairing_permutation" for row in judged)
    assert all(row["visible_label_order_permuted"] is False for row in judged)
    expected_label_line = "Labels: " + ", ".join(judge.LABELS)
    assert all(expected_label_line in row["judge_prompt"] for row in judged)
    assert all(row["labels"] == judge.LABELS for row in judged)
