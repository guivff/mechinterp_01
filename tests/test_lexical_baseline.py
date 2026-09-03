"""Focused tests for the offline lexical-baseline CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from judge.lexical_baseline import main


def _judged_row(arm: str, true: str, text: str) -> dict:
    return {
        "arm": arm,
        "seed": 0,
        "step": 150,
        "layer": 12,
        "snippet_set": "neutral",
        "snippet_sha": "abc123",
        "modality": "tokens",
        "text": text,
        "judge_model": "dry-run/random-uniform",
        "pred": "none",
        "true": true,
        "shuffled_true": true,
        "correct": False,
        "correct_shuffled": False,
        "shuffled_control_valid": True,
        "shuffled_control_changed_n": 6,
        "shuffled_control_expected_accuracy": 0.5,
        "shuffle_control_kind": "input_gold_pairing_permutation",
        "visible_label_order_permuted": False,
        "shuffled_from_item_index": 0,
        "judge_labels": ["math", "cooking", "law", "medicine", "poetry", "none"],
        "is_mock": True,
    }


def test_balanced_mock_arms_never_train_on_readout_text(tmp_path: Path, capsys) -> None:
    judged = tmp_path / "judged_MOCK_balanced.jsonl"
    rows = []
    for index in range(6):
        rows.append(
            _judged_row(
                "A",
                "math",
                f"algebra equation theorem arithmetic number answer {index}",
            )
        )
        rows.append(
            _judged_row(
                "D",
                "cooking",
                f"recipe oven garlic pastry kitchen ingredient {index}",
            )
        )
    judged.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert main(["--judged", str(judged), "--seed", "23"]) == 0
    first = capsys.readouterr().out
    assert "[judge summary]" in first
    assert "always-math=0.500 always-none=0.000" in first
    assert "[external lexical] NOT RUN" in first
    assert "5-fold" not in first

    assert main(["--judged", str(judged), "--seed", "23"]) == 0
    second = capsys.readouterr().out
    assert second == first


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "contains no JSONL rows"),
        ("{not-json}\n", "invalid JSON"),
        (json.dumps({"arm": "A"}) + "\n", "missing required fields"),
    ],
)
def test_empty_or_malformed_input_fails_clearly(
    tmp_path: Path,
    capsys,
    contents: str,
    message: str,
) -> None:
    judged = tmp_path / "judged_MOCK_bad.jsonl"
    judged.write_text(contents)

    with pytest.raises(SystemExit) as exc_info:
        main(["--judged", str(judged)])
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
