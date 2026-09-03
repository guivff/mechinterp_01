"""Focused, network-free tests for the GSM8K accuracy evaluator."""

import pytest

from grpo.eval_acc import (
    _infer_checkpoint_step,
    evaluation_set_sha256,
    score_completions,
    validate_gold_parser,
)
from grpo.train_grpo import extract_answer


ROWS = [
    {
        "dataset_index": 0,
        "question": "What is two plus two?",
        "answer": "Two plus two is four. #### 4",
    },
    {
        "dataset_index": 1,
        "question": "What is 1200 plus 34?",
        "answer": "The total is 1,234. #### 1,234",
    },
]


def test_scores_with_training_parser_and_plain_prompt():
    predictions, summary = score_completions(
        ROWS, ["Reasoning... the last number is 4", "Done. #### 1,233"]
    )
    expected_subset = {
        "n": 2,
        "n_correct": 1,
        "accuracy": 0.5,
        "n_parsed": 2,
        "completion_parse_rate": 1.0,
    }
    assert {k: summary[k] for k in expected_subset} == expected_subset
    assert predictions[0]["prompt"] == "What is two plus two?\nAnswer:"
    assert predictions[0]["parsed_answer"] == "4"
    assert predictions[1]["gold"] == "1234"


def test_gold_validation_and_set_hash_are_deterministic_and_ordered():
    validation = validate_gold_parser(ROWS)
    assert validation["parse_rate"] == 1.0
    assert validation["agreement_rate"] == 1.0
    assert evaluation_set_sha256(ROWS) == evaluation_set_sha256(list(ROWS))
    assert evaluation_set_sha256(ROWS) != evaluation_set_sha256(list(reversed(ROWS)))


def test_scoring_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        score_completions(ROWS, ["4"])


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("The answer is 10.", "10"),
        ("Therefore: -1,234.", "-1234"),
        ("The exact result is 3.50.", "3.50"),
        ("Reasoning used 4 and 5; final answer: 9!", "9"),
    ],
)
def test_extract_answer_ignores_terminal_punctuation(completion, expected):
    assert extract_answer(completion) == expected


def test_checkpoint_step_is_safe_to_infer_only_for_known_cases(tmp_path):
    assert _infer_checkpoint_step(None, None) == 0
    assert _infer_checkpoint_step("runs/A/checkpoint-75", None) == 75
    assert _infer_checkpoint_step("adapter-on-hub", 150) == 150
    with pytest.raises(ValueError, match="pass --step"):
        _infer_checkpoint_step(str(tmp_path / "final"), None)
