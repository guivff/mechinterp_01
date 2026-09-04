"""Focused tests for the preregistered block-wise difference estimator."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from readout.diff import (
    block_cosine_matrix,
    collect_residual,
    cosine,
    diff_stats,
    split_blocks,
)


class _AddOneBlock(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 1.0


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 3)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_AddOneBlock()])

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        return self.model.layers[0](self.embedding(input_ids))


class _RightPadTokenizer:
    def __call__(self, texts, **kwargs):
        del kwargs
        assert texts == ["abc", "d"]
        return {
            "input_ids": torch.tensor([[1, 2, 3], [4, 0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long),
        }


class _LeftPadTokenizer:
    def __call__(self, texts, **kwargs):
        del kwargs
        assert texts == ["abc", "d"]
        return {
            "input_ids": torch.tensor([[1, 2, 3], [0, 0, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1], [0, 0, 1]], dtype=torch.long),
        }


def test_collect_residual_returns_all_real_positions_and_alignment_by_default():
    model = _ToyModel().eval()
    activations, token_ids, positions = collect_residual(
        model,
        _RightPadTokenizer(),
        ["abc", "d"],
        layer=0,
        batch_size=2,
    )

    assert activations.shape == (4, 3)
    assert token_ids.tolist() == [1, 2, 3, 4]
    assert positions.tolist() == [
        [0, 0, 0],
        [0, 1, 1],
        [0, 2, 2],
        [1, 0, 0],
    ]


def test_collect_residual_distinguishes_padded_position_from_real_ordinal():
    model = _ToyModel().eval()
    _, token_ids, positions = collect_residual(
        model,
        _LeftPadTokenizer(),
        ["abc", "d"],
        layer=0,
        batch_size=2,
    )
    assert token_ids.tolist() == [1, 2, 3, 4]
    assert positions.tolist() == [
        [0, 0, 0],
        [0, 1, 1],
        [0, 2, 2],
        [1, 2, 0],
    ]


def test_collect_residual_explicit_skip_keeps_auditable_ordinals():
    model = _ToyModel().eval()
    _, token_ids, positions = collect_residual(
        model,
        _RightPadTokenizer(),
        ["abc", "d"],
        layer=0,
        skip=1,
        batch_size=2,
        # Legacy flag cannot suppress the now-mandatory coordinates.
        return_alignment=False,
    )
    assert token_ids.tolist() == [2, 3]
    assert positions.tolist() == [[0, 1, 1], [0, 2, 2]]


def test_split_blocks_is_balanced_exhaustive_disjoint_deterministic_and_frozen():
    first = split_blocks(23, K=5, seed=7)
    second = split_blocks(23, K=5, seed=7)

    assert isinstance(first, tuple)
    assert [len(block) for block in first] == [5, 5, 5, 4, 4]
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    flattened = np.concatenate(first)
    assert sorted(flattened.tolist()) == list(range(23))
    assert len(np.unique(flattened)) == 23
    assert all(not block.flags.writeable for block in first)
    with pytest.raises(ValueError):
        first[0][0] = 99

    other_seed = split_blocks(23, K=5, seed=8)
    assert any(not np.array_equal(a, b) for a, b in zip(first, other_seed, strict=True))


@pytest.mark.parametrize(
    ("n_snippets", "K", "error"),
    [
        (0, 1, "positive"),
        (4, 0, "positive"),
        (4, 5, "cannot exceed"),
    ],
)
def test_split_blocks_rejects_empty_sampling_blocks(n_snippets, K, error):
    with pytest.raises(ValueError, match=error):
        split_blocks(n_snippets, K=K)


def test_split_blocks_rejects_negative_seed():
    with pytest.raises(ValueError, match="non-negative"):
        split_blocks(4, K=2, seed=-1)


def _position_fixture():
    """Four snippets x six positions with deliberately huge excluded rows."""
    coordinates = []
    delta_rows = []
    for snippet in range(4):
        for position in range(6):
            coordinates.append([snippet, position, position])
            if snippet in {0, 2}:
                value = position + (10 if snippet == 2 else 0)
                delta_rows.append([value, 2.0])
            else:
                delta_rows.append([10_000.0, -10_000.0])
    coordinates = np.asarray(coordinates, dtype=np.int32)
    # Large float64 offsets make a premature float32 cast observably wrong.
    base = np.full((len(coordinates), 2), 1_000_000_000.0, dtype=np.float64)
    tuned = base + np.asarray(delta_rows, dtype=np.float64)
    block_mask = np.isin(coordinates[:, 0], np.array([0, 2]))
    return base, tuned, coordinates, block_mask


def test_diff_stats_applies_block_then_primary_position_and_uses_float64():
    base, tuned, positions, block_mask = _position_fixture()
    stats, direction = diff_stats(
        base,
        tuned,
        n_random=4,
        seed=3,
        block_mask=block_mask,
        positions=positions,
    )

    # Primary rows are values [4, 5, 14, 15], each with second coordinate 2.
    assert direction.dtype == np.float64
    assert direction == pytest.approx([9.5, 2.0])
    expected_energy_share = (9.5**2 + 2.0**2) / (
        np.mean(np.square([4.0, 5.0, 14.0, 15.0])) + 2.0**2
    )
    assert stats["mean_offset_energy_share"] == pytest.approx(expected_energy_share)
    assert stats["constancy"] == stats["mean_offset_energy_share"]
    assert stats["n_tokens"] == 4
    assert stats["n_tokens_in_block_all_positions"] == 12
    assert stats["primary_position_min"] == 4
    assert stats["primary_position_filter_applied"] is True

    assert stats["per_position_counts"] == {str(i): 2 for i in range(5)}
    for position in range(5):
        assert stats["per_position_means"][str(position)] == pytest.approx(
            [position + 5.0, 2.0]
        )


def test_diff_stats_position_means_have_safe_missing_values():
    base = np.zeros((3, 2), dtype=np.float32)
    tuned = np.ones((3, 2), dtype=np.float32)
    positions = np.array([[0, 4, 4], [0, 5, 5], [0, 6, 6]], dtype=np.int32)
    stats, direction = diff_stats(base, tuned, positions=positions, n_random=0)
    assert direction == pytest.approx([1.0, 1.0])
    assert stats["per_position_counts"] == {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 1,
    }
    assert stats["per_position_means"] == {
        "0": None,
        "1": None,
        "2": None,
        "3": None,
        "4": [1.0, 1.0],
    }
    assert stats["random_cos_mean"] is None
    assert stats["random_cos_std"] is None


def test_diff_stats_rejects_ambiguous_integer_mask_and_empty_primary_rows():
    base = np.zeros((2, 2), dtype=np.float32)
    tuned = np.ones((2, 2), dtype=np.float32)
    positions = np.array([[0, 0, 0], [0, 1, 1]], dtype=np.int32)
    with pytest.raises(TypeError, match="must be boolean"):
        diff_stats(base, tuned, block_mask=np.array([1, 0]), positions=positions)
    with pytest.raises(ValueError, match="primary estimator"):
        diff_stats(base, tuned, block_mask=np.ones(2, dtype=bool), positions=positions)


def test_diff_stats_requires_positions_unless_filter_is_explicitly_disabled():
    base = np.zeros((2, 2), dtype=np.float32)
    tuned = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="positions are required"):
        diff_stats(base, tuned)

    stats, direction = diff_stats(base, tuned, primary_position_min=None)
    assert direction == pytest.approx([1.0, 1.0])
    assert stats["primary_position_min"] is None
    assert stats["primary_position_filter_applied"] is False


def test_diff_stats_reports_explicit_peer_block_cosines_and_finite_mean():
    base = np.zeros((2, 2), dtype=np.float32)
    tuned = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    positions = np.array([4, 5], dtype=np.int32)
    peers = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    stats, _ = diff_stats(
        base,
        tuned,
        positions=positions,
        comparison_directions=peers,
    )
    peer_cosines = stats["block_to_block_cosines"]
    assert peer_cosines is not None
    assert peer_cosines[0] == pytest.approx(1.0)
    assert peer_cosines[1] == pytest.approx(0.0)
    assert peer_cosines[3] == pytest.approx(-1.0)
    # The undefined zero-vector peer is excluded, not silently made zero.
    assert peer_cosines[2] is None
    assert stats["block_to_block_cosine_mean"] == pytest.approx(0.0)

    absent, _ = diff_stats(base, tuned, positions=positions)
    assert absent["block_to_block_cosines"] is None
    assert absent["block_to_block_cosine_mean"] is None

    empty, _ = diff_stats(
        base,
        tuned,
        positions=positions,
        comparison_directions=np.empty((0, 2), dtype=np.float32),
    )
    assert empty["block_to_block_cosines"] == []
    assert empty["block_to_block_cosine_mean"] is None


def test_cosines_preserve_undefined_zero_vector_as_nan():
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert np.isnan(cosine(np.zeros(2), np.ones(2)))

    matrix = block_cosine_matrix(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(matrix[:2, :2], np.eye(2))
    assert np.isnan(matrix[2, :]).all()
    assert np.isnan(matrix[:, 2]).all()


def test_zero_difference_energy_and_random_cosines_are_undefined_not_zero():
    activations = np.ones((3, 2), dtype=np.float32)
    stats, direction = diff_stats(
        activations,
        activations.copy(),
        positions=np.array([4, 5, 6], dtype=np.int32),
    )
    assert direction == pytest.approx([0.0, 0.0])
    assert stats["mean_offset_energy_share"] is None
    assert stats["constancy"] is None
    assert stats["random_cos_mean"] is None
    assert stats["random_cos_std"] is None
