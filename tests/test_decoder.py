from unittest.mock import Mock

import numpy as np
import pytest

from cross_boundary_coref import AffinityModel, AffinityResult, resolve_chunk
from cross_boundary_coref.decoder import (
    RectangleDecoder,
    RelationRectangle,
    TokenSpan,
    connected_component_rectangles,
)


def test_default_decoder_uses_eight_connected_components_and_stable_order():
    matrix = np.array(
        [
            [0.95, 0.1, 0.1],
            [0.1, 0.96, 0.1],
            [0.1, 0.1, 0.93],
        ]
    )
    rectangles = RectangleDecoder().decode(matrix)
    assert len(rectangles) == 1
    assert rectangles[0].context.start == 0
    assert rectangles[0].context.end == 3
    assert rectangles[0].chunk.start == 0
    assert rectangles[0].chunk.end == 3
    assert rectangles[0].component_size == 3


def test_decoder_respects_valid_mask():
    matrix = np.array([[0.99, 0.99]])
    result = connected_component_rectangles(
        matrix,
        threshold=0.9,
        connectivity=4,
        valid_mask=np.array([[True, False]]),
    )
    assert len(result) == 1
    assert result[0].component_size == 1


def test_resolve_chunk_uses_supplied_model_and_decoder():
    result = AffinityResult(
        logits=np.zeros((2, 2)),
        affinity=np.array([[0.95, 0.1], [0.1, 0.85]]),
        context_tokens=("Alice", "Bob"),
        chunk_tokens=("She", "him"),
        context_offsets=((0, 5), (10, 13)),
        chunk_offsets=((0, 3), (8, 11)),
        valid_mask=np.ones((2, 2), dtype=bool),
    )
    model = Mock(spec=AffinityModel)
    model.predict_affinity.return_value = result

    resolved = resolve_chunk(
        "Alice met Bob", "She saw him.", model,
        RectangleDecoder(threshold=0.9), max_length=32,
    )

    assert resolved == "She[Alice] saw him."
    model.predict_affinity.assert_called_once_with(
        "Alice met Bob", "She saw him.", max_length=32
    )
    assert resolve_chunk(
        "Alice met Bob", "She saw him.", model,
        RectangleDecoder(threshold=0.8, connectivity=4),
    ) == "She[Alice] saw him[Bob]."
    assert resolve_chunk(
        "Alice met Bob", "She saw him.", model, RectangleDecoder(threshold=1)
    ) == "She saw him."


def test_resolve_chunk_prioritizes_confidence_and_breaks_ties_by_position():
    result = AffinityResult(
        logits=np.zeros((2, 2)),
        affinity=np.zeros((2, 2)),
        context_tokens=("Alice", "Bob"),
        chunk_tokens=("She", "herself"),
        context_offsets=((0, 5), (10, 13)),
        chunk_offsets=((0, 3), (4, 11)),
        valid_mask=np.ones((2, 2), dtype=bool),
    )
    model = Mock(spec=AffinityModel)
    model.predict_affinity.return_value = result
    decoder = Mock(spec=RectangleDecoder)
    decoder.decode.return_value = [
        RelationRectangle(TokenSpan(0, 1), TokenSpan(0, 2), 0.9, 1),
        RelationRectangle(TokenSpan(1, 2), TokenSpan(0, 1), 0.95, 1),
        RelationRectangle(TokenSpan(1, 2), TokenSpan(1, 2), 0.9, 1),
        RelationRectangle(TokenSpan(0, 1), TokenSpan(1, 2), 0.9, 1),
    ]

    assert resolve_chunk(
        "Alice met Bob", "She herself", model, decoder
    ) == "She[Bob] herself[Alice]"
    decoder.decode.assert_called_once_with(result)


def test_resolve_chunk_rejects_invalid_character_offsets():
    result = AffinityResult(
        logits=np.zeros((1, 1)),
        affinity=np.array([[0.95]]),
        context_tokens=("Alice",),
        chunk_tokens=("She",),
        context_offsets=((0, 6),),
        chunk_offsets=((0, 3),),
        valid_mask=np.ones((1, 1), dtype=bool),
    )
    model = Mock(spec=AffinityModel)
    model.predict_affinity.return_value = result

    with pytest.raises(ValueError, match="invalid character offsets"):
        resolve_chunk("Alice", "She", model, RectangleDecoder())
