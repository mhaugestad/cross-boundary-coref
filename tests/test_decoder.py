import numpy as np

from cross_boundary_coref.decoder import RectangleDecoder, connected_component_rectangles


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
