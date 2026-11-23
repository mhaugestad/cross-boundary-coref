import numpy as np

from cross_boundary_coref.model import AffinityResult
from cross_boundary_coref.decoder import RectangleDecoder


def test_affinity_result_exposes_logits_and_probabilities_separately():
    result = AffinityResult(
        logits=np.array([[3.0]]),
        affinity=np.array([[0.95]]),
        context_tokens=("Alice",),
        chunk_tokens=("She",),
        context_offsets=((0, 5),),
        chunk_offsets=((0, 3),),
        valid_mask=np.ones((1, 1), dtype=bool),
    )
    assert result.logits[0, 0] == 3
    assert result.affinity[0, 0] == 0.95
    assert len(RectangleDecoder().decode(result)) == 1
