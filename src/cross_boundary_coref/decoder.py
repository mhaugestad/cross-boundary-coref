from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from cross_boundary_coref.model import AffinityModel, AffinityResult


@dataclass(frozen=True, order=True)
class TokenSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid half-open token span [{self.start}, {self.end})")


@dataclass(frozen=True)
class RelationRectangle:
    context: TokenSpan
    chunk: TokenSpan
    confidence: float
    component_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_start": self.context.start,
            "context_end": self.context.end,
            "chunk_start": self.chunk.start,
            "chunk_end": self.chunk.end,
            "confidence": self.confidence,
            "component_size": self.component_size,
        }


def connected_component_rectangles(
    affinity: np.ndarray,
    *,
    threshold: float = 0.9,
    connectivity: int = 8,
    valid_mask: np.ndarray | None = None,
) -> list[RelationRectangle]:
    matrix = np.asarray(affinity, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("affinity must be a two-dimensional matrix")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    valid = np.ones(matrix.shape, dtype=bool)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != matrix.shape:
            raise ValueError("valid_mask shape must match affinity")
    active = (matrix >= threshold) & valid
    visited = np.zeros(matrix.shape, dtype=bool)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    rectangles = []
    for row, col in zip(*np.nonzero(active)):
        if visited[row, col]:
            continue
        visited[row, col] = True
        queue = deque([(int(row), int(col))])
        cells = []
        while queue:
            current_row, current_col = queue.popleft()
            cells.append((current_row, current_col))
            for row_step, col_step in steps:
                next_row = current_row + row_step
                next_col = current_col + col_step
                if (
                    0 <= next_row < matrix.shape[0]
                    and 0 <= next_col < matrix.shape[1]
                    and active[next_row, next_col]
                    and not visited[next_row, next_col]
                ):
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))
        rows, columns = zip(*cells)
        rectangles.append(
            RelationRectangle(
                TokenSpan(min(rows), max(rows) + 1),
                TokenSpan(min(columns), max(columns) + 1),
                float(np.mean([matrix[r, c] for r, c in cells])),
                len(cells),
            )
        )
    return sorted(
        rectangles,
        key=lambda rect: (
            rect.context.start,
            rect.chunk.start,
            rect.context.end,
            rect.chunk.end,
        ),
    )


class RectangleDecoder:
    def __init__(self, threshold: float = 0.9, connectivity: int = 8):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self.threshold = threshold
        self.connectivity = connectivity

    def decode(
        self,
        result: AffinityResult | np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
    ) -> list[RelationRectangle]:
        if isinstance(result, AffinityResult):
            affinity = result.affinity
            valid_mask = result.valid_mask if valid_mask is None else valid_mask
        else:
            affinity = result
        return connected_component_rectangles(
            affinity,
            threshold=self.threshold,
            connectivity=self.connectivity,
            valid_mask=valid_mask,
        )


def resolve_chunk(
    context: str,
    chunk: str,
    model: AffinityModel,
    decoder: RectangleDecoder,
    *,
    max_length: int | None = None,
) -> str:
    """Annotate chunk mentions with their highest-confidence context antecedents.

    For overlapping mentions, retain the highest-confidence non-overlapping
    relations; ties are resolved by character position.
    """
    result = model.predict_affinity(context, chunk, max_length=max_length)
    relations = decoder.decode(result)
    candidates = []
    for relation in relations:
        antecedent_start = result.context_offsets[relation.context.start][0]
        antecedent_end = result.context_offsets[relation.context.end - 1][1]
        mention_start = result.chunk_offsets[relation.chunk.start][0]
        mention_end = result.chunk_offsets[relation.chunk.end - 1][1]
        if not (
            0 <= antecedent_start < antecedent_end <= len(context)
            and 0 <= mention_start < mention_end <= len(chunk)
        ):
            raise ValueError("Decoded relation has invalid character offsets")
        candidates.append(
            (
                -relation.confidence,
                mention_start,
                mention_end,
                antecedent_start,
                antecedent_end,
            )
        )

    selected = []
    for _, start, end, antecedent_start, antecedent_end in sorted(candidates):
        if any(
            start < other_end and end > other_start
            for other_start, other_end, _ in selected
        ):
            continue
        selected.append((start, end, context[antecedent_start:antecedent_end]))

    annotated = chunk
    for _, end, antecedent in sorted(selected, reverse=True):
        annotated = annotated[:end] + f"[{antecedent}]" + annotated[end:]
    return annotated
