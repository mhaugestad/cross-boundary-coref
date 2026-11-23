from __future__ import annotations

from pathlib import Path
from typing import Sequence

from cross_boundary_coref.decoder import RelationRectangle
from cross_boundary_coref.model import AffinityResult


def plot_affinity(
    result: AffinityResult,
    relations: Sequence[RelationRectangle] | None = None,
    *,
    output_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figure, axis = plt.subplots(
        figsize=(
            max(7, len(result.chunk_tokens) * 0.32),
            max(5, len(result.context_tokens) * 0.28),
        )
    )
    image = axis.imshow(
        result.affinity,
        vmin=0,
        vmax=1,
        cmap="viridis",
        aspect="auto",
    )
    axis.set_xticks(range(len(result.chunk_tokens)), result.chunk_tokens, rotation=90)
    axis.set_yticks(range(len(result.context_tokens)), result.context_tokens)
    axis.set_xlabel("Chunk tokens")
    axis.set_ylabel("Context tokens")
    axis.set_title("Raw token-pair affinity probabilities")
    figure.colorbar(image, ax=axis, label="Sigmoid(logit)")
    for relation in relations or ():
        axis.add_patch(
            Rectangle(
                (relation.chunk.start - 0.5, relation.context.start - 0.5),
                relation.chunk.end - relation.chunk.start,
                relation.context.end - relation.context.start,
                fill=False,
                edgecolor="red",
                linewidth=1.5,
            )
        )
    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=120, bbox_inches="tight")
    return figure, axis
