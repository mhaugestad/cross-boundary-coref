import argparse
from pathlib import Path

from cross_boundary_coref import AffinityModel, RectangleDecoder, plot_affinity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("affinity.png"))
    args = parser.parse_args()
    model = AffinityModel.from_checkpoint(args.checkpoint, device="cpu")
    result = model.predict_affinity(
        context="Marie Curie discovered radium.",
        chunk="She later won another Nobel Prize.",
    )
    relations = RectangleDecoder().decode(result)
    figure, _ = plot_affinity(result, relations, output_path=args.output)
    figure.clear()
    print(args.output)


if __name__ == "__main__":
    main()
