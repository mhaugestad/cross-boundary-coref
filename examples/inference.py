import argparse
from pathlib import Path

from cross_boundary_coref import AffinityModel, RectangleDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    model = AffinityModel.from_checkpoint(args.checkpoint, device="cpu")
    result = model.predict_affinity(
        context="Marie Curie discovered radium.",
        chunk="She later won another Nobel Prize.",
    )
    relations = RectangleDecoder().decode(result)
    print("Affinity matrix shape:", result.affinity.shape)
    print("Affinity values are sigmoid probabilities; model logits are in result.logits.")
    for relation in relations:
        print(relation.to_dict())


if __name__ == "__main__":
    main()
