import argparse
from pathlib import Path

from kegcl.config import load_config
from kegcl.data import GraphArtifact
from kegcl.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["bio", "col", "dip", "k14"], default="bio")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config, args.dataset)
    if args.device:
        config["device"] = args.device
    artifact_path = Path(config["output_root"]) / "data" / f"{args.dataset}.pt"
    artifact = GraphArtifact.load(artifact_path)
    checkpoint = train(config, artifact)
    print(checkpoint)


if __name__ == "__main__":
    main()
