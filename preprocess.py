import argparse
from pathlib import Path

from kegcl.config import load_config
from kegcl.preprocess import build_vocabulary, preprocess_dataset, read_go_slim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["bio", "col", "dip", "k14"], default="bio")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = Path(config["data_root"])
    output_root = Path(config["output_root"])
    annotations = read_go_slim(data_root / config["go_file"])
    vocabulary = build_vocabulary(config["datasets"], data_root, annotations)
    names = list(config["datasets"]) if args.all else [args.dataset]
    for name in names:
        artifact = preprocess_dataset(
            name=name,
            spec=config["datasets"][name],
            data_root=data_root,
            output_root=output_root,
            annotations=annotations,
            vocabulary=vocabulary,
            expression_file=config["expression_file"],
            go_file=config["go_file"],
        )
        print(
            f"{name}: nodes={artifact.num_nodes} features={artifact.num_features} "
            f"retained={artifact.retained_edge_index.size(1)} "
            f"flagged={artifact.flagged_edge_index.size(1)}"
        )


if __name__ == "__main__":
    main()
