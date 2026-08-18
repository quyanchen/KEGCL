import argparse
from pathlib import Path

import torch

from kegcl.clustering import (
    cluster_complexes,
    compose_similarity_features,
    weight_node_attributes,
    write_complexes,
)
from kegcl.config import load_config
from kegcl.data import GraphArtifact
from kegcl.trainer import load_embeddings


def align_embeddings(artifact: GraphArtifact, names, embeddings: torch.Tensor) -> torch.Tensor:
    if names == artifact.node_names:
        return embeddings
    indices = {name: index for index, name in enumerate(names)}
    return torch.stack([embeddings[indices[name]] for name in artifact.node_names], dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["bio", "col", "dip", "k14"], required=True)
    parser.add_argument("--embeddings")
    parser.add_argument("--attachment-threshold", type=float)
    parser.add_argument("--min-complex-size", type=int)
    parser.add_argument("--max-complex-size", type=int)
    parser.add_argument("--redundancy-threshold", type=float)
    parser.add_argument("--attribute-weighting", choices=["binary", "idf"])
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config, args.dataset)
    artifact_path = Path(config["output_root"]) / "data" / f"{args.dataset}.pt"
    artifact = GraphArtifact.load(artifact_path)
    embedding_path = (
        Path(args.embeddings)
        if args.embeddings
        else Path(config["output_root"]) / "runs" / args.dataset / "embeddings.pt"
    )
    names, embeddings = load_embeddings(embedding_path)
    embeddings = align_embeddings(artifact, names, embeddings)
    values = config["clustering"]
    attribute_weighting = args.attribute_weighting or values.get(
        "attribute_weighting",
        "binary",
    )
    attributes = weight_node_attributes(artifact.x, attribute_weighting)
    embeddings = compose_similarity_features(
        embeddings,
        attributes,
        values["embedding_weight"],
    )
    attachment_threshold = (
        args.attachment_threshold
        if args.attachment_threshold is not None
        else values["attachment_threshold"]
    )
    min_complex_size = (
        args.min_complex_size
        if args.min_complex_size is not None
        else values["min_complex_size"]
    )
    max_complex_size = (
        args.max_complex_size
        if args.max_complex_size is not None
        else values["max_complex_size"]
    )
    redundancy_threshold = (
        args.redundancy_threshold
        if args.redundancy_threshold is not None
        else values["redundancy_threshold"]
    )
    cores, complexes = cluster_complexes(
        artifact=artifact,
        embeddings=embeddings,
        min_clique_size=values["min_clique_size"],
        attachment_threshold=attachment_threshold,
        max_cliques=values["max_cliques"],
        min_complex_size=min_complex_size,
        max_complex_size=max_complex_size,
        redundancy_threshold=redundancy_threshold,
    )
    output = (
        Path(args.output)
        if args.output
        else Path(config["output_root"]) / "runs" / args.dataset
    )
    write_complexes(
        output,
        artifact.node_names,
        cores,
        complexes,
        {
            "dataset": args.dataset,
            "attachment_threshold": attachment_threshold,
            "min_complex_size": min_complex_size,
            "max_complex_size": max_complex_size,
            "redundancy_threshold": redundancy_threshold,
            "embedding_weight": values["embedding_weight"],
            "attribute_weighting": attribute_weighting,
        },
    )
    print(f"cores={len(cores)} complexes={len(complexes)} output={output / 'complexes.txt'}")


if __name__ == "__main__":
    main()
