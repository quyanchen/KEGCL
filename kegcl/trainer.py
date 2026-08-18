import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .augment import GraphAugmentor, build_views
from .data import GraphArtifact
from .loss import blocked_codes, edge_info_nce, edge_union
from .model import KEGCL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def build_model(config: Dict[str, Any], input_dim: int) -> KEGCL:
    values = config["model"]
    return KEGCL(
        input_dim=input_dim,
        hidden_dim=values["hidden_dim"],
        projection_dim=values["projection_dim"],
        stages=values["stages"],
        depth_low=values["depth_low"],
        depth_high=values["depth_high"],
        activation=values["activation"],
        dropout=values["dropout"],
    )


def build_augmentors(config: Dict[str, Any]) -> Tuple[GraphAugmentor, GraphAugmentor]:
    values = config["augmentation"]
    return (
        GraphAugmentor(values["add_rate_1"], values["drop_edge_rate_1"], values["drop_feature_rate_1"]),
        GraphAugmentor(values["add_rate_2"], values["drop_edge_rate_2"], values["drop_feature_rate_2"]),
    )


def export_embeddings(
    model: KEGCL,
    artifact: GraphArtifact,
    device: torch.device,
    rng: random.Random,
    samples: int,
) -> torch.Tensor:
    model.eval()
    x = artifact.x.to(device)
    edges = artifact.retained_edge_index.to(device)
    embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(samples):
            first, second = model.sample_depths(rng)
            embeddings.append(model.encode(x, edges, first))
            embeddings.append(model.encode(x, edges, second))
    return torch.stack(embeddings, dim=0).mean(dim=0).cpu()


def write_embedding_tsv(path: Path, node_names: List[str], embeddings: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for node, vector in zip(node_names, embeddings.tolist()):
            handle.write(node + "\t" + "\t".join(f"{value:.8f}" for value in vector) + "\n")


def train(config: Dict[str, Any], artifact: GraphArtifact) -> Path:
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(config["device"])
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    model = build_model(config, artifact.num_features).to(device)
    first_augmentor, second_augmentor = build_augmentors(config)
    values = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=values["learning_rate"],
        weight_decay=values["weight_decay"],
    )
    original_edges = artifact.original_edge_index.to(device)
    blocked = blocked_codes(original_edges, artifact.num_nodes)
    run_dir = Path(config["output_root"]) / "runs" / artifact.name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    resolved_path = run_dir / "config.json"
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with metrics_path.open("w", encoding="utf-8") as metrics:
        for epoch in range(1, values["epochs"] + 1):
            model.train()
            view1, view2 = build_views(
                artifact.x,
                artifact.retained_edge_index,
                artifact.flagged_edge_index,
                first_augmentor,
                second_augmentor,
                generator,
            )
            depths1, depths2 = model.sample_depths(rng)
            x1 = view1.x.to(device)
            x2 = view2.x.to(device)
            edges1 = view1.edge_index.to(device)
            edges2 = view2.edge_index.to(device)
            positives = edge_union(edges1, edges2)
            optimizer.zero_grad()
            _, _, z1, z2 = model(x1, edges1, x2, edges2, depths1, depths2)
            loss, positive_count = edge_info_nce(
                z1=z1,
                z2=z2,
                positive_edge_index=positives,
                original_edge_index=original_edges,
                temperature=values["temperature"],
                negatives_per_positive=values["negatives_per_positive"],
                positive_batch_size=values["positive_batch_size"],
                blocked=blocked,
            )
            loss.backward()
            optimizer.step()
            record = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "depths1": depths1,
                "depths2": depths2,
                "positive_pairs": positive_count,
                "view1_edges": int(edges1.size(1)),
                "view2_edges": int(edges2.size(1)),
            }
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == values["epochs"]:
                print(f"epoch={epoch:04d} loss={record['loss']:.6f} depths={depths1}/{depths2}")
            if values["save_every"] and epoch % values["save_every"] == 0:
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config},
                    run_dir / f"checkpoint-{epoch}.pt",
                )
    checkpoint_path = run_dir / "checkpoint-final.pt"
    torch.save({"epoch": values["epochs"], "model": model.state_dict(), "config": config}, checkpoint_path)
    embeddings = export_embeddings(
        model=model,
        artifact=artifact,
        device=device,
        rng=rng,
        samples=values.get("inference_samples", 4),
    )
    torch.save(
        {"embeddings": embeddings, "node_names": artifact.node_names, "dataset": artifact.name},
        run_dir / "embeddings.pt",
    )
    write_embedding_tsv(run_dir / "embeddings.tsv", artifact.node_names, embeddings)
    return checkpoint_path


def load_embeddings(path: Path) -> Tuple[List[str], torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return payload["node_names"], payload["embeddings"]
