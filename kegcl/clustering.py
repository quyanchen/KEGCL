import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx
import torch
import torch.nn.functional as F

from .data import GraphArtifact


WeightedNeighbors = Dict[int, Dict[int, float]]


def weight_node_attributes(attributes: torch.Tensor, weighting: str) -> torch.Tensor:
    attributes = attributes.float()
    if weighting == "idf":
        frequencies = (attributes > 0).sum(dim=0)
        weights = torch.log((attributes.size(0) + 1) / (frequencies + 1)) + 1.0
        return attributes * weights
    return attributes


def fuse_node_features(
    embeddings: torch.Tensor,
    attributes: torch.Tensor,
    embedding_weight: float,
) -> torch.Tensor:
    embedding_features = F.normalize(embeddings.float(), dim=1) * embedding_weight ** 0.5
    attribute_features = F.normalize(attributes.float(), dim=1) * (1.0 - embedding_weight) ** 0.5
    return torch.cat((embedding_features, attribute_features), dim=1)


def compose_similarity_features(
    embeddings: torch.Tensor,
    attributes: torch.Tensor,
    embedding_weight: float,
) -> torch.Tensor:
    if embedding_weight == 0.0:
        return attributes.float()
    return fuse_node_features(embeddings, attributes, embedding_weight)


def build_weighted_neighbors(
    edge_index: torch.Tensor,
    embeddings: torch.Tensor,
) -> WeightedNeighbors:
    features = F.normalize(embeddings.float(), dim=1)
    neighbors: WeightedNeighbors = {index: {} for index in range(embeddings.size(0))}
    source, target = edge_index
    weights = (features[source] * features[target]).sum(dim=1).tolist()
    for (left, right), weight in zip(edge_index.t().tolist(), weights):
        neighbors[left][right] = weight
        neighbors[right][left] = weight
    return neighbors


def mine_maximal_cliques(
    edge_index: torch.Tensor,
    num_nodes: int,
    min_size: int,
    max_cliques: int = 0,
) -> List[Set[int]]:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_index.t().tolist())
    cliques: List[Set[int]] = []
    for clique in nx.find_cliques(graph):
        if len(clique) >= min_size:
            cliques.append(set(clique))
            if max_cliques and len(cliques) >= max_cliques:
                break
    return cliques


def density_score(nodes: Set[int], neighbors: WeightedNeighbors) -> float:
    if len(nodes) < 2:
        return 0.0
    total = sum(neighbors[left].get(right, 0.0) for left, right in combinations(nodes, 2))
    return total / (len(nodes) * (len(nodes) - 1) / 2)


def select_cores(
    cliques: Iterable[Set[int]],
    neighbors: WeightedNeighbors,
    min_size: int,
) -> List[Set[int]]:
    candidates = [set(clique) for clique in cliques if len(clique) >= min_size]
    cores: List[Set[int]] = []
    while candidates:
        candidates.sort(key=lambda clique: density_score(clique, neighbors), reverse=True)
        seed = candidates[0]
        cores.append(seed)
        residuals: List[Set[int]] = []
        seen = set()
        for clique in candidates[1:]:
            residual = clique - seed if clique & seed else clique
            key = tuple(sorted(residual))
            if len(residual) >= min_size and key not in seen:
                seen.add(key)
                residuals.append(residual)
        candidates = residuals
    return cores


def attachment_score(protein: int, complex_nodes: Set[int], neighbors: WeightedNeighbors) -> float:
    return sum(neighbors[protein].get(node, 0.0) for node in complex_nodes) / len(complex_nodes)


def expand_cores(
    cores: Sequence[Set[int]],
    neighbors: WeightedNeighbors,
    threshold: float,
) -> List[Set[int]]:
    complexes: List[Set[int]] = []
    for core in cores:
        candidates = set().union(*(neighbors[node].keys() for node in core)) - core
        affiliates = {
            protein
            for protein in candidates
            if attachment_score(protein, core, neighbors) >= threshold
        }
        complexes.append(set(core) | affiliates)
    return complexes


def overlap_ratio(first: Set[int], second: Set[int]) -> float:
    return len(first & second) / min(len(first), len(second))


def postprocess_complexes(
    complexes: Sequence[Set[int]],
    neighbors: WeightedNeighbors,
    min_size: int,
    max_size: int,
    redundancy_threshold: float,
) -> List[Set[int]]:
    unique = {
        frozenset(complex_nodes)
        for complex_nodes in complexes
        if len(complex_nodes) >= min_size and (not max_size or len(complex_nodes) <= max_size)
    }
    ranked = sorted(
        (set(complex_nodes) for complex_nodes in unique),
        key=lambda complex_nodes: (
            density_score(complex_nodes, neighbors),
            len(complex_nodes),
        ),
        reverse=True,
    )
    selected: List[Set[int]] = []
    for complex_nodes in ranked:
        is_distinct = all(
            overlap_ratio(complex_nodes, existing) < redundancy_threshold
            for existing in selected
        )
        if is_distinct:
            selected.append(complex_nodes)
    return selected


def cluster_complexes(
    artifact: GraphArtifact,
    embeddings: torch.Tensor,
    min_clique_size: int,
    attachment_threshold: float,
    max_cliques: int = 0,
    min_complex_size: int = 3,
    max_complex_size: int = 0,
    redundancy_threshold: float = 1.01,
) -> Tuple[List[Set[int]], List[Set[int]]]:
    neighbors = build_weighted_neighbors(
        artifact.original_edge_index,
        embeddings,
    )
    cliques = mine_maximal_cliques(
        artifact.original_edge_index,
        artifact.num_nodes,
        min_clique_size,
        max_cliques,
    )
    cores = select_cores(cliques, neighbors, min_clique_size)
    complexes = expand_cores(cores, neighbors, attachment_threshold)
    complexes = postprocess_complexes(
        complexes,
        neighbors,
        min_complex_size,
        max_complex_size,
        redundancy_threshold,
    )
    return cores, complexes


def write_complexes(
    directory: Path,
    node_names: Sequence[str],
    cores: Sequence[Set[int]],
    complexes: Sequence[Set[int]],
    metadata: Dict[str, object],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    named = [[node_names[index] for index in sorted(complex_nodes)] for complex_nodes in complexes]
    with (directory / "complexes.txt").open("w", encoding="utf-8") as handle:
        for complex_nodes in named:
            handle.write(" ".join(complex_nodes) + "\n")
    payload = {
        "metadata": metadata,
        "cores": [[node_names[index] for index in sorted(core)] for core in cores],
        "complexes": named,
    }
    with (directory / "complexes.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
