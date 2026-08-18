import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch

from .data import Edge, GraphArtifact, edges_to_index, read_ppi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_go_slim(path: Path) -> Dict[str, Dict[str, Set[str]]]:
    annotations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 5 and parts[3] in {"F", "P", "C"} and parts[5]:
                annotations[parts[0]][parts[3]].add(parts[5])
    return annotations


def build_vocabulary(
    dataset_specs: Dict[str, Dict[str, str]],
    data_root: Path,
    annotations: Dict[str, Dict[str, Set[str]]],
) -> List[str]:
    proteins = set()
    for spec in dataset_specs.values():
        nodes, _ = read_ppi(data_root / spec["ppi_file"])
        proteins.update(nodes)
    molecular = set()
    biological = set()
    for protein in proteins:
        molecular.update(annotations.get(protein, {}).get("F", set()))
        biological.update(annotations.get(protein, {}).get("P", set()))
    return sorted(molecular) + sorted(biological)


def read_expression(path: Path, proteins: Set[str]) -> Dict[str, List[float]]:
    expression: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            protein = parts[1] if len(parts) == 38 else ""
            if protein in proteins and protein not in expression:
                values = [float(value) for value in parts[2:]]
                expression[protein] = [
                    (values[index] + values[index + 12] + values[index + 24]) / 3.0
                    for index in range(12)
                ]
    return expression


def active_times(values: Sequence[float]) -> Set[int]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sigma = variance ** 0.5
    threshold = mean + sigma * variance / (1.0 + variance)
    return {index for index, value in enumerate(values) if value >= threshold}


def build_features(
    nodes: Sequence[str],
    vocabulary: Sequence[str],
    annotations: Dict[str, Dict[str, Set[str]]],
) -> torch.Tensor:
    term_to_index = {term: index for index, term in enumerate(vocabulary)}
    features = torch.zeros((len(nodes), len(vocabulary)), dtype=torch.float32)
    for row, protein in enumerate(nodes):
        terms = annotations.get(protein, {}).get(
            "F",
            set(),
        ) | annotations.get(protein, {}).get("P", set())
        indices = [term_to_index[term] for term in terms if term in term_to_index]
        if indices:
            features[row, torch.tensor(indices, dtype=torch.long)] = 1.0
    return features


def partition_edges(
    edges: Iterable[Edge],
    temporal: Dict[str, Set[int]],
    annotations: Dict[str, Dict[str, Set[str]]],
) -> Tuple[List[Edge], List[Edge]]:
    retained: List[Edge] = []
    flagged: List[Edge] = []
    for left, right in edges:
        temporal_match = bool(temporal.get(left, set()) & temporal.get(right, set()))
        left_cc = annotations.get(left, {}).get("C", set())
        right_cc = annotations.get(right, {}).get("C", set())
        spatial_match = bool(left_cc & right_cc) or (not left_cc and not right_cc)
        if temporal_match and spatial_match:
            retained.append((left, right))
        else:
            flagged.append((left, right))
    return retained, flagged


def preprocess_dataset(
    name: str,
    spec: Dict[str, str],
    data_root: Path,
    output_root: Path,
    annotations: Dict[str, Dict[str, Set[str]]],
    vocabulary: Sequence[str],
    expression_file: str,
    go_file: str,
) -> GraphArtifact:
    ppi_path = data_root / spec["ppi_file"]
    expression_path = data_root / expression_file
    go_path = data_root / go_file
    nodes, edges = read_ppi(ppi_path)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    expression = read_expression(expression_path, set(nodes))
    temporal = {protein: active_times(values) for protein, values in expression.items()}
    retained, flagged = partition_edges(edges, temporal, annotations)
    activity = torch.zeros((len(nodes), 12), dtype=torch.bool)
    for protein, times in temporal.items():
        if times:
            activity[node_to_index[protein], torch.tensor(sorted(times), dtype=torch.long)] = True
    metadata = {
        "ppi_file": str(ppi_path),
        "expression_file": str(expression_path),
        "go_file": str(go_path),
        "sha256": {
            "ppi": sha256(ppi_path),
            "expression": sha256(expression_path),
            "go": sha256(go_path),
        },
        "num_original_edges": len(edges),
        "num_retained_edges": len(retained),
        "num_flagged_edges": len(flagged),
        "expression_coverage": len(expression),
    }
    artifact = GraphArtifact(
        name=name,
        node_names=list(nodes),
        feature_names=list(vocabulary),
        x=build_features(nodes, vocabulary, annotations),
        active_mask=activity,
        original_edge_index=edges_to_index(edges, node_to_index),
        retained_edge_index=edges_to_index(retained, node_to_index),
        flagged_edge_index=edges_to_index(flagged, node_to_index),
        metadata=metadata,
    )
    artifact_path = output_root / "data" / f"{name}.pt"
    manifest_path = output_root / "data" / f"{name}.json"
    artifact.save(artifact_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **metadata,
                "num_nodes": artifact.num_nodes,
                "num_features": artifact.num_features,
            },
            handle,
            indent=2,
        )
    return artifact
