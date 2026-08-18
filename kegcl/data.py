from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch


Edge = Tuple[str, str]


def canonical_edge(left: str, right: str) -> Edge:
    return (left, right) if left < right else (right, left)


def read_ppi(path: Path) -> Tuple[List[str], List[Edge]]:
    nodes: List[str] = []
    seen_nodes = set()
    edges: List[Edge] = []
    seen_edges = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2 or parts[0] == parts[1]:
                continue
            for node in parts[:2]:
                if node not in seen_nodes:
                    seen_nodes.add(node)
                    nodes.append(node)
            edge = canonical_edge(parts[0], parts[1])
            if edge not in seen_edges:
                seen_edges.add(edge)
                edges.append(edge)
    return nodes, edges


def edges_to_index(edges: Sequence[Edge], node_to_index: Dict[str, int]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    pairs = [(node_to_index[left], node_to_index[right]) for left, right in edges]
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def canonicalize_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    low = torch.minimum(edge_index[0], edge_index[1])
    high = torch.maximum(edge_index[0], edge_index[1])
    pairs = torch.stack((low, high), dim=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    return torch.unique(pairs, dim=0, sorted=True).t().contiguous()


def to_bidirectional(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.clone()
    reverse = edge_index.flip(0)
    return torch.cat((edge_index, reverse), dim=1).contiguous()


@dataclass
class GraphArtifact:
    name: str
    node_names: List[str]
    feature_names: List[str]
    x: torch.Tensor
    active_mask: torch.Tensor
    original_edge_index: torch.Tensor
    retained_edge_index: torch.Tensor
    flagged_edge_index: torch.Tensor
    metadata: Dict[str, Any]

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_features(self) -> int:
        return int(self.x.size(1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "node_names": self.node_names,
            "feature_names": self.feature_names,
            "x": self.x,
            "active_mask": self.active_mask,
            "original_edge_index": self.original_edge_index,
            "retained_edge_index": self.retained_edge_index,
            "flagged_edge_index": self.flagged_edge_index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GraphArtifact":
        return cls(**payload)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "GraphArtifact":
        return cls.from_dict(torch.load(path, map_location="cpu"))
