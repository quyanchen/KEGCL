import random
from typing import List, Sequence, Tuple

import torch
from torch import nn

from .data import to_bidirectional


def normalized_adjacency(edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> torch.Tensor:
    directed = to_bidirectional(edge_index)
    loops = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
    source = torch.cat((directed[0], loops))
    target = torch.cat((directed[1], loops))
    degree = torch.zeros(num_nodes, device=edge_index.device, dtype=dtype)
    degree.scatter_add_(0, target, torch.ones_like(target, dtype=dtype))
    inverse = degree.clamp_min(1).pow(-0.5)
    values = inverse[source] * inverse[target]
    indices = torch.stack((target, source), dim=0)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()


def sample_depth_pair(stages: int, low: int, high: int, rng: random.Random) -> Tuple[List[int], List[int]]:
    if stages < 1 or low < 0 or low >= high:
        raise ValueError("Depth sampling requires stages >= 1 and 0 <= low < high")
    while True:
        first = [rng.randint(low, high) for _ in range(stages)]
        second = [rng.randint(low, high) for _ in range(stages)]
        if sum(first) != sum(second):
            return (first, second) if sum(first) < sum(second) else (second, first)


def activation_layer(name: str, channels: int) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "rrelu":
        return nn.RReLU()
    if name == "prelu":
        return nn.PReLU(channels)
    raise ValueError(f"Unknown activation: {name}")


class RandomizedEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        stages: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * stages
        self.transforms = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1]) for index in range(stages)
        )
        self.activations = nn.ModuleList(
            activation_layer(activation, hidden_dim) for _ in range(stages - 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, depths: Sequence[int]) -> torch.Tensor:
        adjacency = normalized_adjacency(edge_index, x.size(0), x.dtype)
        representation = x
        for index, (transform, depth) in enumerate(zip(self.transforms, depths)):
            for _ in range(depth):
                representation = torch.sparse.mm(adjacency, representation)
            representation = transform(representation)
            if index < len(self.transforms) - 1:
                representation = self.dropout(self.activations[index](representation))
        return representation


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class KEGCL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        projection_dim: int,
        stages: int = 2,
        depth_low: int = 1,
        depth_high: int = 4,
        activation: str = "prelu",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = RandomizedEncoder(input_dim, hidden_dim, stages, activation, dropout)
        self.projector = ProjectionHead(hidden_dim, projection_dim)
        self.stages = stages
        self.depth_low = depth_low
        self.depth_high = depth_high

    def sample_depths(self, rng: random.Random) -> Tuple[List[int], List[int]]:
        return sample_depth_pair(self.stages, self.depth_low, self.depth_high, rng)

    def forward(
        self,
        x1: torch.Tensor,
        edge_index1: torch.Tensor,
        x2: torch.Tensor,
        edge_index2: torch.Tensor,
        depths1: Sequence[int],
        depths2: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.encoder(x1, edge_index1, depths1)
        h2 = self.encoder(x2, edge_index2, depths2)
        return h1, h2, self.projector(h1), self.projector(h2)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, depths: Sequence[int]) -> torch.Tensor:
        return self.encoder(x, edge_index, depths)
