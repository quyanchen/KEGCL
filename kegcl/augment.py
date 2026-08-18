from dataclasses import dataclass
from typing import Tuple

import torch

from .data import canonicalize_index


@dataclass(frozen=True)
class AugmentedView:
    x: torch.Tensor
    edge_index: torch.Tensor


class GraphAugmentor:
    def __init__(self, add_rate: float, drop_edge_rate: float, drop_feature_rate: float) -> None:
        self.add_rate = add_rate
        self.drop_edge_rate = drop_edge_rate
        self.drop_feature_rate = drop_feature_rate

    def __call__(
        self,
        x: torch.Tensor,
        retained_edge_index: torch.Tensor,
        flagged_edge_index: torch.Tensor,
        generator: torch.Generator,
    ) -> AugmentedView:
        retained_mask = torch.rand(retained_edge_index.size(1), generator=generator) >= self.drop_edge_rate
        flagged_mask = torch.rand(flagged_edge_index.size(1), generator=generator) < self.add_rate
        edges = torch.cat(
            (retained_edge_index[:, retained_mask], flagged_edge_index[:, flagged_mask]),
            dim=1,
        )
        feature_mask = torch.rand(x.size(1), generator=generator) >= self.drop_feature_rate
        return AugmentedView(x=x * feature_mask.to(x.dtype), edge_index=canonicalize_index(edges))


def build_views(
    x: torch.Tensor,
    retained_edge_index: torch.Tensor,
    flagged_edge_index: torch.Tensor,
    first: GraphAugmentor,
    second: GraphAugmentor,
    generator: torch.Generator,
) -> Tuple[AugmentedView, AugmentedView]:
    return (
        first(x, retained_edge_index, flagged_edge_index, generator),
        second(x, retained_edge_index, flagged_edge_index, generator),
    )
