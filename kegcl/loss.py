from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .data import canonicalize_index


def edge_union(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return canonicalize_index(torch.cat((first, second), dim=1))


def blocked_codes(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    directed = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    loops = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
    codes = torch.cat((directed[0] * num_nodes + directed[1], loops * num_nodes + loops))
    return torch.unique(codes, sorted=True)


def sample_negative_nodes(
    anchors: torch.Tensor,
    blocked: torch.Tensor,
    num_nodes: int,
    count: int,
) -> torch.Tensor:
    candidates = torch.randint(num_nodes, (anchors.size(0), count), device=anchors.device)
    while True:
        codes = anchors[:, None] * num_nodes + candidates
        positions = torch.searchsorted(blocked, codes)
        matched = positions < blocked.numel()
        values = blocked[positions.clamp_max(blocked.numel() - 1)]
        invalid = matched & (values == codes)
        if not invalid.any():
            return candidates
        candidates[invalid] = torch.randint(num_nodes, (int(invalid.sum()),), device=anchors.device)


def _directional_loss(
    query: torch.Tensor,
    positive_bank: torch.Tensor,
    intra_bank: torch.Tensor,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query_vectors = query[anchors]
    positive_scores = (query_vectors * positive_bank[positives]).sum(dim=1, keepdim=True)
    inter_scores = torch.einsum("bd,bkd->bk", query_vectors, positive_bank[negatives])
    intra_scores = torch.einsum("bd,bkd->bk", query_vectors, intra_bank[negatives])
    logits = torch.cat((positive_scores, inter_scores, intra_scores), dim=1) / temperature
    labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


def edge_info_nce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    positive_edge_index: torch.Tensor,
    original_edge_index: torch.Tensor,
    temperature: float,
    negatives_per_positive: int,
    positive_batch_size: int,
    blocked: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    if positive_edge_index.size(1) > positive_batch_size:
        selection = torch.randperm(positive_edge_index.size(1), device=positive_edge_index.device)[:positive_batch_size]
        positive_edge_index = positive_edge_index[:, selection]
    anchors = torch.cat((positive_edge_index[0], positive_edge_index[1]))
    positives = torch.cat((positive_edge_index[1], positive_edge_index[0]))
    if blocked is None:
        blocked = blocked_codes(original_edge_index, z1.size(0))
    negatives = sample_negative_nodes(anchors, blocked, z1.size(0), negatives_per_positive)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    forward = _directional_loss(z1, z2, z1, anchors, positives, negatives, temperature)
    reverse = _directional_loss(z2, z1, z2, anchors, positives, negatives, temperature)
    return 0.5 * (forward + reverse), int(anchors.size(0))
