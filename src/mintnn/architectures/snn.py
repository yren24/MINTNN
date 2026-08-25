"""Simplicial-style graph regressors used by the MINTNN SNN experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, GCNConv
except ImportError:  # pragma: no cover - only triggered when optional dependency is absent.
    GATv2Conv = None
    GCNConv = None


class FullGraphNodeBlock(nn.Module):
    """Message block for MOF category-node SNNs over a complete category graph."""

    def __init__(self, h_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(h_dim, h_dim, bias=False),
        )
        self.norm = nn.BatchNorm1d(h_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) <= 1:
            neighbor = torch.zeros_like(x)
        else:
            neighbor = (x.sum(dim=1, keepdim=True) - x) / float(x.size(1) - 1)
        update = self.message(torch.cat([x, neighbor], dim=-1))
        batch, nodes, hidden = update.shape
        update = self.norm(update.reshape(batch * nodes, hidden)).reshape(batch, nodes, hidden)
        return x + self.dropout(update)


class CategoryNodeSNN(nn.Module):
    """MOF SNN over atom-category nodes."""

    def __init__(
        self,
        in_dim: int,
        num_nodes: int,
        h_dim: int = 128,
        layers: int = 3,
        dropout: float = 0.1,
        mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.node_init = nn.Linear(in_dim, h_dim, bias=False)
        self.blocks = nn.ModuleList([FullGraphNodeBlock(h_dim, dropout) for _ in range(layers)])
        mlp_hidden = mlp_hidden if mlp_hidden is not None else max(256, h_dim * 2)
        self.head = nn.Sequential(
            nn.Linear((num_nodes + 3) * h_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mlp_hidden, 1),
        )
        nn.init.xavier_uniform_(self.node_init.weight)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.node_init(node_features))
        for block in self.blocks:
            x = block(x)
        x_flat = x.reshape(x.size(0), -1)
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1).values
        x_sum = x.sum(dim=1)
        return self.head(torch.cat([x_flat, x_mean, x_max, x_sum], dim=1))


def build_line_edge_index(pair_endpoints: Sequence[tuple[str, str]]) -> torch.Tensor:
    """Build the directed line graph used by the LD50 pair-edge SNN."""

    src: list[int] = []
    dst: list[int] = []
    for i, (a_i, b_i) in enumerate(pair_endpoints):
        nodes_i = {a_i, b_i}
        for j, (a_j, b_j) in enumerate(pair_endpoints):
            if i != j and nodes_i.intersection({a_j, b_j}):
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


class EdgeOnlyPairSNN(nn.Module):
    """LD50 pair-edge graph SNN with GATv2/GCN message passing."""

    def __init__(
        self,
        in_dim: int,
        num_pairs: int,
        h_dim: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        mlp_hidden: int | None = None,
        conv_type: str = "gatv2",
        pooling: str = "flat_mean_max_sum",
    ) -> None:
        super().__init__()
        if GATv2Conv is None or GCNConv is None:
            raise ImportError("EdgeOnlyPairSNN requires torch-geometric.")
        if conv_type == "gatv2" and h_dim % heads != 0:
            raise ValueError("h_dim must be divisible by heads for GATv2Conv concat=True.")
        self.num_pairs = num_pairs
        self.pooling = pooling
        self.edge_init = nn.Linear(in_dim, h_dim)
        convs = []
        for _ in range(layers):
            if conv_type == "gatv2":
                convs.append(
                    GATv2Conv(
                        h_dim,
                        h_dim // heads,
                        heads=heads,
                        concat=True,
                        add_self_loops=False,
                        dropout=dropout,
                    )
                )
            elif conv_type == "gcn":
                convs.append(GCNConv(h_dim, h_dim, add_self_loops=False))
            else:
                raise ValueError(f"Unsupported conv_type={conv_type}")
        self.layers = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.BatchNorm1d(h_dim) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        if pooling == "flat":
            head_in_dim = num_pairs * h_dim
        elif pooling == "flat_mean_max_sum":
            head_in_dim = (num_pairs + 3) * h_dim
        else:
            raise ValueError(f"Unsupported pooling={pooling}")
        mlp_hidden = mlp_hidden if mlp_hidden is not None else max(256, h_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, edge_attr: torch.Tensor, line_edge_index: torch.Tensor) -> torch.Tensor:
        batch_size = edge_attr.size(0)
        x = self.edge_init(edge_attr).reshape(batch_size * self.num_pairs, -1)
        offsets = (torch.arange(batch_size, device=edge_attr.device) * self.num_pairs).view(-1, 1, 1)
        batched_line_edge_index = (
            line_edge_index.to(edge_attr.device).view(1, 2, -1) + offsets
        ).permute(1, 0, 2).reshape(2, -1)

        for conv, norm in zip(self.layers, self.norms):
            x = x + self.drop(norm(conv(x, batched_line_edge_index)))

        x = x.view(batch_size, self.num_pairs, -1)
        x_flat = x.reshape(batch_size, -1)
        if self.pooling == "flat":
            pooled = x_flat
        else:
            pooled = torch.cat([x_flat, x.mean(dim=1), x.max(dim=1).values, x.sum(dim=1)], dim=1)
        return self.head(pooled).view(-1)
