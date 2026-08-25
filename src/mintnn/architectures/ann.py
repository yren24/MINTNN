"""Fully connected regression network used by the MINTNN ANN baselines."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ANNRegressor(nn.Module):
    """Batch-normalized multilayer perceptron for flattened topological features.

    The final MOF models use hidden dimensions
    ``[2048, 1024, 1024, 512, 512, 64]``. The final LD50 models use
    ``[2048, 2048, 1024, 1024, 512, 64]``.
    """

    def __init__(
        self,
        d_in: int,
        hidden_dims: Sequence[int],
        d_out: int = 1,
        dropout: float = 0.0,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer width.")
        self.input_layer = nn.Linear(d_in, hidden_dims[0], bias=False)
        nn.init.xavier_uniform_(self.input_layer.weight)
        self.bn_input = nn.BatchNorm1d(hidden_dims[0])

        layers = []
        norms = []
        for left, right in zip(hidden_dims[:-1], hidden_dims[1:]):
            layer = nn.Linear(left, right, bias=False)
            nn.init.xavier_uniform_(layer.weight)
            layers.append(layer)
            norms.append(nn.BatchNorm1d(right))
        self.hidden_layers = nn.ModuleList(layers)
        self.bn_hidden = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.output_layer = nn.Linear(hidden_dims[-1], d_out, bias=True)
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn_input(self.input_layer(x))))
        for layer, norm in zip(self.hidden_layers, self.bn_hidden):
            x = self.dropout(F.relu(norm(layer(x))))
        x = self.output_layer(x)
        if self.output_activation == "tanh":
            return torch.tanh(x)
        if self.output_activation != "none":
            raise ValueError(f"Unsupported output_activation={self.output_activation}")
        return x
