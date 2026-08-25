"""One-dimensional CNN regressors for filtration-indexed topological features."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    """Residual Conv1d block followed by length-halving max pooling."""

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.pool = nn.MaxPool1d(2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x + self.block(x))


class CNNRegressor(nn.Module):
    """Final MOF 1D CNN architecture.

    Inputs are shaped as ``(batch, channels, filtration_length)`` after the
    feature-specific reshape used in the MOF scripts.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        pool_size: int = 1,
        fc_hidden: int | None = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.init = nn.Conv1d(in_channels, h_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.blocks = nn.Sequential(
            *[ResidualConvBlock(h_channels, kernel_size=kernel_size, dropout=dropout) for _ in range(num_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        flat_size = h_channels * pool_size
        fc_hidden = flat_size * 2 if fc_hidden is None else fc_hidden
        self.fc = nn.Sequential(
            nn.Linear(flat_size, fc_hidden, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fc_hidden, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.fc(x.reshape(x.size(0), -1))


class CNN1DRegressor(CNNRegressor):
    """Final LD50 CNN naming alias.

    The LD50 final models use ``h_channels=128``, ``num_layers=5``,
    ``kernel_size=3`` and ``pool_size=1``. This subclass keeps that preset while
    reusing the same residual CNN body.
    """

    def __init__(
        self,
        in_channels: int,
        h_channels: int = 128,
        num_layers: int = 5,
        kernel_size: int = 3,
        pool_size: int = 1,
        d_out: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            h_channels=h_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            pool_size=pool_size,
            fc_hidden=h_channels * pool_size * 2,
            dropout=0.0,
        )
        self.fc[-1] = nn.Linear(h_channels * pool_size * 2, d_out, bias=True)
