"""CoPresheaf transformer, referred to as CTNN in the MINTNN paper."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoPresheafConfig:
    combination: int
    num_statis: int
    encoder_h_dim: int = 128
    encoder_heads: int = 4
    encoder_stalk_dim: int = 32
    encoder_dropout: float = 0.05
    encoder_num_layers: int = 3
    low_rank: int = 8
    patch_size: int = 1
    max_len: int = 128
    norm_typ: str = "pre_norm"
    pooling: str = "cls_mean_max"


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 128) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        half = dim // 2
        div = torch.exp(torch.arange(half).float() * -(np.log(10000.0) / half))
        pe[:, :half] = torch.sin(pos * div)
        pe[:, half : 2 * half] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return x + self.pe[:, start:end, :]


class TopoPatchEmbeddings(nn.Module):
    def __init__(self, combination: int, num_statis: int, h_dim: int, patch_size: int) -> None:
        super().__init__()
        self.combination = combination
        self.num_statis = num_statis
        self.patch_size = (patch_size, num_statis)
        self.projection = nn.Conv2d(combination, h_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, topological_features: torch.Tensor) -> torch.Tensor:
        _batch, combination, _length, num_statis = topological_features.shape
        if combination != self.combination or num_statis != self.num_statis:
            raise ValueError(
                f"feature shape mismatch: got combination={combination}, num_statis={num_statis}; "
                f"expected {self.combination}, {self.num_statis}"
            )
        return self.projection(topological_features).flatten(2).transpose(1, 2)


class TopoEmbeddings(nn.Module):
    def __init__(self, combination: int, num_statis: int, h_dim: int, patch_size: int, max_len: int) -> None:
        super().__init__()
        self.patch_embed = TopoPatchEmbeddings(combination, num_statis, h_dim, patch_size)
        self.position_emb = PositionalEncoding(dim=h_dim, max_len=max_len)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, h_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def encode(self, topological_features: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(topological_features)
        batch, _length, dim = x.shape
        cls_tokens = self.cls_token.expand(batch, 1, dim)
        return self.position_emb(torch.cat((cls_tokens, x), dim=1), 0, x.size(1) + 1)


class SheafValueTransformNonlinear(nn.Module):
    def __init__(self, dim: int, heads: int, stalk_dim: int, low_rank: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.stalk_dim = stalk_dim
        self.low_rank = low_rank
        self.u_net = nn.Linear(dim, heads * stalk_dim * low_rank)
        self.v_net = nn.Linear(dim, heads * stalk_dim * low_rank)
        nn.init.xavier_uniform_(self.u_net.weight)
        nn.init.xavier_uniform_(self.v_net.weight)
        nn.init.zeros_(self.u_net.bias)
        nn.init.zeros_(self.v_net.bias)

    def forward(self, x: torch.Tensor, values: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        batch, n_tokens, _dim = x.shape
        u = torch.tanh(self.u_net(x)).view(batch, n_tokens, self.heads, self.stalk_dim, self.low_rank)
        v = torch.tanh(self.v_net(x)).view(batch, n_tokens, self.heads, self.stalk_dim, self.low_rank)
        u = u.permute(0, 2, 1, 3, 4).contiguous()
        v = v.permute(0, 2, 1, 3, 4).contiguous()
        s = torch.einsum("bhnkr,bhnk->bhnr", v, values)
        return torch.einsum("bhnkr,bhnr->bhnk", u, torch.matmul(attn, s))


class SheafTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, stalk_dim: int, low_rank: int, norm_typ: str, dropout: float) -> None:
        super().__init__()
        self.norm_typ = norm_typ
        self.heads = heads
        self.stalk_dim = stalk_dim
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(dim, heads * stalk_dim)
        self.w_k = nn.Linear(dim, heads * stalk_dim)
        self.w_v = nn.Linear(dim, heads * stalk_dim)
        self.w_o = nn.Linear(heads * stalk_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.sheaf_transform = SheafValueTransformNonlinear(dim, heads, stalk_dim, low_rank)
        for layer in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_uniform_(layer.weight)

    def _attention_update(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _dim = x.shape
        q = self.w_q(x).view(batch, length, self.heads, self.stalk_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, length, self.heads, self.stalk_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, length, self.heads, self.stalk_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.stalk_dim)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = self.sheaf_transform(x, v, attn).transpose(1, 2).reshape(batch, length, -1)
        return self.w_o(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_typ == "pre_norm":
            x_norm = self.norm1(x)
            x = x + self.dropout(self._attention_update(x_norm))
            return x + self.dropout(self.ffn(self.norm2(x)))
        if self.norm_typ == "post_norm":
            x2 = self.norm1(x + self.dropout(self._attention_update(x)))
            return self.norm2(x2 + self.dropout(self.ffn(x2)))
        raise ValueError(f"Unsupported norm_typ={self.norm_typ}")


class TopoEncoder(nn.Module):
    def __init__(self, config: CoPresheafConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SheafTransformerLayer(
                    config.encoder_h_dim,
                    config.encoder_heads,
                    config.encoder_stalk_dim,
                    config.low_rank,
                    config.norm_typ,
                    config.encoder_dropout,
                )
                for _ in range(config.encoder_num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class CoPresheafFinetune(nn.Module):
    """Regression head on top of the CoPresheaf transformer encoder."""

    def __init__(self, config: CoPresheafConfig) -> None:
        super().__init__()
        self.dim = config.encoder_h_dim
        self.pooling = config.pooling
        self.topo_embed = TopoEmbeddings(
            config.combination,
            config.num_statis,
            config.encoder_h_dim,
            config.patch_size,
            config.max_len,
        )
        self.encoder = TopoEncoder(config)
        if self.pooling == "cls_mean_max":
            head_dim = config.encoder_h_dim * 3
        elif self.pooling in {"cls", "average"}:
            head_dim = config.encoder_h_dim
        else:
            raise ValueError(f"Unsupported pooling={self.pooling}")
        self.fc = nn.Sequential(
            nn.Linear(head_dim, config.encoder_h_dim * 2),
            nn.ReLU(),
            nn.Linear(config.encoder_h_dim * 2, 1),
        )

    def forward(self, topological_features: torch.Tensor) -> torch.Tensor:
        x = self.encoder(self.topo_embed.encode(topological_features))
        if self.pooling == "average":
            pooled = x.mean(dim=1)
        elif self.pooling == "cls_mean_max":
            token_features = x[:, 1:, :]
            pooled = torch.cat([x[:, 0, :], token_features.mean(dim=1), token_features.max(dim=1).values], dim=1)
        else:
            pooled = x[:, 0, :]
        return self.fc(pooled)
