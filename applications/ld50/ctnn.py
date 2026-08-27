#!/usr/bin/env python3
"""CoPresheaf transformer regression for LD50 topology features.

This mirrors plbind/finetune_copresheaf_final.py while using the LD50
toxicity split, feature folders, and regression metrics.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.stats as sp_stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


torch.set_default_dtype(torch.float32)

DEFAULT_ROOT = Path("data/ld50")
ALL_FEATURES = ["homology", "facet", "lap", "curvature", "forman"]
DEFAULT_SEEDS = [42]
DEFAULT_LRS = [5e-4, 3e-4, 1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5]


def parse_list_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_list_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def normalized_cas_key(name: str):
    name = name.strip()
    slash_match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", name)
    if slash_match:
        month, day, year = slash_match.groups()
        return (str(int(year)), str(int(month)), str(int(day)))
    dash_match = re.fullmatch(r"(\d+)-(\d+)-(\d+)", name)
    if dash_match:
        first, second, third = dash_match.groups()
        return (str(int(first)), str(int(second)), str(int(third)))
    return None


def build_file_index(feature_dir: Path) -> Dict[Tuple[str, str, str], List[str]]:
    index: Dict[Tuple[str, str, str], List[str]] = {}
    for path in feature_dir.glob("*.npy"):
        key = normalized_cas_key(path.stem)
        if key is not None:
            index.setdefault(key, []).append(path.stem)
    return index


def read_split_csv(root: Path, split: str) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    with open(root / f"LD50_{split}.csv", "r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            name = row["filename"].strip()
            if name:
                rows.append((name, float(row["label"])))
    return rows


def resolve_feature_path(csv_name: str, feature_dir: Path, index: Dict[Tuple[str, str, str], List[str]]):
    exact = feature_dir / f"{csv_name}.npy"
    if exact.exists():
        return csv_name, exact
    key = normalized_cas_key(csv_name)
    candidates = index.get(key, []) if key is not None else []
    if len(candidates) == 1:
        stem = candidates[0]
        return stem, feature_dir / f"{stem}.npy"
    return None, None


def default_tag_for_feature(feature: str, preset: str) -> str:
    if preset == "compact":
        return {
            "homology": "PH",
            "facet": "CA",
            "lap": "PL",
            "forman": "FPRC",
            "curvature": "EIC/bidirectional",
        }[feature]
    if preset in {"step01_bond0", "0.1_bond0"}:
        if feature == "homology":
            return "maxfil10_step01_bond0_homdim01"
        if feature == "facet":
            return "maxfil10_step01_bond0_facetdim1"
        if feature == "curvature":
            return "maxfil10_step01_bond0_curvbi"
        return "maxfil10_step01_bond0"
    if preset == "bond0":
        return "maxfil10_bond0_facetdim1" if feature == "facet" else "maxfil10_bond0"
    if preset in {"045", "0.45", "bond045"}:
        return "maxfil10_bond045_facetdim1" if feature == "facet" else "maxfil10_bond045"
    return preset


def feature_folder(root: Path, feature: str, tag: str, split: str) -> Path:
    path = root / "topology_features" / tag / feature / split
    if path.is_dir() or feature != "curvature":
        return path
    legacy_tag = {
        "EIC/bidirectional": "EIC_BI",
        "EIC/single_direction": "EIC",
    }.get(tag)
    if legacy_tag is None:
        return path
    legacy_path = root / "topology_features" / legacy_tag / feature / split
    return legacy_path if legacy_path.is_dir() else path


def feature_to_model_axes(arr, source_name="feature") -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr.transpose(1, 0, 2)
    if arr.ndim == 2:
        return arr[:, None, :]
    if arr.ndim == 1:
        return arr[:, None, None]
    raise ValueError(f"{source_name}: expected 1D, 2D, or 3D feature array, got shape {arr.shape}")


def limit_filtrations(feature: np.ndarray, max_filtrations: int = 100) -> np.ndarray:
    return feature[:, : min(max_filtrations, feature.shape[1]), :]


def infer_feature_shape(root: Path, feature_specs: Sequence[Tuple[str, str]]) -> Tuple[int, int, int]:
    rows = read_split_csv(root, "train")
    csv_name = rows[0][0]
    pieces = []
    for feature, tag in feature_specs:
        fdir = feature_folder(root, feature, tag, "train")
        stem, path = resolve_feature_path(csv_name, fdir, build_file_index(fdir))
        if path is None:
            raise RuntimeError(f"Missing first {feature}/train feature for {csv_name} in {fdir}")
        arr = limit_filtrations(feature_to_model_axes(np.load(path), source_name=str(path)))
        pieces.append(arr)
        print(f"#### CoPresheaf feature {feature}: model_axes={arr.shape} from {path.name} tag={tag}", flush=True)
    n_filtrations = min(piece.shape[1] for piece in pieces)
    stat_dims = {piece.shape[2] for piece in pieces}
    if len(stat_dims) != 1:
        raise ValueError(
            "CoPresheaf patch embedding expects a common num_statis. "
            f"Run one topology at a time or use compatible features; got shapes {[p.shape for p in pieces]}"
        )
    combination = sum(piece.shape[0] for piece in pieces)
    num_statis = stat_dims.pop()
    return combination, n_filtrations, num_statis


def apply_feature_transform(feature: np.ndarray, feature_transform: str) -> np.ndarray:
    if feature_transform == "none":
        return feature
    if feature_transform == "signedlog":
        return (np.sign(feature) * np.log1p(np.abs(feature))).astype(np.float32)
    raise ValueError(f"Unsupported feature_transform={feature_transform}")


def load_one_feature(path: Path, n_keep: int, feature_transform: str = "none") -> np.ndarray:
    feature = feature_to_model_axes(np.load(path), source_name=str(path)).astype(np.float32, copy=False)[:, :n_keep, :]
    return apply_feature_transform(feature, feature_transform)


def get_feature_label(root: Path, feature_specs: Sequence[Tuple[str, str]], feature_transform: str = "none"):
    combination, n_filtrations, num_statis = infer_feature_shape(root, feature_specs)
    print(
        f"#### CoPresheaf combined shape: combination={combination}, "
        f"N={n_filtrations}, num_statis={num_statis}",
        flush=True,
    )

    split_data = {}
    for split in ["train", "test"]:
        rows = read_split_csv(root, split)
        labels = np.asarray([label for _, label in rows], dtype=np.float32)
        per_feature_indices = []
        for feature, tag in feature_specs:
            fdir = feature_folder(root, feature, tag, split)
            if not fdir.is_dir():
                raise FileNotFoundError(f"Missing feature directory: {fdir}")
            per_feature_indices.append((feature, tag, fdir, build_file_index(fdir)))

        features = []
        names = []
        for csv_name, _label in rows:
            pieces = []
            resolved_name = None
            for feature, tag, fdir, index in per_feature_indices:
                stem, path = resolve_feature_path(csv_name, fdir, index)
                if path is None:
                    raise RuntimeError(f"Missing {feature}/{split} feature for {csv_name} in {fdir}")
                resolved_name = resolved_name or stem
                pieces.append(load_one_feature(path, n_filtrations, feature_transform))
            features.append(np.concatenate(pieces, axis=0))
            names.append(resolved_name or csv_name)
        split_data[split] = (np.stack(features).astype(np.float32), labels, names)

    train_fea, train_label, train_names = split_data["train"]
    test_fea, test_label, test_names = split_data["test"]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_flat = train_fea.reshape(len(train_fea), -1)
    test_flat = test_fea.reshape(len(test_fea), -1)
    train_fea = scaler.fit_transform(train_flat).reshape(train_fea.shape).astype(np.float32)
    test_fea = scaler.transform(test_flat).reshape(test_fea.shape).astype(np.float32)
    return train_fea, train_label, test_fea, test_label, train_names, test_names


class FinetuneDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).to(torch.get_default_dtype())
        y = torch.tensor([self.labels[idx]], dtype=torch.get_default_dtype())
        return x, y


def get_finetune_train_test_loader(
    root: Path,
    batch_size: int,
    feature_specs: Sequence[Tuple[str, str]],
    num_workers: int,
    pin_memory: bool,
    feature_transform: str = "none",
):
    train_X, train_Y, test_X, test_Y, train_names, test_names = get_feature_label(root, feature_specs, feature_transform)
    train_data = FinetuneDataset(train_X, train_Y)
    test_data = FinetuneDataset(test_X, test_Y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    print(f"train: {len(train_data)} test: {len(test_data)}", flush=True)
    return train_loader, test_loader, train_names, test_names


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        half = dim // 2
        div = torch.exp(torch.arange(half).float() * -(np.log(10000.0) / half))
        pe[:, :half] = torch.sin(pos * div)
        pe[:, half : 2 * half] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, start, end):
        return x + self.pe[:, start:end, :]


class TopoPatchEmbeddings(nn.Module):
    def __init__(self, combination, num_statis, h_dim, patch_size):
        super().__init__()
        self.combination = combination
        self.num_statis = num_statis
        self.patch_size = (patch_size, num_statis)
        self.projection = nn.Conv2d(combination, h_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, topological_features):
        batch, combination, _length, num_statis = topological_features.shape
        if combination != self.combination or num_statis != self.num_statis:
            raise ValueError(
                f"feature shape mismatch, got combination={combination}, num_statis={num_statis}, "
                f"expected {self.combination}, {self.num_statis}"
            )
        return self.projection(topological_features).flatten(2).transpose(1, 2)


class TopoEmbeddings(nn.Module):
    def __init__(self, combination, num_statis, h_dim, patch_size, max_len):
        super().__init__()
        self.patch_embed = TopoPatchEmbeddings(combination, num_statis, h_dim, patch_size)
        self.position_emb = PositionalEncoding(dim=h_dim, max_len=max_len)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, h_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def encode(self, topological_features):
        x = self.patch_embed(topological_features)
        batch, _length, dim = x.shape
        cls_tokens = self.cls_token.expand(batch, 1, dim)
        embeddings = torch.cat((cls_tokens, x), dim=1)
        return self.position_emb(embeddings, 0, embeddings.size(1))


class SheafValueTransformNonlinear(nn.Module):
    def __init__(self, dim, heads, stalk_dim, low_rank=8, bias=True):
        super().__init__()
        self.heads = heads
        self.stalk_dim = stalk_dim
        self.low_rank = low_rank
        self.u_net = nn.Linear(dim, heads * stalk_dim * low_rank, bias=bias)
        self.v_net = nn.Linear(dim, heads * stalk_dim * low_rank, bias=bias)
        nn.init.xavier_uniform_(self.u_net.weight)
        nn.init.xavier_uniform_(self.v_net.weight)
        if self.u_net.bias is not None:
            nn.init.zeros_(self.u_net.bias)
        if self.v_net.bias is not None:
            nn.init.zeros_(self.v_net.bias)

    def forward(self, x, values, attn):
        batch, n_tokens, _dim = x.shape
        u = torch.tanh(self.u_net(x)).view(batch, n_tokens, self.heads, self.stalk_dim, self.low_rank)
        v = torch.tanh(self.v_net(x)).view(batch, n_tokens, self.heads, self.stalk_dim, self.low_rank)
        u = u.permute(0, 2, 1, 3, 4).contiguous()
        v = v.permute(0, 2, 1, 3, 4).contiguous()
        s = torch.einsum("bhnkr,bhnk->bhnr", v, values)
        s_mix = torch.matmul(attn, s)
        return torch.einsum("bhnkr,bhnr->bhnk", u, s_mix)


class SheafTransformerLayer(nn.Module):
    def __init__(self, dim, heads, stalk_dim, low_rank, norm_typ, dropout):
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
        for layer in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(layer.weight)

    def post_norm_forward(self, x):
        batch, length, _dim = x.shape
        q = self.w_q(x).view(batch, length, self.heads, self.stalk_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, length, self.heads, self.stalk_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, length, self.heads, self.stalk_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.stalk_dim)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = self.sheaf_transform(x, v, attn).transpose(1, 2).reshape(batch, length, -1)
        out = self.w_o(out)
        x2 = self.norm1(x + self.dropout(out))
        return self.norm2(x2 + self.dropout(self.ffn(x2)))

    def pre_norm_forward(self, x1):
        x = self.norm1(x1)
        batch, length, _dim = x.shape
        q = self.w_q(x).view(batch, length, self.heads, self.stalk_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, length, self.heads, self.stalk_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, length, self.heads, self.stalk_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.stalk_dim)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = self.sheaf_transform(x, v, attn).transpose(1, 2).reshape(batch, length, -1)
        out = self.w_o(out)
        x = x1 + self.dropout(out)
        return x + self.dropout(self.ffn(self.norm2(x)))

    def forward(self, x):
        return self.pre_norm_forward(x) if self.norm_typ == "pre_norm" else self.post_norm_forward(x)


class TopoEncoder(nn.Module):
    def __init__(self, dim, heads, stalk_dim, low_rank, dropout, num_layers, norm_typ):
        super().__init__()
        self.layers = nn.ModuleList(
            [SheafTransformerLayer(dim, heads, stalk_dim, low_rank, norm_typ, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CoPresheafFinetune(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.dim = para.encoder_h_dim
        self.pooling = para.pooling
        self.topo_embed = TopoEmbeddings(para.combination, para.num_statis, para.encoder_h_dim, para.patch_size, para.max_len)
        self.encoder = TopoEncoder(
            para.encoder_h_dim,
            para.encoder_heads,
            para.encoder_stalk_dim,
            para.low_rank,
            para.encoder_dropout,
            para.encoder_num_layers,
            para.norm_typ,
        )
        if self.pooling == "cls_mean_max":
            head_dim = para.encoder_h_dim * 3
        elif self.pooling in {"cls", "average"}:
            head_dim = para.encoder_h_dim
        else:
            raise ValueError(f"Unsupported pooling={self.pooling}")
        self.fc = nn.Sequential(
            nn.Linear(head_dim, para.encoder_h_dim * 2),
            nn.ReLU(),
            nn.Linear(para.encoder_h_dim * 2, 1),
        )

    def forward(self, topological_features):
        x = self.encoder(self.topo_embed.encode(topological_features))
        if self.pooling == "average":
            x = x.mean(dim=1)
        elif self.pooling == "cls_mean_max":
            x_cls = x[:, 0, :]
            x_tokens = x[:, 1:, :]
            x = torch.cat([x_cls, x_tokens.mean(dim=1), x_tokens.max(dim=1).values], dim=1)
        else:
            x = x[:, 0, :]
        return self.fc(x)


class Para:
    def __init__(self, args, combination: int, num_filtrations: int, num_statis: int):
        self.finetune_lr = args.lr
        self.finetune_epoch = args.epoch
        self.finetune_batch_size = args.batch_size
        self.combination = combination
        self.num_filtrations = num_filtrations
        self.num_statis = num_statis
        self.encoder_h_dim = args.encoder_h_dim
        self.encoder_heads = args.encoder_heads
        self.encoder_stalk_dim = self.encoder_h_dim // self.encoder_heads
        self.encoder_num_layers = args.encoder_num_layers
        self.max_len = num_filtrations + 1
        self.low_rank = args.low_rank
        self.encoder_dropout = args.encoder_dropout
        self.norm_typ = args.norm_typ
        self.patch_size = args.patch_size
        self.weight_decay = args.weight_decay
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_workers = args.num_workers
        self.pin_memory = not args.no_pin_memory
        self.print_every = args.print_every
        self.feature_transform = args.feature_transform
        self.pooling = args.pooling

    def print_attrs(self):
        print("--- Hyperparameters ---", flush=True)
        for key, value in self.__dict__.items():
            print(f"{key}: {value}", flush=True)
        print("-----------------------", flush=True)


def safe_pcc(true_y, pred_y) -> float:
    true_y = np.asarray(true_y, dtype=np.float64)
    pred_y = np.asarray(pred_y, dtype=np.float64)
    if len(true_y) < 2 or np.std(true_y) == 0 or np.std(pred_y) == 0:
        return float("nan")
    pcc, _ = sp_stats.pearsonr(true_y, pred_y)
    return float(pcc)


def regression_metrics(true_y, pred_y):
    true_y = np.asarray(true_y, dtype=np.float64)
    pred_y = np.asarray(pred_y, dtype=np.float64)
    pcc = safe_pcc(true_y, pred_y)
    mse = mean_squared_error(true_y, pred_y)
    return {
        "pcc": pcc,
        "r2_paper": float(pcc * pcc) if np.isfinite(pcc) else float("nan"),
        "sklearn_r2": float(r2_score(true_y, pred_y)),
        "rmse": float(pow(mse, 0.5)),
        "rmse_x1.36": float(pow(mse, 0.5) * 1.36),
        "mae": float(mean_absolute_error(true_y, pred_y)),
    }


def train_model(model, dl, criterion, optimizer, scheduler, device):
    model.train()
    total_loss, total = 0, 0
    true_y, pred_y = [], []
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)
        true_y.extend(yb.view(-1).detach().cpu().tolist())
        pred_y.extend(logits.view(-1).detach().cpu().tolist())
    pcc = safe_pcc(true_y, pred_y)
    return total_loss / total, pcc, pow(mean_squared_error(true_y, pred_y), 0.5), model


def eval_model(model, dl, criterion, device):
    model.eval()
    total_loss, total = 0, 0
    true_y, pred_y = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
            true_y.extend(yb.view(-1).detach().cpu().tolist())
            pred_y.extend(logits.view(-1).detach().cpu().tolist())
    pcc = safe_pcc(true_y, pred_y)
    return total_loss / total, pcc, pow(mean_squared_error(true_y, pred_y), 0.5)


def collect_prediction(model, dl, device):
    model.eval()
    true_y, pred_y = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            true_y.extend(yb.view(-1).cpu().numpy())
            pred_y.extend(logits.view(-1).cpu().numpy())
    return np.asarray(true_y, dtype=np.float32), np.asarray(pred_y, dtype=np.float32)


def save_prediction(model, dl, device, split_name: str, seed: int, folder: Path, names: Sequence[str]):
    folder.mkdir(parents=True, exist_ok=True)
    true_y, pred_y = collect_prediction(model, dl, device)
    true_file = folder / f"{split_name}-true.npy"
    if not true_file.exists():
        np.save(true_file, true_y)
        np.save(folder / f"{split_name}-names.npy", np.asarray(names))
    np.save(folder / f"{split_name}-seed-{seed}-pred.npy", pred_y)
    return true_y, pred_y


def get_metrics(seeds: Sequence[int], folder: Path, split_name: str = "test"):
    true_y = np.load(folder / f"{split_name}-true.npy")
    pred = [np.load(folder / f"{split_name}-seed-{seed}-pred.npy") for seed in seeds]
    pred_mean = np.mean(np.asarray(pred), axis=0)
    metrics = regression_metrics(true_y, pred_mean)
    np.save(folder / f"{split_name}-ensemble-pred.npy", pred_mean)
    with open(folder / f"{split_name}-ensemble-metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)
    return metrics


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_one_seed(args, para: Para, seed: int, prediction_folder: Path, feature_specs: Sequence[Tuple[str, str]]):
    print(f"\n  -> Running Seed {seed}...", flush=True)
    set_seed(seed)
    train_loader, test_loader, train_names, test_names = get_finetune_train_test_loader(
        Path(args.root),
        para.finetune_batch_size,
        feature_specs,
        para.num_workers,
        para.pin_memory,
        para.feature_transform,
    )
    model = CoPresheafFinetune(para).to(para.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=para.finetune_lr, weight_decay=para.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=para.finetune_lr,
        steps_per_epoch=len(train_loader),
        epochs=para.finetune_epoch,
        pct_start=0.3,
    )

    last_train = last_test = None
    for e in range(para.finetune_epoch):
        train_loss, train_pcc, train_rmse, model = train_model(model, train_loader, criterion, optimizer, scheduler, para.device)
        test_loss, test_pcc, test_rmse = eval_model(model, test_loader, criterion, para.device)
        last_train = (train_loss, train_pcc, train_rmse)
        last_test = (test_loss, test_pcc, test_rmse)
        if e % para.print_every == 0 or e == para.finetune_epoch - 1:
            print(
                f"Epoch: {e + 1:03d}/{para.finetune_epoch} | "
                f"Train Loss: {train_loss:.3f}, PCC: {train_pcc:.3f}, R2: {train_pcc * train_pcc:.3f}, RMSE: {train_rmse:.3f} | "
                f"Test Loss: {test_loss:.3f}, PCC: {test_pcc:.3f}, R2: {test_pcc * test_pcc:.3f}, RMSE: {test_rmse:.3f}",
                flush=True,
            )

    train_true, train_pred = save_prediction(model, train_loader, para.device, "train", seed, prediction_folder, train_names)
    test_true, test_pred = save_prediction(model, test_loader, para.device, "test", seed, prediction_folder, test_names)
    return {
        "seed": seed,
        "last_train": last_train,
        "last_test": last_test,
        "train_metrics": regression_metrics(train_true, train_pred),
        "test_metrics": regression_metrics(test_true, test_pred),
    }


def format_lr(lr: float) -> str:
    return f"{lr:.0e}" if lr < 1e-3 else f"{lr:g}"


def combo_name(feature_specs: Sequence[Tuple[str, str]]) -> str:
    return "+".join(feature for feature, _tag in feature_specs)


def run_grid(args):
    feature_specs = [(feature, args.feature_root_tag or default_tag_for_feature(feature, args.preset)) for feature in args.features]
    feature_configs = {}
    if args.combine_features:
        for r in range(1, len(feature_specs) + 1):
            for combo in itertools.combinations(feature_specs, r):
                feature_configs[combo_name(combo)] = list(combo)
    else:
        feature_configs[combo_name(feature_specs)] = feature_specs

    results_summary = []
    start_time = time.time()
    for feat_name, specs in feature_configs.items():
        combination, n_filtrations, num_statis = infer_feature_shape(Path(args.root), specs)
        for lr in args.lrs:
            for h_dim in args.encoder_h_dims:
                for n_layers in args.encoder_num_layers:
                    args.lr = lr
                    args.encoder_h_dim = h_dim
                    args.encoder_num_layers_value = n_layers
                    local_args = argparse.Namespace(**vars(args))
                    local_args.encoder_num_layers = n_layers
                    para = Para(local_args, combination, n_filtrations, num_statis)
                    para.print_attrs()
                    prediction_folder = Path(args.output_dir) / (
                        f"{feat_name.replace('+', '_plus_')}_copresheaf_LD50_"
                        f"{'+'.join(tag for _f, tag in specs).replace('/', '-')}_"
                        f"seeds{'-'.join(map(str, args.seeds))}_lr{format_lr(lr)}_"
                        f"hdim{h_dim}_elayers{n_layers}_bs{para.finetune_batch_size}_"
                        f"ep{para.finetune_epoch}_minmax_{para.feature_transform}_{para.pooling}"
                    )
                    print("\n" + "=" * 96, flush=True)
                    print(
                        f" COPRESHEAF TESTING | Features: {feat_name} | Specs: {specs} | "
                        f"lr={lr} | hdim={h_dim} | layers={n_layers}",
                        flush=True,
                    )
                    print(f" Prediction folder: {prediction_folder}", flush=True)
                    print("=" * 96, flush=True)

                    per_seed = [run_one_seed(args, para, seed, prediction_folder, specs) for seed in args.seeds]
                    test_metrics = get_metrics(args.seeds, prediction_folder, "test")
                    train_metrics = get_metrics(args.seeds, prediction_folder, "train")
                    row = {
                        "features": feat_name,
                        "feature_specs": specs,
                        "lr": lr,
                        "encoder_h_dim": h_dim,
                        "encoder_num_layers": n_layers,
                        "combination": combination,
                        "num_filtrations": n_filtrations,
                        "num_statis": num_statis,
                        "feature_transform": para.feature_transform,
                        "pooling": para.pooling,
                        "prediction_folder": str(prediction_folder),
                        "train_ensemble": train_metrics,
                        "test_ensemble": test_metrics,
                        "per_seed": per_seed,
                    }
                    results_summary.append(row)
                    with open(prediction_folder / "summary.json", "w") as fp:
                        json.dump(row, fp, indent=2)
                    print(
                        f"--> Result: PCC={test_metrics['pcc']:.6f} | "
                        f"R2={test_metrics['r2_paper']:.6f} | RMSE={test_metrics['rmse']:.6f} | "
                        f"RMSE_x1.36={test_metrics['rmse_x1.36']:.6f}",
                        flush=True,
                    )

    results_summary.sort(key=lambda x: x["test_ensemble"]["pcc"], reverse=True)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / f"copresheaf_grid_summary_{int(time.time())}.csv"
    with open(summary_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "rank",
            "features",
            "lr",
            "encoder_h_dim",
            "encoder_num_layers",
            "combination",
            "num_filtrations",
            "num_statis",
            "feature_transform",
            "pooling",
            "test_pcc",
            "test_R2_paper",
            "test_RMSE",
            "test_RMSE_x1.36",
            "test_MAE",
            "train_pcc",
            "folder",
        ])
        for i, row in enumerate(results_summary, start=1):
            tm = row["test_ensemble"]
            trm = row["train_ensemble"]
            writer.writerow([
                i,
                row["features"],
                row["lr"],
                row["encoder_h_dim"],
                row["encoder_num_layers"],
                row["combination"],
                row["num_filtrations"],
                row["num_statis"],
                row["feature_transform"],
                row["pooling"],
                tm["pcc"],
                tm["r2_paper"],
                tm["rmse"],
                tm["rmse_x1.36"],
                tm["mae"],
                trm["pcc"],
                row["prediction_folder"],
            ])
    print("\n" + "=" * 100, flush=True)
    print(" FINAL COPRESHEAF GRID SEARCH RESULTS (LD50)", flush=True)
    print("=" * 100, flush=True)
    for i, row in enumerate(results_summary, start=1):
        tm = row["test_ensemble"]
        print(
            f"{i:02d}. {row['features']} lr={row['lr']} hdim={row['encoder_h_dim']} "
            f"layers={row['encoder_num_layers']} PCC={tm['pcc']:.6f} R2={tm['r2_paper']:.6f} "
            f"RMSE={tm['rmse']:.6f} folder={row['prediction_folder']}",
            flush=True,
        )
    print(f"summary_csv={summary_path}", flush=True)
    print(f"Sweep Finished. Total Duration: {round((time.time() - start_time) / 60, 1)} minutes", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--features", nargs="+", default=["homology"], choices=ALL_FEATURES)
    ap.add_argument("--feature-root-tag", default=None)
    ap.add_argument("--preset", default="compact")
    ap.add_argument("--combine-features", action="store_true")
    ap.add_argument("--lrs", type=parse_list_floats, default=DEFAULT_LRS)
    ap.add_argument("--seeds", type=parse_list_ints, default=DEFAULT_SEEDS)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--encoder-h-dims", type=parse_list_ints, default=[64, 128, 256])
    ap.add_argument("--encoder-num-layers", type=parse_list_ints, default=[3])
    ap.add_argument("--encoder-heads", type=int, default=4)
    ap.add_argument("--low-rank", type=int, default=8)
    ap.add_argument("--encoder-dropout", type=float, default=0.05)
    ap.add_argument("--norm-typ", choices=["post_norm", "pre_norm"], default="pre_norm")
    ap.add_argument("--patch-size", type=int, default=1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--no-pin-memory", action="store_true")
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--feature-transform", default="none", choices=["none", "signedlog"])
    ap.add_argument("--pooling", default="cls_mean_max", choices=["cls", "average", "cls_mean_max"])
    ap.add_argument("--output-dir", default="results/ld50/copresheaf")
    return ap.parse_args()


if __name__ == "__main__":
    run_grid(parse_args())
