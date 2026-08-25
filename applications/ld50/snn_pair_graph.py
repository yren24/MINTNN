#!/usr/bin/env python3
"""Grid-search edge-only SNN for LD50 topology features.

This mirrors plbind/snn/grid_pair_edgegraph_methods_final.py as closely as
possible while adapting only the LD50 data loading and the 30 molecule
element-pair graph used by toxicity_topology_features.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.stats as sp_stats
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATv2Conv, GCNConv


torch.set_default_dtype(torch.float32)

DEFAULT_ROOT = Path("data/ld50")

BASE_ELEMENTS = ["H", "C", "N", "O"]
ALL_ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
ELEMENT_ORDER = {ele: i for i, ele in enumerate(ALL_ELEMENTS)}
PAIR_ELEMENT_NAMES = [
    (a, b)
    for a in BASE_ELEMENTS
    for b in ALL_ELEMENTS
    if ELEMENT_ORDER[a] < ELEMENT_ORDER[b]
]
PAIR_NAMES = [f"{a}-{b}" for a, b in PAIR_ELEMENT_NAMES]
NUM_PAIRS = len(PAIR_NAMES)


@dataclass(frozen=True)
class MethodConfig:
    name: str
    tag: str
    num_pairs: int = NUM_PAIRS


ALL_FEATURES = ["facet", "curvature", "homology", "forman", "lap"]

DEFAULT_LRS = [
    1e-3,
    9e-4,
    8e-4,
    7e-4,
    6e-4,
    5e-4,
    4e-4,
    3e-4,
    1e-4,
    9e-5,
    8e-5,
    5e-5,
    3e-5,
    2e-5,
    1e-5,
]
DEFAULT_H_DIMS = [128]
DEFAULT_EPOCHS = [50]


def parse_list(text, cast):
    if text is None or text == "":
        return []
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def default_tag_for_feature(feature: str, preset: str) -> str:
    if preset == "compact":
        return {
            "homology": "PH",
            "facet": "CA",
            "lap": "PL",
            "forman": "FPRC",
            "curvature": "EIC",
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


def build_methods(preset: str, override_tag: str | None = None) -> Dict[str, MethodConfig]:
    methods = {}
    for feature in ALL_FEATURES:
        methods[feature] = MethodConfig(
            name=feature,
            tag=override_tag if override_tag else default_tag_for_feature(feature, preset),
        )
    return methods


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    with open(root / f"LD50_{split}.csv", "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["filename"].strip()
            if name:
                rows.append((name, float(row["label"])))
    return rows


def feature_folder(root: Path, feature: str, tag: str, split: str) -> Path:
    return root / "topology_features" / tag / feature / split


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


def feature_to_model_axes(arr, source_name="feature"):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr.transpose(1, 0, 2)
    if arr.ndim == 2:
        return arr[:, None, :]
    if arr.ndim == 1:
        return arr[:, None, None]
    raise ValueError(f"{source_name}: expected 1D, 2D, or 3D feature array, got shape {arr.shape}")


def apply_feature_transform(edge_attr: np.ndarray, feature_transform: str) -> np.ndarray:
    if feature_transform == "none":
        return edge_attr
    if feature_transform == "signedlog":
        return (np.sign(edge_attr) * np.log1p(np.abs(edge_attr))).astype(np.float32)
    raise ValueError(f"Unsupported feature_transform={feature_transform}")


def load_pair_feature(path: Path, num_pairs: int, feature_transform: str = "none"):
    fea = feature_to_model_axes(np.load(path), source_name=str(path))
    if fea.shape[0] < num_pairs:
        raise ValueError(f"{path}: cannot extract {num_pairs} pairs from feature shape {fea.shape}")
    pair_fea = fea[:num_pairs, :100, :]
    edge_attr = pair_fea.reshape(num_pairs, -1).astype(np.float32)
    return apply_feature_transform(edge_attr, feature_transform)


def fit_flat_scaler(rows, feature_dir: Path, index, num_pairs: int, scaler_type: str, feature_transform: str):
    if scaler_type == "standard":
        scaler = StandardScaler()
        for csv_name, _label in rows:
            _stem, path = resolve_feature_path(csv_name, feature_dir, index)
            if path is None:
                raise RuntimeError(f"Missing train feature for {csv_name} in {feature_dir}")
            edge_attr = load_pair_feature(path, num_pairs, feature_transform)
            scaler.partial_fit(edge_attr.reshape(1, -1))
        return scaler
    if scaler_type == "minmax":
        flat_features = []
        for csv_name, _label in rows:
            _stem, path = resolve_feature_path(csv_name, feature_dir, index)
            if path is None:
                raise RuntimeError(f"Missing train feature for {csv_name} in {feature_dir}")
            edge_attr = load_pair_feature(path, num_pairs, feature_transform)
            flat_features.append(edge_attr.reshape(-1))
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(np.asarray(flat_features, dtype=np.float32))
        return scaler
    raise ValueError(f"Unsupported scaler_type={scaler_type}")


def check_train_features(rows, feature_dir: Path, index, num_pairs: int, feature_transform: str):
    for csv_name, _label in rows:
        _stem, path = resolve_feature_path(csv_name, feature_dir, index)
        if path is None:
            raise RuntimeError(f"Missing train feature for {csv_name} in {feature_dir}")
        load_pair_feature(path, num_pairs, feature_transform)


class PairEdgeDataset(Dataset):
    def __init__(self, rows, feature_dir: Path, index, scaler, num_pairs: int, feature_transform: str):
        self.rows = rows
        self.feature_dir = feature_dir
        self.index = index
        self.scaler = scaler
        self.num_pairs = num_pairs
        self.feature_transform = feature_transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        csv_name, label = self.rows[idx]
        stem, path = resolve_feature_path(csv_name, self.feature_dir, self.index)
        if path is None:
            raise RuntimeError(f"Missing feature for {csv_name} in {self.feature_dir}")
        edge_attr = load_pair_feature(path, self.num_pairs, self.feature_transform)
        edge_attr = self.scaler.transform(edge_attr.reshape(1, -1)).reshape(self.num_pairs, -1).astype(np.float32)
        return torch.from_numpy(edge_attr), torch.tensor(np.float32(label), dtype=torch.get_default_dtype()), stem or csv_name


def collate_pair_edges(samples):
    edge_attr, labels, names = zip(*samples)
    return torch.stack(edge_attr, dim=0), torch.stack(labels, dim=0), list(names)


def build_line_edge_index(num_pairs: int):
    if num_pairs != NUM_PAIRS:
        raise ValueError(f"Unsupported num_pairs={num_pairs}; toxicity currently expects {NUM_PAIRS}.")
    src, dst = [], []
    endpoints = PAIR_ELEMENT_NAMES
    for i, (a_i, b_i) in enumerate(endpoints):
        nodes_i = {a_i, b_i}
        for j, (a_j, b_j) in enumerate(endpoints):
            if i == j:
                continue
            if nodes_i.intersection({a_j, b_j}):
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


class EdgeOnlyPairSNN(nn.Module):
    def __init__(self, in_dim, num_pairs, h_dim, heads, layers, dropout, mlp_hidden, conv_type="gatv2", pooling="flat_mean_max_sum"):
        super().__init__()
        if conv_type == "gatv2" and h_dim % heads != 0:
            raise ValueError("h_dim must be divisible by heads for GATv2Conv concat=True.")
        self.num_pairs = num_pairs
        self.conv_type = conv_type
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
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, edge_attr, line_edge_index):
        batch_size = edge_attr.size(0)
        x = self.edge_init(edge_attr).reshape(batch_size * self.num_pairs, -1)
        offsets = (torch.arange(batch_size, device=edge_attr.device) * self.num_pairs).view(-1, 1, 1)
        batched_line_edge_index = (
            line_edge_index.to(edge_attr.device).view(1, 2, -1) + offsets
        ).permute(1, 0, 2).reshape(2, -1)

        for conv, norm in zip(self.layers, self.norms):
            out = conv(x, batched_line_edge_index)
            out = self.drop(norm(out))
            x = x + out

        x = x.view(batch_size, self.num_pairs, -1)
        x_flat = x.reshape(batch_size, -1)
        if self.pooling == "flat":
            return self.head(x_flat).view(-1)
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1).values
        x_sum = x.sum(dim=1)
        x = torch.cat([x_flat, x_mean, x_max, x_sum], dim=1)
        return self.head(x).view(-1)


def metrics_from_arrays(true_y, pred_y):
    true_y = np.asarray(true_y, dtype=np.float64)
    pred_y = np.asarray(pred_y, dtype=np.float64)
    if np.std(true_y) < 1e-12 or np.std(pred_y) < 1e-12:
        pcc = 0.0
    else:
        pcc, _ = sp_stats.pearsonr(true_y, pred_y)
    rmse = mean_squared_error(true_y, pred_y) ** 0.5
    return {
        "pcc": float(pcc),
        "r2_paper": float(pcc * pcc),
        "sklearn_r2": float(r2_score(true_y, pred_y)),
        "rmse": float(rmse),
        "rmse_x1.36": float(rmse * 1.36),
        "mae": float(mean_absolute_error(true_y, pred_y)),
    }


def run_epoch(model, loader, line_edge_index, criterion, optimizer, scheduler, device, train):
    model.train(train)
    total_loss, total = 0.0, 0
    true_y, pred_y = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for edge_attr, label, _names in loader:
            edge_attr = edge_attr.to(device)
            label = label.to(device)
            if train:
                optimizer.zero_grad()
            pred = model(edge_attr, line_edge_index)
            loss = criterion(pred, label)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            bs = edge_attr.size(0)
            total_loss += loss.item() * bs
            total += bs
            true_y.extend(label.detach().cpu().numpy().tolist())
            pred_y.extend(pred.detach().cpu().numpy().tolist())
    metrics = metrics_from_arrays(true_y, pred_y)
    return total_loss / total, metrics


def make_loader(dataset, batch_size, shuffle, num_workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pair_edges,
        generator=generator,
    )


def run_setting(method, config, train_data, test_data, line_edge_index, first_dim, args, lr, h_dim, layer_setting, conv_type, epoch_setting, setting_idx):
    set_seed(args.seed + setting_idx)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = make_loader(train_data, args.batch_size, True, args.num_workers, args.seed + setting_idx)
    test_loader = make_loader(test_data, args.batch_size, False, args.num_workers, args.seed + setting_idx)

    mlp_hidden = args.mlp_hidden if args.mlp_hidden is not None else max(256, h_dim * 2)
    model = EdgeOnlyPairSNN(
        in_dim=first_dim,
        num_pairs=config.num_pairs,
        h_dim=h_dim,
        heads=args.heads,
        layers=layer_setting,
        dropout=args.dropout,
        mlp_hidden=mlp_hidden,
        conv_type=conv_type,
        pooling=args.pooling,
    ).to(device)
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "smoothl1":
        criterion = nn.SmoothL1Loss(beta=args.smoothl1_beta)
    elif args.loss == "huber":
        criterion = nn.HuberLoss(delta=args.smoothl1_beta)
    else:
        raise ValueError(f"Unsupported loss={args.loss}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epoch_setting,
        pct_start=0.3,
    )

    best = {
        "test_pcc": -999.0,
        "test_rmse": None,
        "train_pcc": None,
        "train_rmse": None,
        "epoch": None,
    }
    final = None
    for e in range(epoch_setting):
        train_loss, train_metrics = run_epoch(
            model, train_loader, line_edge_index, criterion, optimizer, scheduler, device, train=True
        )
        test_loss, test_metrics = run_epoch(
            model, test_loader, line_edge_index, criterion, optimizer, scheduler, device, train=False
        )
        final = (train_loss, train_metrics, test_loss, test_metrics)
        print(
            f"{method} setting {setting_idx:03d} lr={lr:g} h_dim={h_dim} epochs={epoch_setting} | "
            f"layers={layer_setting} conv={conv_type} | "
            f"Epoch {e + 1:03d}/{epoch_setting} | "
            f"Train PCC {train_metrics['pcc']:.4f} R2 {train_metrics['r2_paper']:.4f} RMSE {train_metrics['rmse']:.4f} | "
            f"Test PCC {test_metrics['pcc']:.4f} R2 {test_metrics['r2_paper']:.4f} RMSE {test_metrics['rmse']:.4f}",
            flush=True,
        )
        if test_metrics["pcc"] > best["test_pcc"]:
            best.update(
                {
                    "test_pcc": test_metrics["pcc"],
                    "test_rmse": test_metrics["rmse"],
                    "test_r2_paper": test_metrics["r2_paper"],
                    "test_sklearn_r2": test_metrics["sklearn_r2"],
                    "test_mae": test_metrics["mae"],
                    "train_pcc": train_metrics["pcc"],
                    "train_rmse": train_metrics["rmse"],
                    "train_r2_paper": train_metrics["r2_paper"],
                    "epoch": e + 1,
                }
            )

    train_loss, train_metrics, test_loss, test_metrics = final
    return {
        "method": method,
        "feature_tag": config.tag,
        "num_pairs": config.num_pairs,
        "pair_names": "|".join(PAIR_NAMES),
        "input_dim_per_pair": first_dim,
        "lr": lr,
        "h_dim": h_dim,
        "layers": layer_setting,
        "conv_type": conv_type,
        "loss": args.loss,
        "smoothl1_beta": args.smoothl1_beta,
        "scaler": args.scaler,
        "feature_transform": args.feature_transform,
        "pooling": args.pooling,
        "epochs": epoch_setting,
        "best_epoch": best["epoch"],
        "best_train_pcc": best["train_pcc"],
        "best_train_r2_paper": best["train_r2_paper"],
        "best_train_rmse": best["train_rmse"],
        "best_test_pcc": best["test_pcc"],
        "best_test_r2_paper": best["test_r2_paper"],
        "best_test_sklearn_r2": best["test_sklearn_r2"],
        "best_test_rmse": best["test_rmse"],
        "best_test_rmse_x1.36": best["test_rmse"] * 1.36,
        "best_test_mae": best["test_mae"],
        "final_train_pcc": train_metrics["pcc"],
        "final_train_r2_paper": train_metrics["r2_paper"],
        "final_train_rmse": train_metrics["rmse"],
        "final_test_pcc": test_metrics["pcc"],
        "final_test_r2_paper": test_metrics["r2_paper"],
        "final_test_sklearn_r2": test_metrics["sklearn_r2"],
        "final_test_rmse": test_metrics["rmse"],
        "final_test_rmse_x1.36": test_metrics["rmse_x1.36"],
        "final_test_mae": test_metrics["mae"],
    }


def task_grid(method_names, lrs, h_dims, layers_list, conv_types, epochs):
    tasks = []
    for method in method_names:
        for lr in lrs:
            for h_dim in h_dims:
                for layer_setting in layers_list:
                    for conv_type in conv_types:
                        for epoch_setting in epochs:
                            tasks.append((method, lr, h_dim, layer_setting, conv_type, epoch_setting))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Grid search edge-only pair graph SNN across LD50 topology methods.")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--preset", type=str, default="compact")
    parser.add_argument("--feature-root-tag", type=str, default=None)
    parser.add_argument("--method", type=str, default="all", choices=["all"] + sorted(ALL_FEATURES))
    parser.add_argument("--task-id", type=int, default=None, help="Run a single setting from the global task list.")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--lr-list", type=str, default=",".join(f"{x:g}" for x in DEFAULT_LRS))
    parser.add_argument("--h-dim-list", type=str, default=",".join(str(x) for x in DEFAULT_H_DIMS))
    parser.add_argument("--epoch-list", type=str, default=",".join(str(x) for x in DEFAULT_EPOCHS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--layers-list", type=str, default=None)
    parser.add_argument("--conv-type", type=str, default="gatv2", choices=["gatv2", "gcn"])
    parser.add_argument("--conv-type-list", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smoothl1", "huber"])
    parser.add_argument("--smoothl1-beta", type=float, default=1.0)
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"])
    parser.add_argument("--feature-transform", type=str, default="none", choices=["none", "signedlog"])
    parser.add_argument("--pooling", type=str, default="flat_mean_max_sum", choices=["flat", "flat_mean_max_sum"])
    parser.add_argument("--mlp-hidden", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-folder", type=str, default="results/ld50/snn_grid")
    args = parser.parse_args()

    root = Path(args.root)
    methods = build_methods(args.preset, args.feature_root_tag)
    lrs = parse_list(args.lr_list, float) or DEFAULT_LRS
    h_dims = parse_list(args.h_dim_list, int) or DEFAULT_H_DIMS
    epochs = parse_list(args.epoch_list, int) or DEFAULT_EPOCHS
    layers_list = parse_list(args.layers_list, int) if args.layers_list else [args.layers]
    conv_types = parse_list(args.conv_type_list, str) if args.conv_type_list else [args.conv_type]
    method_names = sorted(methods) if args.method == "all" else [args.method]
    tasks = task_grid(method_names, lrs, h_dims, layers_list, conv_types, epochs)

    if args.list_tasks:
        for idx, (method, lr, h_dim, layer_setting, conv_type, epoch_setting) in enumerate(tasks):
            print(f"{idx}: method={method} lr={lr:g} h_dim={h_dim} layers={layer_setting} conv={conv_type} epochs={epoch_setting}")
        print(f"#### Total tasks: {len(tasks)}")
        print(f"#### Pair names ({NUM_PAIRS}): {PAIR_NAMES}")
        print(f"#### Line graph directed edges: {build_line_edge_index(NUM_PAIRS).size(1)}")
        return

    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(tasks):
            raise ValueError(f"task_id={args.task_id} out of range 0..{len(tasks) - 1}")
        tasks = [tasks[args.task_id]]

    os.makedirs(args.output_folder, exist_ok=True)
    train_rows = read_split_csv(root, "train")
    test_rows = read_split_csv(root, "test")
    grouped = {}
    for task in tasks:
        grouped.setdefault(task[0], []).append(task)

    all_rows = []
    for method, method_tasks in grouped.items():
        config = methods[method]
        start = time.time()
        train_dir = feature_folder(root, method, config.tag, "train")
        test_dir = feature_folder(root, method, config.tag, "test")
        train_index = build_file_index(train_dir)
        test_index = build_file_index(test_dir)
        print(
            f"#### Method={method} tag={config.tag} train_dir={train_dir} "
            f"num_pairs={config.num_pairs} settings={len(method_tasks)}",
            flush=True,
        )
        first_stem, first_path = resolve_feature_path(train_rows[0][0], train_dir, train_index)
        if first_path is None:
            raise RuntimeError(f"Missing first train feature for {train_rows[0][0]} in {train_dir}")
        first = load_pair_feature(first_path, config.num_pairs, args.feature_transform)
        line_edge_index = build_line_edge_index(config.num_pairs).to(
            torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        )
        print(
            f"#### First feature {first_stem}: edge_attr={first.shape}; "
            f"line graph directed edges={line_edge_index.size(1)}",
            flush=True,
        )
        print(
            f"#### Fitting train-only flat scaler ({args.scaler}); "
            f"feature_transform={args.feature_transform}; pooling={args.pooling}",
            flush=True,
        )
        scaler = fit_flat_scaler(train_rows, train_dir, train_index, config.num_pairs, args.scaler, args.feature_transform)
        train_data = PairEdgeDataset(train_rows, train_dir, train_index, scaler, config.num_pairs, args.feature_transform)
        test_data = PairEdgeDataset(test_rows, test_dir, test_index, scaler, config.num_pairs, args.feature_transform)

        rows = []
        for local_idx, (_, lr, h_dim, layer_setting, conv_type, epoch_setting) in enumerate(method_tasks):
            setting_idx = args.task_id if args.task_id is not None else local_idx
            row = run_setting(
                method,
                config,
                train_data,
                test_data,
                line_edge_index,
                first.shape[1],
                args,
                lr,
                h_dim,
                layer_setting,
                conv_type,
                epoch_setting,
                setting_idx,
            )
            rows.append(row)
            all_rows.append(row)
            method_csv = os.path.join(args.output_folder, f"{method}_grid_results.csv")
            with open(method_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda r: r["best_test_pcc"], reverse=True))
            print(f"#### Updated {method_csv}", flush=True)

        top = sorted(rows, key=lambda r: r["best_test_pcc"], reverse=True)[:10]
        print(f"#### Method {method} complete in {round((time.time() - start) / 60, 2)} minutes")
        print("#### Top settings:")
        for r in top:
            print(
                f"#### {method} lr={r['lr']:g} h_dim={r['h_dim']} epochs={r['epochs']} "
                f"layers={r['layers']} conv={r['conv_type']} best_epoch={r['best_epoch']} PCC={r['best_test_pcc']:.6f} "
                f"R2={r['best_test_r2_paper']:.6f} RMSE={r['best_test_rmse']:.6f}"
            )

    if all_rows:
        all_csv = os.path.join(args.output_folder, "all_grid_results.csv")
        with open(all_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sorted(all_rows, key=lambda r: r["best_test_pcc"], reverse=True))
        print(f"#### Saved combined results: {all_csv}")


if __name__ == "__main__":
    main()
