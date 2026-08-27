#!/usr/bin/env python3
"""CNN regression for LD50 topology features, mirroring plbind/ltest_cnn_final.py.

Kept consistent with the PLBind CNN logic:
  - topology feature arrays are converted to CNN input as (channels, filtration)
  - train-only StandardScaler is fit on flattened CNN inputs
  - DataLoader defaults match PLBind CNN: batch_size=32, shuffle train,
    num_workers=5, pin_memory=True
  - model is Conv1d init -> residual conv blocks -> AdaptiveAvgPool1d -> FC
  - MSELoss + AdamW(weight_decay=0.05) + OneCycleLR per batch
  - deterministic seeds and seed-ensemble prediction saving

The LD50-specific parts mirror toxicity_ann_ltest_final.py: CSV split loading,
CAS/date-like filename normalization, paper R2=PCC^2, sklearn R2, RMSE, and MAE.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


torch.set_default_dtype(torch.float32)

DEFAULT_ROOT = Path("data/ld50")
ALL_FEATURES = ["homology", "facet", "lap", "curvature", "forman"]
DEFAULT_SEEDS = [42]
DEFAULT_LRS = [1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4]


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
    with open(root / f"LD50_{split}.csv", "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
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
            "curvature": "EIC/single_direction",
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


def _feature_to_model_axes(arr, source_name="feature"):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr.transpose(1, 0, 2)
    if arr.ndim == 2:
        return arr[:, None, :]
    if arr.ndim == 1:
        return arr[:, None, None]
    raise ValueError(f"{source_name}: expected 1D, 2D, or 3D feature array, got shape {arr.shape}")


def _limit_filtrations(feature, max_filtrations=100):
    n_keep = min(max_filtrations, feature.shape[1])
    return feature[:, :n_keep, :]


def _feature_to_cnn_axes(feature):
    combination, n_filtrations, num_statis = feature.shape
    return feature.transpose(1, 0, 2).reshape(n_filtrations, combination * num_statis).T


def apply_feature_transform(feature: np.ndarray, feature_transform: str) -> np.ndarray:
    if feature_transform == "none":
        return feature
    if feature_transform == "signedlog":
        return (np.sign(feature) * np.log1p(np.abs(feature))).astype(np.float32)
    raise ValueError(f"Unsupported feature_transform={feature_transform}")


def load_one_feature(path: Path, n_keep: int | None, feature_transform: str = "none"):
    feature = _feature_to_model_axes(np.load(path), source_name=str(path)).astype(np.float32, copy=False)
    feature = _limit_filtrations(feature) if n_keep is None else feature[:, :n_keep, :]
    feature = apply_feature_transform(feature, feature_transform)
    return _feature_to_cnn_axes(feature)


def get_first_cnn_shape(root: Path, feature_specs: Sequence[Tuple[str, str]], split: str = "train") -> Tuple[int, int]:
    rows = read_split_csv(root, split)
    csv_name = rows[0][0]
    lengths = []
    channels = 0
    for feature, tag in feature_specs:
        fdir = feature_folder(root, feature, tag, split)
        stem, path = resolve_feature_path(csv_name, fdir, build_file_index(fdir))
        if path is None:
            raise RuntimeError(f"Missing first {feature}/{split} feature for {csv_name} in {fdir}")
        arr = _limit_filtrations(_feature_to_model_axes(np.load(path), source_name=str(path)))
        lengths.append(arr.shape[1])
        channels += arr.shape[0] * arr.shape[2]
        print(
            f"#### CNN feature {feature}: raw_model_axes={arr.shape} "
            f"from {path.name} tag={tag}",
            flush=True,
        )
    return channels, min(lengths)


def get_feature_label(root: Path, feature_specs: Sequence[Tuple[str, str]], feature_transform: str = "none"):
    channels, n_filtrations = get_first_cnn_shape(root, feature_specs)
    print(f"#### CNN combined input shape: channels={channels}, N={n_filtrations}", flush=True)

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

    scaler = StandardScaler()
    train_flat = train_fea.reshape(len(train_fea), -1)
    test_flat = test_fea.reshape(len(test_fea), -1)
    train_fea = scaler.fit_transform(train_flat).reshape(train_fea.shape).astype(np.float32)
    test_fea = scaler.transform(test_flat).reshape(test_fea.shape).astype(np.float32)

    return train_fea, train_label, test_fea, test_label, train_names, test_names


class NNDataset(Dataset):
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


def get_nn_train_test_loader(
    root: Path,
    batch_size: int,
    feature_specs: Sequence[Tuple[str, str]],
    num_workers: int,
    pin_memory: bool,
    feature_transform: str = "none",
):
    train_X, train_Y, test_X, test_Y, train_names, test_names = get_feature_label(root, feature_specs, feature_transform)
    train_data = NNDataset(train_X, train_Y)
    test_data = NNDataset(test_X, test_Y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    print(f"Train samples: {len(train_data)} | Test samples: {len(test_data)}", flush=True)
    print(f"CNN input: channels={train_X.shape[1]}, length={train_X.shape[2]}", flush=True)
    return train_loader, test_loader, train_names, test_names


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        p = kernel_size // 2
        if in_channels != out_channels:
            raise ValueError("PLBind residual CNN expects in_channels == out_channels inside residual blocks")
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=p, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool1d(2, ceil_mode=True)

    def forward(self, x):
        x = self.conv(x) + x
        x = self.pool(x)
        return x


class CNN1DModule(nn.Module):
    def __init__(self, in_channels, h_channels, num_layers, kernel_size, pool, d_out):
        super().__init__()
        p = kernel_size // 2
        self.init = nn.Conv1d(in_channels, h_channels, kernel_size=kernel_size, padding=p, bias=False)
        layers = [
            ResidualConvBlock(h_channels, h_channels, kernel_size=kernel_size)
            for _ in range(num_layers)
        ]
        layers.append(nn.AdaptiveAvgPool1d(pool))
        self.conv_block = nn.Sequential(*layers)
        flat_size = h_channels * pool
        self.fc_block = nn.Sequential(
            nn.Linear(flat_size, flat_size * 2, bias=False),
            nn.ReLU(),
            nn.Linear(flat_size * 2, d_out, bias=True),
        )

    def forward(self, x):
        x = self.init(x)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc_block(x)


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

        logits_ = logits.view(-1)
        yb_ = yb.view(-1)
        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)
        true_y.extend(yb_.detach().cpu().tolist())
        pred_y.extend(logits_.detach().cpu().tolist())
    pcc = safe_pcc(true_y, pred_y)
    mse = mean_squared_error(true_y, pred_y)
    return total_loss / total, pcc, pow(mse, 0.5), model


def eval_model(model, dl, criterion, device):
    model.eval()
    total_loss, total = 0, 0
    true_y, pred_y = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            logits_ = logits.view(-1)
            yb_ = yb.view(-1)
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
            true_y.extend(yb_.cpu().tolist())
            pred_y.extend(logits_.cpu().tolist())
    pcc = safe_pcc(true_y, pred_y)
    mse = mean_squared_error(true_y, pred_y)
    return total_loss / total, pcc, pow(mse, 0.5)


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
    pred = []
    for seed in seeds:
        pred.append(np.load(folder / f"{split_name}-seed-{seed}-pred.npy"))
    pred_mean = np.mean(np.asarray(pred), axis=0)
    metrics = regression_metrics(true_y, pred_mean)
    np.save(folder / f"{split_name}-ensemble-pred.npy", pred_mean)
    with open(folder / f"{split_name}-ensemble-metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


class Para:
    def __init__(self, args):
        self.lr = args.lr
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.h_channels = args.h_channels
        self.num_layers = args.num_layers
        self.cnn_kernel = args.kernel
        self.cnn_pool = args.pool
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_workers = args.num_workers
        self.pin_memory = not args.no_pin_memory
        self.print_every = args.print_every
        self.root = Path(args.root)
        self.feature_transform = args.feature_transform
        self.print_attrs()

    def print_attrs(self):
        print("--- Hyperparameters ---", flush=True)
        for k, v in self.__dict__.items():
            print(f"{k}: {v}", flush=True)
        print("-----------------------", flush=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prediction(para: Para, seed: int, prediction_folder: Path, feature_specs: Sequence[Tuple[str, str]]):
    print(f"\n  -> Running Seed {seed}...", flush=True)
    set_seed(seed)
    train_loader, test_loader, train_names, test_names = get_nn_train_test_loader(
        para.root,
        para.batch_size,
        feature_specs,
        para.num_workers,
        para.pin_memory,
        para.feature_transform,
    )

    in_channels = train_loader.dataset.features.shape[1]
    model = CNN1DModule(
        in_channels,
        para.h_channels,
        para.num_layers,
        para.cnn_kernel,
        para.cnn_pool,
        1,
    ).to(para.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=para.lr, weight_decay=para.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=para.lr, steps_per_epoch=len(train_loader), epochs=para.epoch, pct_start=0.3
    )

    last_train = last_test = None
    for e in range(para.epoch):
        train_loss, train_pcc, train_rmse, model = train_model(
            model, train_loader, criterion, optimizer, scheduler, para.device
        )
        test_loss, test_pcc, test_rmse = eval_model(model, test_loader, criterion, para.device)
        last_train = (train_loss, train_pcc, train_rmse)
        last_test = (test_loss, test_pcc, test_rmse)
        if e % para.print_every == 0 or e == para.epoch - 1:
            print(
                f"Epoch {e + 1:03d}/{para.epoch} | "
                f"Train [Loss: {train_loss:.3f}, PCC: {train_pcc:.3f}, R2: {train_pcc * train_pcc:.3f}, RMSE: {train_rmse:.3f}] | "
                f"Test [Loss: {test_loss:.3f}, PCC: {test_pcc:.3f}, R2: {test_pcc * test_pcc:.3f}, RMSE: {test_rmse:.3f}]",
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
    all_feature_specs = []
    for feature in args.features:
        tag = args.feature_root_tag or default_tag_for_feature(feature, args.preset)
        all_feature_specs.append((feature, tag))

    feature_configs = {}
    if args.combine_features:
        for r in range(1, len(all_feature_specs) + 1):
            for combo in itertools.combinations(all_feature_specs, r):
                feature_configs[combo_name(combo)] = list(combo)
    else:
        feature_configs[combo_name(all_feature_specs)] = all_feature_specs

    results_summary = []
    s = time.time()
    for feat_name, feature_specs in feature_configs.items():
        for lr in args.lrs:
            args.lr = lr
            para = Para(args)
            prediction_folder = Path(args.output_dir) / (
                f"{feat_name.replace('+', '_plus_')}_cnn_LD50_"
                f"{'+'.join(tag for _f, tag in feature_specs).replace('/', '-')}_"
                f"seeds{'-'.join(map(str, args.seeds))}_lr{format_lr(lr)}_"
                f"h{para.h_channels}_l{para.num_layers}_bs{para.batch_size}_"
                f"k{para.cnn_kernel}_pool{para.cnn_pool}_ep{para.epoch}_standard_{para.feature_transform}"
            )

            print(f"\n{'=' * 90}", flush=True)
            print(
                f" CNN TESTING | Features: {feat_name} | Specs: {feature_specs} | "
                f"LR: {lr} | h_channels: {para.h_channels} | layers: {para.num_layers}",
                flush=True,
            )
            print(f" Prediction folder: {prediction_folder}", flush=True)
            print(f"{'=' * 90}", flush=True)

            per_seed = []
            for seed in args.seeds:
                per_seed.append(prediction(para, seed, prediction_folder, feature_specs))

            test_metrics = get_metrics(args.seeds, prediction_folder, "test")
            train_metrics = get_metrics(args.seeds, prediction_folder, "train")
            row = {
                "features": feat_name,
                "feature_specs": feature_specs,
                "lr": lr,
                "h_channels": para.h_channels,
                "num_layers": para.num_layers,
                "kernel": para.cnn_kernel,
                "pool": para.cnn_pool,
                "batch_size": para.batch_size,
                "epoch": para.epoch,
                "feature_transform": para.feature_transform,
                "prediction_folder": str(prediction_folder),
                "train_ensemble": train_metrics,
                "test_ensemble": test_metrics,
                "per_seed": per_seed,
            }
            results_summary.append(row)
            with open(prediction_folder / "summary.json", "w") as f:
                json.dump(row, f, indent=2)

            print(
                f"--> Result: PCC: {test_metrics['pcc']:.6f} | "
                f"R2_paper: {test_metrics['r2_paper']:.6f} | "
                f"RMSE: {test_metrics['rmse']:.6f} | RMSE_x1.36: {test_metrics['rmse_x1.36']:.6f}",
                flush=True,
            )

    results_summary.sort(key=lambda x: x["test_ensemble"]["pcc"], reverse=True)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / f"cnn_grid_summary_{int(time.time())}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "features",
            "lr",
            "h_channels",
            "num_layers",
            "kernel",
            "pool",
            "batch_size",
            "epoch",
            "feature_transform",
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
                row["h_channels"],
                row["num_layers"],
                row["kernel"],
                row["pool"],
                row["batch_size"],
                row["epoch"],
                row["feature_transform"],
                tm["pcc"],
                tm["r2_paper"],
                tm["rmse"],
                tm["rmse_x1.36"],
                tm["mae"],
                trm["pcc"],
                row["prediction_folder"],
            ])

    print("\n" + "=" * 100, flush=True)
    print(" FINAL CNN GRID SEARCH RESULTS (LD50)", flush=True)
    print("=" * 100, flush=True)
    for i, row in enumerate(results_summary, start=1):
        tm = row["test_ensemble"]
        print(
            f"{i:02d}. {row['features']} lr={row['lr']} h={row['h_channels']} "
            f"PCC={tm['pcc']:.6f} R2={tm['r2_paper']:.6f} RMSE={tm['rmse']:.6f} "
            f"folder={row['prediction_folder']}",
            flush=True,
        )
    print(f"summary_csv={summary_path}", flush=True)
    print(f"Sweep Finished. Total Duration: {round((time.time() - s) / 60, 1)} minutes", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--features", nargs="+", default=["homology"], choices=ALL_FEATURES)
    ap.add_argument("--feature-root-tag", default=None, help="Override topology_features tag for all selected features.")
    ap.add_argument("--preset", default="compact", help="Feature preset: compact, step01_bond0, bond0, 045, or explicit tag.")
    ap.add_argument("--combine-features", action="store_true", help="Run all non-empty combinations of --features.")
    ap.add_argument("--lrs", type=parse_list_floats, default=DEFAULT_LRS)
    ap.add_argument("--seeds", type=parse_list_ints, default=DEFAULT_SEEDS)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--h-channels", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--no-pin-memory", action="store_true")
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--feature-transform", default="none", choices=["none", "signedlog"])
    ap.add_argument("--output-dir", default="results/ld50/cnn")
    return ap.parse_args()


if __name__ == "__main__":
    run_grid(parse_args())
