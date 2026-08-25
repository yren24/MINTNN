#!/usr/bin/env python3
"""ANN regression for LD50 topology features, mirroring plbind/ltest_ann_final.py.

Key PLBind logic kept intentionally:
  - concatenate flattened .npy feature folders
  - MinMaxScaler(feature_range=(-1, 1)) fit on train only
  - DataLoader(batch_size=32, shuffle train, num_workers=5, pin_memory=True)
  - MLP: Linear(bias=False) + BatchNorm1d + ReLU for input/hidden layers
  - MSELoss + AdamW(weight_decay=0.05) + OneCycleLR stepped every batch
  - deterministic seed setup matching ltest_ann_final.py
  - save per-seed test predictions as <split>-seed-<seed>-pred.npy and true labels once
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.stats as sp_stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

DEFAULT_ROOT = Path("data/ld50")
ALL_FEATURES = ["homology", "facet", "lap", "curvature", "forman"]
DEFAULT_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DEFAULT_LAYERS = [2048, 1024, 1024, 512, 512, 64]


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


def default_tag_for_feature(feature: str, bond: str) -> str:
    if bond == "compact":
        return {
            "homology": "PH",
            "facet": "CA",
            "lap": "PL",
            "forman": "FPRC",
            "curvature": "EIC_BI",
        }[feature]
    if bond == "0":
        return "maxfil10_bond0_facetdim1" if feature == "facet" else "maxfil10_bond0"
    if bond in {"045", "0.45"}:
        return "maxfil10_bond045_facetdim1" if feature == "facet" else "maxfil10_bond045"
    return bond


def feature_folder(root: Path, feature: str, tag: str) -> Path:
    return root / "topology_features" / tag / feature


def get_feature_label(root: Path, feature_specs: Sequence[Tuple[str, str]]):
    """PLBind get_feature_label equivalent for LD50 CSV splits."""
    split_data = {}
    for split in ["train", "test"]:
        rows = read_split_csv(root, split)
        labels = np.asarray([label for _, label in rows], dtype=np.float32)
        per_feature_indices = []
        for feature, tag in feature_specs:
            fdir = feature_folder(root, feature, tag) / split
            if not fdir.is_dir():
                raise FileNotFoundError(f"Missing feature directory: {fdir}")
            per_feature_indices.append((feature, tag, fdir, build_file_index(fdir)))

        features = []
        names = []
        for csv_name, _label in rows:
            combined_features = []
            resolved_name = None
            for feature, tag, fdir, index in per_feature_indices:
                stem, path = resolve_feature_path(csv_name, fdir, index)
                if path is None:
                    raise RuntimeError(f"Missing {feature}/{split} feature for {csv_name} in {fdir}")
                resolved_name = resolved_name or stem
                feat = np.load(path).astype(np.float32).flatten()
                combined_features.append(feat)
            features.append(np.concatenate(combined_features))
            names.append(resolved_name or csv_name)

        split_data[split] = (np.asarray(features, dtype=np.float32), labels, names)

    train_fea, train_label, train_names = split_data["train"]
    test_fea, test_label, test_names = split_data["test"]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_fea = scaler.fit_transform(train_fea).astype(np.float32)
    test_fea = scaler.transform(test_fea).astype(np.float32)

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


def get_nn_train_test_loader(root: Path, batch_size: int, feature_specs: Sequence[Tuple[str, str]], num_workers: int, pin_memory: bool):
    train_X, train_Y, test_X, test_Y, train_names, test_names = get_feature_label(root, feature_specs)

    train_data = NNDataset(train_X, train_Y)
    test_data = NNDataset(test_X, test_Y)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"Train samples: {len(train_data)} | Test samples: {len(test_data)}", flush=True)
    print(f"Total Concatenated Feature Dimension (D_in): {train_X.shape[1]}", flush=True)
    return train_loader, test_loader, train_names, test_names


class MultitaskModule(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MultitaskModule, self).__init__()

        self.input_layer = nn.Linear(D_in, H[0], bias=False)
        nn.init.xavier_uniform_(self.input_layer.weight)
        self.bn_input = nn.BatchNorm1d(H[0])

        self.hiden_layers = nn.ModuleList([
            nn.Linear(H[i], H[i + 1], bias=False) for i in range(len(H) - 1)
        ])
        for hiden_layer in self.hiden_layers:
            nn.init.xavier_uniform_(hiden_layer.weight)

        self.bn_hidden = nn.ModuleList([
            nn.BatchNorm1d(H[i + 1]) for i in range(len(H) - 1)
        ])

        self.output_layer = nn.Linear(H[-1], D_out, bias=True)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, X):
        X = self.input_layer(X)
        X = self.bn_input(X)
        X = F.relu(X)

        for i, hiden_layer in enumerate(self.hiden_layers):
            X = hiden_layer(X)
            X = self.bn_hidden[i](X)
            X = F.relu(X)

        y = self.output_layer(X)
        return y


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


class Para:
    def __init__(self, args):
        self.lr = args.lr
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.layers = args.layers
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_workers = args.num_workers
        self.pin_memory = not args.no_pin_memory
        self.print_every = args.print_every
        self.root = Path(args.root)
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
        para.root, para.batch_size, feature_specs, para.num_workers, para.pin_memory
    )

    D_in = train_loader.dataset.features.shape[1]
    model = MultitaskModule(D_in, para.layers, 1).to(para.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=para.lr, weight_decay=para.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=para.lr, steps_per_epoch=len(train_loader), epochs=para.epoch, pct_start=0.3
    )

    print(f"Start on LD50 using seed {seed}", flush=True)

    last_train = last_test = None
    for e in range(para.epoch):
        train_loss, train_pcc, train_rmse, model = train_model(
            model, train_loader, criterion, optimizer, scheduler, para.device
        )
        test_loss, test_pcc, test_rmse = eval_model(
            model, test_loader, criterion, para.device
        )
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


def get_metrics(seeds: Sequence[int], folder: Path, split_name: str = "test"):
    true_y = np.load(folder / f"{split_name}-true.npy")
    pred = []
    for seed in seeds:
        pred_y = np.load(folder / f"{split_name}-seed-{seed}-pred.npy")
        pred.append(pred_y)
    pred = np.asarray(pred)
    pred_mean = np.mean(pred, axis=0)
    metrics = regression_metrics(true_y, pred_mean)
    np.save(folder / f"{split_name}-ensemble-pred.npy", pred_mean)
    with open(folder / f"{split_name}-ensemble-metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def format_lr(lr: float) -> str:
    return f"{lr:.0e}" if lr < 1e-3 else f"{lr:g}"


def combo_name(feature_specs: Sequence[Tuple[str, str]]) -> str:
    return "+".join(feature for feature, _tag in feature_specs)


def run_grid(args):
    all_feature_specs = []
    for feature in args.features:
        tag = args.feature_root_tag or default_tag_for_feature(feature, args.bond)
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
                f"{feat_name.replace('+', '_plus_')}_ann_LD50_"
                f"{'+'.join(tag for _f, tag in feature_specs).replace('/', '-')}_"
                f"seeds{'-'.join(map(str, args.seeds))}_lr{format_lr(lr)}_"
                f"bs{para.batch_size}_h{'-'.join(map(str, para.layers))}_ep{para.epoch}_minmax"
            )

            print(f"\n{'=' * 80}", flush=True)
            print(f" TESTING | Features: {feat_name} | Specs: {feature_specs} | LR: {lr} | Layers: {para.layers}", flush=True)
            print(f" Prediction folder: {prediction_folder}", flush=True)
            print(f"{'=' * 80}", flush=True)

            per_seed = []
            for seed in args.seeds:
                per_seed.append(prediction(para, seed, prediction_folder, feature_specs))

            test_metrics = get_metrics(args.seeds, prediction_folder, "test")
            train_metrics = get_metrics(args.seeds, prediction_folder, "train")
            row = {
                "features": feat_name,
                "feature_specs": feature_specs,
                "layers": para.layers,
                "lr": lr,
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
    summary_path = out_root / f"ann_grid_summary_{int(time.time())}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "features", "lr", "layers", "test_pcc", "test_R2_paper", "test_RMSE", "test_RMSE_x1.36", "test_MAE", "train_pcc", "folder"])
        for i, row in enumerate(results_summary, start=1):
            tm = row["test_ensemble"]
            trm = row["train_ensemble"]
            writer.writerow([i, row["features"], row["lr"], row["layers"], tm["pcc"], tm["r2_paper"], tm["rmse"], tm["rmse_x1.36"], tm["mae"], trm["pcc"], row["prediction_folder"]])

    print("\n" + "=" * 100, flush=True)
    print(" FINAL GRID SEARCH RESULTS (LD50)", flush=True)
    print("=" * 100, flush=True)
    for i, row in enumerate(results_summary, start=1):
        tm = row["test_ensemble"]
        print(f"{i:02d}. {row['features']} lr={row['lr']} PCC={tm['pcc']:.6f} R2={tm['r2_paper']:.6f} RMSE={tm['rmse']:.6f} folder={row['prediction_folder']}", flush=True)
    print(f"summary_csv={summary_path}", flush=True)
    print(f"Sweep Finished. Total Duration: {round((time.time() - s) / 60, 1)} minutes", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--features", nargs="+", default=["homology"], choices=ALL_FEATURES)
    ap.add_argument("--feature-root-tag", default=None, help="Override topology_features tag for all selected features.")
    ap.add_argument("--bond", default="compact", help="Preset feature tag: compact, step01_bond0, 045, 0, or an explicit tag if --feature-root-tag is not set.")
    ap.add_argument("--combine-features", action="store_true", help="Run all non-empty combinations of --features, like PLBind combo grid.")
    ap.add_argument("--lrs", type=parse_list_floats, default=[1e-4])
    ap.add_argument("--seeds", type=parse_list_ints, default=DEFAULT_SEEDS)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--layers", type=parse_list_ints, default=DEFAULT_LAYERS)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--no-pin-memory", action="store_true")
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--output-dir", default="results/ld50/ann")
    return ap.parse_args()


if __name__ == "__main__":
    run_grid(parse_args())
