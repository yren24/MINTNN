import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from ann import (
    PROPERTY_DEFAULT,
    TOPOLOGY_CHOICES,
    inverse_target,
    parse_topologies,
    set_seed,
    split_indices,
)
from gbt import load_dataset


torch.set_default_dtype(torch.float32)


#### Known MOF feature layouts from mof_topology_features.py.
#### Each feature is converted to CNN input (channels, sequence_length).
KNOWN_LAYOUTS = {
    "homology": (9, 3, 120),     # category, Betti dimension, filtration
    "lap": (9, 120, 10),         # 9-category callZeroLap: category, filtration, feature channels
    "lap_8cat": (8, 120, 8),     # legacy/current Lap without Call: category, filtration, statistics
    "forman": (9, 120, 20),      # category, filtration, Forman-curvature channels
    "curvature": (9, 10, 10),    # legacy category, tau, statistics
    "curvature_internal": (8, 49, 10),  # C0-C7 internal PLBind-style curvature
    "curvature_internal_dual": (8, 49, 20),  # signed-log + clipped raw C0-C7 curvature
    "facet": (9, 605),           # category, 5*121 curves without summaries
}

DEFAULT_FEATURE_ROOTS = {
    "homology": "data/mof/features/PH",
    "facet": "data/mof/features/CA",
    "forman": "data/mof/features/FPRC",
    "lap": "data/mof/features/PL",
}


def parse_feature_root_map(text):
    roots = dict(DEFAULT_FEATURE_ROOTS)
    if not text:
        return roots
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --feature-root-map item: {item}")
        key, value = item.split("=", 1)
        roots[key.strip()] = value.strip()
    return roots


def get_feature_path(feature_dir, feature_root_map, topology, property_name, mofid):
    if feature_dir and feature_dir != "auto":
        root = feature_dir
    else:
        root = parse_feature_root_map(feature_root_map)[topology]
    return os.path.join(root, topology, property_name, f"{mofid}.npy")


def make_x_scaler(args):
    if args.scaler == "minmax":
        return MinMaxScaler(feature_range=(-1, 1))
    if args.scaler == "standard":
        return StandardScaler()
    raise ValueError(f"Unknown scaler: {args.scaler}")


def topology_vector_to_cnn(arr, topology, feature_layout="auto"):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if feature_layout == "flat":
        return arr.reshape(1, -1)

    if topology == "homology" and arr.size == np.prod(KNOWN_LAYOUTS["homology"]):
        return arr.reshape(9, 3, 120).reshape(27, 120)

    if topology == "lap" and arr.size == np.prod(KNOWN_LAYOUTS["lap"]):
        # 9-category Lap+homology dim1/dim2 Call-zero-Lap layout: category x filtration x feature.
        return arr.reshape(9, 120, 10).transpose(1, 0, 2).reshape(120, -1).T

    if topology == "lap" and arr.size == np.prod(KNOWN_LAYOUTS["lap_8cat"]):
        return arr.reshape(8, 120, 8).transpose(1, 0, 2).reshape(120, -1).T

    if topology == "forman" and arr.size == np.prod(KNOWN_LAYOUTS["forman"]):
        return arr.reshape(9, 120, 20).transpose(1, 0, 2).reshape(120, -1).T

    if topology == "curvature" and arr.size == np.prod(KNOWN_LAYOUTS["curvature_internal_dual"]):
        return arr.reshape(8, 49, 20).transpose(1, 0, 2).reshape(49, -1).T

    if topology == "curvature" and arr.size == np.prod(KNOWN_LAYOUTS["curvature_internal"]):
        return arr.reshape(8, 49, 10).transpose(1, 0, 2).reshape(49, -1).T

    if topology == "curvature" and arr.size == np.prod(KNOWN_LAYOUTS["curvature"]):
        return arr.reshape(9, 10, 10).transpose(1, 0, 2).reshape(10, -1).T

    if topology == "facet" and arr.size == np.prod(KNOWN_LAYOUTS["facet"]):
        #### Facet nostats input already excludes interval summaries and atom count.
        return arr.reshape(9, 5, 121).reshape(45, 121)

    #### Fallback keeps the script usable for future feature variants.
    return arr.reshape(1, -1)


def pad_and_concat_feature_parts(parts):
    max_len = max(part.shape[1] for part in parts)
    padded = []
    for part in parts:
        if part.shape[1] < max_len:
            part = np.pad(part, ((0, 0), (0, max_len - part.shape[1])), mode="constant")
        padded.append(part)
    return np.concatenate(padded, axis=0).astype(np.float32, copy=False)


def load_cnn_features(mofids, topologies, property_name, feature_dir, feature_layout="auto", feature_root_map=None):
    rows = []
    kept = []
    missing = []
    part_shapes = {}

    for mofid in mofids:
        parts = []
        missing_this = False
        for topology in topologies:
            path = get_feature_path(feature_dir, feature_root_map, topology, property_name, mofid)
            if not os.path.exists(path):
                missing_this = True
                break
            part = topology_vector_to_cnn(np.load(path), topology, feature_layout=feature_layout)
            part_shapes.setdefault(topology, part.shape)
            if part.shape != part_shapes[topology]:
                raise ValueError(
                    f"Inconsistent CNN feature shape for topology={topology}, mofid={mofid}: "
                    f"got {part.shape}, expected {part_shapes[topology]}"
                )
            parts.append(part)
        if missing_this:
            missing.append(mofid)
            continue
        rows.append(pad_and_concat_feature_parts(parts))
        kept.append(mofid)

    if not rows:
        raise RuntimeError(f"No complete CNN features found for topologies={topologies}, property={property_name}")
    return np.asarray(kept), np.stack(rows).astype(np.float32), missing, part_shapes


class CNNDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.labels[idx])


def make_loader(x, y, batch_size, shuffle):
    dataset = CNNDataset(x, y)
    #### Keep pin_memory disabled, matching the MOF ANN CPU-pressure fix.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.pool = nn.MaxPool1d(2, ceil_mode=True)

    def forward(self, x):
        x = x + self.block(x)
        return self.pool(x)


class CNNRegressor(nn.Module):
    def __init__(self, in_channels, h_channels, num_layers, kernel_size, pool_size, fc_hidden, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        #### Match plbind/ltest_cnn3.py: initial layer is only Conv1d.
        self.init = nn.Conv1d(in_channels, h_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.blocks = nn.Sequential(
            *[ResidualConvBlock(h_channels, kernel_size=kernel_size, dropout=dropout) for _ in range(num_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        flat_size = h_channels * pool_size
        #### Match plbind/ltest_cnn3.py default FC width; --fc-hidden can still override it.
        fc_hidden = flat_size * 2 if fc_hidden is None else fc_hidden
        self.fc = nn.Sequential(
            nn.Linear(flat_size, fc_hidden, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fc_hidden, 1, bias=True),
        )

    def forward(self, x):
        x = self.init(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


def transform_targets(y_train, y_val, y_test, args):
    if args.target_transform == "log10":
        if np.any(y_train <= 0) or np.any(y_val <= 0) or np.any(y_test <= 0):
            raise ValueError("target_transform=log10 requires positive labels.")
        y_train_model = np.log10(y_train).astype(np.float32)
        y_val_model = np.log10(y_val).astype(np.float32)
        y_test_model = np.log10(y_test).astype(np.float32)
    elif args.target_transform == "none":
        y_train_model = y_train.astype(np.float32)
        y_val_model = y_val.astype(np.float32)
        y_test_model = y_test.astype(np.float32)
    else:
        raise ValueError(f"Unknown target transform: {args.target_transform}")

    if args.target_scaler == "standard":
        y_scaler = StandardScaler()
    elif args.target_scaler == "minmax":
        y_scaler = MinMaxScaler(feature_range=(-1, 1))
    elif args.target_scaler == "none":
        y_scaler = None
    else:
        raise ValueError(f"Unknown target scaler: {args.target_scaler}")

    target_bounds = (float(y_train_model.min()), float(y_train_model.max()))
    if y_scaler is None:
        return y_train_model, y_val_model, y_test_model, y_scaler, target_bounds

    y_train_scaled = y_scaler.fit_transform(y_train_model.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val_model.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_test_scaled = y_scaler.transform(y_test_model.reshape(-1, 1)).reshape(-1).astype(np.float32)
    return y_train_scaled, y_val_scaled, y_test_scaled, y_scaler, target_bounds


def load_cnn_data(args, topologies):
    mofids, y_all = load_dataset(args.property, args.data_dir)
    kept, x, missing, part_shapes = load_cnn_features(
        mofids,
        topologies,
        args.property,
        args.feature_dir,
        feature_layout=args.feature_layout,
        feature_root_map=args.feature_root_map,
    )
    keep_mask = np.isin(mofids, kept)
    y = y_all[keep_mask].astype(np.float32)

    print(f"#### property={args.property} topologies={'+'.join(topologies)}", flush=True)
    print(f"#### labels={len(mofids)} CNN features={x.shape} missing={len(missing)} part_shapes={part_shapes}", flush=True)
    if missing:
        print("#### first missing: " + ",".join(missing[:10]), flush=True)
    return kept, x, y, missing, part_shapes


def prepare_fold(args, topologies, data=None, split_index=None):
    if data is None:
        kept, x, y, _, _ = load_cnn_data(args, topologies)
    else:
        kept, x, y = data

    split = split_indices(args, x.shape[0], split_index)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    x_train = x[train_idx]
    x_val = x[val_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    mof_test = kept[test_idx]

    channels, length = x.shape[1], x.shape[2]
    scaler = make_x_scaler(args)
    if args.csca_global_x_scaler:
        x_scaled = scaler.fit_transform(x.reshape(x.shape[0], -1)).astype(np.float32)
        x_scaled = x_scaled.reshape(x.shape[0], channels, length)
        x_train = x_scaled[train_idx]
        x_val = x_scaled[val_idx]
        x_test = x_scaled[test_idx]
    else:
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_val_flat = x_val.reshape(x_val.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        x_train = scaler.fit_transform(x_train_flat).astype(np.float32).reshape(x_train.shape[0], channels, length)
        x_val = scaler.transform(x_val_flat).astype(np.float32).reshape(x_val.shape[0], channels, length)
        x_test = scaler.transform(x_test_flat).astype(np.float32).reshape(x_test.shape[0], channels, length)

    y_train, y_val, y_test, y_scaler, target_bounds = transform_targets(y_train, y_val, y_test, args)
    return x_train, y_train, x_val, y_val, x_test, y_test, mof_test, y_scaler, target_bounds, split


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)
    return total_loss / max(1, total)


def evaluate_scaled(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
            y_true.extend(yb.view(-1).cpu().numpy().tolist())
            y_pred.extend(pred.view(-1).cpu().numpy().tolist())
    return total_loss / max(1, total), np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)


def evaluate_original_scale(model, loader, criterion, device, y_scaler, target_transform, target_bounds=None):
    loss, true_scaled, pred_scaled = evaluate_scaled(model, loader, criterion, device)
    true_y = inverse_target(true_scaled, y_scaler, target_transform, target_bounds)
    pred_y = inverse_target(pred_scaled, y_scaler, target_transform, target_bounds)
    r2 = r2_score(true_y, pred_y)
    mae = mean_absolute_error(true_y, pred_y)
    rmse = mean_squared_error(true_y, pred_y) ** 0.5
    return loss, r2, mae, rmse, true_y, pred_y


def run_one(args, topologies=None, data=None, split_index=None, write_outputs=True):
    if topologies is None:
        topologies = parse_topologies(args)
    set_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    x_train, y_train, x_val, y_val, x_test, y_test, mof_test, y_scaler, target_bounds, split = prepare_fold(
        args, topologies, data=data, split_index=split_index
    )
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    model = CNNRegressor(
        in_channels=x_train.shape[1],
        h_channels=args.h_channels,
        num_layers=args.cnn_layers,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        fc_hidden=args.fc_hidden,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=args.pct_start,
    )

    print(
        f"#### CNN input={tuple(x_train.shape[1:])} h_channels={args.h_channels} layers={args.cnn_layers} "
        f"kernel={args.kernel_size} pool={args.pool_size} fc_hidden={args.fc_hidden} lr={args.lr} "
        f"epochs={args.epochs} batch={args.batch_size} seed={args.seed} "
        f"split={split['split_label']} x_scaler={args.scaler} "
        f"target_transform={args.target_transform} target_scaler={args.target_scaler} device={device}",
        flush=True,
    )

    best_val_r2 = -np.inf
    best_epoch = -1
    best_state = None
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        train_eval_loss, train_r2, train_mae, train_rmse, _, _ = evaluate_original_scale(
            model, train_loader, criterion, device, y_scaler, args.target_transform, target_bounds
        )
        val_loss, val_r2, val_mae, val_rmse, _, _ = evaluate_original_scale(
            model, val_loader, criterion, device, y_scaler, args.target_transform, target_bounds
        )
        test_epoch_loss, test_epoch_r2, test_epoch_mae, test_epoch_rmse, _, _ = evaluate_original_scale(
            model, test_loader, criterion, device, y_scaler, args.target_transform, target_bounds
        )
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if args.print_every <= 1 or (epoch + 1) % args.print_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} | "
                f"Train step_loss={train_loss:.4e} eval_R2={train_r2:.4f} MAE={train_mae:.4e} RMSE={train_rmse:.4e} | "
                f"Val R2={val_r2:.4f} MAE={val_mae:.4e} RMSE={val_rmse:.4e} | "
                f"Test R2={test_epoch_r2:.4f} MAE={test_epoch_mae:.4e} RMSE={test_epoch_rmse:.4e}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_r2, test_mae, test_rmse, true_y, pred_y = evaluate_original_scale(
        model, test_loader, criterion, device, y_scaler, args.target_transform, target_bounds
    )

    output_name = args.output_name or "+".join(topologies)
    output_dir = os.path.join(args.output_dir, output_name, args.property)
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{args.property}_{output_name}_{split['split_label']}_modelseed{args.seed}_cnn"
    result_path = os.path.join(output_dir, stem + ".txt")
    pred_path = os.path.join(output_dir, stem + "_predictions.csv")

    if write_outputs:
        with open(result_path, "w") as f:
            f.write(f"[{args.property}] CNN topologies={output_name} Split {split['split_label']}, Model seed {args.seed}\n")
            f.write("Split protocol: CSCA-style random 80/10/10\n")
            f.write(f"Split index: {split['split_index']}\n")
            f.write(f"Data seed: {split['data_seed']}\n")
            f.write(f"Feature scaler: {args.scaler}\n")
            f.write(f"CSCA global X scaler: {args.csca_global_x_scaler}\n")
            f.write(f"Target transform: {args.target_transform}\n")
            f.write(f"Target scaler: {args.target_scaler}\n")
            f.write(f"Best val epoch: {best_epoch}\n")
            f.write(f"Best val R2: {best_val_r2:.6f}\n")
            f.write(f"Test R2: {test_r2:.6f}\n")
            f.write(f"Test MAE: {test_mae:.6e}\n")
            f.write(f"Test RMSE: {test_rmse:.6e}\n")

        with open(pred_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "data_seed", "MOFRefcodes", "true", "pred"])
            writer.writeheader()
            for mofid, yt, yp in zip(mof_test, true_y, pred_y):
                writer.writerow({
                    "split": split["split_index"],
                    "data_seed": split["data_seed"],
                    "MOFRefcodes": mofid,
                    "true": yt,
                    "pred": yp,
                })

    print(
        f"[{args.property} {output_name} CNN] {split['split_label']}, Model seed {args.seed}: "
        f"R2={test_r2:.4f}, MAE={test_mae:.4e}, RMSE={test_rmse:.4e}",
        flush=True,
    )
    if write_outputs:
        print(f"#### saved {result_path}", flush=True)
        print(f"#### saved {pred_path}", flush=True)

    return {
        "split": split["split_index"],
        "data_seed": split["data_seed"],
        "split_label": split["split_label"],
        "model_seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_R2": float(best_val_r2),
        "test_R2": float(test_r2),
        "test_MAE": float(test_mae),
        "test_RMSE": float(test_rmse),
        "mof_test": mof_test,
        "true_y": true_y,
        "pred_y": pred_y,
        "result_path": result_path,
        "pred_path": pred_path,
    }


def run_many_csca_splits(args):
    topologies = parse_topologies(args)
    original_seed = args.seed
    kept, x, y, _, _ = load_cnn_data(args, topologies)
    data = (kept, x, y)
    output_name = args.output_name or "+".join(topologies)
    output_dir = os.path.join(args.output_dir, output_name, args.property)
    os.makedirs(output_dir, exist_ok=True)
    npy_dir = os.path.join(output_dir, "npy_predictions")
    if args.save_seed_npy or args.save_ensemble_npy:
        os.makedirs(npy_dir, exist_ok=True)

    split_list = [args.split] if args.split is not None else list(range(args.n_splits))
    seed_list = [original_seed] if args.seed_count <= 1 else list(range(args.seed_start, args.seed_start + args.seed_count))
    rows = []
    pred_rows = []
    y_true_all = []
    y_pred_all = []
    id_all = []

    for split_index in split_list:
        seed_results = []
        for model_seed in seed_list:
            args.seed = model_seed
            result = run_one(
                args,
                topologies=topologies,
                data=data,
                split_index=split_index,
                write_outputs=(len(split_list) == 1 and len(seed_list) == 1),
            )
            seed_results.append(result)
            rows.append({
                "split": result["split"],
                "data_seed": result["data_seed"],
                "model_seed": result["model_seed"],
                "best_epoch": result["best_epoch"],
                "best_val_R2": result["best_val_R2"],
                "test_R2": result["test_R2"],
                "test_MAE": result["test_MAE"],
                "test_RMSE": result["test_RMSE"],
                "test_size": int(len(result["true_y"])),
            })
            if args.save_seed_npy:
                prefix = f"{args.property}_{output_name}_split{result['split']}_dataseed{result['data_seed']}_modelseed{result['model_seed']}"
                np.save(os.path.join(npy_dir, prefix + "_pred.npy"), result["pred_y"].astype(np.float32))
                np.save(os.path.join(npy_dir, prefix + "_true.npy"), result["true_y"].astype(np.float32))
                np.save(os.path.join(npy_dir, prefix + "_mofids.npy"), np.asarray(result["mof_test"]).astype(str))
            if not args.save_ensemble_npy:
                for mofid, yt, yp in zip(result["mof_test"], result["true_y"], result["pred_y"]):
                    pred_rows.append({
                        "split": result["split"],
                        "data_seed": result["data_seed"],
                        "model_seed": result["model_seed"],
                        "MOFRefcodes": mofid,
                        "true": float(yt),
                        "pred": float(yp),
                    })

        split_true = seed_results[0]["true_y"]
        split_mofids = np.asarray(seed_results[0]["mof_test"])
        split_pred_mean = np.mean(np.stack([r["pred_y"] for r in seed_results], axis=0), axis=0)
        y_true_all.append(split_true)
        y_pred_all.append(split_pred_mean)
        id_all.append(split_mofids)
        if args.save_ensemble_npy:
            first = seed_results[0]
            ensemble_label = f"ensemble{seed_list[0]}-{seed_list[-1]}"
            for mofid, yt, yp in zip(split_mofids, split_true, split_pred_mean):
                pred_rows.append({
                    "split": first["split"],
                    "data_seed": first["data_seed"],
                    "model_seed": ensemble_label,
                    "MOFRefcodes": mofid,
                    "true": float(yt),
                    "pred": float(yp),
                })
        if (args.save_seed_npy or args.save_ensemble_npy) and len(seed_results) > 1:
            first = seed_results[0]
            prefix = f"{args.property}_{output_name}_split{first['split']}_dataseed{first['data_seed']}_ensemble{seed_list[0]}-{seed_list[-1]}"
            np.save(os.path.join(npy_dir, prefix + "_pred_mean.npy"), split_pred_mean.astype(np.float32))
            np.save(os.path.join(npy_dir, prefix + "_true.npy"), split_true.astype(np.float32))
            np.save(os.path.join(npy_dir, prefix + "_mofids.npy"), split_mofids.astype(str))

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    id_all = np.concatenate(id_all)
    concat_r2 = float(r2_score(y_true_all, y_pred_all))
    concat_mae = float(mean_absolute_error(y_true_all, y_pred_all))
    concat_rmse = float(mean_squared_error(y_true_all, y_pred_all) ** 0.5)

    dedup = {}
    for mofid, yt, yp in zip(id_all, y_true_all, y_pred_all):
        rec = dedup.setdefault(str(mofid), {"true": float(yt), "pred": []})
        rec["pred"].append(float(yp))
    dedup_true = np.asarray([rec["true"] for rec in dedup.values()], dtype=float)
    dedup_pred = np.asarray([np.mean(rec["pred"]) for rec in dedup.values()], dtype=float)
    dedup_r2 = float(r2_score(dedup_true, dedup_pred))
    dedup_mae = float(mean_absolute_error(dedup_true, dedup_pred))
    dedup_rmse = float(mean_squared_error(dedup_true, dedup_pred) ** 0.5)

    split_csv = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_cnn_seed_metrics.csv")
    pred_csv = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_cnn_seed_predictions.csv")
    summary_txt = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_cnn_summary.txt")
    meta_json = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_cnn_meta.json")

    with open(split_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "data_seed", "model_seed", "best_epoch", "best_val_R2", "test_R2", "test_MAE", "test_RMSE", "test_size"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(pred_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "data_seed", "model_seed", "MOFRefcodes", "true", "pred"])
        writer.writeheader()
        writer.writerows(pred_rows)

    with open(summary_txt, "w") as f:
        f.write(f"[{args.property}] CNN topologies={output_name} CSCA-style splits\n")
        f.write(f"Splits run: {','.join(str(x) for x in split_list)}\n")
        f.write(f"Data seed start: {args.data_seed_start}\n")
        f.write(f"Model seeds: {','.join(str(x) for x in seed_list)}\n")
        f.write(f"Feature scaler: {args.scaler}\n")
        f.write(f"CSCA global X scaler: {args.csca_global_x_scaler}\n")
        f.write(f"Target transform: {args.target_transform}\n")
        f.write(f"Target scaler: {args.target_scaler}\n")
        f.write(f"Concat R2: {concat_r2:.8f}\n")
        f.write(f"Concat MAE: {concat_mae:.8e}\n")
        f.write(f"Concat RMSE: {concat_rmse:.8e}\n")
        f.write(f"Dedup R2: {dedup_r2:.8f}\n")
        f.write(f"Dedup MAE: {dedup_mae:.8e}\n")
        f.write(f"Dedup RMSE: {dedup_rmse:.8e}\n")

    with open(meta_json, "w") as f:
        json.dump({
            "property": args.property,
            "topologies": topologies,
            "output_name": output_name,
            "split_protocol": "csca_style_random_80_10_10",
            "splits_run": split_list,
            "data_seed_start": args.data_seed_start,
            "model_seeds": seed_list,
            "h_channels": args.h_channels,
            "cnn_layers": args.cnn_layers,
            "kernel_size": args.kernel_size,
            "pool_size": args.pool_size,
            "fc_hidden": args.fc_hidden,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "feature_layout": args.feature_layout,
            "feature_root_map": args.feature_root_map,
            "scaler": args.scaler,
            "csca_global_x_scaler": args.csca_global_x_scaler,
            "target_transform": args.target_transform,
            "target_scaler": args.target_scaler,
            "concat_R2": concat_r2,
            "concat_MAE": concat_mae,
            "concat_RMSE": concat_rmse,
            "dedup_R2": dedup_r2,
            "dedup_MAE": dedup_mae,
            "dedup_RMSE": dedup_rmse,
        }, f, indent=2)

    print(f"#### saved {summary_txt}", flush=True)
    print(f"#### saved {split_csv}", flush=True)
    print(f"#### saved {pred_csv}", flush=True)
    print(f"#### saved {meta_json}", flush=True)
    if args.save_seed_npy or args.save_ensemble_npy:
        print(f"#### saved npy predictions in {npy_dir}", flush=True)
    print(f"#### CSCA-style CNN concat R2={concat_r2:.6f} MAE={concat_mae:.6e} RMSE={concat_rmse:.6e}", flush=True)
    args.seed = original_seed


def main():
    parser = argparse.ArgumentParser(description="Run 1D CNN on MOF topology features.")
    parser.add_argument("--property", default=PROPERTY_DEFAULT)
    parser.add_argument("--topology", default="homology", choices=TOPOLOGY_CHOICES)
    parser.add_argument("--topologies", default=None, help="Comma-separated topology list to concatenate, e.g. homology,facet.")
    parser.add_argument("--split", type=int, default=None, help="0-based CSCA split index.")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--data-seed-start", type=int, default=23)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-start", type=int, default=13)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--data-dir", default="data/mof/2STD")
    parser.add_argument("--feature-dir", default="auto", help="Use auto for current fil120 roots, or set one common root containing topology/property/*.npy.")
    parser.add_argument("--feature-root-map", default=None, help="Optional comma list like homology=DIR,facet=DIR,forman=DIR,lap=DIR.")
    parser.add_argument("--output-dir", default="results/mof/cnn")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--feature-layout", choices=["auto", "flat"], default="auto")
    parser.add_argument("--h-channels", type=int, default=128)
    parser.add_argument("--cnn-layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--pool-size", type=int, default=1)
    parser.add_argument("--fc-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pct-start", type=float, default=0.3)
    parser.add_argument("--scaler", choices=["minmax", "standard"], default="minmax")
    parser.add_argument("--csca-global-x-scaler", dest="csca_global_x_scaler", action="store_true", default=True)
    parser.add_argument("--no-csca-global-x-scaler", dest="csca_global_x_scaler", action="store_false")
    parser.add_argument("--target-transform", choices=["none", "log10"], default="log10")
    parser.add_argument("--target-scaler", choices=["none", "standard", "minmax"], default="minmax")
    parser.add_argument("--save-seed-npy", action="store_true")
    parser.add_argument("--save-ensemble-npy", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_many_csca_splits(args)


if __name__ == "__main__":
    main()
