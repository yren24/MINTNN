import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from ann import inverse_target, set_seed, split_indices
from gbt import load_dataset


torch.set_default_dtype(torch.float32)

PROPERTY_DEFAULT = "O2uptakemolkg"
TOPOLOGY_CHOICES = ["homology", "lap", "facet", "forman", "curvature"]

DEFAULT_FEATURE_ROOTS = {
    "homology": "data/mof/features/PH",
    "facet": "data/mof/features/CA",
    "forman": "data/mof/features/FPRC",
    "lap": "data/mof/features/PL",
    "curvature": "data/mof/features/EIC",
}


def parse_topologies(args):
    if args.topologies:
        topologies = [x.strip() for x in args.topologies.split(",") if x.strip()]
    else:
        topologies = [args.topology]
    bad = [x for x in topologies if x not in TOPOLOGY_CHOICES]
    if bad:
        raise ValueError(f"Unknown topologies: {bad}. Choices: {TOPOLOGY_CHOICES}")
    return topologies


def parse_feature_root_map(text):
    roots = dict(DEFAULT_FEATURE_ROOTS)
    if not text:
        return roots
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        roots[key.strip()] = value.strip()
    return roots


def get_feature_path(args, topology, property_name, mofid):
    if args.feature_dir and args.feature_dir != "auto":
        root = args.feature_dir
    else:
        root = parse_feature_root_map(args.feature_root_map)[topology]
    return os.path.join(root, topology, property_name, f"{mofid}.npy")


def topology_array_to_nodes(arr, topology):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        # Return node/category axis first, flatten the remaining feature axes.
        if arr.shape[0] in (8, 9):
            return arr.reshape(arr.shape[0], -1).astype(np.float32, copy=False)
        if arr.shape[1] in (8, 9):
            return arr.transpose(1, 0, 2).reshape(arr.shape[1], -1).astype(np.float32, copy=False)
        raise ValueError(f"{topology}: cannot infer category axis from shape {arr.shape}")

    flat = arr.reshape(-1).astype(np.float32, copy=False)
    if topology == "homology":
        if flat.size % 9 != 0:
            raise ValueError(f"homology feature width {flat.size} is not divisible by 9")
        return flat.reshape(9, -1)
    if topology == "facet":
        if flat.size % 9 != 0:
            raise ValueError(f"facet feature width {flat.size} is not divisible by 9")
        return flat.reshape(9, -1)
    if topology == "forman":
        if flat.size % 9 != 0:
            raise ValueError(f"forman feature width {flat.size} is not divisible by 9")
        return flat.reshape(9, -1)
    if topology == "curvature":
        if flat.size == 8 * 49 * 10:
            return flat.reshape(8, -1)
        if flat.size == 9 * 10 * 10:
            return flat.reshape(9, -1)
        if flat.size % 8 != 0:
            raise ValueError(f"curvature feature width {flat.size} is not divisible by 8")
        return flat.reshape(8, -1)

    if topology == "lap":
        if flat.size == 9 * 120 * 10:
            return flat.reshape(9, -1)
        if flat.size % 8 != 0:
            raise ValueError(f"lap feature width {flat.size} is not divisible by 8")
        return flat.reshape(8, -1)
    raise ValueError(f"Unsupported topology: {topology}")


def pad_node_features(nodes, width):
    if nodes.shape[1] == width:
        return nodes
    out = np.zeros((nodes.shape[0], width), dtype=np.float32)
    out[:, : nodes.shape[1]] = nodes
    return out


def load_snn_features(mofids, topologies, property_name, args):
    rows = []
    kept = []
    missing = []
    part_shapes = {}

    for mofid in mofids:
        parts = []
        missing_this = False
        for topology in topologies:
            path = get_feature_path(args, topology, property_name, mofid)
            if not os.path.exists(path):
                missing_this = True
                break
            nodes = topology_array_to_nodes(np.load(path), topology)
            part_shapes.setdefault(topology, tuple(nodes.shape))
            if tuple(nodes.shape) != part_shapes[topology]:
                raise ValueError(
                    f"Inconsistent SNN node feature shape for topology={topology}, mofid={mofid}: "
                    f"got {nodes.shape}, expected {part_shapes[topology]}"
                )
            parts.append(nodes)
        if missing_this:
            missing.append(mofid)
            continue
        max_width = max(part.shape[1] for part in parts)
        rows.append(np.concatenate([pad_node_features(part, max_width) for part in parts], axis=0))
        kept.append(mofid)

    if not rows:
        raise RuntimeError(f"No complete SNN features found for topologies={topologies}, property={property_name}")
    return np.asarray(kept), np.stack(rows).astype(np.float32), missing, part_shapes


def load_snn_data(args, topologies):
    mofids, y_all = load_dataset(args.property, args.data_dir)
    kept, x, missing, part_shapes = load_snn_features(mofids, topologies, args.property, args)
    keep_mask = np.isin(mofids, kept)
    y = y_all[keep_mask].astype(np.float32)

    print(f"#### property={args.property} topologies={'+'.join(topologies)}", flush=True)
    print(f"#### labels={len(mofids)} SNN node features={x.shape} missing={len(missing)} part_shapes={part_shapes}", flush=True)
    if missing:
        print("#### first missing: " + ",".join(missing[:10]), flush=True)
    return kept, x, y, missing, part_shapes


class NodeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.labels[idx])


def make_loader(x, y, batch_size, shuffle):
    dataset = NodeDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


class FullGraphNodeBlock(nn.Module):
    def __init__(self, h_dim, dropout):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(h_dim, h_dim, bias=False),
        )
        self.norm = nn.BatchNorm1d(h_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: (batch, nodes, h_dim). Complete graph without edge features.
        if x.size(1) <= 1:
            neighbor = torch.zeros_like(x)
        else:
            neighbor = (x.sum(dim=1, keepdim=True) - x) / float(x.size(1) - 1)
        update = self.message(torch.cat([x, neighbor], dim=-1))
        b, n, h = update.shape
        update = self.norm(update.reshape(b * n, h)).reshape(b, n, h)
        return x + self.dropout(update)


class CategoryNodeSNN(nn.Module):
    def __init__(self, in_dim, num_nodes, h_dim, layers, dropout, mlp_hidden=None):
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

    def forward(self, node_features):
        x = F.relu(self.node_init(node_features))
        for block in self.blocks:
            x = block(x)
        x_flat = x.reshape(x.size(0), -1)
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1).values
        x_sum = x.sum(dim=1)
        return self.head(torch.cat([x_flat, x_mean, x_max, x_sum], dim=1))


def make_x_scaler(args):
    if args.scaler == "minmax":
        return MinMaxScaler(feature_range=(-1, 1))
    if args.scaler == "standard":
        return StandardScaler()
    raise ValueError(f"Unknown scaler: {args.scaler}")


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


def prepare_fold(args, topologies, data=None, split_index=None):
    if data is None:
        kept, x, y, _, _ = load_snn_data(args, topologies)
    else:
        kept, x, y = data

    split = split_indices(args, x.shape[0], split_index)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    mof_test = kept[test_idx]

    nodes, width = x.shape[1], x.shape[2]
    scaler = make_x_scaler(args)
    if args.split_style == "csca" and args.csca_global_x_scaler:
        x_scaled = scaler.fit_transform(x.reshape(x.shape[0], -1)).astype(np.float32).reshape(x.shape[0], nodes, width)
        x_train, x_val, x_test = x_scaled[train_idx], x_scaled[val_idx], x_scaled[test_idx]
    else:
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_val_flat = x_val.reshape(x_val.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        x_train = scaler.fit_transform(x_train_flat).astype(np.float32).reshape(x_train.shape[0], nodes, width)
        x_val = scaler.transform(x_val_flat).astype(np.float32).reshape(x_val.shape[0], nodes, width)
        x_test = scaler.transform(x_test_flat).astype(np.float32).reshape(x_test.shape[0], nodes, width)

    y_train, y_val, y_test, y_scaler, target_bounds = transform_targets(y_train, y_val, y_test, args)
    return x_train, y_train, x_val, y_val, x_test, y_test, mof_test, y_scaler, target_bounds, split


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
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
            xb, yb = xb.to(device), yb.to(device)
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

    model = CategoryNodeSNN(
        in_dim=x_train.shape[2],
        num_nodes=x_train.shape[1],
        h_dim=args.h_dim,
        layers=args.layers,
        dropout=args.dropout,
        mlp_hidden=args.mlp_hidden,
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
        f"#### SNN input={tuple(x_train.shape[1:])} h_dim={args.h_dim} layers={args.layers} lr={args.lr} "
        f"epochs={args.epochs} batch={args.batch_size} seed={args.seed} split_style={args.split_style} "
        f"split={split['split_label']} x_scaler={args.scaler} csca_global_x_scaler={args.csca_global_x_scaler} "
        f"target_transform={args.target_transform} target_scaler={args.target_scaler} device={device}",
        flush=True,
    )

    best_val_r2 = -np.inf
    best_epoch = -1
    best_state = None
    final_metrics = None
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        _, train_r2, train_mae, train_rmse, _, _ = evaluate_original_scale(
            model, train_loader, criterion, device, y_scaler, args.target_transform, target_bounds
        )
        _, val_r2, val_mae, val_rmse, _, _ = evaluate_original_scale(
            model, val_loader, criterion, device, y_scaler, args.target_transform, target_bounds
        )
        _, test_epoch_r2, test_epoch_mae, test_epoch_rmse, _, _ = evaluate_original_scale(
            model, test_loader, criterion, device, y_scaler, args.target_transform, target_bounds
        )
        final_metrics = (test_epoch_r2, test_epoch_mae, test_epoch_rmse)
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

    final_test_r2, final_test_mae, final_test_rmse = final_metrics
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_r2, test_mae, test_rmse, true_y, pred_y = evaluate_original_scale(
        model, test_loader, criterion, device, y_scaler, args.target_transform, target_bounds
    )

    output_name = args.output_name or "+".join(topologies)
    output_dir = os.path.join(args.output_dir, output_name, args.property)
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{args.property}_{output_name}_{split['split_label']}_modelseed{args.seed}_snn"
    result_path = os.path.join(output_dir, stem + ".txt")
    pred_path = os.path.join(output_dir, stem + "_predictions.csv")

    if write_outputs:
        with open(result_path, "w") as f:
            f.write(f"[{args.property}] SNN topologies={output_name} Split {split['split_label']}, Model seed {args.seed}\n")
            f.write(f"Split style: {args.split_style}\n")
            f.write(f"Split index: {split['split_index']}\n")
            f.write(f"Data seed: {split['data_seed']}\n")
            f.write(f"Feature scaler: {args.scaler}\n")
            f.write(f"CSCA global X scaler: {args.csca_global_x_scaler}\n")
            f.write(f"Target transform: {args.target_transform}\n")
            f.write(f"Target scaler: {args.target_scaler}\n")
            f.write(f"Input shape: {tuple(x_train.shape[1:])}\n")
            f.write(f"H dim: {args.h_dim}\n")
            f.write(f"Layers: {args.layers}\n")
            f.write(f"Best val epoch: {best_epoch}\n")
            f.write(f"Best val R2: {best_val_r2:.6f}\n")
            f.write(f"Best-epoch Test R2: {test_r2:.6f}\n")
            f.write(f"Best-epoch Test MAE: {test_mae:.6e}\n")
            f.write(f"Best-epoch Test RMSE: {test_rmse:.6e}\n")
            f.write(f"Final-epoch Test R2: {final_test_r2:.6f}\n")
            f.write(f"Final-epoch Test MAE: {final_test_mae:.6e}\n")
            f.write(f"Final-epoch Test RMSE: {final_test_rmse:.6e}\n")

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
        f"[{args.property} {output_name} SNN] {split['split_label']}, Model seed {args.seed}: "
        f"R2={test_r2:.4f}, MAE={test_mae:.4e}, RMSE={test_rmse:.4e}",
        flush=True,
    )
    if write_outputs:
        print(f"#### saved {result_path}", flush=True)
        print(f"#### saved {pred_path}", flush=True)

    return {
        "split_style": args.split_style,
        "split": split["split_index"],
        "data_seed": split["data_seed"],
        "split_label": split["split_label"],
        "model_seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_R2": float(best_val_r2),
        "test_R2": float(test_r2),
        "test_MAE": float(test_mae),
        "test_RMSE": float(test_rmse),
        "final_test_R2": float(final_test_r2),
        "final_test_MAE": float(final_test_mae),
        "final_test_RMSE": float(final_test_rmse),
        "mof_test": mof_test,
        "true_y": true_y,
        "pred_y": pred_y,
        "result_path": result_path,
        "pred_path": pred_path,
    }


def run_many_csca_splits(args):
    topologies = parse_topologies(args)
    original_seed = args.seed
    kept, x, y, _, _ = load_snn_data(args, topologies)
    data = (kept, x, y)
    if args.dry_run:
        print(f"#### dry run loaded data={x.shape}; no training launched.", flush=True)
        return

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
                "final_test_R2": result["final_test_R2"],
                "final_test_MAE": result["final_test_MAE"],
                "final_test_RMSE": result["final_test_RMSE"],
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

    split_csv = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_snn_seed_metrics.csv")
    pred_csv = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_snn_seed_predictions.csv")
    summary_txt = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_snn_summary.txt")
    meta_json = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_snn_meta.json")

    with open(split_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "data_seed", "model_seed", "best_epoch", "best_val_R2",
                "test_R2", "test_MAE", "test_RMSE",
                "final_test_R2", "final_test_MAE", "final_test_RMSE", "test_size",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(pred_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "data_seed", "model_seed", "MOFRefcodes", "true", "pred"])
        writer.writeheader()
        writer.writerows(pred_rows)

    with open(summary_txt, "w") as f:
        f.write(f"[{args.property}] SNN topologies={output_name} CSCA-style splits\n")
        f.write("Splits run: " + ",".join(str(x) for x in split_list) + "\n")
        f.write(f"Data seed start: {args.data_seed_start}\n")
        f.write("Model seeds: " + ",".join(str(x) for x in seed_list) + "\n")
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
            "split_style": args.split_style,
            "splits_run": split_list,
            "data_seed_start": args.data_seed_start,
            "model_seeds": seed_list,
            "h_dim": args.h_dim,
            "layers": args.layers,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
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
    print(f"#### CSCA-style SNN concat R2={concat_r2:.6f} MAE={concat_mae:.6e} RMSE={concat_rmse:.6e}", flush=True)
    args.seed = original_seed


def main():
    parser = argparse.ArgumentParser(description="Run category-node SNN on MOF topology features.")
    parser.add_argument("--property", default=PROPERTY_DEFAULT)
    parser.add_argument("--topology", default="homology", choices=TOPOLOGY_CHOICES)
    parser.add_argument("--topologies", default=None, help="Comma-separated topology list; nodes are concatenated across methods.")
    parser.add_argument("--split-style", choices=["kfold", "csca"], default="kfold")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--data-seed-start", type=int, default=23)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-start", type=int, default=13)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--data-dir", default="data/mof/2STD")
    parser.add_argument("--feature-dir", default="auto")
    parser.add_argument("--feature-root-map", default=None)
    parser.add_argument("--output-dir", default="results/mof/snn")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--h-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-hidden", type=int, default=None)
    parser.add_argument("--pct-start", type=float, default=0.3)
    parser.add_argument("--scaler", choices=["minmax", "standard"], default="minmax")
    parser.add_argument("--csca-global-x-scaler", dest="csca_global_x_scaler", action="store_true", default=True)
    parser.add_argument("--no-csca-global-x-scaler", dest="csca_global_x_scaler", action="store_false")
    parser.add_argument("--target-transform", choices=["none", "log10"], default="log10")
    parser.add_argument("--target-scaler", choices=["none", "standard", "minmax"], default="minmax")
    parser.add_argument("--save-seed-npy", action="store_true")
    parser.add_argument("--save-ensemble-npy", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.split_style == "csca":
        run_many_csca_splits(args)
    else:
        if args.dry_run:
            topologies = parse_topologies(args)
            _, x, _, _, _ = load_snn_data(args, topologies)
            print(f"#### dry run loaded data={x.shape}; no training launched.", flush=True)
            return
        run_one(args)


if __name__ == "__main__":
    main()
