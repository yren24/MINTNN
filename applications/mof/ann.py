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
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from gbt import load_dataset


PROPERTY_DEFAULT = "O2uptakemolkg"
TOPOLOGY_CHOICES = ["homology", "lap", "facet", "forman", "curvature"]


def parse_int_list(text):
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_topologies(args):
    if args.topologies:
        topologies = [x.strip() for x in args.topologies.split(",") if x.strip()]
    else:
        topologies = [args.topology]
    bad = [x for x in topologies if x not in TOPOLOGY_CHOICES]
    if bad:
        raise ValueError(f"Unknown topologies: {bad}. Choices: {TOPOLOGY_CHOICES}")
    return topologies


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_concat_features(mofids, topologies, property_name, feature_dir):
    rows = []
    kept = []
    missing = []
    widths = {}

    for mofid in mofids:
        parts = []
        missing_this = False
        for topology in topologies:
            path = os.path.join(feature_dir, topology, property_name, f"{mofid}.npy")
            if not os.path.exists(path):
                missing_this = True
                break
            arr = np.load(path).reshape(-1)
            widths.setdefault(topology, arr.shape[0])
            if arr.shape[0] != widths[topology]:
                raise ValueError(
                    f"Inconsistent width for topology={topology}, mofid={mofid}: "
                    f"got {arr.shape[0]}, expected {widths[topology]}"
                )
            parts.append(arr)
        if missing_this:
            missing.append(mofid)
            continue
        rows.append(np.concatenate(parts))
        kept.append(mofid)

    if not rows:
        raise RuntimeError(f"No complete features found for topologies={topologies}, property={property_name}")
    return np.asarray(kept), np.asarray(rows, dtype=np.float32), missing, widths


class NNDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.from_numpy(self.labels[idx])
        return x, y


class ANNRegressor(nn.Module):
    def __init__(self, d_in, hidden_dims, d_out=1, dropout=0.0, output_activation="none"):
        super().__init__()
        self.input_layer = nn.Linear(d_in, hidden_dims[0], bias=False)
        nn.init.xavier_uniform_(self.input_layer.weight)
        self.bn_input = nn.BatchNorm1d(hidden_dims[0])

        layers = []
        bns = []
        for i in range(len(hidden_dims) - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False)
            nn.init.xavier_uniform_(layer.weight)
            layers.append(layer)
            bns.append(nn.BatchNorm1d(hidden_dims[i + 1]))
        self.hidden_layers = nn.ModuleList(layers)
        self.bn_hidden = nn.ModuleList(bns)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.output_layer = nn.Linear(hidden_dims[-1], d_out, bias=True)
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_activation = output_activation

    def forward(self, x):
        x = self.input_layer(x)
        x = self.bn_input(x)
        x = F.relu(x)
        x = self.dropout(x)

        for layer, bn in zip(self.hidden_layers, self.bn_hidden):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.output_layer(x)
        if self.output_activation == "tanh":
            x = torch.tanh(x)
        return x


def make_loader(x, y, batch_size, shuffle):
    dataset = NNDataset(x, y)
    #### Disable pinned memory for MOF ANN grids to reduce CPU / memory-transfer pressure.
    # return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []
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
        y_true.extend(yb.view(-1).detach().cpu().numpy().tolist())
        y_pred.extend(pred.view(-1).detach().cpu().numpy().tolist())

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return total_loss / total, r2, rmse, mae


def evaluate(model, loader, criterion, device):
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

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return total_loss / total, r2, rmse, mae, np.asarray(y_true), np.asarray(y_pred)


def load_ann_data(args, topologies):
    mofids, y_all = load_dataset(args.property, args.data_dir)
    kept, x, missing, widths = load_concat_features(mofids, topologies, args.property, args.feature_dir)
    keep_mask = np.isin(mofids, kept)
    y = y_all[keep_mask].astype(np.float32)

    print(f"#### property={args.property} topologies={'+'.join(topologies)}", flush=True)
    print(f"#### labels={len(mofids)} features={x.shape} missing={len(missing)} widths={widths}", flush=True)
    if missing:
        print("#### first missing: " + ",".join(missing[:10]), flush=True)

    return kept, x, y, missing, widths


def split_indices(args, n_samples, split_index=None):
    indices = np.arange(n_samples)
    if args.split_style == "kfold":
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.repeat)
        folds = list(kf.split(indices))
        fold = args.fold if split_index is None else split_index
        if fold < 0 or fold >= len(folds):
            raise ValueError(f"Fold index {fold} is out of range for {len(folds)} folds.")
        train_idx, valtest_idx = folds[fold]
        val_idx, test_idx = train_test_split(
            valtest_idx,
            test_size=0.5,
            random_state=args.repeat * 10 + fold,
            shuffle=True,
        )
        return {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "split_label": f"repeat{args.repeat}_fold{fold}",
            "split_index": fold,
            "data_seed": args.repeat * 10 + fold,
        }

    if args.split_style == "csca":
        split = 0 if split_index is None else split_index
        if split < 0 or split >= args.n_splits:
            raise ValueError(f"CSCA split index {split} is out of range for n_splits={args.n_splits}.")
        data_seed = args.data_seed_start + split
        train_idx, tmp_idx = train_test_split(
            indices,
            test_size=0.20,
            random_state=data_seed,
            shuffle=True,
        )
        val_idx, test_idx = train_test_split(
            tmp_idx,
            test_size=0.50,
            random_state=data_seed,
            shuffle=True,
        )
        return {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "split_label": f"csca_split{split}_dataseed{data_seed}",
            "split_index": split,
            "data_seed": data_seed,
        }

    raise ValueError(f"Unknown split_style: {args.split_style}")


def make_x_scaler(args):
    if args.scaler == "minmax":
        return MinMaxScaler(feature_range=(-1, 1))
    if args.scaler == "standard":
        return StandardScaler()
    raise ValueError(f"Unknown scaler: {args.scaler}")


def prepare_fold(args, topologies, data=None, split_index=None):
    if data is None:
        kept, x, y, _, _ = load_ann_data(args, topologies)
    else:
        kept, x, y = data

    split = split_indices(args, x.shape[0], split_index)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    mof_test = kept[test_idx]

    scaler = make_x_scaler(args)
    if args.split_style == "csca" and args.csca_global_x_scaler:
        x_scaled = scaler.fit_transform(x).astype(np.float32)
        x_train = x_scaled[train_idx]
        x_val = x_scaled[val_idx]
        x_test = x_scaled[test_idx]
    else:
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_val = scaler.transform(x_val).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)

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

    if y_scaler is not None:
        y_train_scaled = y_scaler.fit_transform(y_train_model.reshape(-1, 1)).reshape(-1).astype(np.float32)
        y_val_scaled = y_scaler.transform(y_val_model.reshape(-1, 1)).reshape(-1).astype(np.float32)
        y_test_scaled = y_scaler.transform(y_test_model.reshape(-1, 1)).reshape(-1).astype(np.float32)
    else:
        y_train_scaled = y_train_model.astype(np.float32)
        y_val_scaled = y_val_model.astype(np.float32)
        y_test_scaled = y_test_model.astype(np.float32)

    return x_train, y_train_scaled, x_val, y_val_scaled, x_test, y_test_scaled, mof_test, y_scaler, target_bounds, split


def inverse_target(values, y_scaler, target_transform, target_bounds=None):
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if y_scaler is None:
        unscaled = values.reshape(-1)
    else:
        unscaled = y_scaler.inverse_transform(values).reshape(-1)
    if target_bounds is not None:
        unscaled = np.clip(unscaled, target_bounds[0], target_bounds[1])
    if target_transform == "log10":
        return np.power(10.0, unscaled)
    return unscaled


def evaluate_original_scale(model, loader, criterion, device, y_scaler, target_transform, target_bounds=None):
    loss, scaled_r2, scaled_rmse, scaled_mae, true_scaled, pred_scaled = evaluate(model, loader, criterion, device)
    true_y = inverse_target(true_scaled, y_scaler, target_transform, target_bounds)
    pred_y = inverse_target(pred_scaled, y_scaler, target_transform, target_bounds)
    r2 = r2_score(true_y, pred_y)
    rmse = mean_squared_error(true_y, pred_y) ** 0.5
    mae = mean_absolute_error(true_y, pred_y)
    return loss, r2, rmse, mae, true_y, pred_y, scaled_r2, scaled_rmse, scaled_mae


def run_one(args, topologies=None, data=None, split_index=None, write_outputs=True):
    if topologies is None:
        topologies = parse_topologies(args)
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    x_train, y_train, x_val, y_val, x_test, y_test, mof_test, y_scaler, target_bounds, split = prepare_fold(
        args, topologies, data=data, split_index=split_index
    )
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    hidden_dims = parse_int_list(args.hidden_dims)
    model = ANNRegressor(
        x_train.shape[1],
        hidden_dims,
        dropout=args.dropout,
        output_activation=args.output_activation,
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
        f"#### ANN d_in={x_train.shape[1]} hidden={hidden_dims} lr={args.lr} "
        f"epochs={args.epochs} batch={args.batch_size} seed={args.seed} "
        f"split_style={args.split_style} split={split['split_label']} "
        f"x_scaler={args.scaler} target_transform={args.target_transform} "
        f"target_scaler={args.target_scaler} output_activation={args.output_activation} device={device}",
        flush=True,
    )

    best_val_r2 = -np.inf
    best_state = None
    best_epoch = -1
    for epoch in range(args.epochs):
        train_loss, train_r2, train_rmse, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        train_eval_loss, train_eval_r2, train_eval_rmse, train_eval_mae, _, _, _, _, _ = evaluate_original_scale(
            model, train_loader, criterion, device, y_scaler, args.target_transform, target_bounds
        )
        val_loss, val_r2, val_rmse, val_mae, _, _, _, _, _ = evaluate_original_scale(
            model, val_loader, criterion, device, y_scaler, args.target_transform, target_bounds
        )
        test_epoch_loss, test_epoch_r2, test_epoch_rmse, test_epoch_mae, _, _, _, _, _ = evaluate_original_scale(
            model, test_loader, criterion, device, y_scaler, args.target_transform, target_bounds
        )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"Train step_loss={train_loss:.4e} step_R2={train_r2:.4f} "
            f"eval_R2={train_eval_r2:.4f} MAE={train_eval_mae:.4e} RMSE={train_eval_rmse:.4e} | "
            f"Val R2={val_r2:.4f} MAE={val_mae:.4e} RMSE={val_rmse:.4e} | "
            f"Test R2={test_epoch_r2:.4f} MAE={test_epoch_mae:.4e} RMSE={test_epoch_rmse:.4e}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_r2, test_rmse, test_mae, true_y, pred_y, _, _, _ = evaluate_original_scale(
        model, test_loader, criterion, device, y_scaler, args.target_transform, target_bounds
    )

    output_name = args.output_name or "+".join(topologies)
    output_dir = os.path.join(args.output_dir, output_name, args.property)
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{args.property}_{output_name}_{split['split_label']}_modelseed{args.seed}_ann"
    result_path = os.path.join(output_dir, stem + ".txt")
    pred_path = os.path.join(output_dir, stem + "_predictions.csv")

    if write_outputs:
        with open(result_path, "w") as f:
            f.write(f"[{args.property}] ANN topologies={output_name} Split {split['split_label']}, Model seed {args.seed}\n")
            f.write(f"Split style: {args.split_style}\n")
            f.write(f"Split index: {split['split_index']}\n")
            f.write(f"Data seed: {split['data_seed']}\n")
            f.write(f"Feature scaler: {args.scaler}\n")
            f.write(f"CSCA global X scaler: {args.csca_global_x_scaler}\n")
            f.write(f"Target transform: {args.target_transform}\n")
            f.write(f"Target scaler: {args.target_scaler}\n")
            f.write(f"Output activation: {args.output_activation}\n")
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
        f"[{args.property} {output_name} ANN] {split['split_label']}, Model seed {args.seed}: "
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
        "mof_test": mof_test,
        "true_y": true_y,
        "pred_y": pred_y,
        "result_path": result_path,
        "pred_path": pred_path,
    }


def run_many_csca_splits(args):
    topologies = parse_topologies(args)
    original_seed = args.seed
    kept, x, y, _, _ = load_ann_data(args, topologies)
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
            #### For ensemble-only runs, avoid keeping per-seed prediction rows.
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
        #### Save only the split-level ensemble prediction when requested, without per-seed npy files.
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

    split_csv = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_ann_seed_metrics.csv")
    pred_csv = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_ann_seed_predictions.csv")
    summary_txt = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_ann_summary.txt")
    meta_json = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_ann_meta.json")

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
        f.write(f"[{args.property}] ANN topologies={output_name} CSCA-style splits\n")
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
            "split_style": args.split_style,
            "splits_run": split_list,
            "data_seed_start": args.data_seed_start,
            "model_seeds": seed_list,
            "hidden_dims": parse_int_list(args.hidden_dims),
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
    if args.save_seed_npy:
        print(f"#### saved seed npy predictions in {npy_dir}", flush=True)
    print(f"#### CSCA-style ANN concat R2={concat_r2:.6f} MAE={concat_mae:.6e} RMSE={concat_rmse:.6e}", flush=True)
    args.seed = original_seed


def main():
    parser = argparse.ArgumentParser(description="Run ANN on MOF topology features.")
    parser.add_argument("--property", default=PROPERTY_DEFAULT)
    parser.add_argument("--topology", default="homology", choices=TOPOLOGY_CHOICES)
    parser.add_argument("--topologies", default=None, help="Comma-separated topology list to concatenate, e.g. homology,facet.")
    parser.add_argument("--split-style", choices=["kfold", "csca"], default="kfold")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0, help="0-based fold index, matching uahpc-style GBT split.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split", type=int, default=None, help="0-based CSCA split index. For example, --split 5 uses data_seed=28 when data_seed_start=23.")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of CSCA-style random 80/10/10 splits.")
    parser.add_argument("--data-seed-start", type=int, default=23, help="CSCA-style data split seed start.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-start", type=int, default=13, help="First ANN model seed for CSCA-style seed ensemble.")
    parser.add_argument("--seed-count", type=int, default=1, help="Number of ANN model seeds to run for each CSCA split.")
    parser.add_argument("--data-dir", default="data/mof/2STD")
    parser.add_argument("--feature-dir", default="data/mof/features/PH")
    parser.add_argument("--output-dir", default="results/mof/ann")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--hidden-dims", default="2048,1024,1024,512,512,64")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--output-activation", choices=["none", "tanh"], default="none")
    parser.add_argument("--pct-start", type=float, default=0.3)
    parser.add_argument("--scaler", choices=["minmax", "standard"], default="minmax")
    parser.add_argument("--csca-global-x-scaler", dest="csca_global_x_scaler", action="store_true", default=True, help="For CSCA-style splits, fit the feature scaler on all X before splitting, matching train.csca.py.")
    parser.add_argument("--no-csca-global-x-scaler", dest="csca_global_x_scaler", action="store_false", help="Fit the feature scaler on train split only.")
    parser.add_argument("--target-transform", choices=["none", "log10"], default="log10")
    parser.add_argument("--target-scaler", choices=["none", "standard", "minmax"], default="minmax")
    parser.add_argument("--save-seed-npy", action="store_true", help="Save per-split per-model-seed pred/true/mofids npy files.")
    parser.add_argument("--save-ensemble-npy", action="store_true", help="Save only per-split ensemble mean pred/true/mofids npy files.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.split_style == "csca":
        run_many_csca_splits(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
