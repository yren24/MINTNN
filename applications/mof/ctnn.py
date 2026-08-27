
import argparse
import csv
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

COMMON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from copresheaf_model import Finetune  # noqa: E402
from ann import inverse_target, parse_topologies, set_seed, split_indices  # noqa: E402
from gbt import load_dataset  # noqa: E402


torch.set_default_dtype(torch.float32)

PROPERTY_DEFAULT = "O2uptakemolkg"
TOPOLOGY_CHOICES = ["homology", "lap", "facet", "forman"]

DEFAULT_FEATURE_ROOTS = {
    "homology": "data/mof/features/PH",
    "facet": "data/mof/features/CA",
    "forman": "data/mof/features/FPRC",
    "lap": "data/mof/features/PL",
}


class TopologyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32).reshape(-1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).to(torch.get_default_dtype())
        y = torch.tensor([self.labels[idx]], dtype=torch.get_default_dtype())
        return x, y


def make_loader(x, y, batch_size, shuffle):
    dataset = TopologyDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


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
    roots = parse_feature_root_map(args.feature_root_map)
    if args.feature_dir and args.feature_dir != "auto":
        root = args.feature_dir
    else:
        root = roots[topology]
    return os.path.join(root, topology, property_name, f"{mofid}.npy")


def topology_array_to_copresheaf(arr, topology):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        # Accept either (combination, filtration, stats) or plbind-style (filtration, combination, stats).
        if arr.shape[0] in (8, 9) and arr.shape[1] >= 10:
            return arr.astype(np.float32, copy=False)
        return arr.transpose(1, 0, 2).astype(np.float32, copy=False)

    flat = arr.reshape(-1).astype(np.float32, copy=False)
    if topology == "homology":
        # Stored as category x homology_dim x filtration.
        if flat.size % (9 * 3) != 0:
            raise ValueError(f"homology feature width {flat.size} is not divisible by 9*3")
        n_filtrations = flat.size // (9 * 3)
        return flat.reshape(9, 3, n_filtrations).transpose(0, 2, 1)

    if topology == "facet":
        # No-stat fil120 folder stores category x facet_curve x filtration, currently 9*5*121.
        if flat.size % (9 * 5) != 0:
            raise ValueError(f"facet feature width {flat.size} is not divisible by 9*5")
        n_filtrations = flat.size // (9 * 5)
        return flat.reshape(9, 5, n_filtrations).transpose(0, 2, 1)

    if topology == "forman":
        if flat.size % (9 * 120) == 0:
            return flat.reshape(9, 120, flat.size // (9 * 120))
        if flat.size % 9 == 0:
            per_cat = flat.size // 9
            raise ValueError(f"Cannot infer forman filtration/stat split from per-category width {per_cat}")
        raise ValueError(f"forman feature width {flat.size} is not divisible by 9")

    if topology == "curvature":
        if flat.size == 8 * 49 * 20:
            return flat.reshape(8, 49, 20)
        if flat.size == 8 * 49 * 10:
            return flat.reshape(8, 49, 10)
        if flat.size == 9 * 10 * 10:
            return flat.reshape(9, 10, 10)
        if flat.size % 8 == 0:
            per_cat = flat.size / 8
            raise ValueError(f"Cannot infer curvature tau/stat split from per-category width {per_cat}")
        raise ValueError(f"curvature feature width {flat.size} is not compatible with known curvature layouts")

    if topology == "lap":
        if flat.size == 9 * 120 * 10:
            return flat.reshape(9, 120, 10)
        if flat.size % (8 * 120) == 0:
            return flat.reshape(8, 120, flat.size // (8 * 120))
        if flat.size % 8 == 0:
            per_cat = flat.size // 8
            raise ValueError(f"Cannot infer lap filtration/stat split from per-category width {per_cat}")
        raise ValueError(f"lap feature width {flat.size} is not divisible by 8")

    raise ValueError(f"Unsupported topology: {topology}")


def pad_part(part, length, num_statis):
    c, l, s = part.shape
    if l > length or s > num_statis:
        raise ValueError(f"Cannot pad part shape {part.shape} to length={length}, num_statis={num_statis}")
    if l == length and s == num_statis:
        return part
    out = np.zeros((c, length, num_statis), dtype=np.float32)
    out[:, :l, :s] = part
    return out


def load_copresheaf_features(mofids, topologies, property_name, args):
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
            part = topology_array_to_copresheaf(np.load(path), topology)
            part_shapes.setdefault(topology, tuple(part.shape))
            if tuple(part.shape) != part_shapes[topology]:
                raise ValueError(
                    f"Inconsistent copresheaf feature shape for topology={topology}, mofid={mofid}: "
                    f"got {part.shape}, expected {part_shapes[topology]}"
                )
            parts.append(part)
        if missing_this:
            missing.append(mofid)
            continue

        max_len = max(part.shape[1] for part in parts)
        max_statis = max(part.shape[2] for part in parts)
        rows.append(np.concatenate([pad_part(part, max_len, max_statis) for part in parts], axis=0))
        kept.append(mofid)

    if not rows:
        raise RuntimeError(f"No complete copresheaf features found for topologies={topologies}, property={property_name}")
    return np.asarray(kept), np.stack(rows).astype(np.float32), missing, part_shapes


def load_copresheaf_data(args, topologies):
    mofids, y_all = load_dataset(args.property, args.data_dir)
    kept, x, missing, part_shapes = load_copresheaf_features(mofids, topologies, args.property, args)
    keep_mask = np.isin(mofids, kept)
    y = y_all[keep_mask].astype(np.float32)

    topology_label = "+".join(topologies)
    print(f"#### property={args.property} topologies={topology_label}", flush=True)
    print(f"#### labels={len(mofids)} copresheaf features={x.shape} missing={len(missing)} part_shapes={part_shapes}", flush=True)
    if missing:
        print("#### first missing: " + ",".join(missing[:10]), flush=True)
    return kept, x, y, missing, part_shapes


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
        kept, x, y, _, _ = load_copresheaf_data(args, topologies)
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

    c, l, s = x.shape[1], x.shape[2], x.shape[3]
    scaler = make_x_scaler(args)
    if args.csca_global_x_scaler:
        x_scaled = scaler.fit_transform(x.reshape(x.shape[0], -1)).astype(np.float32)
        x_scaled = x_scaled.reshape(x.shape[0], c, l, s)
        x_train = x_scaled[train_idx]
        x_val = x_scaled[val_idx]
        x_test = x_scaled[test_idx]
    else:
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_val_flat = x_val.reshape(x_val.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        x_train = scaler.fit_transform(x_train_flat).astype(np.float32).reshape(x_train.shape[0], c, l, s)
        x_val = scaler.transform(x_val_flat).astype(np.float32).reshape(x_val.shape[0], c, l, s)
        x_test = scaler.transform(x_test_flat).astype(np.float32).reshape(x_test.shape[0], c, l, s)

    y_train, y_val, y_test, y_scaler, target_bounds = transform_targets(y_train, y_val, y_test, args)
    return x_train, y_train, x_val, y_val, x_test, y_test, mof_test, y_scaler, target_bounds, split


def build_para(args, x_shape, device):
    combination, num_filtrations, num_statis = x_shape
    encoder_stalk_dim = args.encoder_h_dim // args.encoder_heads
    decoder_stalk_dim = args.decoder_h_dim // args.decoder_heads
    return SimpleNamespace(
        combination=int(combination),
        num_filtrations=int(num_filtrations),
        num_statis=int(num_statis),
        encoder_h_dim=int(args.encoder_h_dim),
        encoder_heads=int(args.encoder_heads),
        encoder_stalk_dim=int(encoder_stalk_dim),
        encoder_num_layers=int(args.encoder_num_layers),
        decoder_h_dim=int(args.decoder_h_dim),
        decoder_heads=int(args.decoder_heads),
        decoder_stalk_dim=int(decoder_stalk_dim),
        decoder_num_layers=int(args.decoder_num_layers),
        max_len=int(num_filtrations) + 1,
        low_rank=int(args.low_rank),
        encoder_dropout=float(args.encoder_dropout),
        decoder_dropout=float(args.decoder_dropout),
        mask_ratio=float(args.mask_ratio),
        mask_typ=args.mask_typ,
        norm_typ=args.norm_typ,
        patch_size=int(args.patch_size),
        weight_decay=float(args.weight_decay),
        device=device,
    )


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

    para = build_para(args, x_train.shape[1:], device)
    if para.num_filtrations % para.patch_size != 0:
        raise ValueError(f"num_filtrations={para.num_filtrations} must be divisible by patch_size={para.patch_size}")

    model = Finetune(para, model_path=args.model_path, use_pretrain=args.use_pretrain, freeze_encoder=args.freeze_encoder).to(device)
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
        f"#### CoPresheaf input={tuple(x_train.shape[1:])} lr={args.lr} epochs={args.epochs} "
        f"batch={args.batch_size} seed={args.seed} split={split['split_label']} "
        f"encoder_h={args.encoder_h_dim} encoder_layers={args.encoder_num_layers} heads={args.encoder_heads} "
        f"x_scaler={args.scaler} csca_global_x_scaler={args.csca_global_x_scaler} "
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
    stem = f"{args.property}_{output_name}_{split['split_label']}_modelseed{args.seed}_copresheaf"
    result_path = os.path.join(output_dir, stem + ".txt")
    pred_path = os.path.join(output_dir, stem + "_predictions.csv")

    if write_outputs:
        with open(result_path, "w") as f:
            f.write(f"[{args.property}] CoPresheaf topologies={output_name} Split {split['split_label']}, Model seed {args.seed}\n")
            f.write("Split protocol: CSCA-style random 80/10/10\n")
            f.write(f"Split index: {split['split_index']}\n")
            f.write(f"Data seed: {split['data_seed']}\n")
            f.write(f"Feature scaler: {args.scaler}\n")
            f.write(f"CSCA global X scaler: {args.csca_global_x_scaler}\n")
            f.write(f"Target transform: {args.target_transform}\n")
            f.write(f"Target scaler: {args.target_scaler}\n")
            f.write(f"Input shape: {tuple(x_train.shape[1:])}\n")
            f.write(f"Encoder h dim: {args.encoder_h_dim}\n")
            f.write(f"Encoder heads: {args.encoder_heads}\n")
            f.write(f"Encoder layers: {args.encoder_num_layers}\n")
            f.write(f"Low rank: {args.low_rank}\n")
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
                    "split": split['split_index'],
                    "data_seed": split['data_seed'],
                    "MOFRefcodes": mofid,
                    "true": yt,
                    "pred": yp,
                })

    print(
        f"[{args.property} {output_name} CoPresheaf] {split['split_label']}, Model seed {args.seed}: "
        f"R2={test_r2:.4f}, MAE={test_mae:.4e}, RMSE={test_rmse:.4e}",
        flush=True,
    )
    if write_outputs:
        print(f"#### saved {result_path}", flush=True)
        print(f"#### saved {pred_path}", flush=True)

    return {
        "split": split['split_index'],
        "data_seed": split['data_seed'],
        "split_label": split['split_label'],
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
    kept, x, y, _, _ = load_copresheaf_data(args, topologies)
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
                "split": result['split'],
                "data_seed": result['data_seed'],
                "model_seed": result['model_seed'],
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
                        "split": result['split'],
                        "data_seed": result['data_seed'],
                        "model_seed": result['model_seed'],
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
                    "split": first['split'],
                    "data_seed": first['data_seed'],
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

    split_csv = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_copresheaf_seed_metrics.csv")
    pred_csv = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_copresheaf_seed_predictions.csv")
    summary_txt = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_copresheaf_summary.txt")
    meta_json = os.path.join(output_dir, f"{args.property}_{output_name}_csca_style_copresheaf_meta.json")

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
        f.write(f"[{args.property}] CoPresheaf topologies={output_name} CSCA-style splits\n")
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
            "split_protocol": "csca_style_random_80_10_10",
            "splits_run": split_list,
            "data_seed_start": args.data_seed_start,
            "model_seeds": seed_list,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "encoder_h_dim": args.encoder_h_dim,
            "encoder_heads": args.encoder_heads,
            "encoder_num_layers": args.encoder_num_layers,
            "decoder_h_dim": args.decoder_h_dim,
            "decoder_heads": args.decoder_heads,
            "decoder_num_layers": args.decoder_num_layers,
            "low_rank": args.low_rank,
            "patch_size": args.patch_size,
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
    print(f"#### CSCA-style CoPresheaf concat R2={concat_r2:.6f} MAE={concat_mae:.6e} RMSE={concat_rmse:.6e}", flush=True)
    args.seed = original_seed


def main():
    parser = argparse.ArgumentParser(description="Run CoPresheaf transformer on MOF topology features.")
    parser.add_argument("--property", default=PROPERTY_DEFAULT)
    parser.add_argument("--topology", default="homology", choices=TOPOLOGY_CHOICES)
    parser.add_argument("--topologies", default=None, help="Comma-separated topology list to concatenate along the channel axis.")
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--data-seed-start", type=int, default=23)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-start", type=int, default=13)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--data-dir", default="data/mof/2STD")
    parser.add_argument("--feature-dir", default="auto", help="Set to a common root, or use auto for per-topology fil120 roots.")
    parser.add_argument("--feature-root-map", default=None, help="Optional comma list like homology=DIR,facet=DIR.")
    parser.add_argument("--output-dir", default="results/mof/copresheaf")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--pct-start", type=float, default=0.3)
    parser.add_argument("--encoder-h-dim", type=int, default=128)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--encoder-num-layers", type=int, default=3)
    parser.add_argument("--decoder-h-dim", type=int, default=512)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--decoder-num-layers", type=int, default=3)
    parser.add_argument("--low-rank", type=int, default=8)
    parser.add_argument("--encoder-dropout", type=float, default=0.1)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--mask-typ", choices=["random", "span"], default="span")
    parser.add_argument("--norm-typ", choices=["pre_norm", "post_norm"], default="post_norm")
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--model-path", default="", help="Optional pretrained copresheaf checkpoint path.")
    parser.add_argument("--use-pretrain", action="store_true", help="Load --model-path into the copresheaf backbone.")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--scaler", choices=["minmax", "standard"], default="minmax")
    parser.add_argument("--csca-global-x-scaler", dest="csca_global_x_scaler", action="store_true", default=True)
    parser.add_argument("--no-csca-global-x-scaler", dest="csca_global_x_scaler", action="store_false")
    parser.add_argument("--target-transform", choices=["none", "log10"], default="log10")
    parser.add_argument("--target-scaler", choices=["none", "standard", "minmax"], default="minmax")
    parser.add_argument("--save-seed-npy", action="store_true")
    parser.add_argument("--save-ensemble-npy", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Load features and print shapes without training.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.use_pretrain and not args.model_path:
        raise ValueError("--use-pretrain requires --model-path")

    run_many_csca_splits(args)


if __name__ == "__main__":
    main()
