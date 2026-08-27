#!/usr/bin/env python3
"""Seed-ensemble runner for LD50 pair-edge SNN topology models.

This reuses toxicity_snn_grid_pair_edgegraph_methods_final.py for data loading,
feature transforms, graph construction, model definition, and metrics. It trains
one fixed final-epoch setting across multiple seeds, saves per-seed predictions,
and averages predictions for the final ensemble.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import snn_pair_graph as snn


def parse_list(text, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def predict(model, loader, line_edge_index, device):
    model.eval()
    true_y, pred_y, names = [], [], []
    with torch.no_grad():
        for edge_attr, label, batch_names in loader:
            edge_attr = edge_attr.to(device)
            pred = model(edge_attr, line_edge_index)
            true_y.extend(label.cpu().numpy().tolist())
            pred_y.extend(pred.detach().cpu().numpy().tolist())
            names.extend(batch_names)
    return np.asarray(true_y, dtype=np.float64), np.asarray(pred_y, dtype=np.float64), np.asarray(names, dtype=str)


def make_criterion(args):
    if args.loss == "mse":
        return nn.MSELoss()
    if args.loss == "smoothl1":
        return nn.SmoothL1Loss(beta=args.smoothl1_beta)
    if args.loss == "huber":
        return nn.HuberLoss(delta=args.smoothl1_beta)
    raise ValueError(f"Unsupported loss={args.loss}")


def run_seed(seed, args, config, train_data, test_data, line_edge_index, first_dim, out_dir):
    snn.set_seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = snn.make_loader(train_data, args.batch_size, True, args.num_workers, seed)
    train_eval_loader = snn.make_loader(train_data, args.batch_size, False, args.num_workers, seed)
    test_loader = snn.make_loader(test_data, args.batch_size, False, args.num_workers, seed)
    mlp_hidden = args.mlp_hidden if args.mlp_hidden is not None else max(256, args.h_dim * 2)

    model = snn.EdgeOnlyPairSNN(
        in_dim=first_dim,
        num_pairs=config.num_pairs,
        h_dim=args.h_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        mlp_hidden=mlp_hidden,
        conv_type=args.conv_type,
        pooling=args.pooling,
    ).to(device)
    criterion = make_criterion(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,
    )

    final_train = final_test = None
    for epoch in range(args.epochs):
        train_loss, train_metrics = snn.run_epoch(
            model, train_loader, line_edge_index, criterion, optimizer, scheduler, device, train=True
        )
        test_loss, test_metrics = snn.run_epoch(
            model, test_loader, line_edge_index, criterion, optimizer, scheduler, device, train=False
        )
        final_train = (train_loss, train_metrics)
        final_test = (test_loss, test_metrics)
        print(
            f"{args.method} seed={seed} Epoch {epoch + 1:03d}/{args.epochs} | "
            f"Train PCC {train_metrics['pcc']:.4f} R2 {train_metrics['r2_paper']:.4f} RMSE {train_metrics['rmse']:.4f} | "
            f"Test PCC {test_metrics['pcc']:.4f} R2 {test_metrics['r2_paper']:.4f} RMSE {test_metrics['rmse']:.4f}",
            flush=True,
        )

    train_true, train_pred, train_names = predict(model, train_eval_loader, line_edge_index, device)
    test_true, test_pred, test_names = predict(model, test_loader, line_edge_index, device)
    np.save(out_dir / f"train-seed-{seed}-true.npy", train_true)
    np.save(out_dir / f"train-seed-{seed}-pred.npy", train_pred)
    np.save(out_dir / f"train-seed-{seed}-names.npy", train_names)
    np.save(out_dir / f"test-seed-{seed}-true.npy", test_true)
    np.save(out_dir / f"test-seed-{seed}-pred.npy", test_pred)
    np.save(out_dir / f"test-seed-{seed}-names.npy", test_names)
    return {
        "seed": seed,
        "train": final_train[1],
        "test": final_test[1],
    }


def align_seed_split(split_name, seed, true, pred, names, true_ref, names_ref):
    names = np.asarray(names, dtype=str)
    if names_ref is None:
        if len(np.unique(names)) != len(names):
            raise RuntimeError(f"{split_name}: duplicate sample names for seed {seed}")
        return true, pred, names

    if len(np.unique(names)) != len(names):
        raise RuntimeError(f"{split_name}: duplicate sample names for seed {seed}")
    if len(names) != len(names_ref) or set(names.tolist()) != set(names_ref.tolist()):
        missing = sorted(set(names_ref.tolist()) - set(names.tolist()))
        extra = sorted(set(names.tolist()) - set(names_ref.tolist()))
        raise RuntimeError(
            f"{split_name}: sample-name set mismatch for seed {seed}; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    index = {name: idx for idx, name in enumerate(names.tolist())}
    order = np.asarray([index[name] for name in names_ref.tolist()], dtype=np.int64)
    true = true[order]
    pred = pred[order]
    if not np.allclose(true_ref, true):
        raise RuntimeError(f"{split_name}: target mismatch after name alignment for seed {seed}")
    return true, pred, names_ref


def save_ensemble(split_name, out_dir, seeds):
    true_ref = names_ref = None
    preds = []
    for seed in seeds:
        true = np.load(out_dir / f"{split_name}-seed-{seed}-true.npy")
        pred = np.load(out_dir / f"{split_name}-seed-{seed}-pred.npy")
        names = np.load(out_dir / f"{split_name}-seed-{seed}-names.npy", allow_pickle=False)
        if true_ref is None:
            true_ref, pred, names_ref = align_seed_split(split_name, seed, true, pred, names, None, None)
        else:
            _, pred, _ = align_seed_split(split_name, seed, true, pred, names, true_ref, names_ref)
        preds.append(pred)
    pred_mean = np.mean(np.stack(preds, axis=0), axis=0)
    metrics = snn.metrics_from_arrays(true_ref, pred_mean)
    np.save(out_dir / f"{split_name}-ensemble-true.npy", true_ref)
    np.save(out_dir / f"{split_name}-ensemble-pred.npy", pred_mean)
    np.save(out_dir / f"{split_name}-ensemble-names.npy", names_ref)
    with (out_dir / f"{split_name}-ensemble-metrics.json").open("w") as fp:
        json.dump(metrics, fp, indent=2, sort_keys=True)
    return metrics


def write_ensemble_summary(args, config, seeds, train_metrics, test_metrics, out_dir):
    summary = {
        "method": args.method,
        "feature_tag": config.tag,
        "seeds": seeds,
        "lr": args.lr,
        "h_dim": args.h_dim,
        "layers": args.layers,
        "epochs": args.epochs,
        "conv_type": args.conv_type,
        "loss": args.loss,
        "scaler": args.scaler,
        "feature_transform": args.feature_transform,
        "pooling": args.pooling,
        "train_ensemble": train_metrics,
        "test_ensemble": test_metrics,
    }
    with (out_dir / "ensemble_summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="Run final-epoch LD50 SNN seed ensemble for one topology.")
    parser.add_argument("--root", type=str, default=str(snn.DEFAULT_ROOT))
    parser.add_argument("--preset", type=str, default="compact")
    parser.add_argument("--feature-root-tag", type=str, default=None)
    parser.add_argument("--method", type=str, required=True, choices=sorted(snn.ALL_FEATURES))
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--h-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--conv-type", type=str, default="gatv2", choices=["gatv2", "gcn"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smoothl1", "huber"])
    parser.add_argument("--smoothl1-beta", type=float, default=1.0)
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"])
    parser.add_argument("--feature-transform", type=str, default="none", choices=["none", "signedlog"])
    parser.add_argument("--pooling", type=str, default="flat_mean_max_sum", choices=["flat", "flat_mean_max_sum"])
    parser.add_argument("--mlp-hidden", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="42,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-folder", type=str, default="results/ld50/snn")
    parser.add_argument(
        "--ensemble-only",
        action="store_true",
        help="Skip training and rebuild ensemble metrics from saved per-seed prediction files.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output_folder)
    out_dir = output_root / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_list(args.seeds, int)

    methods = snn.build_methods(args.preset, args.feature_root_tag)
    config = methods[args.method]

    if args.ensemble_only:
        train_metrics = save_ensemble("train", out_dir, seeds)
        test_metrics = save_ensemble("test", out_dir, seeds)
        write_ensemble_summary(args, config, seeds, train_metrics, test_metrics, out_dir)
        print(
            f"#### ENSEMBLE {args.method} test PCC={test_metrics['pcc']:.6f} "
            f"R2={test_metrics['r2_paper']:.6f} RMSE={test_metrics['rmse']:.6f} MAE={test_metrics['mae']:.6f}",
            flush=True,
        )
        return

    train_rows = snn.read_split_csv(root, "train")
    test_rows = snn.read_split_csv(root, "test")
    train_dir = snn.feature_folder(root, args.method, config.tag, "train")
    test_dir = snn.feature_folder(root, args.method, config.tag, "test")
    train_index = snn.build_file_index(train_dir)
    test_index = snn.build_file_index(test_dir)
    first_stem, first_path = snn.resolve_feature_path(train_rows[0][0], train_dir, train_index)
    if first_path is None:
        raise RuntimeError(f"Missing first train feature for {train_rows[0][0]} in {train_dir}")
    first = snn.load_pair_feature(first_path, config.num_pairs, args.feature_transform)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    line_edge_index = snn.build_line_edge_index(config.num_pairs).to(device)

    print(
        f"#### SNN ensemble method={args.method} tag={config.tag} first={first_stem} "
        f"edge_attr={first.shape} line_edges={line_edge_index.size(1)} seeds={seeds}",
        flush=True,
    )
    print(
        f"#### lr={args.lr:g} h_dim={args.h_dim} layers={args.layers} epochs={args.epochs} "
        f"conv={args.conv_type} scaler={args.scaler} transform={args.feature_transform} pooling={args.pooling}",
        flush=True,
    )
    scaler = snn.fit_flat_scaler(train_rows, train_dir, train_index, config.num_pairs, args.scaler, args.feature_transform)
    train_data = snn.PairEdgeDataset(train_rows, train_dir, train_index, scaler, config.num_pairs, args.feature_transform)
    test_data = snn.PairEdgeDataset(test_rows, test_dir, test_index, scaler, config.num_pairs, args.feature_transform)

    seed_rows = []
    for seed in seeds:
        metrics = run_seed(seed, args, config, train_data, test_data, line_edge_index, first.shape[1], out_dir)
        row = {
            "method": args.method,
            "seed": seed,
            "lr": args.lr,
            "h_dim": args.h_dim,
            "layers": args.layers,
            "epochs": args.epochs,
            "conv_type": args.conv_type,
            "loss": args.loss,
            "scaler": args.scaler,
            "feature_transform": args.feature_transform,
            "pooling": args.pooling,
        }
        for split_name in ("train", "test"):
            for key, value in metrics[split_name].items():
                row[f"{split_name}_{key}"] = value
        seed_rows.append(row)
        with (out_dir / "seed_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(seed_rows)

    train_metrics = save_ensemble("train", out_dir, seeds)
    test_metrics = save_ensemble("test", out_dir, seeds)
    write_ensemble_summary(args, config, seeds, train_metrics, test_metrics, out_dir)
    print(
        f"#### ENSEMBLE {args.method} test PCC={test_metrics['pcc']:.6f} "
        f"R2={test_metrics['r2_paper']:.6f} RMSE={test_metrics['rmse']:.6f} MAE={test_metrics['mae']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
