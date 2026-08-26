#!/usr/bin/env python3
"""Build consensus predictions from trained MINTNN component outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_COLUMNS = {"component", "pred_path", "true_path", "names_path"}


def np_module():
    import numpy as np

    return np


def resolve_path(path_text: str, manifest_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (manifest_dir / path).resolve()


def load_array(path: Path) -> np.ndarray:
    np = np_module()
    if path.suffix != ".npy":
        raise ValueError(f"Only .npy arrays are supported: {path}")
    return np.load(path, allow_pickle=False)


def deduplicate(names: np.ndarray, true: np.ndarray, pred: np.ndarray) -> tuple[list[str], dict[str, tuple[float, float]]]:
    np = np_module()
    values: dict[str, dict[str, list[float] | float]] = {}
    order: list[str] = []
    for raw_name, raw_true, raw_pred in zip(names, true, pred):
        name = str(raw_name)
        if name not in values:
            values[name] = {"true": float(raw_true), "preds": []}
            order.append(name)
        elif not np.isclose(float(values[name]["true"]), float(raw_true), rtol=1e-6, atol=1e-6):
            raise ValueError(f"Conflicting true values for sample {name}")
        values[name]["preds"].append(float(raw_pred))

    collapsed = {
        name: (float(values[name]["true"]), float(np.mean(values[name]["preds"])))
        for name in order
    }
    return order, collapsed


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return rows


def load_component(row: dict[str, str], manifest_dir: Path) -> dict:
    pred = load_array(resolve_path(row["pred_path"], manifest_dir)).reshape(-1)
    true = load_array(resolve_path(row["true_path"], manifest_dir)).reshape(-1)
    names = load_array(resolve_path(row["names_path"], manifest_dir)).reshape(-1)
    if not (len(pred) == len(true) == len(names)):
        raise ValueError(
            f"Length mismatch for component {row['component']}: "
            f"pred={len(pred)} true={len(true)} names={len(names)}"
        )
    order, values = deduplicate(names, true, pred)
    weight = float(row.get("weight") or 1.0)
    if weight <= 0:
        raise ValueError(f"Component weight must be positive for {row['component']}")
    return {
        "name": row["component"],
        "weight": weight,
        "order": order,
        "values": values,
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    np = np_module()
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if denom == 0 else float(1.0 - np.sum(err * err) / denom)
    pcc = float("nan")
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "n": int(len(y_true)),
        "pcc": pcc,
        "r2": r2,
        "r2_paper": float("nan") if np.isnan(pcc) else pcc * pcc,
        "mae": mae,
        "rmse": rmse,
        "rmse_x1.36": rmse * 1.36,
    }


def build_consensus(components: list[dict]) -> tuple[list[dict], dict[str, float]]:
    np = np_module()
    common = set(components[0]["values"])
    for component in components[1:]:
        common.intersection_update(component["values"])
    if not common:
        raise ValueError("No shared sample names across components.")

    order = [name for name in components[0]["order"] if name in common]
    weight_sum = sum(component["weight"] for component in components)
    rows = []
    for name in order:
        true_values = [component["values"][name][0] for component in components]
        if not np.allclose(true_values, true_values[0], rtol=1e-6, atol=1e-6):
            raise ValueError(f"Conflicting true values across components for sample {name}")
        pred = sum(
            component["weight"] * component["values"][name][1]
            for component in components
        ) / weight_sum
        row = {
            "name": name,
            "true": float(true_values[0]),
            "prediction": float(pred),
        }
        for component in components:
            row[f"pred_{component['name']}"] = float(component["values"][name][1])
        rows.append(row)

    y_true = np.asarray([row["true"] for row in rows], dtype=np.float64)
    y_pred = np.asarray([row["prediction"] for row in rows], dtype=np.float64)
    return rows, metrics(y_true, y_pred)


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV with component,pred_path,true_path,names_path[,weight].")
    parser.add_argument("--output-dir", default="results/consensus")
    parser.add_argument("--name", default="consensus")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest_rows = read_manifest(manifest_path)
    components = [load_component(row, manifest_path.parent) for row in manifest_rows]
    rows, summary = build_consensus(components)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / f"{args.name}_predictions.csv", rows)
    with (output_dir / f"{args.name}_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
