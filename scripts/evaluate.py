#!/usr/bin/env python3
"""Compute regression metrics from a prediction CSV file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    import numpy as np
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }
    if len(y_true) > 1:
        metrics["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
    else:
        metrics["pearson_r"] = float("nan")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Prediction CSV path.")
    parser.add_argument("--true-column", default="y_true")
    parser.add_argument("--pred-column", default="y_pred")
    parser.add_argument("--output-json", default=None, help="Optional path for JSON metrics.")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    frame = pd.read_csv(args.csv)
    required = [args.true_column, args.pred_column]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    values = frame[required].dropna()
    y_true = values[args.true_column].to_numpy(dtype=np.float64)
    y_pred = values[args.pred_column].to_numpy(dtype=np.float64)
    metrics = compute_metrics(y_true, y_pred)

    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
