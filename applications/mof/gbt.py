import argparse
import csv
import math
import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


PROPERTY_DEFAULT = "O2uptakemolkg"


def read_xlsx_rows(xlsx_path):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    parsed = []
    for row in sheet.findall(".//a:sheetData/a:row", ns):
        values = {}
        for cell in row.findall("a:c", ns):
            ref = cell.get("r", "")
            col = "".join(ch for ch in ref if ch.isalpha())
            ctype = cell.get("t")
            value = ""
            if ctype == "inlineStr":
                value = "".join(t.text or "" for t in cell.findall(".//a:t", ns))
            else:
                v = cell.find("a:v", ns)
                if v is not None:
                    value = v.text or ""
                    if ctype == "s":
                        value = shared[int(value)]
            values[col] = value
        parsed.append(values)
    headers = parsed[0]
    col_to_name = {col: name for col, name in headers.items() if name}
    rows = []
    for raw in parsed[1:]:
        row = {}
        for col, name in col_to_name.items():
            row[name] = raw.get(col, "")
        rows.append(row)
    return rows


def load_dataset(property_name, data_dir):
    xlsx_path = os.path.join(data_dir, f"{property_name}.xlsx")
    rows = read_xlsx_rows(xlsx_path)
    mofids = []
    y = []
    for row in rows:
        mofid = row.get("MOFRefcodes", "")
        value = row.get(property_name, "")
        if not mofid or value in ("", None):
            continue
        try:
            target = float(value)
        except ValueError:
            continue
        if math.isnan(target):
            continue
        mofids.append(str(mofid))
        y.append(target)
    return np.asarray(mofids), np.asarray(y, dtype=np.float64)


def load_features(mofids, topology, property_name, feature_dir):
    features = []
    kept = []
    missing = []
    folder = os.path.join(feature_dir, topology, property_name)
    for mofid in mofids:
        path = os.path.join(folder, f"{mofid}.npy")
        if not os.path.exists(path):
            missing.append(mofid)
            continue
        features.append(np.load(path).reshape(-1))
        kept.append(mofid)
    if not features:
        raise RuntimeError(f"No features found in {folder}")
    width = features[0].shape[0]
    bad_width = [m for m, x in zip(kept, features) if x.shape[0] != width]
    if bad_width:
        raise ValueError(f"Inconsistent feature widths; first bad MOFs: {bad_width[:5]}")
    return np.asarray(kept), np.asarray(features, dtype=np.float32), missing


def run_cv(args):
    mofids, y_all = load_dataset(args.property, args.data_dir)
    kept, X, missing = load_features(mofids, args.topology, args.property, args.feature_dir)
    keep_mask = np.isin(mofids, kept)
    y = y_all[keep_mask]

    print(f"#### property={args.property} topology={args.topology}", flush=True)
    print(f"#### loaded labels={len(mofids)} features={X.shape} missing={len(missing)}", flush=True)
    if missing:
        print("#### first missing: " + ",".join(missing[:10]), flush=True)
    if X.shape[0] < args.folds:
        raise ValueError(f"Need at least {args.folds} feature files to run CV, found {X.shape[0]}. Generate more features first.")

    mae_all = []
    pcc_all = []
    fold_rows = []
    pred_rows = []

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.repeat)
    folds = list(kf.split(X))
    if args.fold is not None:
        if args.fold < 0 or args.fold >= len(folds):
            raise ValueError(f"Fold index {args.fold} is out of range for {len(folds)} folds.")
        fold_items = [(args.fold, folds[args.fold])]
    else:
        fold_items = list(enumerate(folds))

    for fold_index, (train_idx, valtest_idx) in fold_items:
        fold_label = fold_index + 1
        X_train, X_valtest = X[train_idx], X[valtest_idx]
        y_train, y_valtest = y[train_idx], y[valtest_idx]
        mof_valtest = kept[valtest_idx]

        X_val, X_test, y_val, y_test, mof_val, mof_test = train_test_split(
            X_valtest,
            y_valtest,
            mof_valtest,
            test_size=0.5,
            random_state=args.repeat * 10 + fold_index,
            shuffle=True,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.5,
            max_features="sqrt",
            learning_rate=args.learning_rate,
            random_state=args.repeat * 10 + fold_index,
        )
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        pcc = pearsonr(y_test, y_pred)[0]
        r2 = pcc ** 2
        mae_all.append(mae)
        pcc_all.append(pcc)
        fold_rows.append({
            "repeat": args.repeat,
            "fold": fold_index,
            "test_size": len(y_test),
            "mae": mae,
            "pcc": pcc,
            "r2_from_pcc": r2,
        })
        for mofid, yt, yp in zip(mof_test, y_test, y_pred):
            pred_rows.append({"repeat": args.repeat, "fold": fold_index, "MOFRefcodes": mofid, "true": yt, "pred": yp})
        print(f"[{args.property} {args.topology}] Repeat {args.repeat + 1}, Fold {fold_label}: MAE = {mae:.3e}, PCC = {pcc:.4f}, R2 = {r2:.4f}", flush=True)

    if args.fold is not None:
        output_dir = os.path.join(args.output_dir, args.property)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{args.property}_repeat{args.repeat}_fold{args.fold}.txt")
        with open(output_file, "w") as f:
            f.write(f"[{args.property}] Repeat {args.repeat + 1}, Fold {args.fold + 1} Results:\n")
            f.write(f"MAE: {mae_all[0]:.4e}\n")
            f.write(f"R2:  {np.square(pcc_all[0]):.4f}\n")
        print(f"#### saved {output_file}", flush=True)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"{args.property}_{args.topology}_repeat{args.repeat}_gbt"
    result_txt = os.path.join(args.output_dir, stem + "_results.txt")
    fold_csv = os.path.join(args.output_dir, stem + "_fold_metrics.csv")
    pred_csv = os.path.join(args.output_dir, stem + "_predictions.csv")

    with open(result_txt, "w") as f:
        f.write(f"[{args.property}] topology={args.topology} {args.folds}-fold CV x 1 repeat (repeat={args.repeat})\n")
        f.write(f"Feature shape: {X.shape}\n")
        f.write(f"Missing features: {len(missing)}\n")
        f.write(f"Average MAE: {np.mean(mae_all):.6e} +/- {np.std(mae_all):.6e}\n")
        f.write(f"Average PCC: {np.mean(pcc_all):.6f} +/- {np.std(pcc_all):.6f}\n")
        f.write(f"Average R2_from_PCC: {np.mean(np.square(pcc_all)):.6f} +/- {np.std(np.square(pcc_all)):.6f}\n")

    with open(fold_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["repeat", "fold", "test_size", "mae", "pcc", "r2_from_pcc"])
        writer.writeheader()
        writer.writerows(fold_rows)

    with open(pred_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["repeat", "fold", "MOFRefcodes", "true", "pred"])
        writer.writeheader()
        writer.writerows(pred_rows)

    print(f"#### saved {result_txt}", flush=True)
    print(f"#### saved {fold_csv}", flush=True)
    print(f"#### saved {pred_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run CSMOF1-style GBT on MOF topology features.")
    parser.add_argument("--property", default=PROPERTY_DEFAULT)
    parser.add_argument("--topology", required=True, choices=["homology", "lap", "facet", "forman", "curvature"])
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=None, help="Optional 0-based fold index for uahpc-style single-fold runs.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--data-dir", default="data/mof/2STD")
    parser.add_argument("--feature-dir", default="data/mof/features/PH")
    parser.add_argument("--output-dir", default="results/mof/gbt")
    parser.add_argument("--n-estimators", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    args = parser.parse_args()
    run_cv(args)


if __name__ == "__main__":
    main()
