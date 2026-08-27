#!/usr/bin/env python3
"""Dispatch MINTNN component training commands."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys

from utils import (
    APPLICATION_SCRIPTS,
    DEFAULT_LD50_SNN_LR,
    INVARIANT_TO_TOPOLOGY,
    LD50_INVARIANTS,
    MOF_INVARIANTS,
    MOF_TARGETS,
    REPO_ROOT,
    has_option,
    normalize_application,
    normalize_architecture,
    normalize_invariant,
)


ARCH_TO_MODEL = {
    "ANN": "ann",
    "CNN": "cnn",
    "SNN": "snn",
    "CTNN": "copresheaf",
}

CURVATURE_BIDIRECTIONAL_ARCHES = {"ANN", "CTNN"}


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def clean_value(value, integer=False):
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if integer or number.is_integer():
        return str(int(number))
    return f"{number:g}"


def add_option(command, passthrough, option, value, integer=False):
    value = clean_value(value, integer=integer)
    if value is None or has_option(passthrough, option):
        return
    command.extend([option, value])


def normalize_hidden_dims(value):
    value = clean_value(value)
    if value is None:
        return None
    return value.replace("-", ",")


def final_row_for_mof(app, architecture, topology):
    target = MOF_TARGETS[app]
    path = REPO_ROOT / "metadata" / ("mof_o2_final_models.csv" if target == "O2uptakemolkg" else "mof_n2_final_models.csv")
    model_key = ARCH_TO_MODEL[architecture]
    for row in read_csv_rows(path):
        row_model = (row.get("model") or row.get("model_group") or "").lower()
        if row_model == model_key and row.get("topology") == topology:
            return row
    return None


def final_row_for_ld50(architecture, topology):
    path = REPO_ROOT / "metadata" / "ld50_final_20_models.csv"
    family = "Copresheaf" if architecture == "CTNN" else architecture
    needs_bidirectional = topology == "curvature" and architecture in CURVATURE_BIDIRECTIONAL_ARCHES
    for row in read_csv_rows(path):
        if row.get("nn_family") != family or row.get("topology") != topology:
            continue
        tag = row.get("feature_tag", "")
        if topology == "curvature":
            has_curvbi = "curvbi" in tag
            if needs_bidirectional and not has_curvbi:
                continue
            if not needs_bidirectional and has_curvbi:
                continue
        return row
    return None


def ld50_feature_root_tag(architecture, topology, invariant):
    if topology != "curvature":
        return invariant
    if architecture in CURVATURE_BIDIRECTIONAL_ARCHES:
        return "EIC/bidirectional"
    return "EIC/single_direction"


def apply_mof_final_defaults(command, passthrough, app, architecture, topology):
    if architecture == "GBT":
        return
    row = final_row_for_mof(app, architecture, topology)
    if row is None:
        return

    add_option(command, passthrough, "--split-style", "csca")
    add_option(command, passthrough, "--lr", row.get("lr"))
    add_option(command, passthrough, "--epochs", row.get("epochs"), integer=True)
    add_option(command, passthrough, "--batch-size", row.get("batch_size"), integer=True)
    add_option(command, passthrough, "--weight-decay", row.get("weight_decay"))
    add_option(command, passthrough, "--dropout", row.get("dropout"))
    add_option(command, passthrough, "--scaler", row.get("x_scaler") or row.get("scaler"))
    add_option(command, passthrough, "--target-transform", row.get("target_transform"))
    add_option(command, passthrough, "--target-scaler", row.get("target_scaler"))
    add_option(command, passthrough, "--seed-start", "13", integer=True)
    add_option(command, passthrough, "--seed-count", "10", integer=True)

    if architecture == "ANN":
        hidden_dims = normalize_hidden_dims(row.get("hidden_dims"))
        add_option(command, passthrough, "--hidden-dims", hidden_dims)
    elif architecture == "CNN":
        add_option(command, passthrough, "--h-channels", row.get("h_channels"), integer=True)
        add_option(command, passthrough, "--cnn-layers", row.get("cnn_num_layers") or row.get("cnn_layers"), integer=True)
        add_option(command, passthrough, "--kernel-size", row.get("kernel_size"), integer=True)
        add_option(command, passthrough, "--pool-size", row.get("pool_size"), integer=True)
        add_option(command, passthrough, "--fc-hidden", row.get("fc_hidden"), integer=True)
    elif architecture == "SNN":
        add_option(command, passthrough, "--h-dim", row.get("h_dim"), integer=True)
        add_option(command, passthrough, "--layers", row.get("snn_layers") or row.get("layers"), integer=True)
    elif architecture == "CTNN":
        add_option(command, passthrough, "--encoder-h-dim", row.get("encoder_h_dim"), integer=True)
        add_option(command, passthrough, "--encoder-heads", row.get("encoder_heads"), integer=True)
        add_option(command, passthrough, "--encoder-num-layers", row.get("encoder_num_layers"), integer=True)
        add_option(command, passthrough, "--decoder-h-dim", row.get("decoder_h_dim"), integer=True)
        add_option(command, passthrough, "--decoder-heads", row.get("decoder_heads"), integer=True)
        add_option(command, passthrough, "--decoder-num-layers", row.get("decoder_num_layers"), integer=True)
        add_option(command, passthrough, "--low-rank", row.get("low_rank"), integer=True)
        add_option(command, passthrough, "--patch-size", row.get("patch_size"), integer=True)


def apply_ld50_final_defaults(command, passthrough, architecture, topology):
    row = final_row_for_ld50(architecture, topology)
    if row is None:
        return

    seeds = row.get("seeds") or "42,1,2,3,4,5,6,7,8,9"
    if architecture == "SNN":
        add_option(command, passthrough, "--lr", row.get("lr"))
        add_option(command, passthrough, "--epochs", row.get("epochs"), integer=True)
        add_option(command, passthrough, "--h-dim", row.get("h_dim_or_channels"), integer=True)
        add_option(command, passthrough, "--layers", row.get("layers"), integer=True)
        add_option(command, passthrough, "--batch-size", row.get("batch_size"), integer=True)
        add_option(command, passthrough, "--feature-transform", row.get("feature_transform"))
        add_option(command, passthrough, "--pooling", row.get("pooling"))
        add_option(command, passthrough, "--seeds", seeds)
        return

    add_option(command, passthrough, "--lrs", row.get("lr"))
    add_option(command, passthrough, "--seeds", seeds)
    add_option(command, passthrough, "--epoch", row.get("epochs"), integer=True)
    add_option(command, passthrough, "--batch-size", row.get("batch_size"), integer=True)

    if architecture == "ANN":
        add_option(command, passthrough, "--layers", "2048,2048,1024,1024,512,64")
    elif architecture == "CNN":
        add_option(command, passthrough, "--feature-transform", row.get("feature_transform"))
        add_option(command, passthrough, "--h-channels", row.get("h_dim_or_channels"), integer=True)
        add_option(command, passthrough, "--num-layers", row.get("layers"), integer=True)
        add_option(command, passthrough, "--kernel", "3", integer=True)
        add_option(command, passthrough, "--pool", "1", integer=True)
    elif architecture == "CTNN":
        add_option(command, passthrough, "--feature-transform", row.get("feature_transform"))
        add_option(command, passthrough, "--encoder-h-dims", row.get("h_dim_or_channels"), integer=True)
        add_option(command, passthrough, "--encoder-num-layers", row.get("layers"), integer=True)
        add_option(command, passthrough, "--encoder-heads", "4", integer=True)
        add_option(command, passthrough, "--low-rank", "8", integer=True)
        add_option(command, passthrough, "--encoder-dropout", "0.05")
        add_option(command, passthrough, "--norm-typ", "pre_norm")
        add_option(command, passthrough, "--weight-decay", "0.01")
        add_option(command, passthrough, "--pooling", row.get("pooling"))


def build_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    app = normalize_application(args.application)
    invariant = normalize_invariant(args.invariant)
    architecture = normalize_architecture(args.architecture)
    topology = INVARIANT_TO_TOPOLOGY[invariant]

    family = "ld50" if app == "ld50" else "mof"
    if family == "mof" and invariant not in MOF_INVARIANTS:
        raise ValueError(f"{invariant} is not released for the MOF application.")
    if family == "ld50" and invariant not in LD50_INVARIANTS:
        raise ValueError(f"{invariant} is not released for the LD50 application.")

    script = APPLICATION_SCRIPTS[family].get(architecture)
    if script is None:
        raise ValueError(f"{architecture} is not available for {family}.")

    command = [sys.executable, str(script)]
    if family == "mof":
        property_name = MOF_TARGETS[app]
        command.extend(["--property", property_name, "--topology", topology])
        if architecture in {"ANN", "GBT"}:
            command.extend(["--feature-dir", f"data/mof/features/{invariant}"])
        if args.final_defaults:
            apply_mof_final_defaults(command, passthrough, app, architecture, topology)
    else:
        feature_root_tag = ld50_feature_root_tag(architecture, topology, invariant)
        if architecture == "SNN":
            command.extend(["--method", topology, "--feature-root-tag", feature_root_tag])
            if not args.final_defaults and not has_option(passthrough, "--lr"):
                command.extend(["--lr", DEFAULT_LD50_SNN_LR[invariant]])
        else:
            command.extend(["--features", topology, "--feature-root-tag", feature_root_tag])
        if args.final_defaults:
            apply_ld50_final_defaults(command, passthrough, architecture, topology)

    command.extend(passthrough)
    return command


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run a MINTNN application component defined by application, "
            "mathematical invariant, and neural architecture."
        )
    )
    parser.add_argument("--application", required=True, help="mof_o2, mof_n2, mof, or ld50")
    parser.add_argument("--invariant", required=True, help="PH, PL, CA, EIC, or FPRC")
    parser.add_argument("--architecture", required=True, help="ANN, CNN, SNN, CTNN, or GBT")
    parser.add_argument("--no-final-defaults", dest="final_defaults", action="store_false", help="Use application script defaults instead of metadata final-component settings.")
    parser.set_defaults(final_defaults=True)
    parser.add_argument("--print-command", action="store_true", help="Print the resolved command without running it.")
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()
    try:
        command = build_command(args, passthrough)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("Resolved command:")
    print(shlex.join(command))
    if args.print_command:
        return 0

    result = subprocess.run(command)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
