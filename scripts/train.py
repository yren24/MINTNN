#!/usr/bin/env python3
"""Dispatch MINTNN component training commands."""

from __future__ import annotations

import argparse
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
    has_option,
    normalize_application,
    normalize_architecture,
    normalize_invariant,
)


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
    else:
        if architecture == "SNN":
            command.extend(["--method", topology, "--feature-root-tag", invariant])
            if not has_option(passthrough, "--lr"):
                command.extend(["--lr", DEFAULT_LD50_SNN_LR[invariant]])
        else:
            command.extend(["--features", topology, "--feature-root-tag", invariant])

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
    parser.add_argument("--invariant", required=True, help="PH, PL, CA, EIC, EIC_BI, or FPRC")
    parser.add_argument("--architecture", required=True, help="ANN, CNN, SNN, CTNN, or GBT")
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
