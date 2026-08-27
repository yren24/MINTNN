"""Shared command-building utilities for MINTNN application scripts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

INVARIANT_TO_TOPOLOGY = {
    "PH": "homology",
    "PL": "lap",
    "CA": "facet",
    "EIC": "curvature",
    "FPRC": "forman",
}

LEGACY_INVARIANT_ALIASES = {
    "EIC_BI": "EIC",
}

MOF_TARGETS = {
    "mof": "O2uptakemolkg",
    "mof_o2": "O2uptakemolkg",
    "mof_n2": "N2uptakemolkg",
}

APPLICATION_SCRIPTS = {
    "mof": {
        "ANN": REPO_ROOT / "applications" / "mof" / "ann.py",
        "CNN": REPO_ROOT / "applications" / "mof" / "cnn.py",
        "SNN": REPO_ROOT / "applications" / "mof" / "snn.py",
        "CTNN": REPO_ROOT / "applications" / "mof" / "ctnn.py",
        "GBT": REPO_ROOT / "applications" / "mof" / "gbt.py",
    },
    "ld50": {
        "ANN": REPO_ROOT / "applications" / "ld50" / "ann.py",
        "CNN": REPO_ROOT / "applications" / "ld50" / "cnn.py",
        "SNN": REPO_ROOT / "applications" / "ld50" / "snn.py",
        "CTNN": REPO_ROOT / "applications" / "ld50" / "ctnn.py",
    },
}

MOF_INVARIANTS = {"PH", "PL", "CA", "FPRC"}
LD50_INVARIANTS = {"PH", "PL", "CA", "EIC", "FPRC"}

DEFAULT_LD50_SNN_LR = {
    "PH": "8e-4",
    "PL": "8e-4",
    "CA": "8e-4",
    "EIC": "8e-4",
    "FPRC": "8e-4",
}


def has_option(argv: list[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in argv)


def normalize_application(value: str) -> str:
    app = value.strip().lower().replace("-", "_")
    if app not in {"mof", "mof_o2", "mof_n2", "ld50"}:
        raise ValueError(f"Unsupported application: {value}")
    return app


def normalize_invariant(value: str) -> str:
    invariant = value.strip().upper().replace("-", "_")
    invariant = LEGACY_INVARIANT_ALIASES.get(invariant, invariant)
    if invariant not in INVARIANT_TO_TOPOLOGY:
        raise ValueError(f"Unsupported invariant: {value}")
    return invariant


def normalize_architecture(value: str) -> str:
    arch = value.strip().upper()
    if arch == "COPRESHEAF":
        return "CTNN"
    return arch
