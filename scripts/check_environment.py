#!/usr/bin/env python3
"""Check that the MINTNN runtime dependencies can be imported."""

from __future__ import annotations

import argparse
import importlib
import sys


REQUIRED_MODULES = [
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("sklearn", "scikit-learn"),
    ("torch", "PyTorch"),
    ("pandas", "pandas"),
    ("torch_geometric", "PyTorch Geometric"),
]


def check_module(module_name: str, label: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"[missing] {label}: {exc}")
        return False
    version = getattr(module, "__version__", "unknown")
    print(f"[ok] {label}: {version}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    print(f"python: {sys.executable}")
    ok = True
    for module_name, label in REQUIRED_MODULES:
        ok = check_module(module_name, label) and ok

    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
            print(f"torch.cuda.device_name.0: {torch.cuda.get_device_name(0)}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
