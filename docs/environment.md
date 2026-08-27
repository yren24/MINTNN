# Environment Notes

The public MINTNN code paths use NumPy, SciPy, scikit-learn, PyTorch, pandas, and PyTorch Geometric.

## Recommended Files

- `environment.yml` is the recommended full reproduction environment. It targets Linux GPU or HPCC-style systems with CUDA 12.1, PyTorch 2.4.1, and PyTorch Geometric.
- `requirements.txt` is a pip fallback listing direct runtime dependencies. For GPU/SNN runs, Conda is preferred because PyTorch Geometric wheels must match the PyTorch and CUDA build.

## Architecture-Specific Notes

ANN, CNN, CTNN, GBT, evaluation, and consensus scripts do not require PyTorch Geometric.

SNN components require PyTorch Geometric because they use `GATv2Conv` and `GCNConv`. The recommended environment target is:

```text
PyTorch 2.4.1
CUDA 12.1
PyTorch Geometric >= 2.4
```

Full paper-scale reproduction is expected to run on a GPU workstation or cluster. GitHub hosts the code and configuration; it does not provide compute.

After creating an environment, verify it with:

```bash
python scripts/check_environment.py
python scripts/check_environment.py --include-snn
```
