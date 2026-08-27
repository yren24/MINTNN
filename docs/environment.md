# Environment Notes

The public MINTNN code paths use a small set of direct Python dependencies:
NumPy, SciPy, scikit-learn, PyTorch, pandas, and PyTorch Geometric.

## Recommended Files

- `requirements.txt` lists direct runtime dependencies for the public scripts.
- `environment.yml` is a portable Conda environment for CPU or user-managed CUDA installs.
- `envs/environment-hpcc-cu121.yml` is the recommended Linux GPU environment for reproducing the released runs with PyTorch 2.4.1 and CUDA 12.1.
- `envs/requirements-hpcc-cu121.txt` preserves the pip-compiled CUDA 12.1 reference lock from the original `embed_nn` workspace and adds the public-release utilities.

## Architecture-Specific Notes

ANN, CNN, CTNN, GBT, evaluation, and consensus scripts do not require PyTorch Geometric.

SNN components require PyTorch Geometric because they use `GATv2Conv` and `GCNConv`. On Linux GPU systems, install PyTorch Geometric with wheels that match the active PyTorch and CUDA versions. The HPCC reference target is:

```text
PyTorch 2.4.1
CUDA 12.1
PyTorch Geometric >= 2.4
```

For full HPCC reproduction across all released architecture families, use `envs/environment-hpcc-cu121.yml`.

After creating an environment, verify it with:

```bash
python scripts/check_environment.py
python scripts/check_environment.py --include-snn
```
