# MINTNN

## Mathematical Invariant-Enabled Topological Neural Networks

**MINTNN** is a molecular and materials learning framework that couples multiscale mathematical invariant representations with neural network architectures. The repository provides the application code used to evaluate how complementary topological and geometric descriptors interact with different neural processing mechanisms.

<p align="center">
  <img src="assets/mintnn_workflow.png" alt="MINTNN workflow" width="72%">
</p>

## Precomputed Feature Download

The released feature archives are hosted on Google Drive:

**Google Drive:** [MINTNN final feature archives](https://drive.google.com/drive/folders/1Y5hwU1WikPbT5pzhIL0ocmJcf9MWE1mv)

The folder is expected to contain:

```text
MINTNN_MOF_final_features.zip
MINTNN_LD50_final_features.zip
```

After downloading and extracting the archives, the recommended local organization is:

```text
data/
├── mof/
│   ├── 2STD/
│   │   ├── O2uptakemolkg.xlsx
│   │   └── N2uptakemolkg.xlsx
│   └── features/
│       ├── PH/
│       ├── PL/
│       ├── CA/
│       └── FPRC/
│
└── ld50/
    ├── LD50_train.csv
    ├── LD50_test.csv
    └── topology_features/
        ├── PH/
        ├── PL/
        ├── CA/
        ├── EIC/
        ├── EIC_BI/
        └── FPRC/
```

A concise manifest of the released feature groups is provided in:

```text
data/feature_manifest.csv
```

The feature archives are intentionally not stored directly in the Git repository because they are large generated artifacts.

## Installation

Clone the repository:

```bash
git clone https://github.com/yren24/MINTNN.git
cd MINTNN
```

Create the recommended Conda environment:

```bash
conda env create -f environment.yml
conda activate mintnn
```

Alternatively, install the Python requirements into an existing environment:

```bash
pip install -r requirements.txt
```

The graph-based SNN and CTNN components require PyTorch Geometric. For CUDA-specific installations, follow the PyTorch Geometric installation command matching your PyTorch and CUDA versions.

## Preparing the Data

Download the two feature archives from Google Drive and extract them from the repository root:

```bash
unzip MINTNN_MOF_final_features.zip
unzip MINTNN_LD50_final_features.zip
```

This should create the `data/mof/` and `data/ld50/` directories shown above. The application scripts use these compact feature family names directly: `PH`, `PL`, `CA`, `EIC`, `EIC_BI`, and `FPRC`.

## Training

Each component is defined by an application, an invariant family, and a neural architecture:

```text
application + invariant + architecture
```

Examples:

```text
LD50 + EIC + CTNN
MOF O2 + CA + CNN
MOF N2 + FPRC + GBT
```

The helper dispatcher in `scripts/train.py` builds the appropriate application command:

```bash
python scripts/train.py \
    --application ld50 \
    --invariant EIC \
    --architecture CTNN
```

MOF O2 uptake with CA-CNN:

```bash
python scripts/train.py \
    --application mof_o2 \
    --invariant CA \
    --architecture CNN
```

MOF N2 uptake with PH-GBT:

```bash
python scripts/train.py \
    --application mof_n2 \
    --invariant PH \
    --architecture GBT
```

Arguments not consumed by the dispatcher are forwarded to the application script, so component-specific settings can still be supplied:

```bash
python scripts/train.py \
    --application ld50 \
    --invariant FPRC \
    --architecture SNN \
    --epochs 300 \
    --batch-size 256
```

The underlying application scripts can also be called directly from `applications/mof/` or `applications/ld50/`.

## Evaluation

Training scripts write component outputs under `results/` by default. A lightweight metric utility is provided for prediction CSV files:

```bash
python scripts/evaluate.py \
    --csv results/example_predictions.csv \
    --true-column y_true \
    --pred-column y_pred
```

## Metadata

The `metadata/` directory contains workbook-derived summaries used to organize the released feature groups and final application components. These files document which invariant-architecture combinations were retained in the cleaned release without storing large feature tensors in Git.

## Citation

If you use this repository, please cite the accompanying MINTNN manuscript.

## License

License terms are provided in `LICENSE`.
