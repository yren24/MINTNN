# Data and Feature Archives

The final `.npy` features should live outside Git history and be distributed through Google Drive.

## Download Link

[MINTNN final feature archives](https://drive.google.com/drive/folders/1Y5hwU1WikPbT5pzhIL0ocmJcf9MWE1mv)

Expected zip filenames:

- `MINTNN_MOF_final_features.zip`
- `MINTNN_LD50_final_features.zip`

## MOF Layout

Place MOF labels and features under:

```text
data/mof/
  2STD/
    O2uptakemolkg.xlsx
    N2uptakemolkg.xlsx
  features/
    PH/
      homology/O2uptakemolkg/*.npy
      homology/N2uptakemolkg/*.npy
    CA/
      facet/O2uptakemolkg/*.npy
      facet/N2uptakemolkg/*.npy
    FPRC/
      forman/O2uptakemolkg/*.npy
      forman/N2uptakemolkg/*.npy
    PL/
      lap/O2uptakemolkg/*.npy
      lap/N2uptakemolkg/*.npy
```

The final MOF code defaults to the `callZeroLap` Laplacian feature folder, matching the final result tables.

## LD50 Layout

Place LD50 labels and features under:

```text
data/ld50/
  LD50_train.csv
  LD50_test.csv
  topology_features/
    PH/homology/train/*.npy
    PH/homology/test/*.npy
    CA/facet/train/*.npy
    CA/facet/test/*.npy
    PL/lap/train/*.npy
    PL/lap/test/*.npy
    FPRC/forman/train/*.npy
    FPRC/forman/test/*.npy
    EIC/single_direction/curvature/train/*.npy
    EIC/single_direction/curvature/test/*.npy
    EIC/bidirectional/curvature/train/*.npy
    EIC/bidirectional/curvature/test/*.npy
```

The LD50 CSV files must contain `filename` and `label` columns.
