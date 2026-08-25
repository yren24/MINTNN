# Data and Feature Archives

Large precomputed feature tensors are distributed outside Git.

Download the final archives from:

[MINTNN final feature archives](https://drive.google.com/drive/folders/1Y5hwU1WikPbT5pzhIL0ocmJcf9MWE1mv)

Expected archive names:

```text
MINTNN_MOF_final_features.zip
MINTNN_LD50_final_features.zip
```

Extract both archives from the repository root:

```bash
unzip MINTNN_MOF_final_features.zip
unzip MINTNN_LD50_final_features.zip
```

Expected directories after extraction:

```text
data/
├── mof/
│   ├── 2STD/
│   └── features/
│       ├── PH/
│       ├── PL/
│       ├── CA/
│       └── FPRC/
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

If the Google Drive folder is used for public release, set sharing to `Anyone with the link` with viewer access.
