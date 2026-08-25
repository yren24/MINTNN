# Feature Summary

This document summarizes the final feature versions used in the MOF and LD50 code paths. The CSV files in `metadata/` are the source of truth for exact model settings.

## MOF

MOF features are grouped by nine atom/category channels: `C`, `H`, `O`, `N`, `Zn`, `Cu`, `OZn`, `ZnO`, and `Call`.

| topology family | folder | shape after model reshape |
| --- | --- | --- |
| PH | `PH` | 9 x 120 x 3 |
| CA/facet | `CA` | 9 x 121 x 5 |
| FPRC/Forman | `FPRC` | 9 x 120 x 20 |
| PL | `PL` | 9 x 120 x 10 |

The final result tables replaced the older MOF Laplacian feature with the `callZeroLap` feature. The `Call` category stores zero-filled Laplacian channels and appends `Call` Betti-1 and Betti-2 channels.

## LD50

LD50 features use 30 element-pair channels from:

```text
base elements = H, C, N, O
all elements = H, C, N, O, F, P, S, Cl, Br, I
pair rule = a in base, b in all, element_order[a] < element_order[b]
```

| topology family | final tag | raw shape | model axes |
| --- | --- | --- | --- |
| PH | `PH` | 100 x 30 x 2 | 30 x 100 x 2 |
| CA/facet | `CA` | 100 x 30 x 2 | 30 x 100 x 2 |
| PL | `PL` | 100 x 30 x 8 | 30 x 100 x 8 |
| FPRC/Forman | `FPRC` | 100 x 30 x 20 | 30 x 100 x 20 |
| EIC/curvature | `EIC` | 49 x 30 x 10 | 30 x 49 x 10 |
| EIC/curvature bidirectional | `EIC_BI` | 49 x 30 x 20 | 30 x 49 x 20 |

The final LD50 setting uses maximum filtration 10, step 0.1, and `bond_delta=0`, with covalent bonds removed from the modified filtration matrix.
