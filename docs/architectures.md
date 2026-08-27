# Architecture Summary

MINTNN evaluates the same mathematical invariant families with four neural architectures.

## ANN

`src/mintnn/architectures/ann.py`

The ANN is a batch-normalized MLP on flattened feature vectors:

```text
Linear(bias=False) -> BatchNorm1d -> ReLU -> hidden blocks -> Linear(1)
```

Final hidden dimensions:

- MOF: `2048, 1024, 1024, 512, 512, 64`
- LD50: `2048, 2048, 1024, 1024, 512, 64`

## CNN

`src/mintnn/architectures/cnn.py`

The CNN treats the filtration or interaction-scale axis as a one-dimensional sequence:

```text
Conv1d -> residual Conv1d blocks -> adaptive average pooling -> MLP head
```

Final presets:

- MOF: `h_channels=128`, `num_layers=3`, `kernel_size=3`, `pool_size=1`, `fc_hidden=256`
- LD50: `h_channels=128`, `num_layers=5`, `kernel_size=3`, `pool_size=1`

## SNN

`src/mintnn/architectures/snn.py`

Two task-specific SNN forms are included:

- MOF category-node SNN: each atom/category channel is a node in a complete graph-like message block.
- LD50 pair-edge SNN: each of the 30 element-pair feature channels is a graph node; a directed line graph connects pair nodes that share an element.

The final LD50 pair-edge SNN uses `GATv2Conv` by default and requires `torch-geometric`.

## CTNN / CoPresheaf

`src/mintnn/architectures/ctnn.py`

The CTNN is implemented as a CoPresheaf transformer:

```text
topological feature patches -> positional embedding + CLS token
-> low-rank sheaf transformer blocks -> regression head
```

The final LD50 CTNN uses `pre_norm`, `pooling=cls_mean_max`, `low_rank=8`, `encoder_heads=4`, and three encoder layers. The MOF CoPresheaf backbone used by the MOF CTNN script is stored in `applications/mof/copresheaf_model.py`.
