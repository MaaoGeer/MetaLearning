# Project Overview

The refactored research story is:

```text
Few-shot adaptation for unknown NIDS attacks using a learned optimizer on a
single-direction LSTM classifier.
```

The project intentionally removes supervised pretraining, TCN, Attention,
Transformer, MLP baselines, and architecture ablation code from the mainline.

## Training

`train_meta.py`:

1. loads config and dataset;
2. builds LOAO raw-flow splits;
3. windows each split independently;
4. samples N-way K-shot episodes;
5. builds a randomly initialized single-direction LSTM classifier;
6. trains only the LSTM Meta Optimizer with query loss.

## Evaluation

`scripts/run_experiments.py` compares:

- MetaOpt;
- SGD;
- Adam.

All methods share the same model, initialization, data split, support/query
episodes, and adaptable parameter set.

## Thesis Claim Boundary

The code supports the claim:

> A learned coordinate-wise LSTM optimizer can be evaluated as a few-shot
> adaptation rule for a temporal LSTM NIDS classifier under LOAO unknown-attack
> splits.

It does not claim:

- production real-time IDS;
- TCN/Attention/Transformer superiority;
- future-flow attack prediction;
- supervised-pretraining transfer learning.
