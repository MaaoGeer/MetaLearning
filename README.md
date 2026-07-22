# LSTM + Meta Optimizer + Few-shot NIDS

This repository is a research-oriented offline experiment framework for few-shot
network intrusion detection.

The mainline is intentionally narrow:

```text
Raw Flow
  -> Temporal Window
  -> Single-direction LSTM
  -> Last Hidden State
  -> Linear Classifier
  -> Meta Optimizer / Adam / SGD
  -> Few-shot Adaptation
```

There is no supervised pretraining stage. `train_meta.py` saves one shared random
initialization plus the learned LSTM Meta Optimizer. `scripts/run_experiments.py`
then compares MetaOpt, Adam, and SGD from the same initialization and the same
few-shot episodes.

## Structure

```text
configs/
  base.yaml
  datasets/
src/
  data/              # dataset loading, LOAO split, temporal windows, episode sampling
  models/            # single-direction LSTM classifier
  meta_optimizer/    # LSTM learned optimizer plus SGD/Adam functional baselines
  meta_learning/     # functional inner/outer loops
  trainer/           # meta trainer and evaluation adapter
  evaluation/        # metrics and adaptation-speed summaries
scripts/
  run_experiments.py
  run_fast_adaptation_matrix.py
train_meta.py
tests/
```

## Train Meta Optimizer

```bash
python train_meta.py --config configs/base.yaml --dataset configs/datasets/cicids2017.yaml
```

Artifact:

```text
checkpoints/meta_artifacts.pt
```

It contains:

- shared random LSTM initialization;
- learned LSTM Meta Optimizer weights;
- data/model metadata required for fair evaluation.

## Compare Optimizers

```bash
python scripts/run_experiments.py --artifacts checkpoints/meta_artifacts.pt --out outputs/experiments
```

The comparison uses:

- identical model architecture;
- identical random initialization;
- identical validation/test episodes;
- validation-only LR search for Adam and SGD;
- test-only final metrics.

## Matrix Experiments

```bash
python scripts/run_fast_adaptation_matrix.py --quick --dry-run
python scripts/run_fast_adaptation_matrix.py --unknowns all --shots 1,5,10,20 --seeds 0,1,2,3,4
```

Use the matrix runner for thesis-grade multi-unknown, multi-shot, multi-seed
experiments and paired MetaOpt-vs-Adam significance summaries.

## Core Configuration

```yaml
model:
  arch: "lstm"
  lstm:
    hidden_size: 32
    num_layers: 1
    dropout: 0.0

meta:
  inner_steps: 20
  first_order: false
  meta_optimizer_lr: 0.001
```

## Prediction Target

The default label strategy is `last`:

```text
X_t = [flow_{t-L+1}, ..., flow_t]
y_t = label(flow_t)
```

So the task is temporal-context flow classification, not next-flow prediction.

## Tests

```bash
python -m pytest tests -q
```
