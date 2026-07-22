"""Component tests for the LSTM + Meta Optimizer + Few-shot mainline."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.loao import SplitArrays
from src.data.preprocessing import FeatureStandardizer, build_class_index
from src.data.task_builder import AdaptationTaskSampler, build_windowed_dataset
from src.data.task_sampler import FewShotTaskSampler
from src.evaluation.adaptation_speed import (
    adaptation_selection_key,
    compute_speed,
    summarize_adaptation,
)
from src.evaluation.metrics import compute_metrics
from src.models import LSTMClassifier
from src.models.factory import build_base_learner
from src.utils.config import Config


def _toy_frame(n_per_class: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    centers = {"benign": 0.0, "dos": 3.0, "ddos": -3.0}
    order = 0
    for cls, center in centers.items():
        feats = rng.normal(center, 1.0, size=(n_per_class, 5))
        for i in range(n_per_class):
            row = {f"f{j}": float(feats[i, j]) for j in range(5)}
            row["label"] = cls
            row["__order__"] = float(order)
            order += 1
            rows.append(row)
    return pd.DataFrame(rows)


def test_standardizer():
    df = _toy_frame()
    cols = [f"f{j}" for j in range(5)]
    x = FeatureStandardizer(feature_columns=cols).fit(df).transform(df)
    assert x.shape == (len(df), 5)


def test_sequence_and_sampler():
    df = _toy_frame(n_per_class=120, seed=2)
    cols = [f"f{j}" for j in range(5)]
    feats = FeatureStandardizer(feature_columns=cols).fit(df).transform(df)
    c2i = build_class_index(["benign", "dos", "ddos"])
    split = SplitArrays(features=feats, labels=df["label"].to_numpy(), order=df["__order__"].to_numpy())
    ds = build_windowed_dataset(split, c2i, window_size=8, stride=8,
                                windowing_mode="temporal", label_strategy="last")
    sampler = FewShotTaskSampler(ds, n_way=3, k_shot=2, q_query=2, seed=0)
    task = sampler.sample_task()
    assert task.support_x.shape[0] == 6


def test_window_size_one_is_nontemporal():
    df = _toy_frame(n_per_class=20, seed=1)
    cols = [f"f{j}" for j in range(5)]
    feats = FeatureStandardizer(feature_columns=cols).fit(df).transform(df)
    c2i = build_class_index(["benign", "dos", "ddos"])
    split = SplitArrays(features=feats, labels=df["label"].to_numpy(), order=df["__order__"].to_numpy())
    ds = build_windowed_dataset(split, c2i, window_size=1, stride=1,
                                windowing_mode="temporal", label_strategy="last")
    assert ds.features.shape[1] == 1


def test_lstm_classifier_uses_last_hidden_state():
    torch.manual_seed(0)
    model = LSTMClassifier(feature_dim=5, n_classes=2, hidden_size=8)
    x = torch.randn(6, 8, 5)
    logits, emb = model(x, return_embedding=True)
    assert logits.shape == (6, 2)
    assert emb.shape == (6, 8)
    assert model.lstm.bidirectional is False


def test_factory_rejects_non_lstm_architectures():
    cfg = Config({"model": {"arch": "unsupported_arch"}})
    with pytest.raises(ValueError):
        build_base_learner(cfg, feature_dim=5, window_size=8, n_classes=2)


def test_factory_builds_lstm():
    cfg = Config({
        "model": {
            "arch": "lstm",
            "lstm": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0},
        }
    })
    model = build_base_learner(cfg, feature_dim=5, window_size=8, n_classes=2)
    assert isinstance(model, LSTMClassifier)
    assert model(torch.randn(4, 8, 5)).shape == (4, 2)


def test_adaptation_task_sampler_forces_unknown():
    df = _toy_frame(n_per_class=120, seed=3)
    cols = [f"f{j}" for j in range(5)]
    feats = FeatureStandardizer(feature_columns=cols).fit(df).transform(df)
    c2i = build_class_index(["benign", "dos", "ddos"])
    split = SplitArrays(features=feats, labels=df["label"].to_numpy(), order=df["__order__"].to_numpy())
    ds = build_windowed_dataset(split, c2i, window_size=8, stride=8,
                                windowing_mode="temporal", label_strategy="last")
    sampler = AdaptationTaskSampler(
        ds, unknown_idx=c2i["ddos"], ref_indices=list(c2i.values()),
        mode="binary", k_shot=2, q_query=2, benign_idx=c2i["benign"], seed=0)
    task = sampler.sample_task()
    assert c2i["ddos"] in task.global_classes


def test_metrics_and_speed():
    logits = torch.tensor([[2.0, 0.1], [0.0, 3.0], [0.0, 1.5]])
    targets = torch.tensor([0, 1, 1])
    metrics = compute_metrics(logits, targets, num_classes=2, attack_class_indices=[1])
    assert metrics.accuracy == 1.0
    assert metrics.attack_recall == 1.0
    speed = compute_speed([0.5, 0.7, 0.82], [0.8], max_steps=3)
    assert speed.speeds[0.8] == 3


def test_fast_adaptation_selection_prefers_reach_then_speed():
    fast = compute_speed([0.81, 0.82], [0.8], max_steps=2)
    slow = compute_speed([0.70, 0.81], [0.8], max_steps=2)
    fast_summary = summarize_adaptation([fast], 0.8, [0, 1, 2])
    slow_summary = summarize_adaptation([slow], 0.8, [0, 1, 2])
    assert adaptation_selection_key(fast_summary) > adaptation_selection_key(slow_summary)


def test_config_override_parses_lists():
    cfg = Config({"compare": {"shots": [1]}})
    updated = cfg.apply_overrides(["compare.shots=[1, 5, 10, 20]"])
    assert updated.compare.shots == [1, 5, 10, 20]
