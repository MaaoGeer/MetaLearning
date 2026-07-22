"""Model factory for the research-refactored LSTM-only project."""

from __future__ import annotations

import torch.nn as nn

from ..utils.config import Config
from .lstm import LSTMClassifier


def build_base_learner(
    cfg: Config,
    feature_dim: int,
    window_size: int,
    n_classes: int,
) -> nn.Module:
    """Build the only supported base learner.

    The project mainline is intentionally narrow:
        temporal window -> single-direction LSTM -> last hidden state -> linear classifier.
    """
    arch = str(cfg.model.get("arch", "lstm")).lower()
    if arch not in {"lstm", "lstm_only"}:
        raise ValueError(
            f"Unsupported model.arch={arch!r}. The refactored project only supports 'lstm'."
        )

    m = cfg.model.get("lstm", {})
    return LSTMClassifier(
        feature_dim=feature_dim,
        n_classes=n_classes,
        hidden_size=int(m.get("hidden_size", 32)),
        num_layers=int(m.get("num_layers", 1)),
        dropout=float(m.get("dropout", 0.0)),
    )
