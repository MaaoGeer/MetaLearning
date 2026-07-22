"""Shared builders and artifact I/O for the LSTM-only meta-learning project."""

from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn as nn

from .meta_optimizer.lstm_optimizer import LSTMOptimizer
from .models.factory import build_base_learner
from .utils.config import Config
from .utils.logger import get_logger

logger = get_logger(__name__)


def task_n_way(cfg: Config) -> int:
    """Return the number of local classes in each episode."""
    mode = str(cfg.data.get("task_mode", "nway")).lower()
    return 2 if mode == "binary" else int(cfg.data.n_way)


def build_meta_model(cfg: Config, feature_dim: int, window_size: int) -> nn.Module:
    """Build the randomly initialized LSTM base learner used by all methods."""
    return build_base_learner(
        cfg,
        feature_dim=feature_dim,
        window_size=window_size,
        n_classes=task_n_way(cfg),
    )


def build_meta_optimizer(cfg: Config) -> LSTMOptimizer:
    """Build the learned LSTM meta optimizer."""
    m = cfg.meta_optimizer
    meta_cfg = cfg.get("meta", {})
    return LSTMOptimizer(
        hidden_size=int(m.hidden_size),
        num_layers=int(m.num_layers),
        preprocess=bool(m.preprocess),
        preprocess_p=float(m.preprocess_p),
        output_scale=float(m.output_scale),
        use_learnable_lr=bool(m.use_learnable_lr),
        update_norm_clip=m.get("update_norm_clip", meta_cfg.get("update_norm_clip", 1.0)),
    )


def resolve_adapt_names(model: nn.Module, cfg: Config) -> List[str]:
    """Resolve which base-learner tensors are adapted in the inner loop."""
    scope = str(cfg.meta.get("adapt_scope", "full")).lower()
    all_params = list(model.named_parameters())
    if scope == "full":
        names = [name for name, _ in all_params]
    elif scope in {"head_only", "last_linear", "lstm_frozen"}:
        names = [
            name for name, _ in all_params
            if name.startswith("classifier.") or name.startswith("head.")
            or name.startswith("fc.") or name.startswith("linear.")
        ]
    else:
        raise ValueError(
            "Unsupported meta.adapt_scope=%r. Expected full/head_only/last_linear/lstm_frozen."
            % scope
        )
    if not names:
        raise ValueError(f"meta.adapt_scope={scope!r} did not match any model parameters.")

    param_lookup = dict(all_params)
    total = sum(param_lookup[name].numel() for name in names)
    logger.info(
        "Adaptation scope: %s (%d tensors, %d params)",
        scope,
        len(names),
        total,
    )
    logger.info("Adapted parameter names: %s", ", ".join(names))
    return names


def save_artifacts(
    path: str,
    meta_init_state: Dict[str, torch.Tensor],
    meta_opt_state: Dict[str, torch.Tensor],
    cfg: Config,
    extra: Dict,
) -> None:
    """Save the shared random initialization and learned optimizer."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "meta_init_state": {k: v.detach().cpu() for k, v in meta_init_state.items()},
        "meta_opt_state": {k: v.detach().cpu() for k, v in meta_opt_state.items()},
        "config": cfg.to_dict(),
        "extra": extra,
    }
    torch.save(payload, path)
    logger.info("Saved meta-learning artifacts: %s", path)


def load_artifacts(path: str) -> dict:
    """Load meta-learning artifacts."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "weights_only=True load failed; falling back to full load. "
            "Only do this for trusted artifacts: %s",
            exc,
        )
        return torch.load(path, map_location="cpu", weights_only=False)
