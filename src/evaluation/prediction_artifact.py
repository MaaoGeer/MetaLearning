"""Compact, schema-versioned storage for per-task prediction trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = 1


def write_prediction_trajectories(
    path: str | Path,
    records: Sequence[Mapping[str, object]],
) -> Path:
    """Store logits/labels for every recorded task and step in compressed NPZ."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("prediction trajectory artifact requires at least one record")

    logits = np.stack([
        torch.stack(list(record["step_logits"])).cpu().numpy().astype(np.float32)
        for record in records
    ])
    labels = np.stack([
        torch.as_tensor(record["labels"]).cpu().numpy().astype(np.int64)
        for record in records
    ])
    metadata_keys = ("experiment", "shot", "method", "split", "task_id")
    arrays = {
        "logits": logits,
        "labels": labels,
        **{
            key: np.asarray([record[key] for record in records])
            for key in metadata_keys
        },
    }
    np.savez_compressed(destination, **arrays)
    schema = {
        "schema_version": SCHEMA_VERSION,
        "format": "npz",
        "logits_shape": list(logits.shape),
        "labels_shape": list(labels.shape),
        "step_axis_includes_zero": True,
        "metadata_fields": list(metadata_keys),
        "metrics_reconstructable": [
            "accuracy", "precision", "recall", "macro_f1", "roc_auc",
            "pr_auc", "attack_recall", "false_positive_rate",
            "brier_score", "ece",
        ],
    }
    destination.with_suffix(destination.suffix + ".schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination

