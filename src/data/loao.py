"""Leave-One-Attack-Out splitting at raw-flow level."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from .base_dataset import LoadResult
from .preprocessing import FeatureStandardizer, build_class_index

logger = get_logger(__name__)


@dataclass
class SplitArrays:
    """Array representation of one raw split."""

    features: np.ndarray
    labels: np.ndarray
    order: np.ndarray
    row_id: Optional[np.ndarray] = None
    segment_id: Optional[np.ndarray] = None
    source_file: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.features.shape[0]


@dataclass
class LOAOResult:
    """LOAO split result. Windowing happens after these raw splits are isolated."""

    feature_columns: List[str]
    known_classes: List[str]
    known_class_to_idx: Dict[str, int]
    unknown_class: str
    train: SplitArrays
    eval: SplitArrays
    test: SplitArrays
    unknown: SplitArrays
    standardizer: FeatureStandardizer
    class_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _subset(df: pd.DataFrame, classes: List[str], max_per_class: int,
            rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for cls in classes:
        sub = df[df["label"] == cls]
        if max_per_class and max_per_class > 0 and len(sub) > max_per_class:
            idx = rng.choice(sub.index.values, size=max_per_class, replace=False)
            sub = sub.loc[idx]
        parts.append(sub)
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, axis=0).sort_values("__order__").reset_index(drop=True)


def _ordered_or_shuffled(
    df: pd.DataFrame,
    split_mode: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if split_mode == "random":
        return df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    return df.sort_values("__order__").reset_index(drop=True)


def _slice_train_eval_test(
    df: pd.DataFrame,
    eval_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one ordered class block into train/eval/test."""
    n = len(df)
    if n == 0:
        empty = df.iloc[0:0]
        return empty, empty, empty

    n_test = max(0, int(n * test_ratio))
    n_eval = max(1, int(n * eval_ratio)) if eval_ratio > 0 else 0
    while n_test + n_eval >= n and (n_test + n_eval) > 0:
        if n_test > 0:
            n_test -= 1
        elif n_eval > 1:
            n_eval -= 1
        else:
            break

    n_train = max(0, n - n_eval - n_test)
    train_df = df.iloc[:n_train]
    eval_df = df.iloc[n_train:n_train + n_eval]
    test_df = df.iloc[n_train + n_eval:] if n_test > 0 else df.iloc[0:0]
    return train_df, eval_df, test_df


def _merge(parts: List[pd.DataFrame], fallback: pd.DataFrame) -> pd.DataFrame:
    nonempty = [part for part in parts if len(part) > 0]
    if not nonempty:
        return fallback.iloc[0:0]
    return pd.concat(nonempty, axis=0).sort_values("__order__").reset_index(drop=True)


def _global_temporal_split(
    known_df: pd.DataFrame,
    eval_ratio: float,
    test_ratio: float,
    split_mode: str,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = _ordered_or_shuffled(known_df, split_mode, rng)
    if len(ordered) < 2:
        raise ValueError(f"Too few known samples for train/eval/test split: n={len(ordered)}")
    return _slice_train_eval_test(ordered, eval_ratio, test_ratio)


def _per_class_temporal_split(
    known_df: pd.DataFrame,
    known_space: List[str],
    eval_ratio: float,
    test_ratio: float,
    split_mode: str,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts, eval_parts, test_parts = [], [], []
    for cls in known_space:
        cls_df = known_df[known_df["label"] == cls]
        if len(cls_df) == 0:
            continue
        cls_ordered = _ordered_or_shuffled(cls_df, split_mode, rng)
        train_df, eval_df, test_df = _slice_train_eval_test(
            cls_ordered, eval_ratio, test_ratio)
        train_parts.append(train_df)
        eval_parts.append(eval_df)
        test_parts.append(test_df)
    return (
        _merge(train_parts, known_df),
        _merge(eval_parts, known_df),
        _merge(test_parts, known_df),
    )


def _class_dist(df: pd.DataFrame) -> Dict[str, int]:
    if len(df) == 0:
        return {}
    return {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}


def _limit_train_fraction(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"data.train_fraction must be in (0, 1], got {fraction}")
    if fraction >= 1.0 or len(df) == 0:
        return df
    parts = []
    for label in df["label"].unique():
        class_df = df[df["label"] == label].sort_values("__order__")
        keep = max(1, int(np.floor(len(class_df) * fraction)))
        parts.append(class_df.iloc[:keep])
    return pd.concat(parts, axis=0).sort_values("__order__").reset_index(drop=True)


def _check_required_classes(
    meta_train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    unknown_df: pd.DataFrame,
    known_space: List[str],
    granularity: str,
) -> None:
    issues: List[str] = []
    for split_name, split_df in (("meta_train", meta_train_df), ("meta_val(eval)", eval_df)):
        present = set(split_df["label"].unique().tolist()) if len(split_df) else set()
        missing = [cls for cls in known_space if cls not in present]
        if missing:
            issues.append(
                f"{split_name} missing known classes {missing}; dist={_class_dist(split_df)}"
            )
    if len(unknown_df) == 0:
        issues.append("unknown split is empty")

    if not issues:
        return
    msg = "LOAO split class sufficiency check failed: " + "; ".join(issues)
    if granularity == "global_temporal":
        logger.warning("%s | global_temporal can create class imbalance.", msg)
    else:
        raise ValueError(
            msg + ". Increase max_per_class/eval_ratio or adjust known/unknown classes."
        )


def build_loao(
    load_result: LoadResult,
    known_classes: List[str],
    unknown_class: str,
    include_benign: bool = True,
    eval_ratio: float = 0.2,
    test_ratio: float = 0.0,
    split_mode: str = "temporal",
    split_granularity: str = "per_class_temporal",
    max_per_class: int = 20000,
    train_fraction: float = 1.0,
    standardize: bool = True,
    seed: int = 42,
) -> LOAOResult:
    """Build raw-flow LOAO splits, then fit scaling only on meta-train."""
    granularity = str(split_granularity).lower()
    if granularity not in {"per_class_temporal", "global_temporal"}:
        raise ValueError(
            f"Unknown split_granularity={split_granularity}; "
            "expected per_class_temporal or global_temporal."
        )
    if split_mode == "random":
        logger.warning("data.split_mode=random is an ablation setting and breaks temporal order.")

    rng = np.random.default_rng(seed)
    df = load_result.df
    feat_cols = load_result.feature_columns
    available = set(df["label"].unique())

    if unknown_class not in available:
        raise ValueError(f"unknown_class {unknown_class!r} is not in labels {sorted(available)}")

    known_space = list(known_classes)
    if include_benign and "benign" in available and "benign" not in known_space:
        known_space = ["benign"] + known_space
    missing = [cls for cls in known_space if cls not in available]
    if missing:
        raise ValueError(f"known_classes contains labels missing from dataset: {missing}")
    if unknown_class in known_space:
        raise ValueError(f"unknown_class {unknown_class!r} cannot also be in known_classes")

    known_df = _subset(df, known_space, max_per_class, rng)
    unknown_df = _subset(df, [unknown_class], max_per_class, rng)

    if granularity == "global_temporal":
        logger.warning(
            "split_granularity=global_temporal can cause class imbalance in NIDS datasets."
        )
        meta_train_df, eval_df, test_df = _global_temporal_split(
            known_df, eval_ratio, test_ratio, split_mode, rng)
    else:
        meta_train_df, eval_df, test_df = _per_class_temporal_split(
            known_df, known_space, eval_ratio, test_ratio, split_mode, rng)

    meta_train_df = _limit_train_fraction(meta_train_df, float(train_fraction))
    _check_required_classes(meta_train_df, eval_df, unknown_df, known_space, granularity)

    standardizer = FeatureStandardizer(feature_columns=feat_cols, enabled=standardize).fit(
        meta_train_df)

    def to_split(split_df: pd.DataFrame) -> SplitArrays:
        if len(split_df) == 0:
            return SplitArrays(
                features=np.zeros((0, len(feat_cols)), dtype=np.float32),
                labels=np.array([], dtype=object),
                order=np.array([], dtype=float),
                row_id=np.array([], dtype=np.int64),
                segment_id=np.array([], dtype=np.int64),
            )
        row_id = (
            split_df["__row_id__"].to_numpy()
            if "__row_id__" in split_df.columns
            else np.arange(len(split_df), dtype=np.int64)
        )
        segment_id = (
            split_df["__segment_id__"].to_numpy()
            if "__segment_id__" in split_df.columns
            else np.zeros(len(split_df), dtype=np.int64)
        )
        source_file = (
            split_df["__source_file__"].to_numpy()
            if "__source_file__" in split_df.columns
            else None
        )
        return SplitArrays(
            features=standardizer.transform(split_df),
            labels=split_df["label"].to_numpy(),
            order=split_df["__order__"].to_numpy(),
            row_id=row_id,
            segment_id=segment_id,
            source_file=source_file,
        )

    class_to_idx = build_class_index(known_space)
    class_distribution = {
        "train": _class_dist(meta_train_df),
        "eval": _class_dist(eval_df),
        "test": _class_dist(test_df),
        "unknown": _class_dist(unknown_df),
    }

    logger.info(
        "LOAO: known=%s | unknown=%s | split_mode=%s | granularity=%s",
        known_space,
        unknown_class,
        split_mode,
        granularity,
    )
    logger.info("known-train fraction: %.6f", train_fraction)
    logger.info(
        "Raw samples train=%d eval=%d test=%d unknown=%d",
        len(meta_train_df),
        len(eval_df),
        len(test_df),
        len(unknown_df),
    )
    logger.info("===== split class distribution =====")
    for split_name, dist in class_distribution.items():
        logger.info("  %-8s: %s", split_name, dist)
    logger.info("====================================")

    return LOAOResult(
        feature_columns=feat_cols,
        known_classes=known_space,
        known_class_to_idx=class_to_idx,
        unknown_class=unknown_class,
        train=to_split(meta_train_df),
        eval=to_split(eval_df),
        test=to_split(test_df),
        unknown=to_split(unknown_df),
        standardizer=standardizer,
        class_distribution=class_distribution,
    )
