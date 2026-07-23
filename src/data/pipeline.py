"""Data pipeline assembly for offline temporal few-shot NIDS."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.config import Config
from ..utils.logger import get_logger
from .base_dataset import LoadResult
from .dataset import IntrusionDataset, merge_windowed_datasets
from .leakage import audit_pipeline_splits
from .loao import LOAOResult, SplitArrays, build_loao
from .preprocessing import build_class_index
from .registry import build_dataset
from .task_builder import (
    AdaptationTaskSampler,
    build_windowed_dataset,
    make_meta_sampler,
)
from .task_sampler import FewShotTaskSampler

logger = get_logger(__name__)

CACHE_SCHEMA_VERSION = "v3_lstm_meta_no_pretrain"


@dataclass
class DataBundle:
    """Assembled data context."""

    feature_dim: int
    window_size: int
    stride: int
    known_classes: List[str]
    known_class_to_idx: Dict[str, int]
    n_known: int
    unknown_class: str
    meta_train_dataset: IntrusionDataset
    meta_val_dataset: IntrusionDataset
    meta_train_sampler: FewShotTaskSampler
    meta_val_sampler: FewShotTaskSampler
    loao: LOAOResult
    _adapt_class_to_idx: Dict[str, int]
    _adapt_dataset: IntrusionDataset
    _adapt_val_dataset: IntrusionDataset
    _adapt_test_dataset: IntrusionDataset

    @property
    def adapt_val_dataset(self) -> IntrusionDataset:
        return self._adapt_val_dataset

    @property
    def adapt_test_dataset(self) -> IntrusionDataset:
        return self._adapt_test_dataset

    def make_adaptation_sampler(
        self,
        k_shot: int,
        q_query: int,
        mode: str = "binary",
        n_way: int = 2,
        seed: Optional[int] = None,
        disallow_support_query_overlap: bool = True,
        disallow_internal_overlap: bool = True,
        split: str = "test",
    ) -> AdaptationTaskSampler:
        if split == "val":
            dataset = self._adapt_val_dataset
        elif split == "test":
            dataset = self._adapt_test_dataset
        elif split == "all":
            dataset = self._adapt_dataset
        else:
            raise ValueError(f"Unknown adaptation split={split!r}; expected val|test|all")

        unknown_idx = self._adapt_class_to_idx[self.unknown_class]
        benign_idx = self._adapt_class_to_idx.get("benign")
        ref_indices = [
            idx for name, idx in self._adapt_class_to_idx.items()
            if name != self.unknown_class
        ]
        return AdaptationTaskSampler(
            dataset=dataset,
            unknown_idx=unknown_idx,
            ref_indices=ref_indices,
            mode=mode,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            benign_idx=benign_idx,
            seed=seed,
            disallow_support_query_overlap=disallow_support_query_overlap,
            disallow_internal_overlap=disallow_internal_overlap,
        )


def _cache_key(cfg: Config) -> Dict[str, object]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "data_name": str(cfg.data.get("name", "")),
        "data_root": os.path.abspath(str(cfg.data.get("root", ""))),
        "label_mapping": dict(cfg.data.get("label_mapping", {}) or {}),
    }


def cache_key(cfg: Config) -> Dict[str, object]:
    """Public, serialization-safe cache provenance used in run receipts."""
    return _cache_key(cfg)


def _cache_paths(cfg: Config) -> str:
    cache_dir = str(cfg.data.get("cache_dir", "outputs/cache"))
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.sha1(
        json.dumps(_cache_key(cfg), sort_keys=True).encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f"clean_{digest}.npz")


def _load_cache_npz(path: str) -> LoadResult:
    data = np.load(path, allow_pickle=True)
    feature_columns = list(data["feature_columns"])
    df = pd.DataFrame(data["features"], columns=feature_columns)
    df["label"] = data["labels"]
    df["__order__"] = data["order"]
    if "row_id" in data.files:
        df["__row_id__"] = data["row_id"]
    if "segment_id" in data.files:
        df["__segment_id__"] = data["segment_id"]
    if "source_file" in data.files:
        df["__source_file__"] = data["source_file"]

    cache_key: Optional[dict] = None
    if "cache_key_json" in data.files:
        try:
            cache_key = json.loads(str(data["cache_key_json"]))
        except (ValueError, TypeError):
            cache_key = None
    return LoadResult(
        df=df,
        feature_columns=feature_columns,
        stats={"source": "cache", "cache_key": cache_key},
    )


def _save_cache_npz(cache_file: str, result: LoadResult, cache_key: dict) -> None:
    df = result.df
    arrays = dict(
        features=df[result.feature_columns].to_numpy(dtype=np.float32),
        labels=df["label"].to_numpy().astype(object),
        order=df["__order__"].to_numpy(),
        feature_columns=np.array(result.feature_columns, dtype=object),
        cache_key_json=json.dumps(cache_key, sort_keys=True),
    )
    if "__row_id__" in df.columns:
        arrays["row_id"] = df["__row_id__"].to_numpy()
    if "__segment_id__" in df.columns:
        arrays["segment_id"] = df["__segment_id__"].to_numpy()
    if "__source_file__" in df.columns:
        arrays["source_file"] = df["__source_file__"].to_numpy().astype(object)
    np.savez(cache_file, **arrays)


def _load_clean(cfg: Config) -> LoadResult:
    cache_file = _cache_paths(cfg)
    use_cache = bool(cfg.data.get("use_cache", True))
    expected = _cache_key(cfg)
    if use_cache and os.path.exists(cache_file):
        cached = _load_cache_npz(cache_file)
        got = cached.stats.get("cache_key")
        if got == expected:
            logger.info("Using cleaned cache: %s", cache_file)
            return cached
        logger.warning("Cleaned cache key mismatch or missing; rebuilding cache.")

    label_map = dict(cfg.data.get("label_mapping", {}) or {})
    ds = build_dataset(str(cfg.data.name), str(cfg.data.root), label_mapping=label_map)
    result = ds.load()
    if use_cache:
        _save_cache_npz(cache_file, result, expected)
    return result


def _temporal_subsplit(split: SplitArrays, val_ratio: float) -> Tuple[SplitArrays, SplitArrays]:
    if len(split) == 0:
        return split, split

    seg = split.segment_id if split.segment_id is not None else np.zeros(len(split), dtype=np.int64)
    row = split.row_id if split.row_id is not None else np.arange(len(split), dtype=np.int64)
    src = split.source_file
    val_idx: List[int] = []
    test_idx: List[int] = []
    for segment in np.unique(seg):
        pos = np.where(seg == segment)[0]
        pos = pos[np.argsort(split.order[pos], kind="stable")]
        n = len(pos)
        n_val = max(1, int(round(n * val_ratio))) if n > 1 else n
        n_val = min(n_val, n - 1) if n > 1 else n
        val_idx.extend(pos[:n_val].tolist())
        test_idx.extend(pos[n_val:].tolist())

    def take(indices: List[int]) -> SplitArrays:
        idx_arr = np.array(sorted(indices), dtype=np.int64)
        return SplitArrays(
            features=split.features[idx_arr],
            labels=split.labels[idx_arr],
            order=split.order[idx_arr],
            row_id=row[idx_arr],
            segment_id=seg[idx_arr],
            source_file=src[idx_arr] if src is not None else None,
        )

    return take(val_idx), take(test_idx)


def _maybe_window(
    split: SplitArrays,
    class_to_idx: Dict[str, int],
    window_size: int,
    stride: int,
    cfg: Config,
) -> Optional[IntrusionDataset]:
    if len(split) < window_size:
        return None
    try:
        return build_windowed_dataset(split, class_to_idx, window_size, stride, cfg=cfg)
    except ValueError as exc:
        logger.warning("Skipping windowing for undersized split: %s", exc)
        return None


def _build_adapt_dataset(
    eval_split: SplitArrays,
    unknown_split: SplitArrays,
    class_to_idx: Dict[str, int],
    window_size: int,
    stride: int,
    cfg: Config,
) -> Optional[IntrusionDataset]:
    parts: List[IntrusionDataset] = []
    for split in (eval_split, unknown_split):
        ds = _maybe_window(split, class_to_idx, window_size, stride, cfg)
        if ds is not None and len(ds) > 0:
            parts.append(ds)
    if not parts:
        return None
    return merge_windowed_datasets(parts)


def _raw_class_counts(split: SplitArrays) -> Dict[str, int]:
    if len(split) == 0:
        return {}
    values, counts = np.unique(split.labels, return_counts=True)
    return {str(label): int(count) for label, count in zip(values, counts)}


def _window_class_counts(dataset: Optional[IntrusionDataset]) -> Dict[int, int]:
    if dataset is None:
        return {}
    return {int(cls): int(len(indices)) for cls, indices in dataset.class_to_indices.items()}


def _log_split_audit(
    name: str,
    split: SplitArrays,
    dataset: Optional[IntrusionDataset],
) -> None:
    logger.info(
        "Split audit [%s]: raw_rows=%d raw_class_counts=%s windows=%d "
        "window_class_counts=%s",
        name,
        len(split),
        _raw_class_counts(split),
        len(dataset) if dataset is not None else 0,
        _window_class_counts(dataset),
    )


def build_pipeline(cfg: Config, seed: int = 42) -> DataBundle:
    dcfg = cfg.data
    window_size = int(dcfg.window_size)
    stride = int(dcfg.get("stride", max(1, window_size // 2)))
    disallow_overlap = bool(dcfg.get("disallow_support_query_overlap", True))
    disallow_internal_overlap = bool(dcfg.get("disallow_internal_overlap", True))
    strict_adapt_test = bool(dcfg.get("strict_adapt_test", False))

    if "unknown_class" not in dcfg and "unknown_classes" in dcfg:
        unknowns = list(dcfg.unknown_classes)
        if len(unknowns) == 1:
            dcfg.unknown_class = unknowns[0]
        else:
            raise ValueError("Use scripts to iterate over multiple unknown classes.")

    load_result = _load_clean(cfg)
    loao = build_loao(
        load_result,
        known_classes=list(dcfg.known_classes),
        unknown_class=str(dcfg.unknown_class),
        include_benign=bool(dcfg.get("include_benign", True)),
        eval_ratio=float(dcfg.get("eval_ratio", 0.2)),
        test_ratio=float(dcfg.get("test_ratio", 0.0)),
        split_mode=str(dcfg.get("split_mode", "temporal")),
        split_granularity=str(dcfg.get("split_granularity", "per_class_temporal")),
        max_per_class=int(dcfg.get("max_per_class", 20000)),
        train_fraction=float(dcfg.get("train_fraction", 1.0)),
        standardize=bool(dcfg.get("standardize", True)),
        seed=seed,
    )

    feature_dim = len(loao.feature_columns)
    known_c2i = loao.known_class_to_idx
    meta_train_ds = build_windowed_dataset(loao.train, known_c2i, window_size, stride, cfg=cfg)
    meta_val_ds = build_windowed_dataset(loao.eval, known_c2i, window_size, stride, cfg=cfg)

    allowed = list(known_c2i.values())
    mode = str(dcfg.get("task_mode", "nway")).lower()
    meta_n_way = 2 if mode == "binary" else int(dcfg.n_way)
    binary_pair_mode = str(dcfg.get("binary_pair_mode", "any_two")).lower()
    benign_class = known_c2i.get("benign")
    meta_train_sampler = make_meta_sampler(
        meta_train_ds,
        allowed,
        n_way=meta_n_way,
        k_shot=int(dcfg.k_shot),
        q_query=int(dcfg.q_query),
        seed=seed,
        disallow_support_query_overlap=disallow_overlap,
        disallow_internal_overlap=disallow_internal_overlap,
        binary_pair_mode=binary_pair_mode,
        benign_class=benign_class,
    )
    meta_val_sampler = make_meta_sampler(
        meta_val_ds,
        allowed,
        n_way=meta_n_way,
        k_shot=int(dcfg.k_shot),
        q_query=int(dcfg.q_query),
        seed=seed + 1,
        disallow_support_query_overlap=disallow_overlap,
        disallow_internal_overlap=disallow_internal_overlap,
        binary_pair_mode=binary_pair_mode,
        benign_class=benign_class,
    )

    if str(dcfg.get("window_label_strategy", "last")) == "any_attack":
        adapt_c2i = {"benign": 0, loao.unknown_class: 1, "attack": 1}
    else:
        adapt_c2i = build_class_index(loao.known_classes + [loao.unknown_class])

    adapt_val_ratio = float(dcfg.get("adapt_val_ratio", 0.5))
    eval_val_sp, eval_test_sp = _temporal_subsplit(loao.eval, adapt_val_ratio)
    unk_val_sp, unk_test_sp = _temporal_subsplit(loao.unknown, adapt_val_ratio)
    if len(loao.test) < window_size:
        if strict_adapt_test:
            raise ValueError(
                "strict_adapt_test=true requires an independent loao.test split "
                f"with at least window_size={window_size} raw rows; got {len(loao.test)}. "
                "Set data.test_ratio > 0 or disable strict_adapt_test only for ablations."
            )
        logger.warning(
            "loao.test has %d raw rows < window_size=%d; falling back to eval split "
            "for adapt_test because strict_adapt_test=false.",
            len(loao.test),
            window_size,
        )
        eval_test_source = eval_test_sp
    else:
        eval_test_source = loao.test

    adapt_val_ds = _build_adapt_dataset(
        eval_val_sp, unk_val_sp, adapt_c2i, window_size, stride, cfg)
    adapt_test_ds = _build_adapt_dataset(
        eval_test_source, unk_test_sp, adapt_c2i, window_size, stride, cfg)
    adapt_full_ds = _build_adapt_dataset(
        loao.eval, loao.unknown, adapt_c2i, window_size, stride, cfg)
    if adapt_val_ds is None or adapt_test_ds is None:
        raise ValueError(
            "Not enough adaptation data to build disjoint val/test windows. "
            "Reduce window_size/stride/k_shot/q_query or increase max_per_class."
        )
    unknown_idx = adapt_c2i[loao.unknown_class]
    val_unknown_windows = len(adapt_val_ds.class_to_indices.get(unknown_idx, []))
    test_unknown_windows = len(adapt_test_ds.class_to_indices.get(unknown_idx, []))
    logger.info(
        "Unknown window audit: class=%s idx=%d val_windows=%d test_windows=%d "
        "required_K_plus_Q=%d",
        loao.unknown_class,
        unknown_idx,
        val_unknown_windows,
        test_unknown_windows,
        int(dcfg.k_shot) + int(dcfg.q_query),
    )

    audit_pipeline_splits(loao)
    _log_split_audit("meta_train", loao.train, meta_train_ds)
    _log_split_audit("meta_val", loao.eval, meta_val_ds)
    _log_split_audit("adapt_val_known", eval_val_sp, None)
    _log_split_audit("adapt_val_unknown", unk_val_sp, None)
    _log_split_audit("adapt_test_known", eval_test_source, None)
    _log_split_audit("adapt_test_unknown", unk_test_sp, None)
    _log_split_audit("adapt_val", SplitArrays(
        features=np.zeros((0, feature_dim), dtype=np.float32),
        labels=np.array([], dtype=object),
        order=np.array([], dtype=float),
    ), adapt_val_ds)
    _log_split_audit("adapt_test", SplitArrays(
        features=np.zeros((0, feature_dim), dtype=np.float32),
        labels=np.array([], dtype=object),
        order=np.array([], dtype=float),
    ), adapt_test_ds)
    logger.info(
        "DataBundle: F=%d L=%d stride=%d mode=%s label=%s | meta_train=%d "
        "meta_val=%d adapt_val=%d adapt_test=%d windows",
        feature_dim,
        window_size,
        stride,
        dcfg.get("windowing_mode", "temporal"),
        dcfg.get("window_label_strategy", "last"),
        len(meta_train_ds),
        len(meta_val_ds),
        len(adapt_val_ds),
        len(adapt_test_ds),
    )

    return DataBundle(
        feature_dim=feature_dim,
        window_size=window_size,
        stride=stride,
        known_classes=loao.known_classes,
        known_class_to_idx=known_c2i,
        n_known=len(known_c2i),
        unknown_class=loao.unknown_class,
        meta_train_dataset=meta_train_ds,
        meta_val_dataset=meta_val_ds,
        meta_train_sampler=meta_train_sampler,
        meta_val_sampler=meta_val_sampler,
        loao=loao,
        _adapt_class_to_idx=adapt_c2i,
        _adapt_dataset=adapt_full_ds if adapt_full_ds is not None else adapt_test_ds,
        _adapt_val_dataset=adapt_val_ds,
        _adapt_test_dataset=adapt_test_ds,
    )
