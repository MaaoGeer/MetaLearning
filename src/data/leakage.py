"""Leakage audit helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger
from .dataset import IntrusionDataset
from .loao import LOAOResult, SplitArrays
from .windowing import raw_ranges_overlap

logger = get_logger(__name__)


def _seg(ds: IntrusionDataset, i: int) -> Optional[int]:
    if ds.segment_id is None:
        return None
    return int(ds.segment_id[i])


def windows_overlap_between(
    ds_a: IntrusionDataset,
    idx_a: int,
    ds_b: IntrusionDataset,
    idx_b: int,
) -> bool:
    """Return whether two windows cover at least one identical raw row."""
    seg_a, seg_b = _seg(ds_a, idx_a), _seg(ds_b, idx_b)
    if seg_a is not None and seg_b is not None and seg_a != seg_b:
        return False

    if ds_a.row_ids is not None and ds_b.row_ids is not None:
        rows_a = np.asarray(ds_a.row_ids[idx_a], dtype=np.int64)
        rows_b = np.asarray(ds_b.row_ids[idx_b], dtype=np.int64)
        return bool(np.intersect1d(rows_a, rows_b, assume_unique=False).size)

    if (
        ds_a.row_start is not None and ds_a.row_end is not None
        and ds_b.row_start is not None and ds_b.row_end is not None
    ):
        return raw_ranges_overlap(
            int(ds_a.row_start[idx_a]), int(ds_a.row_end[idx_a]),
            int(ds_b.row_start[idx_b]), int(ds_b.row_end[idx_b]),
        )

    if ds_a is ds_b and ds_a.raw_start is not None and ds_a.raw_end is not None:
        return raw_ranges_overlap(
            int(ds_a.raw_start[idx_a]), int(ds_a.raw_end[idx_a]),
            int(ds_b.raw_start[idx_b]), int(ds_b.raw_end[idx_b]),
        )
    return False


def window_indices_overlap(ds: IntrusionDataset, i: int, j: int) -> bool:
    if ds.raw_start is not None and ds.raw_end is not None:
        return raw_ranges_overlap(
            int(ds.raw_start[i]), int(ds.raw_end[i]),
            int(ds.raw_start[j]), int(ds.raw_end[j]),
        )
    return windows_overlap_between(ds, i, ds, j)


def check_support_query_overlap(
    support_wids: Iterable[int],
    query_wids: Iterable[int],
    dataset: IntrusionDataset,
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for support_id in support_wids:
        for query_id in query_wids:
            if window_indices_overlap(dataset, support_id, query_id):
                pairs.append((support_id, query_id))
    return pairs


def datasets_window_row_disjoint(ds_a: IntrusionDataset, ds_b: IntrusionDataset) -> bool:
    for i in range(len(ds_a)):
        for j in range(len(ds_b)):
            if windows_overlap_between(ds_a, i, ds_b, j):
                return False
    return True


def raw_sample_sets_overlap(a: SplitArrays, b: SplitArrays) -> bool:
    if len(a.order) == 0 or len(b.order) == 0:
        return False
    if a.row_id is not None and b.row_id is not None:
        return bool(set(a.row_id.tolist()) & set(b.row_id.tolist()))
    set_a = set(zip(a.order.tolist(), a.labels.tolist()))
    set_b = set(zip(b.order.tolist(), b.labels.tolist()))
    return bool(set_a & set_b)


def check_unknown_not_in_meta_train(loao: LOAOResult) -> bool:
    train_labels = set(loao.train.labels.tolist())
    if loao.unknown_class in train_labels:
        logger.error("Leakage: unknown_class %s appears in meta-train.", loao.unknown_class)
        return False
    return True


def audit_pipeline_splits(loao: LOAOResult) -> Dict[str, bool]:
    """Audit raw split disjointness and unknown-label isolation."""
    results: Dict[str, bool] = {
        "unknown_not_in_meta_train": check_unknown_not_in_meta_train(loao),
        "train_eval_disjoint": not raw_sample_sets_overlap(loao.train, loao.eval),
        "train_test_disjoint": not raw_sample_sets_overlap(loao.train, loao.test),
        "eval_test_disjoint": not raw_sample_sets_overlap(loao.eval, loao.test),
        "train_unknown_disjoint": not raw_sample_sets_overlap(loao.train, loao.unknown),
        "eval_unknown_disjoint": not raw_sample_sets_overlap(loao.eval, loao.unknown),
        "test_unknown_disjoint": not raw_sample_sets_overlap(loao.test, loao.unknown),
    }

    logger.info("===== leakage audit =====")
    for key, ok in results.items():
        level = logger.info if ok else logger.warning
        level("  %s: %s", key, "PASS" if ok else "FAIL/RISK")
    logger.info("=========================")
    return results
