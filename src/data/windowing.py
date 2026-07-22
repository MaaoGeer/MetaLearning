"""时序窗口构建: temporal (因果滑窗) 与 classwise (旧逻辑, 消融用)。

Temporal 模式 (默认):
    按全局时间序排列原始样本, 在每个 segment(同一 CSV/capture session) 内构建因果滑窗:
        X_t = [x_{t-L+1}, ..., x_t]
        y_t 由 window_label_strategy 决定 (默认 last = 末样本标签)
    不使用未来样本; 窗口不跨 segment 边界; 模拟 offline temporal few-shot / adaptation 评估。

Classwise 模式 (消融):
    按类别分组后类内滑窗, 窗口标签恒为该类 (旧行为)。

窗口元数据 (用于泄漏检查):
    raw_start/raw_end:   窗口在所属 split 排序后数组中的局部位置(含), 兼容旧代码。
    row_start/row_end:   窗口覆盖的全局原始行号(含), 跨 split/dataset 唯一。
    order_start/order_end: 窗口覆盖的时间序范围(含)。
    segment_id:          窗口所属 segment(同一 CSV/session)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)

BENIGN_NAMES = frozenset({"benign", "normal"})
TIE_BREAK_CHOICES = frozenset({"last", "smallest", "benign", "error"})


@dataclass
class WindowBuildResult:
    """窗口化结果 + 元数据(用于泄漏检查)。"""

    features: np.ndarray          # [M, L, F]
    labels: np.ndarray            # [M] int64 全局类索引
    raw_start: np.ndarray         # [M] 窗口在 split 排序数组中的局部起始位置(含)
    raw_end: np.ndarray           # [M] 窗口在 split 排序数组中的局部结束位置(含)
    row_start: np.ndarray         # [M] 全局原始行号起始(含)
    row_end: np.ndarray           # [M] 全局原始行号结束(含)
    order_start: np.ndarray       # [M] 时间序起始(含)
    order_end: np.ndarray         # [M] 时间序结束(含)
    segment_id: np.ndarray        # [M] 窗口所属 segment
    row_ids: Optional[np.ndarray] = None  # [M, L] 每个窗口实际覆盖的原始行号
    label_names: Optional[np.ndarray] = None  # [M] str, 可选


def _benign_indices(class_to_idx: Dict[str, int]) -> set:
    return {idx for name, idx in class_to_idx.items() if str(name).lower() in BENIGN_NAMES}


def _validate_any_attack(class_to_idx: Dict[str, int]) -> None:
    """any_attack 仅适用于二分类标签空间, 或显式提供 'attack' 别名。"""
    if "attack" in class_to_idx:
        return
    benign = _benign_indices(class_to_idx)
    non_benign = {idx for name, idx in class_to_idx.items()
                  if idx not in benign and str(name).lower() != "attack"}
    if len(non_benign) > 1:
        raise ValueError(
            "window_label_strategy='any_attack' 需要二分类标签空间或显式 'attack' 别名; "
            f"当前非 benign 类有 {len(non_benign)} 个 ({sorted(non_benign)})。"
            " 请用 task_mode=binary, 或在 class_to_idx 中提供 'attack' 映射。")


def _resolve_window_label(
    win_label_idx: np.ndarray,
    win_label_names: Optional[np.ndarray],
    strategy: str,
    class_to_idx: Dict[str, int],
    majority_tie_break: str = "last",
) -> int:
    """根据策略为单个窗口分配标签(返回全局类索引)。"""
    strategy = strategy.lower()
    if strategy == "last":
        return int(win_label_idx[-1])

    if strategy == "majority":
        vals, counts = np.unique(win_label_idx, return_counts=True)
        max_count = counts.max()
        tied = [int(v) for v, c in zip(vals, counts) if c == max_count]
        if len(tied) == 1:
            return tied[0]
        tb = majority_tie_break.lower()
        if tb == "smallest":
            return min(tied)
        if tb == "benign":
            benign = _benign_indices(class_to_idx)
            for t in tied:
                if t in benign:
                    return t
            # 平票且无 benign: 退化为 last
            for v in reversed(win_label_idx.tolist()):
                if int(v) in tied:
                    return int(v)
            return tied[0]
        if tb == "error":
            raise ValueError(f"majority 平票且 majority_tie_break='error': 候选类 {tied}")
        # 默认 "last": 取窗口内最后一个属于平票集合的样本标签
        tied_set = set(tied)
        for v in reversed(win_label_idx.tolist()):
            if int(v) in tied_set:
                return int(v)
        return tied[0]

    if strategy == "any_attack":
        benign_idx = class_to_idx.get("benign", class_to_idx.get("normal", 0))
        attack_idx = class_to_idx.get("attack", 1)
        if win_label_names is not None:
            for name in win_label_names:
                if str(name).lower() not in BENIGN_NAMES:
                    return attack_idx
            return benign_idx
        benign_set = _benign_indices(class_to_idx) or {benign_idx}
        for v in win_label_idx:
            if int(v) not in benign_set:
                return attack_idx
        return benign_idx

    if strategy == "classwise":
        return int(win_label_idx[0])

    raise ValueError(f"未知 window_label_strategy: {strategy}")


def build_temporal_windows(
    flat_features: np.ndarray,
    labels_idx: np.ndarray,
    order: np.ndarray,
    window_size: int,
    stride: int,
    label_strategy: str = "last",
    label_names: Optional[np.ndarray] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
    row_id: Optional[np.ndarray] = None,
    segment_id: Optional[np.ndarray] = None,
    majority_tie_break: str = "last",
) -> WindowBuildResult:
    """按全局时间序在每个 segment 内构建因果滑窗(窗口不跨 segment)。"""
    if class_to_idx is None:
        class_to_idx = {str(i): i for i in np.unique(labels_idx)}
    if label_strategy.lower() == "any_attack":
        _validate_any_attack(class_to_idx)

    n = flat_features.shape[0]
    if window_size > n:
        raise ValueError(f"样本数 {n} < window_size {window_size}, 无法构建 temporal 窗口。")

    if row_id is None:
        row_id = np.arange(n, dtype=np.int64)
    else:
        row_id = np.asarray(row_id)
    if segment_id is None:
        segment_id = np.zeros(n, dtype=np.int64)
    else:
        segment_id = np.asarray(segment_id)

    # 先按 (segment, order) 稳定排序: segment 内保持时间序, 窗口绝不跨 segment。
    sort_idx = np.lexsort((order, segment_id))
    feats = flat_features[sort_idx]
    labs = labels_idx[sort_idx]
    names = label_names[sort_idx] if label_names is not None else None
    rows = row_id[sort_idx]
    segs = segment_id[sort_idx]
    orders = np.asarray(order)[sort_idx]

    seqs: List[np.ndarray] = []
    seq_labels: List[int] = []
    raw_starts: List[int] = []
    raw_ends: List[int] = []
    row_starts: List[int] = []
    row_ends: List[int] = []
    window_row_ids: List[np.ndarray] = []
    order_starts: List[float] = []
    order_ends: List[float] = []
    seg_ids: List[int] = []
    out_names: List[str] = []

    # 在每个 segment 的连续块内独立滑窗。
    for seg in np.unique(segs):
        seg_positions = np.where(segs == seg)[0]
        seg_len = len(seg_positions)
        if seg_len < window_size:
            continue
        for local in range(0, seg_len - window_size + 1, stride):
            start = int(seg_positions[local])
            end = int(seg_positions[local + window_size - 1])
            # 该 segment 内连续(seg_positions 为排序后连续区间), start..end 即窗口。
            win_x = feats[start:end + 1]
            win_lab = labs[start:end + 1]
            win_names = names[start:end + 1] if names is not None else None
            y = _resolve_window_label(win_lab, win_names, label_strategy,
                                      class_to_idx, majority_tie_break)
            seqs.append(win_x)
            seq_labels.append(y)
            raw_starts.append(start)
            raw_ends.append(end)
            row_starts.append(int(rows[start:end + 1].min()))
            row_ends.append(int(rows[start:end + 1].max()))
            window_row_ids.append(np.asarray(rows[start:end + 1], dtype=np.int64))
            order_starts.append(float(orders[start]))
            order_ends.append(float(orders[end]))
            seg_ids.append(int(seg))
            if names is not None:
                out_names.append(str(names[end]))

    if not seqs:
        raise ValueError("temporal 滑窗未生成任何窗口, 请检查 stride/window_size/segment 长度。")

    return WindowBuildResult(
        features=np.stack(seqs, axis=0).astype(np.float32),
        labels=np.array(seq_labels, dtype=np.int64),
        raw_start=np.array(raw_starts, dtype=np.int64),
        raw_end=np.array(raw_ends, dtype=np.int64),
        row_start=np.array(row_starts, dtype=np.int64),
        row_end=np.array(row_ends, dtype=np.int64),
        row_ids=np.stack(window_row_ids, axis=0),
        order_start=np.array(order_starts, dtype=np.float64),
        order_end=np.array(order_ends, dtype=np.float64),
        segment_id=np.array(seg_ids, dtype=np.int64),
        label_names=np.array(out_names, dtype=object) if out_names else None,
    )


def build_classwise_windows(
    flat_features: np.ndarray,
    labels_idx: np.ndarray,
    window_size: int,
    stride: int,
    label_names: Optional[np.ndarray] = None,
    row_id: Optional[np.ndarray] = None,
    segment_id: Optional[np.ndarray] = None,
) -> WindowBuildResult:
    """按类内滑窗 (旧逻辑, 仅消融)。"""
    n = flat_features.shape[0]
    if row_id is None:
        row_id = np.arange(n, dtype=np.int64)
    else:
        row_id = np.asarray(row_id)
    if segment_id is None:
        segment_id = np.zeros(n, dtype=np.int64)
    else:
        segment_id = np.asarray(segment_id)

    seqs: List[np.ndarray] = []
    seq_labels: List[int] = []
    raw_starts: List[int] = []
    raw_ends: List[int] = []
    row_starts: List[int] = []
    row_ends: List[int] = []
    window_row_ids: List[np.ndarray] = []
    seg_ids: List[int] = []
    out_names: List[str] = []

    for cls in np.unique(labels_idx):
        cls_mask = labels_idx == cls
        cls_idx = np.where(cls_mask)[0]
        cls_feats = flat_features[cls_idx]
        m = cls_feats.shape[0]
        pos = 0
        while pos + window_size <= m:
            win_x = cls_feats[pos:pos + window_size]
            raw_s = int(cls_idx[pos])
            raw_e = int(cls_idx[pos + window_size - 1])
            seqs.append(win_x)
            seq_labels.append(int(cls))
            raw_starts.append(raw_s)
            raw_ends.append(raw_e)
            covered = cls_idx[pos:pos + window_size]
            row_starts.append(int(row_id[covered].min()))
            row_ends.append(int(row_id[covered].max()))
            window_row_ids.append(np.asarray(row_id[covered], dtype=np.int64))
            seg_ids.append(int(segment_id[raw_e]))
            if label_names is not None:
                out_names.append(str(label_names[raw_e]))
            pos += stride

    if not seqs:
        raise ValueError(f"classwise 滑窗失败: 每类需 >= window_size({window_size})。")

    raw_start_arr = np.array(raw_starts, dtype=np.int64)
    raw_end_arr = np.array(raw_ends, dtype=np.int64)
    return WindowBuildResult(
        features=np.stack(seqs, axis=0).astype(np.float32),
        labels=np.array(seq_labels, dtype=np.int64),
        raw_start=raw_start_arr,
        raw_end=raw_end_arr,
        row_start=np.array(row_starts, dtype=np.int64),
        row_end=np.array(row_ends, dtype=np.int64),
        row_ids=np.stack(window_row_ids, axis=0),
        order_start=raw_start_arr.astype(np.float64),
        order_end=raw_end_arr.astype(np.float64),
        segment_id=np.array(seg_ids, dtype=np.int64),
        label_names=np.array(out_names, dtype=object) if out_names else None,
    )


def build_windows(
    split_features: np.ndarray,
    labels_idx: np.ndarray,
    order: np.ndarray,
    window_size: int,
    stride: int,
    windowing_mode: str = "temporal",
    label_strategy: str = "last",
    label_names: Optional[np.ndarray] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
    row_id: Optional[np.ndarray] = None,
    segment_id: Optional[np.ndarray] = None,
    majority_tie_break: str = "last",
) -> WindowBuildResult:
    """统一入口。"""
    mode = windowing_mode.lower()
    if mode == "temporal":
        return build_temporal_windows(
            split_features, labels_idx, order, window_size, stride,
            label_strategy=label_strategy, label_names=label_names,
            class_to_idx=class_to_idx, row_id=row_id, segment_id=segment_id,
            majority_tie_break=majority_tie_break)
    if mode == "classwise":
        logger.warning("使用 classwise 滑窗(消融模式), 非真实连续流量。")
        return build_classwise_windows(
            split_features, labels_idx, window_size, stride,
            label_names=label_names, row_id=row_id, segment_id=segment_id)
    raise ValueError(f"未知 windowing_mode: {windowing_mode}")


def raw_ranges_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    """两窗口在(局部位置或行号)区间上是否重叠。"""
    return not (e1 < s2 or e2 < s1)
