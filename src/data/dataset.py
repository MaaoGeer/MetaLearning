"""时序入侵检测数据集封装 (含窗口元数据)。"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.logger import get_logger
from .windowing import WindowBuildResult, build_windows

logger = get_logger(__name__)


class IntrusionDataset(Dataset):
    """时序入侵检测窗口数据集。

    属性:
        features: [N, seq_len, feature_dim]
        labels:   [N] 全局类别索引
        raw_start/raw_end:   窗口在所属 split 排序数组中的局部位置(含); 合并数据集时会加 offset
        row_start/row_end:   窗口覆盖的全局原始行号(含), 跨 split/dataset 唯一
        order_start/order_end: 窗口覆盖的时间序范围(含)
        segment_id:          窗口所属 segment(同一 CSV/session)
        class_to_indices: 类 → 窗口下标列表
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 sequence_length: int,
                 raw_start: Optional[np.ndarray] = None,
                 raw_end: Optional[np.ndarray] = None,
                 row_start: Optional[np.ndarray] = None,
                 row_end: Optional[np.ndarray] = None,
                 row_ids: Optional[np.ndarray] = None,
                 order_start: Optional[np.ndarray] = None,
                 order_end: Optional[np.ndarray] = None,
                 segment_id: Optional[np.ndarray] = None) -> None:
        assert features.ndim == 3, "features 应为 [N, seq_len, feat]"
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.sequence_length = sequence_length
        self.feature_dim = features.shape[2]
        self.raw_start = raw_start
        self.raw_end = raw_end
        self.row_start = row_start
        self.row_end = row_end
        self.row_ids = row_ids
        self.order_start = order_start
        self.order_end = order_end
        self.segment_id = segment_id
        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, y in enumerate(labels.tolist()):
            self.class_to_indices.setdefault(int(y), []).append(idx)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @property
    def classes(self) -> List[int]:
        return sorted(self.class_to_indices.keys())

    @classmethod
    def from_window_result(cls, result: WindowBuildResult,
                           sequence_length: int) -> "IntrusionDataset":
        return cls(result.features, result.labels, sequence_length,
                   raw_start=result.raw_start, raw_end=result.raw_end,
                   row_start=result.row_start, row_end=result.row_end,
                   row_ids=result.row_ids,
                   order_start=result.order_start, order_end=result.order_end,
                   segment_id=result.segment_id)

    @staticmethod
    def build_sequences(
        flat_features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int,
        stride: Optional[int] = None,
        order: Optional[np.ndarray] = None,
        windowing_mode: Optional[str] = None,
        label_strategy: Optional[str] = None,
        label_names: Optional[np.ndarray] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> tuple:
        """构建窗口 (旧 API)。

        注意: 主管线已改用 task_builder.build_windowed_dataset(默认 temporal/last)。
        此处仅为兼容旧调用; 未显式指定 windowing_mode 时回退 classwise 并 warning。
        """
        if windowing_mode is None:
            logger.warning(
                "IntrusionDataset.build_sequences() 未显式指定 windowing_mode, "
                "回退到旧 classwise 逻辑(非真实连续流量)。主管线请使用 "
                "task_builder.build_windowed_dataset(temporal/last)。")
            windowing_mode = "classwise"
            if label_strategy is None:
                label_strategy = "classwise"
        if label_strategy is None:
            label_strategy = "last"
        if stride is None:
            stride = sequence_length
        if order is None:
            order = np.arange(len(labels), dtype=np.float64)
        result = build_windows(
            flat_features, labels, order, sequence_length, stride,
            windowing_mode=windowing_mode,
            label_strategy=label_strategy,
            label_names=label_names,
            class_to_idx=class_to_idx,
        )
        return result.features, result.labels


def merge_windowed_datasets(datasets: List[IntrusionDataset]) -> IntrusionDataset:
    """安全合并多个已窗口化的 IntrusionDataset。

    - features / labels concat。
    - row_start/row_end/order_start/order_end/segment_id 直接拼接(全局唯一, 不冲突)。
    - raw_start/raw_end 为各 split 内局部位置, 合并时给后续 dataset 加 offset, 避免误判重叠。
    - class_to_indices 由 __init__ 重新构建。
    """
    datasets = [d for d in datasets if len(d) > 0]
    if not datasets:
        raise ValueError("merge_windowed_datasets: 没有可合并的非空数据集。")
    if len(datasets) == 1:
        return datasets[0]

    seq_len = datasets[0].sequence_length
    feats = np.concatenate([d.features.numpy() for d in datasets], axis=0)
    labels = np.concatenate([d.labels.numpy() for d in datasets], axis=0)

    def _cat(attr: str, dtype) -> Optional[np.ndarray]:
        if any(getattr(d, attr) is None for d in datasets):
            return None
        return np.concatenate([np.asarray(getattr(d, attr), dtype=dtype) for d in datasets], axis=0)

    row_start = _cat("row_start", np.int64)
    row_end = _cat("row_end", np.int64)
    row_ids = _cat("row_ids", np.int64)
    order_start = _cat("order_start", np.float64)
    order_end = _cat("order_end", np.float64)

    # segment_id: 给每个 dataset 分配独立命名空间, 避免不同来源 segment 撞号。
    seg_parts: List[np.ndarray] = []
    seg_offset = 0
    seg_ok = all(d.segment_id is not None for d in datasets)
    for d in datasets:
        if seg_ok:
            seg = np.asarray(d.segment_id, dtype=np.int64) + seg_offset
            seg_parts.append(seg)
            seg_offset = int(seg.max()) + 1 if len(seg) else seg_offset
    segment_id = np.concatenate(seg_parts, axis=0) if seg_ok else None

    # raw_start/raw_end: 局部位置, 加 offset 保证跨 dataset 全局唯一(不被误判重叠)。
    raw_parts_s: List[np.ndarray] = []
    raw_parts_e: List[np.ndarray] = []
    raw_ok = all(d.raw_start is not None and d.raw_end is not None for d in datasets)
    raw_offset = 0
    for d in datasets:
        if raw_ok:
            rs = np.asarray(d.raw_start, dtype=np.int64) + raw_offset
            re = np.asarray(d.raw_end, dtype=np.int64) + raw_offset
            raw_parts_s.append(rs)
            raw_parts_e.append(re)
            raw_offset = int(re.max()) + 1 if len(re) else raw_offset
    raw_start = np.concatenate(raw_parts_s, axis=0) if raw_ok else None
    raw_end = np.concatenate(raw_parts_e, axis=0) if raw_ok else None

    return IntrusionDataset(
        feats, labels, seq_len,
        raw_start=raw_start, raw_end=raw_end,
        row_start=row_start, row_end=row_end,
        row_ids=row_ids,
        order_start=order_start, order_end=order_end,
        segment_id=segment_id)
