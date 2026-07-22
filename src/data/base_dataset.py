"""统一数据集接口 BaseDataset。

设计目的:
    为多个入侵检测数据集(CICIDS2017 / CSE-CIC-IDS2018 / UNSW-NB15 / TON-IoT)提供
    统一的"发现→读取→清洗→标签规范化"流程, 使配置 `data.name` 即可切换数据集。

子类只需实现:
    - file_glob():            CSV 发现的相对通配(默认 **/*.csv)
    - label_column_candidates(): 可能的标签列名
    - timestamp_column_candidates(): 可能的时间戳列名(用于时序窗排序; 没有则用行序)
    - normalize_label(raw):   把数据集原始攻击名映射到统一 taxonomy

通用清洗在基类完成并输出统计日志:
    删全空列 / 去重列 / 选数值特征 / Inf±→NaN / NaN填补 / 去重样本 / 标签规范化。
"""

from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# 统一攻击 taxonomy（各数据集子类把原始标签映射到这些规范名）。
CANONICAL_CLASSES = [
    "benign", "normal", "dos", "ddos", "botnet", "portscan",
    "webattack", "bruteforce", "infiltration", "heartbleed",
    "reconnaissance", "exploits", "fuzzers", "generic", "analysis",
    "backdoor", "shellcode", "worms", "other",
]


@dataclass
class LoadResult:
    """数据加载结果。"""

    df: pd.DataFrame                       # 含数值特征列 + 'label'(规范名) + '__order__'
    feature_columns: List[str]
    label_column: str = "label"
    order_column: str = "__order__"
    stats: Dict[str, object] = field(default_factory=dict)


class BaseDataset(ABC):
    """入侵检测数据集统一接口。"""

    def __init__(self, root: str) -> None:
        self.root = root

    # ----------------------- 子类需实现 ----------------------- #
    @abstractmethod
    def file_glob(self) -> str:
        """相对 root 的递归通配, 例如 '**/*.csv'。"""

    @abstractmethod
    def label_column_candidates(self) -> List[str]:
        ...

    @abstractmethod
    def timestamp_column_candidates(self) -> List[str]:
        ...

    @abstractmethod
    def normalize_label(self, raw: str) -> Optional[str]:
        """原始标签 → 统一 taxonomy 名; 返回 None 表示丢弃该类。"""

    # ----------------------- 通用实现 ----------------------- #
    def discover_files(self) -> List[str]:
        """递归发现所有 CSV(不假设已合并为单文件)。"""
        pattern = os.path.join(self.root, self.file_glob())
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            raise FileNotFoundError(
                f"未在 {self.root} 下发现匹配 {self.file_glob()} 的 CSV 文件。")
        logger.info("发现 %d 个 CSV 文件 (root=%s)。", len(files), self.root)
        for f in files:
            logger.info("  - %s", os.path.relpath(f, self.root))
        return files

    def _read_concat(self, files: List[str]) -> pd.DataFrame:
        """逐文件读取并纵向拼接, 统一列名(strip)。

        每个 CSV 视为一个 segment(同一 capture session), 写入:
            __source_file__: 相对 root 的文件名
            __segment_id__:  每个 CSV 一个递增 segment id
            __row_id__:      全局原始行号(拼接顺序)
        以便后续 temporal windowing 禁止跨 segment 建窗, 并做跨 split 泄漏审计。
        """
        frames = []
        total_rows = 0
        for seg_id, f in enumerate(files):
            df = pd.read_csv(f, low_memory=False, encoding="latin-1")
            df.columns = [str(c).strip() for c in df.columns]
            df["__source_file__"] = os.path.relpath(f, self.root)
            df["__segment_id__"] = seg_id
            total_rows += len(df)
            frames.append(df)
        merged = pd.concat(frames, axis=0, ignore_index=True)
        merged["__row_id__"] = np.arange(len(merged), dtype=np.int64)
        logger.info("读取合计 %d 行; 拼接后 %d 行 x %d 列; segments=%d。",
                    total_rows, merged.shape[0], merged.shape[1], len(files))
        return merged

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    # ----------------------- 主流程 ----------------------- #
    def load(self) -> LoadResult:
        """发现→读取→清洗→标签规范化, 返回 LoadResult。"""
        stats: Dict[str, object] = {}
        files = self.discover_files()
        df = self._read_concat(files)
        stats["raw_shape"] = tuple(df.shape)

        label_col = self._find_column(df, self.label_column_candidates())
        if label_col is None:
            raise ValueError(f"未找到标签列, 候选={self.label_column_candidates()}")
        ts_col = self._find_column(df, self.timestamp_column_candidates())

        # 1) 去重列(同名重复列在 pandas 读后会变 .1 后缀, 这里删除完全重复内容的列)。
        before_cols = df.shape[1]
        df = df.loc[:, ~df.columns.duplicated()]
        # 删除内容完全重复的冗余列(如 'Fwd Header Length.1')。
        dup_content_cols = self._duplicate_content_columns(df, protect={label_col})
        if dup_content_cols:
            df = df.drop(columns=dup_content_cols)
        stats["dropped_duplicate_columns"] = before_cols - df.shape[1]

        # 2) 保留时间序(若有时间戳, 解析为排序键; 否则用原始行序)。
        if ts_col is not None:
            order = pd.to_datetime(df[ts_col], errors="coerce")
            order_rank = order.rank(method="first").ffill().bfill()
            df["__order__"] = order_rank.to_numpy()
            stats["temporal_order"] = f"by timestamp column '{ts_col}'"
        else:
            df["__order__"] = np.arange(len(df), dtype=np.float64)
            stats["temporal_order"] = "by original row order (no timestamp)"

        # 3) 选数值特征列(排除标签/时间戳/明显的标识列)。
        feature_cols = self._select_numeric_features(df, label_col, ts_col)
        stats["n_feature_candidates"] = len(feature_cols)

        # 4) Inf/-Inf → NaN。
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # 5) 删全空列(全为 NaN 的特征列)。
        all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
        if all_nan_cols:
            df = df.drop(columns=all_nan_cols)
            feature_cols = [c for c in feature_cols if c not in all_nan_cols]
        stats["dropped_all_nan_columns"] = len(all_nan_cols)

        # 6) 标签规范化 + 丢弃无法映射的类。
        df["label"] = df[label_col].astype(str).map(self.normalize_label)
        n_before_label = len(df)
        df = df[df["label"].notna()].copy()
        stats["dropped_unmapped_label_rows"] = n_before_label - len(df)

        # 7) NaN 行填补: 用列中位数。
        n_nan_cells = int(df[feature_cols].isna().sum().sum())
        medians = df[feature_cols].median(numeric_only=True)
        df[feature_cols] = df[feature_cols].fillna(medians)
        # 仍有 NaN(整列中位数也为NaN)的列直接置0。
        df[feature_cols] = df[feature_cols].fillna(0.0)
        stats["filled_nan_cells"] = n_nan_cells

        # 8) 去重样本(基于特征+标签)。
        n_before_dup = len(df)
        df = df.drop_duplicates(subset=feature_cols + ["label"]).reset_index(drop=True)
        stats["dropped_duplicate_rows"] = n_before_dup - len(df)

        # 类型收敛。
        df[feature_cols] = df[feature_cols].astype(np.float32)
        stats["clean_shape"] = (len(df), len(feature_cols))
        stats["class_distribution"] = df["label"].value_counts().to_dict()

        self._log_stats(stats)
        meta_cols = [c for c in ["__order__", "__row_id__", "__segment_id__", "__source_file__"]
                     if c in df.columns]
        return LoadResult(df=df[feature_cols + ["label"] + meta_cols],
                          feature_columns=feature_cols, stats=stats)

    # ----------------------- 工具 ----------------------- #
    @staticmethod
    def _duplicate_content_columns(df: pd.DataFrame, protect: set) -> List[str]:
        """找出内容与前面某列完全相同的冗余列(保留首个)。"""
        seen: Dict[bytes, str] = {}
        dups: List[str] = []
        meta_cols = {"__order__", "__row_id__", "__segment_id__", "__source_file__"}
        for col in df.columns:
            if col in protect or col in meta_cols:
                continue
            series = df[col]
            if series.dtype == object:
                continue
            key = pd.util.hash_pandas_object(series, index=False).values.tobytes()
            if key in seen:
                dups.append(col)
            else:
                seen[key] = col
        return dups

    def _select_numeric_features(self, df: pd.DataFrame, label_col: str,
                                 ts_col: Optional[str]) -> List[str]:
        """选择可用作特征的数值列。"""
        exclude = {label_col, "__order__", "label",
                   "__row_id__", "__segment_id__", "__source_file__"}
        if ts_col is not None:
            exclude.add(ts_col)
        # 常见标识列(若存在)排除。
        for ident in ["Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
                      "Source Port", "Src Port", "Destination Port", "Dst Port",
                      "Protocol", "Fwd Header Length.1"]:
            real = self._find_column(df, [ident])
            if real is not None:
                exclude.add(real)

        feature_cols: List[str] = []
        for col in df.columns:
            if col in exclude:
                continue
            coerced = pd.to_numeric(df[col], errors="coerce")
            # 若大部分可转为数值则视为数值特征。
            if coerced.notna().mean() >= 0.95:
                df[col] = coerced
                feature_cols.append(col)
        return feature_cols

    @staticmethod
    def _log_stats(stats: Dict[str, object]) -> None:
        logger.info("===== 数据清洗统计 =====")
        logger.info("原始形状: %s", stats.get("raw_shape"))
        logger.info("删除重复列: %s", stats.get("dropped_duplicate_columns"))
        logger.info("删除全空列: %s", stats.get("dropped_all_nan_columns"))
        logger.info("时序排序: %s", stats.get("temporal_order"))
        logger.info("丢弃未映射标签行: %s", stats.get("dropped_unmapped_label_rows"))
        logger.info("填补 NaN 单元格: %s", stats.get("filled_nan_cells"))
        logger.info("删除重复样本: %s", stats.get("dropped_duplicate_rows"))
        logger.info("清洗后形状: %s", stats.get("clean_shape"))
        logger.info("类别分布: %s", stats.get("class_distribution"))
        logger.info("========================")
