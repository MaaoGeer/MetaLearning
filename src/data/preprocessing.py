"""特征标准化与标签编码（清洗已在 BaseDataset 完成）。

设计要点:
    - StandardScaler 仅在 known-train 子集上 fit, 其余(known-eval / unknown)仅 transform,
      避免任何信息泄漏(unknown 攻击的统计量绝不能参与训练标准化)。
    - 标签编码在"给定类别集合"内映射到连续索引, 任务构造时再做局部重映射。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureStandardizer:
    """数值特征标准化器。"""

    feature_columns: List[str]
    scaler: StandardScaler = None  # type: ignore[assignment]
    enabled: bool = True

    def fit(self, df: pd.DataFrame) -> "FeatureStandardizer":
        if self.enabled:
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.feature_columns].astype(np.float64).values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_columns].astype(np.float64).values
        if self.enabled and self.scaler is not None:
            x = self.scaler.transform(x)
        return x.astype(np.float32)


def build_class_index(classes: List[str]) -> Dict[str, int]:
    """把类别名映射到连续索引(全局类别空间)。"""
    return {c: i for i, c in enumerate(sorted(classes))}
