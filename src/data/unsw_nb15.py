"""UNSW-NB15 数据集。"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import pandas as pd

from .base_dataset import BaseDataset, LoadResult
from .label_utils import apply_custom_mapping, map_unsw_attack_cat


class UNSWNB15Dataset(BaseDataset):
    """UNSW-NB15: 使用 attack_cat 作为攻击类型, label 列作二值辅助。"""

    def __init__(self, root: str, custom_label_mapping: Optional[dict] = None) -> None:
        super().__init__(root)
        self.custom_label_mapping = custom_label_mapping or {}

    def file_glob(self) -> str:
        return "**/*.csv"

    def label_column_candidates(self) -> List[str]:
        return ["attack_cat", "Label", "label"]

    def timestamp_column_candidates(self) -> List[str]:
        return []

    def normalize_label(self, raw: str) -> Optional[str]:
        if self.custom_label_mapping:
            m = apply_custom_mapping(raw, self.custom_label_mapping)
            if m:
                return m
        return map_unsw_attack_cat(raw)

    def discover_files(self) -> List[str]:
        """优先 training-set + testing-set, 否则递归所有 CSV。"""
        preferred = [
            os.path.join(self.root, "UNSW_NB15_training-set.csv"),
            os.path.join(self.root, "UNSW_NB15_testing-set.csv"),
        ]
        existing = [f for f in preferred if os.path.isfile(f)]
        if existing:
            return existing
        return super().discover_files()
