"""CSE-CIC-IDS2018 数据集。"""

from __future__ import annotations

from typing import List, Optional

from .base_dataset import BaseDataset
from .label_utils import map_cic_family, norm_label


class CSECICIDS2018Dataset(BaseDataset):
    """CSE-CIC-IDS2018 (CICIDS2018)。"""

    def file_glob(self) -> str:
        return "**/*TrafficForML*.csv"

    def label_column_candidates(self) -> List[str]:
        return ["Label"]

    def timestamp_column_candidates(self) -> List[str]:
        return ["Timestamp"]

    def normalize_label(self, raw: str) -> Optional[str]:
        mapped = map_cic_family(raw)
        if mapped:
            return mapped
        n = norm_label(raw)
        if "bruteforce" in n:
            return "bruteforce"
        return None
