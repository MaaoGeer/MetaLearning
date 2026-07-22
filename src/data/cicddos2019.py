"""CIC-DDoS2019 / CICIDS2019 数据集模板 (需将 CSV 放入 datasets/CICDDoS2019/)。"""

from __future__ import annotations

from typing import List, Optional

from .base_dataset import BaseDataset
from .label_utils import map_cic_family, norm_label


class CICDDoS2019Dataset(BaseDataset):
    """CIC-DDoS2019 模板实现。"""

    def file_glob(self) -> str:
        return "**/*.csv"

    def label_column_candidates(self) -> List[str]:
        return ["Label", "label", " Label"]

    def timestamp_column_candidates(self) -> List[str]:
        return ["Timestamp", "timestamp"]

    def normalize_label(self, raw: str) -> Optional[str]:
        mapped = map_cic_family(raw)
        if mapped:
            return mapped
        n = norm_label(raw)
        if "syn" in n and "flood" in n:
            return "dos"
        if "udp" in n or "ldap" in n or "mssql" in n:
            return "ddos"
        if n == "benign" or n == "bening":
            return "benign"
        return "other"
