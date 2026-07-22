"""数据集注册表。"""

from __future__ import annotations

from typing import Dict, Optional, Type

from .base_dataset import BaseDataset
from .cicddos2019 import CICDDoS2019Dataset
from .cicids2017 import CICIDS2017Dataset
from .cse_cic_ids2018 import CSECICIDS2018Dataset
from .unsw_nb15 import UNSWNB15Dataset

_REGISTRY: Dict[str, Type[BaseDataset]] = {
    "cicids2017": CICIDS2017Dataset,
    "cse_cic_ids2018": CSECICIDS2018Dataset,
    "cicids2018": CSECICIDS2018Dataset,
    "unsw_nb15": UNSWNB15Dataset,
    "cicddos2019": CICDDoS2019Dataset,
    "cicids2019": CICDDoS2019Dataset,
}


def build_dataset(name: str, root: str,
                  label_mapping: Optional[Dict[str, str]] = None) -> BaseDataset:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"未注册的数据集 '{name}'; 可用: {list(_REGISTRY)}")
    cls = _REGISTRY[key]
    if key in ("unsw_nb15",):
        return cls(root=root, custom_label_mapping=label_mapping or {})
    return cls(root=root)


def available_datasets() -> list:
    return list(_REGISTRY)
