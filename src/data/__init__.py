"""数据子包: 统一数据集接口、清洗、LOAO、时序窗口、Few-shot 任务构造。"""

from .base_dataset import BaseDataset, LoadResult, CANONICAL_CLASSES
from .cicids2017 import CICIDS2017Dataset
from .registry import build_dataset, available_datasets
from .preprocessing import FeatureStandardizer, build_class_index
from .dataset import IntrusionDataset
from .task_sampler import FewShotTaskSampler, MetaTask
from .loao import LOAOResult, SplitArrays, build_loao
from .task_builder import (
    build_windowed_dataset,
    make_meta_sampler,
    AdaptationTaskSampler,
)
from .pipeline import DataBundle, build_pipeline
from .windowing import build_windows, build_temporal_windows
from .leakage import audit_pipeline_splits, check_support_query_overlap

__all__ = [
    "BaseDataset", "LoadResult", "CANONICAL_CLASSES",
    "CICIDS2017Dataset",
    "build_dataset", "available_datasets",
    "FeatureStandardizer", "build_class_index",
    "IntrusionDataset",
    "FewShotTaskSampler", "MetaTask",
    "LOAOResult", "SplitArrays", "build_loao",
    "build_windowed_dataset", "make_meta_sampler", "AdaptationTaskSampler",
    "DataBundle", "build_pipeline",
    "build_windows", "build_temporal_windows",
    "audit_pipeline_splits", "check_support_query_overlap",
]
