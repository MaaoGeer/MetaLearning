"""评估子包: 指标计算、Adaptation Speed、few-shot 评估器。"""

from .metrics import ClassificationMetrics, compute_metrics, aggregate_logits
from .evaluator import FewShotEvaluator
from .adaptation_speed import SpeedResult, compute_speed, aggregate_speeds
from .task_manifest import (
    load_tasks_from_manifest,
    manifest_raw_row_ids,
    read_task_manifest,
    sha256_file,
    write_task_manifest,
)

__all__ = [
    "ClassificationMetrics", "compute_metrics", "aggregate_logits",
    "FewShotEvaluator",
    "SpeedResult", "compute_speed", "aggregate_speeds",
    "load_tasks_from_manifest", "manifest_raw_row_ids", "read_task_manifest", "sha256_file",
    "write_task_manifest",
]
