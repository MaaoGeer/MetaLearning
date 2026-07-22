"""分类指标: Accuracy / Precision / Recall / F1 / ROC-AUC / 混淆矩阵 / per-class recall。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """聚合的分类指标。"""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    n_samples: int = 0
    per_class_recall: Dict[int, float] = field(default_factory=dict)
    attack_recall: Optional[float] = None  # 非 benign(局部标签1或攻击类) 的 recall

    false_positive_rate: Optional[float] = None

    def as_dict(self) -> dict:
        d = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "macro_f1": self.macro_f1 if self.macro_f1 is not None else self.f1,
            "weighted_f1": self.weighted_f1 if self.weighted_f1 is not None else float("nan"),
            "roc_auc": self.roc_auc if self.roc_auc is not None else float("nan"),
            "pr_auc": self.pr_auc if self.pr_auc is not None else float("nan"),
            "n_samples": self.n_samples,
            "per_class_recall": {str(k): v for k, v in self.per_class_recall.items()},
            "attack_recall": self.attack_recall if self.attack_recall is not None else float("nan"),
            "false_positive_rate": (
                self.false_positive_rate
                if self.false_positive_rate is not None else float("nan")
            ),
        }
        return d


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    attack_class_indices: Optional[List[int]] = None,
) -> ClassificationMetrics:
    """根据 logits 与真实标签计算指标。"""
    if num_classes is None:
        num_classes = logits.shape[1]
    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    y_true = targets.cpu().numpy().astype(int)

    acc = float(accuracy_score(y_true, preds))
    precision = float(precision_score(y_true, preds, average="macro", zero_division=0))
    recall = float(recall_score(y_true, preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))
    f1 = macro_f1

    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    try:
        if num_classes == 2:
            roc_auc = float(roc_auc_score(y_true, probs[:, 1]))
            pr_auc = float(average_precision_score(y_true, probs[:, 1]))
        elif len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(
                y_true, probs, multi_class="ovr", average="macro",
                labels=list(range(num_classes))))
            one_hot = np.eye(num_classes, dtype=np.float32)[y_true]
            pr_auc = float(average_precision_score(one_hot, probs, average="macro"))
    except (ValueError, IndexError):
        roc_auc = None
        pr_auc = None

    cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))
    per_class: Dict[int, float] = {}
    for c in range(num_classes):
        denom = cm[c, :].sum()
        per_class[c] = float(cm[c, c] / denom) if denom > 0 else 0.0

    attack_recall: Optional[float] = None
    false_positive_rate: Optional[float] = None
    if attack_class_indices:
        recalls = [per_class.get(i, 0.0) for i in attack_class_indices if cm[i, :].sum() > 0]
        attack_recall = float(np.mean(recalls)) if recalls else None
    elif num_classes == 2:
        attack_recall = per_class.get(1, 0.0)
    if num_classes == 2 and cm.shape == (2, 2):
        tn, fp = float(cm[0, 0]), float(cm[0, 1])
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return ClassificationMetrics(
        accuracy=acc, precision=precision, recall=recall, f1=f1,
        roc_auc=roc_auc, pr_auc=pr_auc, macro_f1=macro_f1, weighted_f1=weighted_f1,
        confusion=cm, n_samples=int(len(y_true)),
        per_class_recall=per_class, attack_recall=attack_recall,
        false_positive_rate=false_positive_rate,
    )


def aggregate_logits(logits_list: List[torch.Tensor],
                     targets_list: List[torch.Tensor]) -> tuple:
    if not logits_list:
        raise ValueError("空的 logits 列表。")
    return torch.cat(logits_list, dim=0), torch.cat(targets_list, dim=0)
