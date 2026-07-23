"""实验脚本 (run_experiments) 的 pooled 指标与公平性测试。"""

import importlib.util
import os

import torch

from src.evaluation.adaptation_speed import SpeedResult
from src.evaluation.metrics import ClassificationMetrics
from src.trainer.adapter import AdaptOutcome

_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts", "run_experiments.py")


def _load_run_experiments():
    spec = importlib.util.spec_from_file_location("run_experiments", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_outcome(logits, targets):
    sr = SpeedResult(f1_trajectory=[0.5, 0.9], speeds={0.8: 2}, reached={0.8: True},
                     final_f1=0.9, max_steps=2)
    m = ClassificationMetrics(accuracy=1.0, precision=1.0, recall=1.0, f1=1.0,
                              roc_auc=None, n_samples=len(targets))
    return AdaptOutcome(speed=sr, final_metrics=m,
                        final_logits=logits, final_targets=targets)


def test_pooled_metrics_include_confusion_and_per_class_recall():
    mod = _load_run_experiments()
    # 两个 task 的 query 预测; n_way=2, attack_idx=1。
    o1 = _make_outcome(torch.tensor([[2.0, 0.1], [0.0, 3.0]]), torch.tensor([0, 1]))
    o2 = _make_outcome(torch.tensor([[3.0, 0.0], [0.1, 2.0]]), torch.tensor([0, 1]))
    pooled = mod._pooled_metrics([o1, o2], n_way=2, attack_idx=1)
    assert pooled is not None
    assert pooled.confusion.shape == (2, 2)
    assert set(pooled.per_class_recall.keys()) == {0, 1}
    assert pooled.attack_recall is not None
    # 全部预测正确 → 混淆矩阵对角。
    assert int(pooled.confusion[0, 0]) == 2
    assert int(pooled.confusion[1, 1]) == 2


def test_pooled_metrics_none_when_no_logits():
    mod = _load_run_experiments()
    o = AdaptOutcome(
        speed=SpeedResult(f1_trajectory=[0.5], speeds={}, reached={}, final_f1=0.5, max_steps=1),
        final_metrics=ClassificationMetrics(1.0, 1.0, 1.0, 1.0, None),
        final_logits=None, final_targets=None)
    assert mod._pooled_metrics([o], n_way=2, attack_idx=1) is None


def test_select_validation_stop_step_uses_mean_curve_and_earliest_tie():
    mod = _load_run_experiments()
    base = ClassificationMetrics(1.0, 1.0, 1.0, 1.0, None, n_samples=1)
    o1 = AdaptOutcome(
        speed=SpeedResult(
            f1_trajectory=[0.3, 0.8, 0.8],
            metric_trajectories={"macro_f1": [0.3, 0.8, 0.8]},
            max_steps=2),
        final_metrics=base)
    o2 = AdaptOutcome(
        speed=SpeedResult(
            f1_trajectory=[0.4, 0.9, 0.9],
            metric_trajectories={"macro_f1": [0.4, 0.9, 0.9]},
            max_steps=2),
        final_metrics=base)
    assert mod.select_validation_stop_step([o1, o2]) == 1


def test_grid_boundary_status_uses_json_native_booleans():
    mod = _load_run_experiments()
    grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    interior = mod._grid_boundary_status(0.1, grid, "Adam")
    upper = mod._grid_boundary_status(0.3, grid, "Adam")

    assert interior["at_lower_boundary"] is False
    assert interior["at_upper_boundary"] is False
    assert upper["at_lower_boundary"] is False
    assert upper["at_upper_boundary"] is True
