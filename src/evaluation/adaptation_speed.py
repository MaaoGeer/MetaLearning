"""Adaptation Speed —— 本论文核心指标。

严格定义:
    在固定 query 集上, 记录每个内循环步 t 的 query-F1(macro)。
    AdaptationSpeed(τ) = min{ t : F1_t ≥ τ }（达到目标 F1 阈值所需的优化步数）。
    若在 max_steps 内未达到 τ, 记为 censored = max_steps（并标记 reached=False）。

不使用 Accuracy 作为达标判据(按要求)。最终对 SGD / Adam / Meta Optimizer 给出
达到 τ 所需步数, 用于证明 Meta Optimizer 收敛更快。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SpeedResult:
    """单次适配的速度测量结果。"""

    f1_trajectory: List[float]                 # 每步 query-F1
    metric_trajectories: Dict[str, List[float]] = field(default_factory=dict)
    speeds: Dict[float, int] = field(default_factory=dict)   # τ → 步数(未达=max_steps)
    reached: Dict[float, bool] = field(default_factory=dict) # τ → 是否达到
    final_f1: float = 0.0
    max_steps: int = 0
    # Compatibility-only reproduction of the historical calculation that omitted
    # step 0. New analysis must use ``speeds``.
    speeds_deprecated_excluding_step0: Dict[float, int] = field(default_factory=dict)


def compute_speed(f1_trajectory: List[float], target_f1_grid: List[float],
                  max_steps: Optional[int] = None,
                  trajectory_includes_step_zero: bool = False) -> SpeedResult:
    """根据 F1 轨迹计算各目标阈值的 Adaptation Speed。

    Args:
        f1_trajectory: query-F1 trajectory. By default item i is after step i+1.
            When ``trajectory_includes_step_zero=True``, item i is exactly step i.
        target_f1_grid: 目标 F1 阈值列表。
        max_steps: 上限步数; 默认取轨迹长度。

    Returns:
        SpeedResult。
    """
    n = len(f1_trajectory)
    cap = max_steps if max_steps is not None else n
    speeds: Dict[float, int] = {}
    reached: Dict[float, bool] = {}
    deprecated_speeds: Dict[float, int] = {}
    for tau in target_f1_grid:
        step_reached: Optional[int] = None
        for i, f1 in enumerate(f1_trajectory):
            if f1 >= tau:
                step_reached = i if trajectory_includes_step_zero else i + 1
                break
        if step_reached is None:
            speeds[tau] = cap
            reached[tau] = False
        else:
            speeds[tau] = step_reached
            reached[tau] = True
        legacy_values = f1_trajectory[1:] if trajectory_includes_step_zero else f1_trajectory
        deprecated_speeds[tau] = next(
            (index + 1 for index, value in enumerate(legacy_values) if value >= tau),
            cap,
        )
    return SpeedResult(
        f1_trajectory=list(f1_trajectory),
        metric_trajectories={"macro_f1": list(f1_trajectory)},
        speeds=speeds, reached=reached,
        final_f1=f1_trajectory[-1] if f1_trajectory else 0.0,
        max_steps=cap,
        speeds_deprecated_excluding_step0=deprecated_speeds,
    )


def summarize_adaptation(
    results: List[SpeedResult],
    target_f1: float,
    checkpoints: Optional[List[int]] = None,
) -> Dict[str, object]:
    """Summarize early adaptation, threshold crossing, and stability."""
    import numpy as np

    checkpoints = checkpoints or [0, 1, 2, 5, 10, 20, 50]
    base = aggregate_speeds(results, target_f1)
    if not results:
        return {**base, "checkpoints": {}}

    def metric_values(result: SpeedResult, metric: str) -> List[float]:
        if metric == "macro_f1":
            return result.metric_trajectories.get(metric, result.f1_trajectory)
        return result.metric_trajectories.get(metric, [])

    metric_names = sorted({
        metric for result in results for metric in result.metric_trajectories
    } | {"macro_f1"})
    checkpoint_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for step in checkpoints:
        by_metric: Dict[str, Dict[str, float]] = {}
        for metric in metric_names:
            values = []
            for result in results:
                trajectory = metric_values(result, metric)
                if trajectory:
                    index = min(max(int(step), 0), len(trajectory) - 1)
                    values.append(float(trajectory[index]))
            arr = np.asarray(values, dtype=float)
            by_metric[metric] = {
                "mean": float(np.nanmean(arr)) if len(arr) else float("nan"),
                "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n": int(len(arr)),
            }
        checkpoint_summary[str(step)] = by_metric

    curve_aucs, best_f1s, final_f1s, drops = [], [], [], []
    for result in results:
        trajectory = np.asarray(metric_values(result, "macro_f1"), dtype=float)
        if not len(trajectory):
            continue
        x = np.arange(len(trajectory), dtype=float)
        integrate = getattr(np, "trapezoid", np.trapz)
        curve_aucs.append(float(integrate(trajectory, x=x) / max(float(x[-1]), 1.0)))
        best = float(np.nanmax(trajectory))
        final = float(trajectory[-1])
        best_f1s.append(best)
        final_f1s.append(final)
        drops.append(best - final)

    return {
        **base,
        "checkpoints": checkpoint_summary,
        "curve_auc_mean": float(np.mean(curve_aucs)) if curve_aucs else float("nan"),
        "curve_auc_std": float(np.std(curve_aucs, ddof=1)) if len(curve_aucs) > 1 else 0.0,
        "final_f1_mean": float(np.mean(final_f1s)) if final_f1s else float("nan"),
        "descriptive_only_test_oracle": {
            "best_f1_mean": (
                float(np.mean(best_f1s)) if best_f1s else float("nan")
            ),
            "post_peak_drop_mean": (
                float(np.mean(drops)) if drops else float("nan")
            ),
            "post_peak_drop_std": (
                float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0
            ),
            "allowed_for_selection": False,
            "description": (
                "Descriptive test-trajectory oracle; never use for model, "
                "hyperparameter, or method selection."
            ),
        },
    }


def adaptation_selection_key(summary: Dict[str, object]) -> tuple:
    """Lexicographic validation objective aligned with fast adaptation."""
    return (
        float(summary.get("reach_rate", 0.0)),
        -float(summary.get("mean_steps", float("inf"))),
        float(summary.get("curve_auc_mean", float("-inf"))),
        float(summary.get("final_f1_mean", float("-inf"))),
    )


def aggregate_speeds(results: List[SpeedResult], target_f1: float) -> Dict[str, float]:
    """对多次适配(多任务/多种子)的速度做聚合统计。"""
    import numpy as np

    steps = np.array([r.speeds.get(target_f1, r.max_steps) for r in results], dtype=float)
    reached = np.array([1.0 if r.reached.get(target_f1, False) else 0.0 for r in results])
    finals = np.array([r.final_f1 for r in results], dtype=float)
    return {
        "target_f1": target_f1,
        "mean_steps": float(steps.mean()) if len(steps) else float("nan"),
        "std_steps": float(steps.std()) if len(steps) else float("nan"),
        "median_steps": float(np.median(steps)) if len(steps) else float("nan"),
        "reach_rate": float(reached.mean()) if len(reached) else 0.0,
        "mean_final_f1": float(finals.mean()) if len(finals) else float("nan"),
        "n": int(len(results)),
    }
