"""外循环 (Outer Loop): 聚合一个 meta-batch 任务的 query loss → 元损失。

流程:
    for task in meta_batch:
        θ_K = InnerLoop.adapt(θ_0, task, adapt_names)     # 仅适配子集 θ_a
        query_logits = f(query_x; θ_K)
        task_loss = CE(query_logits, query_y)
    meta_loss = mean(task_loss)
    trainer 执行 meta_loss.backward() 更新 Meta Optimizer φ; θ0 是共享随机初始化。

Meta Optimizer 如何学习:
    query loss 衡量"用 meta_opt 适配后, base learner 在未见样本上的表现"。反传把误差沿
    "query loss → θ_K → 各步 Δθ → meta_opt 权重 φ" 传播, 训练 φ 成为快速泛化的更新器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..data.task_sampler import MetaTask
from .functional import functional_forward
from .inner_loop import InnerLoop


@dataclass
class MetaBatchResult:
    """一个 meta-batch 的结果。"""

    meta_loss: torch.Tensor
    query_acc: float
    avg_support_loss: float
    query_logits: List[torch.Tensor] = field(default_factory=list)
    query_targets: List[torch.Tensor] = field(default_factory=list)
    query_loss_by_step: Dict[int, float] = field(default_factory=dict)
    weighted_contribution_by_step: Dict[int, float] = field(default_factory=dict)
    sampled_horizons: List[int] = field(default_factory=list)


def normalized_step_weights(
    steps: Sequence[int],
    weighting: str = "uniform",
    custom_weights: Optional[Sequence[float] | Mapping[int, float]] = None,
    early_heavy_power: float = 0.5,
) -> Dict[int, float]:
    """Return normalized, positive weights for supervised query-loss steps."""
    unique_steps = sorted({int(step) for step in steps if int(step) > 0})
    if not unique_steps:
        raise ValueError("multi-step query objective requires at least one positive step")
    mode = str(weighting).lower()
    if mode == "uniform":
        raw = [1.0] * len(unique_steps)
    elif mode == "early_heavy":
        raw = [float(step) ** (-float(early_heavy_power)) for step in unique_steps]
    elif mode == "custom":
        if isinstance(custom_weights, Mapping):
            raw = [float(custom_weights.get(step, 0.0)) for step in unique_steps]
        else:
            values = list(custom_weights or [])
            if len(values) != len(unique_steps):
                raise ValueError(
                    "custom query-loss weights must align with supervised_steps"
                )
            raw = [float(value) for value in values]
    else:
        raise ValueError(
            "query objective weighting must be uniform, early_heavy, or custom"
        )
    if any(value < 0 for value in raw) or sum(raw) <= 0:
        raise ValueError("query-loss weights must be non-negative with positive sum")
    total = float(sum(raw))
    return {step: value / total for step, value in zip(unique_steps, raw)}


class OuterLoop:
    """封装 meta-batch 的外循环计算。"""

    def __init__(self, model: nn.Module, inner_loop: InnerLoop,
                 adapt_names: Optional[List[str]] = None,
                 query_loss_fn: Optional[nn.Module] = None,
                 query_objective: Optional[Mapping] = None,
                 random_horizon: Optional[Mapping] = None,
                 seed: int = 0) -> None:
        self.model = model
        self.inner_loop = inner_loop
        self.adapt_names = adapt_names
        self.query_loss_fn = query_loss_fn or nn.CrossEntropyLoss()
        objective = query_objective or {}
        self.objective_mode = str(objective.get("mode", "final_only")).lower()
        if self.objective_mode not in {"final_only", "multi_step"}:
            raise ValueError("meta.query_objective.mode must be final_only or multi_step")
        self.supervised_steps = [
            int(step) for step in objective.get(
                "supervised_steps", [self.inner_loop.inner_steps]
            )
        ]
        self.weighting = str(objective.get("weighting", "uniform"))
        self.custom_weights = objective.get("custom_weights", [])
        if (
            self.weighting.lower() == "custom"
            and not isinstance(self.custom_weights, Mapping)
        ):
            values = list(self.custom_weights or [])
            if len(values) != len(self.supervised_steps):
                raise ValueError(
                    "custom query-loss weights must align with supervised_steps"
                )
            self.custom_weights = {
                step: float(value)
                for step, value in zip(self.supervised_steps, values)
            }
        self.early_heavy_power = float(objective.get("early_heavy_power", 0.5))
        self.include_sampled_horizon = bool(
            objective.get("include_sampled_horizon", True)
        )
        horizon = random_horizon or {}
        self.random_horizon_enabled = bool(horizon.get("enabled", False))
        self.min_horizon = int(horizon.get("min_steps", 1))
        self.max_horizon = int(
            horizon.get("max_steps", self.inner_loop.inner_steps)
        )
        if self.min_horizon < 1 or self.max_horizon < self.min_horizon:
            raise ValueError("invalid meta.random_horizon min_steps/max_steps")
        if (
            self.random_horizon_enabled
            and self.max_horizon > int(self.inner_loop.inner_steps)
        ):
            raise ValueError(
                "meta.random_horizon.max_steps cannot exceed meta.inner_steps"
            )
        self._horizon_rng = random.Random(int(seed))

    def sample_horizon(self) -> int:
        """Sample a deterministic-by-seed training horizon."""
        if not self.random_horizon_enabled:
            return int(self.inner_loop.inner_steps)
        return self._horizon_rng.randint(self.min_horizon, self.max_horizon)

    def _supervision_for_horizon(self, horizon: int) -> List[int]:
        if self.objective_mode == "final_only":
            return [int(horizon)]
        steps = [step for step in self.supervised_steps if 0 < step <= horizon]
        if self.include_sampled_horizon and horizon not in steps:
            steps.append(int(horizon))
        return sorted(set(steps or [int(horizon)]))

    def run_meta_batch(self, tasks: List[MetaTask], init_params=None,
                       collect_outputs: bool = False) -> MetaBatchResult:
        """对一批任务执行内循环 + query 评估, 返回聚合元损失。"""
        if init_params is None:
            init_params = {n: p for n, p in self.model.named_parameters()}

        device = next(self.model.parameters()).device
        total_loss = torch.zeros((), device=device)
        correct, total, support_loss_sum = 0, 0, 0.0
        logits_list: List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []
        loss_sums: Dict[int, float] = {}
        contribution_sums: Dict[int, float] = {}
        loss_counts: Dict[int, int] = {}
        sampled_horizons: List[int] = []

        for task in tasks:
            horizon = self.sample_horizon()
            sampled_horizons.append(horizon)
            supervised_steps = self._supervision_for_horizon(horizon)
            params_by_step: Dict[int, Dict[str, torch.Tensor]] = {}

            def record_fn(step_index: int, params: Dict[str, torch.Tensor]) -> None:
                actual_step = int(step_index) + 1
                if actual_step in supervised_steps:
                    params_by_step[actual_step] = params

            inner = self.inner_loop.adapt(
                init_params, task, adapt_names=self.adapt_names,
                record_fn=record_fn if self.objective_mode == "multi_step" else None,
                inner_steps=horizon,
            )
            if self.objective_mode == "final_only":
                params_by_step[horizon] = inner.adapted_params

            custom_weights = self.custom_weights
            if self.weighting.lower() == "custom" and isinstance(custom_weights, Mapping):
                custom_weights = dict(custom_weights)
                custom_weights.setdefault(
                    horizon,
                    float(custom_weights.get(max(custom_weights), 1.0))
                    if custom_weights else 1.0,
                )
            weights = normalized_step_weights(
                supervised_steps,
                weighting=self.weighting,
                custom_weights=custom_weights,
                early_heavy_power=self.early_heavy_power,
            )
            task_loss = torch.zeros((), device=device)
            query_logits = None
            for step in supervised_steps:
                step_params = params_by_step.get(step)
                if step_params is None:
                    raise RuntimeError(f"missing adapted parameters for supervised step {step}")
                step_logits = functional_forward(self.model, step_params, task.query_x)
                step_loss = self.query_loss_fn(step_logits, task.query_y)
                contribution = weights[step] * step_loss
                task_loss = task_loss + contribution
                loss_sums[step] = loss_sums.get(step, 0.0) + float(step_loss.detach())
                contribution_sums[step] = (
                    contribution_sums.get(step, 0.0) + float(contribution.detach())
                )
                loss_counts[step] = loss_counts.get(step, 0) + 1
                if step == horizon:
                    query_logits = step_logits
            if query_logits is None:
                query_logits = functional_forward(
                    self.model, inner.adapted_params, task.query_x
                )
            total_loss = total_loss + task_loss

            with torch.no_grad():
                preds = query_logits.argmax(dim=1)
                correct += int((preds == task.query_y).sum())
                total += task.query_y.numel()
            if inner.support_losses:
                support_loss_sum += inner.support_losses[-1]
            if collect_outputs:
                logits_list.append(query_logits.detach().float().cpu())
                targets_list.append(task.query_y.detach().cpu())

        n_tasks = max(len(tasks), 1)
        return MetaBatchResult(
            meta_loss=total_loss / n_tasks,
            query_acc=correct / max(total, 1),
            avg_support_loss=support_loss_sum / n_tasks,
            query_logits=logits_list,
            query_targets=targets_list,
            query_loss_by_step={
                step: loss_sums[step] / loss_counts[step] for step in loss_sums
            },
            weighted_contribution_by_step={
                step: contribution_sums[step] / n_tasks
                for step in contribution_sums
            },
            sampled_horizons=sampled_horizons,
        )
