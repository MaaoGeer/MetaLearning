"""Few-shot 评估器: 在若干评估任务上做内循环适配并汇总指标。

评估与训练的区别:
    - 评估时不需要二阶图 (create_graph=False), 但仍需对内循环求一阶梯度来执行
      Meta Optimizer 更新, 因此整个适配过程必须在 torch.enable_grad() 下运行。
    - 评估只统计 query 集预测, 不更新 Meta Optimizer / θ0（不调用 backward/step）。
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..data.task_sampler import FewShotTaskSampler, MetaTask
from ..meta_learning.functional import functional_forward
from ..meta_learning.inner_loop import InnerLoop
from ..meta_optimizer.lstm_optimizer import LSTMOptimizer
from ..utils.logger import get_logger
from .metrics import ClassificationMetrics, aggregate_logits, compute_metrics
from .adaptation_speed import summarize_adaptation

logger = get_logger(__name__)


class FewShotEvaluator:
    """在采样任务上评估元学习系统。"""

    def __init__(self, model: nn.Module, meta_opt: LSTMOptimizer,
                 inner_steps: int = 5, device: Optional[torch.device] = None,
                 adapt_names: Optional[List[str]] = None) -> None:
        self.model = model
        self.meta_opt = meta_opt
        self.device = device or next(model.parameters()).device
        self.adapt_names = adapt_names
        # 评估用一阶适配即可（不需要二阶图）。
        self.inner_loop = InnerLoop(
            model=model, meta_opt=meta_opt, inner_steps=inner_steps,
            tbptt_steps=0, first_order=True,
        )

    @torch.enable_grad()
    def evaluate(self, sampler: FewShotTaskSampler, num_tasks: int = 100,
                 desc: str = "eval") -> ClassificationMetrics:
        """采样 num_tasks 个任务, 适配后在 query 集上汇总指标。"""
        was_training = self.model.training
        self.model.eval()       # 关闭 dropout, 保证评估确定性
        logits_all: List[torch.Tensor] = []
        targets_all: List[torch.Tensor] = []
        n_way = sampler.n_way

        for _ in range(num_tasks):
            task: MetaTask = sampler.sample_task().to(self.device)
            init_params = {n: p for n, p in self.model.named_parameters()}
            inner = self.inner_loop.adapt(init_params, task, adapt_names=self.adapt_names)
            with torch.no_grad():
                query_logits = functional_forward(self.model, inner.adapted_params, task.query_x)
            logits_all.append(query_logits.detach().float().cpu())
            targets_all.append(task.query_y.detach().cpu())

        if was_training:
            self.model.train()

        logits, targets = aggregate_logits(logits_all, targets_all)
        metrics = compute_metrics(logits, targets, num_classes=n_way)
        logger.info("[%s] %s", desc, metrics)
        return metrics

    @torch.enable_grad()
    def evaluate_tasks(
        self,
        tasks: List[MetaTask],
        n_way: int,
        desc: str = "eval",
    ) -> ClassificationMetrics:
        """Evaluate a pre-sampled fixed task pool."""
        was_training = self.model.training
        self.model.eval()
        logits_all: List[torch.Tensor] = []
        targets_all: List[torch.Tensor] = []

        for raw_task in tasks:
            task: MetaTask = raw_task.to(self.device)
            init_params = {n: p for n, p in self.model.named_parameters()}
            inner = self.inner_loop.adapt(init_params, task, adapt_names=self.adapt_names)
            with torch.no_grad():
                query_logits = functional_forward(self.model, inner.adapted_params, task.query_x)
            logits_all.append(query_logits.detach().float().cpu())
            targets_all.append(task.query_y.detach().cpu())

        if was_training:
            self.model.train()

        logits, targets = aggregate_logits(logits_all, targets_all)
        metrics = compute_metrics(logits, targets, num_classes=n_way)
        logger.info("[%s] %s", desc, metrics)
        return metrics

    @torch.enable_grad()
    def evaluate_task_curve(
        self,
        tasks: List[MetaTask],
        n_way: int,
        checkpoints: List[int],
        target_f1: float = 0.8,
        desc: str = "eval-curve",
    ) -> Tuple[ClassificationMetrics, Dict[str, object]]:
        """Evaluate fixed tasks at every step used for checkpoint selection."""
        from ..trainer.adapter import FewShotAdapter

        positive_steps = sorted({int(step) for step in checkpoints if int(step) > 0})
        if not positive_steps:
            raise ValueError("validation curve requires at least one positive checkpoint")
        was_training = self.model.training
        self.model.eval()
        adapter = FewShotAdapter(self.model, self.device)
        init_params = {name: param for name, param in self.model.named_parameters()}
        attack_indices = [1] if n_way == 2 else [n_way - 1]
        outcomes = [
            adapter.adapt_once(
                init_params,
                raw_task,
                self.meta_opt,
                self.adapt_names or list(init_params),
                n_way,
                max_steps=max(positive_steps),
                target_f1_grid=[target_f1],
                attack_class_indices=attack_indices,
            )
            for raw_task in tasks
        ]
        logits, targets = aggregate_logits(
            [outcome.final_logits for outcome in outcomes],
            [outcome.final_targets for outcome in outcomes],
        )
        metrics = compute_metrics(
            logits, targets, num_classes=n_way,
            attack_class_indices=attack_indices,
        )
        summary = summarize_adaptation(
            [outcome.speed for outcome in outcomes],
            target_f1=target_f1,
            checkpoints=[0] + positive_steps,
        )
        if was_training:
            self.model.train()
        logger.info(
            "[%s] curve_auc=%.4f final_f1=%.4f checkpoints=%s",
            desc,
            summary["curve_auc_mean"],
            summary["final_f1_mean"],
            positive_steps,
        )
        return metrics, summary
