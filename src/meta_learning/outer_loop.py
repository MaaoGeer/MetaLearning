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
from typing import List, Optional

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


class OuterLoop:
    """封装 meta-batch 的外循环计算。"""

    def __init__(self, model: nn.Module, inner_loop: InnerLoop,
                 adapt_names: Optional[List[str]] = None,
                 query_loss_fn: Optional[nn.Module] = None) -> None:
        self.model = model
        self.inner_loop = inner_loop
        self.adapt_names = adapt_names
        self.query_loss_fn = query_loss_fn or nn.CrossEntropyLoss()

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

        for task in tasks:
            inner = self.inner_loop.adapt(init_params, task, adapt_names=self.adapt_names)
            query_logits = functional_forward(self.model, inner.adapted_params, task.query_x)
            task_loss = self.query_loss_fn(query_logits, task.query_y)
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
        )
