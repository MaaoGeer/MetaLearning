"""内循环 (Inner Loop): 用优化器在支持集上适配 Base Learner 的可适配参数子集。

关键点(本版新增):
    - 只适配指定的参数子集 θ_a(adapt_names), 其余参数冻结(作为固定特征器)。
      这对应 adaptation_scope=head/last_block/full, 显著影响显存与二阶可训练性。
    - 优化器是鸭子类型: 任何实现 init_state/step/detach_state 的对象都可(LSTM Meta
      Optimizer / 手工 SGD / 手工 Adam), 因此内循环代码对三种优化器完全一致——
      这是公平对比的工程基础。
    - 支持每步记录钩子 record_fn(step, merged_params), 用于测量 Adaptation Speed。

梯度/显存:
    - create_graph=True(¬first_order) 保留二阶图, 让 query loss 反传到 Meta Optimizer φ。
    - 仅对 θ_a 求梯度; 冻结参数不进入 autograd.grad 的 inputs, 二阶图更小。
    - tbptt_steps>0 周期性 detach θ_a 与优化器状态, 截断 BPTT。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ..data.task_sampler import MetaTask
from .functional import functional_forward

RecordFn = Callable[[int, Dict[str, torch.Tensor]], None]


@dataclass
class InnerResult:
    """内循环产物。"""

    adapted_params: Dict[str, torch.Tensor]   # 完整参数字典(冻结部分 + 适配后的 θ_a)
    final_state: dict
    support_losses: List[float] = field(default_factory=list)


class InnerLoop:
    """封装单任务内循环适配逻辑(可适配参数子集 + 任意优化器)。"""

    def __init__(self, model: nn.Module, meta_opt, inner_steps: int = 5,
                 tbptt_steps: int = 0, first_order: bool = False,
                 loss_fn: Optional[nn.Module] = None) -> None:
        self.model = model
        self.meta_opt = meta_opt
        self.inner_steps = inner_steps
        self.tbptt_steps = tbptt_steps
        self.first_order = first_order
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def adapt(self, init_params: Dict[str, torch.Tensor], task: MetaTask,
              adapt_names: Optional[List[str]] = None,
              record_fn: Optional[RecordFn] = None,
              inner_steps: Optional[int] = None) -> InnerResult:
        """在支持集上执行内循环适配。

        Args:
            init_params: 完整初始参数 θ(通常为 model.named_parameters())。
            task: 含 support/query 的元任务。
            adapt_names: 需适配的参数名列表; None 表示全部适配。
            record_fn: 可选, 每步更新后回调 record_fn(step, merged_params)(用于 Adaptation Speed)。
            inner_steps: 覆盖默认步数(测试期可设更大)。

        Returns:
            InnerResult。
        """
        steps = inner_steps if inner_steps is not None else self.inner_steps
        full = dict(init_params)
        if adapt_names is None:
            adapt_names = list(full.keys())
        adapt_set = set(adapt_names)

        # 冻结部分(常量, 跨步不变); 适配部分(随步更新)。
        frozen = {n: p for n, p in full.items() if n not in adapt_set}
        adaptable = {n: full[n] for n in adapt_names}

        state = self.meta_opt.init_state(adaptable)
        support_losses: List[float] = []
        create_graph = not self.first_order

        for step in range(steps):
            merged = {**frozen, **adaptable}
            logits = functional_forward(self.model, merged, task.support_x)
            loss = self.loss_fn(logits, task.support_y)
            support_losses.append(float(loss.detach()))

            grads = torch.autograd.grad(
                loss, list(adaptable.values()),
                create_graph=create_graph, retain_graph=create_graph,
                allow_unused=False)
            grad_dict = {name: g for name, g in zip(adaptable.keys(), grads)}

            updates, state = self.meta_opt.step(grad_dict, state)
            adaptable = {name: adaptable[name] + updates[name] for name in adaptable}

            if record_fn is not None:
                record_fn(step, {**frozen, **adaptable})

            is_last = step == steps - 1
            # 截断条件:
            #   - 一阶模式(测试期适配): 每步都 detach, 切断跨步计算图, 显存 O(1);
            #   - 二阶模式 + tbptt_steps>0: 周期性 detach, 截断 BPTT。
            detach_now = self.first_order or (
                self.tbptt_steps and self.tbptt_steps > 0
                and (step + 1) % self.tbptt_steps == 0)
            if detach_now and not is_last:
                adaptable = {n: p.detach().clone().requires_grad_(True)
                             for n, p in adaptable.items()}
                state = self.meta_opt.detach_state(state)

        return InnerResult(adapted_params={**frozen, **adaptable},
                           final_state=state, support_losses=support_losses)
