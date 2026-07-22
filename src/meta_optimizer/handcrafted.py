"""手工优化器 (SGD / Adam) 的函数式可微实现, 用于与 LSTM Meta Optimizer 对比。

为什么需要它:
    论文实验要求"与 SGD/Adam 对比"。这里实现与 LSTMOptimizer 完全相同的接口
    (init_state / step / detach_state), 使其能直接替换进 InnerLoop, 得到标准 MAML
    (内循环 = 固定 SGD/Adam) 作为基线。更新同样以函数式、非 in-place 方式进行,
    保留计算图, 支持二阶 MAML。

与 LSTM Meta Optimizer 的本质区别:
    本类**没有可学习参数**, 更新规则是人手设计且固定的; 而 LSTM Meta Optimizer
    的更新规则由神经网络学习得到。这正是对比的核心。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

ParamState = List[Tuple[torch.Tensor, torch.Tensor]]
OptState = Dict[str, ParamState]


class HandcraftedOptimizer(nn.Module):
    """固定规则优化器: kind ∈ {'sgd', 'adam'}。"""

    def __init__(self, kind: str = "sgd", lr: float = 0.1,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        super().__init__()
        assert kind in {"sgd", "adam"}, f"不支持的优化器: {kind}"
        self.kind = kind
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._step_count = 0

    def init_state(self, params: Dict[str, torch.Tensor]) -> OptState:
        """SGD 无状态; Adam 维护 (m, v)。统一用 (a, b) 元组承载, 便于 detach。"""
        self._step_count = 0
        state: OptState = {}
        for name, p in params.items():
            if self.kind == "adam":
                m = torch.zeros_like(p)
                v = torch.zeros_like(p)
                state[name] = [(m, v)]
            else:
                state[name] = []
        return state

    def step(self, grads: Dict[str, torch.Tensor], state: OptState
             ) -> Tuple[Dict[str, torch.Tensor], OptState]:
        """返回 Δθ（应取 θ+Δθ）与新状态。"""
        updates: Dict[str, torch.Tensor] = {}
        new_state: OptState = {}
        if self.kind == "sgd":
            for name, g in grads.items():
                updates[name] = -self.lr * g
                new_state[name] = state.get(name, [])
            return updates, new_state

        # Adam
        self._step_count += 1
        t = self._step_count
        for name, g in grads.items():
            m_prev, v_prev = state[name][0]
            m = self.beta1 * m_prev + (1 - self.beta1) * g
            v = self.beta2 * v_prev + (1 - self.beta2) * (g * g)
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            updates[name] = -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            new_state[name] = [(m, v)]
        return updates, new_state

    @staticmethod
    def detach_state(state: OptState) -> OptState:
        detached: OptState = {}
        for name, layers in state.items():
            detached[name] = [(a.detach(), b.detach()) for (a, b) in layers]
        return detached
