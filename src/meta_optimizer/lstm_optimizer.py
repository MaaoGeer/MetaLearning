"""坐标级 LSTM Meta Optimizer (Learning to learn by gradient descent by gradient descent)。

核心思想:
    用一个 LSTM 网络学习"参数更新规则", 取代手工设计的 SGD/Adam。对被优化网络
    (Base Learner) 的每一个参数坐标, 该 LSTM 接收其(预处理后的)梯度, 输出该坐标
    的更新量:

        θ_{t+1} = θ_t + g(∇_t, hidden_t)

    所有坐标共享同一套 LSTM 权重(coordinate-wise / 参数共享), 但各自维护独立的
    hidden state。这样 Meta Optimizer 的参数量与被优化网络规模无关, 且能学到
    与坐标无关的通用更新规则。

为什么是"神经网络学习如何优化另一个神经网络":
    Meta Optimizer 的权重 φ 通过"被优化网络在 query set 上的损失"反向传播来更新。
    也就是说, φ 不是靠人定的规则, 而是被训练成"能让 base learner 快速下降 query loss"
    的更新器。这正是 learning-to-optimize 的本质。

hidden state 如何更新:
    每个内循环步, 把该步梯度送入 LSTMCell, 更新 (h, c); 下一步复用。hidden state
    让优化器具备"记忆"(类似动量/自适应学习率的可学习版本)。为防止显存爆炸与梯度
    爆炸, trainer 会按 truncated BPTT 周期性 detach hidden state（见 detach_state）。

梯度如何传播:
    更新量 g(∇_t, h_t) 进入 θ_{t+1}, 进而影响后续所有 inner loss 与最终 query loss。
    meta_loss.backward() 时, 梯度沿"query loss → 各步更新量 → LSTM 权重 φ"回传,
    从而更新 φ。注意: 对 base learner 的二阶项由 base learner 的 double-backward 提供,
    本优化器自身只需一次反传。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import preprocess_gradients

# 每个参数的 hidden state: 各层的 (h, c)。
ParamState = List[Tuple[torch.Tensor, torch.Tensor]]
# 整个被优化网络的 state: name → ParamState。
MetaOptState = Dict[str, ParamState]


class LSTMOptimizer(nn.Module):
    """坐标级 LSTM 优化器。"""

    def __init__(self, hidden_size: int = 20, num_layers: int = 2,
                 preprocess: bool = True, preprocess_p: float = 10.0,
                 output_scale: float = 0.1, use_learnable_lr: bool = True,
                 update_norm_clip: Optional[float] = 1.0) -> None:
        """
        Args:
            hidden_size: LSTM 隐状态维度。
            num_layers: 堆叠的 LSTMCell 层数。
            preprocess: 是否对梯度做 log+sign 预处理。
            preprocess_p: 预处理超参 p。
            output_scale: 输出更新量的固定缩放（稳定内循环）。
            use_learnable_lr: 是否额外学习一个全局可学习步长。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.preprocess = preprocess
        self.preprocess_p = preprocess_p
        self.output_scale = output_scale
        self.update_norm_clip = (
            float(update_norm_clip)
            if update_norm_clip is not None and float(update_norm_clip) > 0
            else 0.0
        )
        self.last_raw_updates: Dict[str, torch.Tensor] = {}
        self.last_clip_scales: Dict[str, float] = {}

        input_size = 2 if preprocess else 1
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_size if layer == 0 else hidden_size
            self.cells.append(nn.LSTMCell(in_dim, hidden_size))

        # 把隐状态映射为单坐标的更新量。初始化为极小, 使训练初期更新温和、稳定。
        self.output = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)

        if use_learnable_lr:
            # 用 softplus 保证为正; 初值约 1.0。
            self.raw_lr = nn.Parameter(torch.tensor(0.5413))  # softplus(0.5413)≈1.0
        else:
            self.register_parameter("raw_lr", None)

    # ------------------------------------------------------------------ #
    @property
    def learnable_lr(self) -> torch.Tensor:
        if self.raw_lr is None:
            return torch.ones((), device=self.output.weight.device)
        return F.softplus(self.raw_lr)

    def init_state(self, params: Dict[str, torch.Tensor]) -> MetaOptState:
        """为每个参数初始化全零 hidden state。

        hidden state 形状 [numel, hidden_size]: 把参数张量的每个坐标当成 batch 元素。
        每个新任务开始前都应重新 init_state, 避免跨任务的 hidden state 泄漏。
        """
        state: MetaOptState = {}
        for name, p in params.items():
            numel = p.numel()
            device = p.device
            dtype = p.dtype
            layers: ParamState = []
            for _ in range(self.num_layers):
                h = torch.zeros(numel, self.hidden_size, device=device, dtype=dtype)
                c = torch.zeros(numel, self.hidden_size, device=device, dtype=dtype)
                layers.append((h, c))
            state[name] = layers
        return state

    # ------------------------------------------------------------------ #
    def step(self, grads: Dict[str, torch.Tensor], state: MetaOptState
             ) -> Tuple[Dict[str, torch.Tensor], MetaOptState]:
        """对所有参数执行一步坐标级更新, 返回更新量与新 hidden state。

        Args:
            grads: name → 梯度张量（与参数同形）。
            state: 当前 hidden state。

        Returns:
            (updates, new_state):
                updates[name] 与参数同形, 表示 Δθ（已含步长/缩放, 通常应取 θ+Δθ）。
                new_state 为更新后的 hidden state。
        """
        updates: Dict[str, torch.Tensor] = {}
        new_state: MetaOptState = {}
        lr = self.learnable_lr
        self.last_raw_updates = {}
        self.last_clip_scales = {}

        for name, grad in grads.items():
            shape = grad.shape
            flat = grad.reshape(-1, 1)                 # [numel, 1]
            if self.preprocess:
                x = preprocess_gradients(flat, self.preprocess_p).reshape(-1, 2)
            else:
                x = flat

            layer_states = state[name]
            new_layers: ParamState = []
            inp = x
            for layer_idx, cell in enumerate(self.cells):
                h, c = layer_states[layer_idx]
                h, c = cell(inp, (h, c))
                inp = h
                new_layers.append((h, c))

            delta = self.output(inp).reshape(shape)    # [*shape]
            # 学习到的更新方向 × 固定缩放 × 可学习步长。
            raw_update = delta * self.output_scale * lr
            update = raw_update
            clip_scale = torch.ones((), device=raw_update.device, dtype=raw_update.dtype)
            if self.update_norm_clip > 0:
                update_norm = torch.linalg.vector_norm(raw_update)
                cap = torch.as_tensor(
                    self.update_norm_clip, device=raw_update.device, dtype=raw_update.dtype)
                clip_scale = torch.clamp(
                    cap / torch.clamp(update_norm, min=1e-12), max=1.0)
                update = raw_update * clip_scale
            updates[name] = update
            self.last_raw_updates[name] = raw_update.detach()
            self.last_clip_scales[name] = float(clip_scale.detach().cpu())
            new_state[name] = new_layers

        return updates, new_state

    # ------------------------------------------------------------------ #
    @staticmethod
    def detach_state(state: MetaOptState) -> MetaOptState:
        """detach hidden state, 截断 BPTT, 防止二阶图无限增长与显存爆炸。"""
        detached: MetaOptState = {}
        for name, layers in state.items():
            detached[name] = [(h.detach(), c.detach()) for (h, c) in layers]
        return detached
