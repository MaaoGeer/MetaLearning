"""双向自定义 LSTM (CustomLSTM)。

为什么不用 nn.LSTM:
    nn.LSTM 在 GPU 上调用 cuDNN, 其 backward **不支持 double-backward**
    (create_graph=True)。而元学习内循环必须保留二阶计算图, 让 query loss 的
    梯度穿过"内循环更新"反传到 Meta Optimizer。若用 cuDNN LSTM, 会在二阶
    反传时报错 / 断图。因此这里用纯 tensor 算子手写 LSTM, 保证 double-backward。

实现:
    标准 LSTM 单元, 按时间步显式展开。所有运算 (matmul/sigmoid/tanh/add/mul)
    均支持二阶导数。支持多层、双向、层间 dropout。

数据流:
    输入 [B, T, in] → 逐时间步更新 (h, c) → 输出 [B, T, hidden*dir]
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLSTMCell(nn.Module):
    """单个 LSTM cell（手写, double-backward 安全）。"""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 把四个门 (i, f, g, o) 的权重合并为一个矩阵, 提高效率。
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / (self.hidden_size ** 0.5)
        for weight in (self.weight_ih, self.weight_hh):
            nn.init.uniform_(weight, -std, std)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)
        # forget gate 偏置初始化为 1, 利于长程记忆。
        with torch.no_grad():
            self.bias_hh[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, x: torch.Tensor,
                state: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, in]; state: (h [B, H], c [B, H]) → (h', c')"""
        h, c = state
        gates = (F.linear(x, self.weight_ih, self.bias_ih)
                 + F.linear(h, self.weight_hh, self.bias_hh))
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        # 注意: 使用非 inplace 运算 (无 mul_/add_), 避免破坏二阶计算图。
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class CustomLSTM(nn.Module):
    """多层双向 LSTM。"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bidirectional: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = dropout

        self.fwd_cells = nn.ModuleList()
        self.bwd_cells = nn.ModuleList() if bidirectional else None
        for layer in range(num_layers):
            layer_in = input_size if layer == 0 else hidden_size * self.num_directions
            self.fwd_cells.append(CustomLSTMCell(layer_in, hidden_size))
            if bidirectional:
                self.bwd_cells.append(CustomLSTMCell(layer_in, hidden_size))

        self.output_size = hidden_size * self.num_directions

    def _run_direction(self, cell: CustomLSTMCell, seq: List[torch.Tensor],
                       reverse: bool) -> List[torch.Tensor]:
        """沿一个方向展开 LSTM, 返回每个时间步的 h 列表。"""
        batch = seq[0].shape[0]
        device = seq[0].device
        dtype = seq[0].dtype
        h = torch.zeros(batch, cell.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch, cell.hidden_size, device=device, dtype=dtype)
        order = reversed(range(len(seq))) if reverse else range(len(seq))
        outputs: List[Optional[torch.Tensor]] = [None] * len(seq)
        for t in order:
            h, c = cell(seq[t], (h, c))
            outputs[t] = h
        return outputs  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, in] → [B, T, hidden*dir]"""
        # 拆成时间步列表, 便于显式展开。
        seq = [x[:, t, :] for t in range(x.shape[1])]
        layer_input = seq
        for layer in range(self.num_layers):
            fwd_out = self._run_direction(self.fwd_cells[layer], layer_input, reverse=False)
            if self.bidirectional:
                bwd_out = self._run_direction(self.bwd_cells[layer], layer_input, reverse=True)
                step_out = [torch.cat([f, b], dim=1) for f, b in zip(fwd_out, bwd_out)]
            else:
                step_out = fwd_out
            # 层间 dropout（最后一层不加）。
            if self.dropout > 0.0 and layer < self.num_layers - 1:
                step_out = [F.dropout(s, p=self.dropout, training=self.training)
                            for s in step_out]
            layer_input = step_out
        return torch.stack(layer_input, dim=1)  # [B, T, hidden*dir]
