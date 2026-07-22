"""梯度预处理 (Andrychowicz et al. 2016, Appendix A)。

为什么需要预处理:
    原始梯度的量级跨越多个数量级, 直接喂给 LSTM 数值不稳定、难以学习。
    论文提出对梯度做 log + sign 的两维编码, 把量级与方向解耦, 显著提升
    Meta Optimizer 的可训练性与泛化。

公式 (对每个坐标 g, 参数 p>0):
    若 |g| >= e^{-p}:   ( log|g| / p ,  sign(g) )
    否则:               ( -1        ,  e^p · g  )

实现:
    全程用可微算子 (log/abs/sign/where), 不破坏二阶计算图。
    sign 在 0 处导数为 0, 但仅作为方向特征, 不影响 Meta Optimizer 的元梯度
    通过另一支路 (log 项 / 小梯度线性项) 正常回传。
"""

from __future__ import annotations

import math

import torch


def preprocess_gradients(grad: torch.Tensor, p: float = 10.0) -> torch.Tensor:
    """对梯度做 log+sign 两维编码。

    Args:
        grad: 任意形状的梯度张量。
        p: 预处理超参数（论文取 10）。

    Returns:
        在最后一维拼接的 [*grad.shape, 2] 张量。
    """
    eps = math.exp(-p)
    abs_grad = grad.abs()
    big_mask = (abs_grad >= eps).to(grad.dtype)

    # 大梯度分支: (log|g|/p, sign(g))
    safe_abs = abs_grad.clamp_min(eps)          # 避免 log(0)
    log_term = torch.log(safe_abs) / p
    sign_term = torch.sign(grad)

    # 小梯度分支: (-1, e^p · g)
    neg_one = torch.full_like(grad, -1.0)
    scaled = math.exp(p) * grad

    first = big_mask * log_term + (1.0 - big_mask) * neg_one
    second = big_mask * sign_term + (1.0 - big_mask) * scaled

    return torch.stack([first, second], dim=-1)
