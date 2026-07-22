"""函数式前向工具。

为什么需要函数式前向:
    元学习内循环要用"任意一组参数张量"做前向, 并保留计算图, 以便:
      1. 通过 autograd.grad(create_graph=True) 求对参数的梯度;
      2. 用 Meta Optimizer 输出的更新量做 θ_{t+1} = θ_t + Δθ (非 in-place);
      3. 让 query loss 的梯度穿过整个内循环, 反传到 Meta Optimizer。
    普通 module.forward() 只能用注册在 module 上的 nn.Parameter, 无法换参数,
    且 in-place 更新会破坏计算图。torch.func.functional_call 提供"无状态前向":
    用外部传入的参数字典执行前向, 完美满足上述需求。

梯度如何传播:
    functional_call 不复制计算图, 它把传入的参数张量直接接到模型的算子上,
    因此从 logits 反传可一路回到这些参数张量(及其上游的 Meta Optimizer)。
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.func import functional_call


def functional_forward(model: torch.nn.Module, params: Dict[str, torch.Tensor],
                       x: torch.Tensor, **kwargs):
    """用外部参数字典对 model 做无状态前向。

    Args:
        model: 提供网络结构的模块（其自身参数值被 params 覆盖, 仅 buffers 复用）。
        params: name → 参数张量（须覆盖 model 的全部可训练参数）。
        x: 输入。
        **kwargs: 透传给 model.forward 的其它参数（如 return_embedding）。

    Returns:
        model.forward(x, **kwargs) 的输出。
    """
    return functional_call(model, params, args=(x,), kwargs=kwargs)


def clone_param_dict(params: Dict[str, torch.Tensor],
                     detach: bool = False) -> Dict[str, torch.Tensor]:
    """克隆参数字典。

    Args:
        params: 原始参数字典。
        detach: True 则切断与上游的计算图（用于 truncated BPTT 边界）。

    Returns:
        新的参数字典（张量为新对象, 避免 in-place 干扰）。
    """
    out: Dict[str, torch.Tensor] = {}
    for name, p in params.items():
        if detach:
            out[name] = p.detach().clone().requires_grad_(p.requires_grad)
        else:
            out[name] = p
    return out
