"""随机种子固定, 保证实验可复现。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """固定 Python / NumPy / PyTorch 的随机种子。

    Args:
        seed: 随机种子。
        deterministic: 是否启用 cuDNN 确定性模式（牺牲少量速度换可复现）。

    说明:
        元学习需要 create_graph=True 的二阶梯度, 我们的 Base Learner 使用
        自定义 CustomLSTM（非 cuDNN）来保证 double-backward, 因此这里即便
        关闭 cudnn.benchmark 也不影响元梯度正确性。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
