"""设备管理: CUDA / CPU / 多 GPU 解析与张量搬运。"""

from __future__ import annotations

from typing import Any, Dict

import torch


def resolve_device(prefer: str = "auto") -> torch.device:
    """解析训练设备。

    Args:
        prefer: "auto" | "cuda" | "cuda:0" | "cpu"。auto 时优先 CUDA。

    Returns:
        torch.device 实例。
    """
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer.startswith("cuda") and not torch.cuda.is_available():
        # 优雅降级, 避免在无 GPU 机器上直接崩溃。
        return torch.device("cpu")
    return torch.device(prefer)


def move_to_device(obj: Any, device: torch.device) -> Any:
    """递归地把张量 / dict / list / tuple 搬到指定设备。"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj


def gpu_count() -> int:
    """返回可用 GPU 数量。"""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0
