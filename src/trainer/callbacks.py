"""训练回调: 早停 与 checkpoint 管理。"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """基于监控指标的早停。"""

    def __init__(self, patience: int = 10, mode: str = "max",
                 min_delta: float = 1e-4) -> None:
        """
        Args:
            patience: 容忍多少次评估无提升。
            mode: "max"（越大越好, 如 f1）或 "min"（越小越好, 如 loss）。
            min_delta: 视为"提升"的最小变化量。
        """
        assert mode in {"max", "min"}
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.num_bad = 0
        self.should_stop = False

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        """更新状态, 返回本次是否为新的最优。"""
        improved = self._is_better(value)
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
                logger.info("早停触发: 连续 %d 次评估无提升 (best=%.4f)。",
                            self.patience, self.best if self.best is not None else float("nan"))
        return improved


class CheckpointManager:
    """checkpoint 保存与加载。"""

    def __init__(self, ckpt_dir: str, save_best: bool = True,
                 save_last: bool = True) -> None:
        self.ckpt_dir = ckpt_dir
        self.save_best = save_best
        self.save_last = save_last
        os.makedirs(ckpt_dir, exist_ok=True)

    def save(self, state: Dict[str, Any], is_best: bool, tag: str = "last") -> str:
        """保存 checkpoint, 返回路径。"""
        last_path = os.path.join(self.ckpt_dir, f"{tag}.pt")
        if self.save_last:
            torch.save(state, last_path)
        if is_best and self.save_best:
            best_path = os.path.join(self.ckpt_dir, "best.pt")
            torch.save(state, best_path)
            logger.info("保存最优 checkpoint → %s", best_path)
        return last_path

    @staticmethod
    def load(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        """加载 checkpoint。"""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("weights_only=True 加载失败, 回退到完整加载(请确保文件可信): %s", exc)
            return torch.load(path, map_location=map_location, weights_only=False)
