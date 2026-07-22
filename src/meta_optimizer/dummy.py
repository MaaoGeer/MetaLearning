"""SGD-like dummy meta optimizer for adapter-path diagnostics."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

ParamState = List[Tuple[torch.Tensor, torch.Tensor]]
DummyState = Dict[str, ParamState]


class DummyMetaOptimizer(nn.Module):
    """Return ``delta = -lr * grad`` through the MetaOpt interface."""

    def __init__(self, lr: float = 0.1) -> None:
        super().__init__()
        self.lr = float(lr)
        self.last_raw_updates: Dict[str, torch.Tensor] = {}
        self.last_clip_scales: Dict[str, float] = {}

    def init_state(self, params: Dict[str, torch.Tensor]) -> DummyState:
        return {name: [] for name in params}

    def step(
        self,
        grads: Dict[str, torch.Tensor],
        state: DummyState,
    ) -> Tuple[Dict[str, torch.Tensor], DummyState]:
        updates = {name: -self.lr * grad for name, grad in grads.items()}
        self.last_raw_updates = {
            name: update.detach() for name, update in updates.items()
        }
        self.last_clip_scales = {name: 1.0 for name in updates}
        return updates, {name: state.get(name, []) for name in grads}

    @staticmethod
    def detach_state(state: DummyState) -> DummyState:
        return state
