"""Update and gradient diagnostics for few-shot adaptation.

The diagnostics are intentionally optimizer-agnostic: they observe the gradients
and the proposed parameter deltas produced inside the same functional inner loop
used by MetaOpt, SGD, and Adam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import torch


@dataclass
class UpdateStats:
    """Aggregated statistics for one method/task/inner-step."""

    step: int
    group: str
    parameter: str
    grad_norm: float
    raw_update_norm: float
    update_norm: float
    clipped_update_norm: float
    update_to_grad_ratio: float
    raw_update_to_grad_ratio: float
    cosine_update_grad: float
    clip_scale: float
    was_clipped: int
    n_params: int


@dataclass
class UpdateTrace:
    """Per-task update trace collected during adaptation."""

    rows: List[UpdateStats] = field(default_factory=list)


def parameter_group(name: str) -> str:
    """Map a parameter name to a paper-friendly group."""
    if name.startswith("lstm."):
        prefix = "lstm"
    elif name.startswith("classifier."):
        prefix = "classifier"
    else:
        prefix = name.split(".", 1)[0]

    suffix = "bias" if name.endswith("bias") else "weight"
    return f"{prefix}.{suffix}"


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def summarize_update_step(
    step: int,
    grads: Dict[str, torch.Tensor],
    updates: Dict[str, torch.Tensor],
    raw_updates: Dict[str, torch.Tensor] | None = None,
) -> List[UpdateStats]:
    """Summarize gradient/update relation for one inner step."""
    raw_updates = raw_updates or {}
    grouped: Dict[str, List[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    for name, grad in grads.items():
        if name not in updates:
            continue
        raw_update = raw_updates.get(name, updates[name])
        grouped.setdefault(parameter_group(name), []).append((name, grad, updates[name], raw_update))
        grouped.setdefault("all", []).append((name, grad, updates[name], raw_update))
        grouped.setdefault("parameter", []).append((name, grad, updates[name], raw_update))

    rows: List[UpdateStats] = []
    eps = 1e-12
    for group, pairs in grouped.items():
        if group == "parameter":
            iter_pairs = [
                (name, [(name, grad, update, raw_update)])
                for name, grad, update, raw_update in pairs
            ]
        else:
            iter_pairs = [("", pairs)]
        for parameter, selected in iter_pairs:
            flat_g = torch.cat([g.detach().reshape(-1).float() for _, g, _, _ in selected])
            flat_u = torch.cat([u.detach().reshape(-1).float() for _, _, u, _ in selected])
            flat_raw = torch.cat([r.detach().reshape(-1).float() for _, _, _, r in selected])
            grad_norm = torch.linalg.vector_norm(flat_g)
            update_norm = torch.linalg.vector_norm(flat_u)
            raw_norm = torch.linalg.vector_norm(flat_raw)
            denom = torch.clamp(grad_norm * update_norm, min=eps)
            cosine = torch.dot(flat_u, flat_g) / denom
            clip_scale = torch.clamp(update_norm / torch.clamp(raw_norm, min=eps), max=1.0)
            rows.append(UpdateStats(
                step=int(step),
                group=group,
                parameter=parameter,
                grad_norm=_safe_float(grad_norm),
                raw_update_norm=_safe_float(raw_norm),
                update_norm=_safe_float(update_norm),
                clipped_update_norm=_safe_float(update_norm),
                update_to_grad_ratio=_safe_float(update_norm / torch.clamp(grad_norm, min=eps)),
                raw_update_to_grad_ratio=_safe_float(raw_norm / torch.clamp(grad_norm, min=eps)),
                cosine_update_grad=_safe_float(cosine),
                clip_scale=_safe_float(clip_scale),
                was_clipped=int(_safe_float(raw_norm - update_norm) > 1e-8),
                n_params=int(flat_g.numel()),
            ))
    return rows


def update_rows_to_dicts(
    rows: Iterable[UpdateStats],
    *,
    experiment: str,
    shot: int,
    method: str,
    task_id: int,
) -> List[dict]:
    """Convert update stats to CSV/JSON-friendly rows."""
    output = []
    for row in rows:
        output.append({
            "experiment": experiment,
            "shot": int(shot),
            "method": method,
            "task_id": int(task_id),
            "step": row.step,
            "group": row.group,
            "parameter": row.parameter,
            "grad_norm": row.grad_norm,
            "raw_update_norm": row.raw_update_norm,
            "update_norm": row.update_norm,
            "clipped_update_norm": row.clipped_update_norm,
            "update_to_grad_ratio": row.update_to_grad_ratio,
            "raw_update_to_grad_ratio": row.raw_update_to_grad_ratio,
            "cosine_update_grad": row.cosine_update_grad,
            "clip_scale": row.clip_scale,
            "was_clipped": row.was_clipped,
            "n_params": row.n_params,
        })
    return output
