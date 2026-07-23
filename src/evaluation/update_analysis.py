"""Update and gradient diagnostics for few-shot adaptation.

The diagnostics are intentionally optimizer-agnostic: they observe the gradients
and the proposed parameter deltas produced inside the same functional inner loop
used by MetaOpt, SGD, and Adam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np
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
    anchor_update_norm: float
    residual_update_norm: float
    gate_mean: float
    trust_scale: float
    was_trust_limited: int


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
    anchor_updates: Dict[str, torch.Tensor] | None = None,
    residual_updates: Dict[str, torch.Tensor] | None = None,
    gate_values: Dict[str, float] | None = None,
    trust_scales: Dict[str, float] | None = None,
    clip_scales: Dict[str, float] | None = None,
) -> List[UpdateStats]:
    """Summarize gradient/update relation for one inner step."""
    raw_updates = raw_updates or {}
    anchor_updates = anchor_updates or {}
    residual_updates = residual_updates or {}
    gate_values = gate_values or {}
    trust_scales = trust_scales or {}
    clip_scales = clip_scales or {}
    grouped: Dict[str, List[tuple[
        str, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, float, float, float,
    ]]] = {}
    for name, grad in grads.items():
        if name not in updates:
            continue
        raw_update = raw_updates.get(name, updates[name])
        entry = (
            name, grad, updates[name], raw_update,
            anchor_updates.get(name, torch.zeros_like(updates[name])),
            residual_updates.get(name, raw_update),
            float(gate_values.get(name, float("nan"))),
            float(trust_scales.get(name, 1.0)),
            float(clip_scales.get(name, float("nan"))),
        )
        grouped.setdefault(parameter_group(name), []).append(entry)
        grouped.setdefault("all", []).append(entry)
        grouped.setdefault("parameter", []).append(entry)

    rows: List[UpdateStats] = []
    eps = 1e-12
    for group, pairs in grouped.items():
        if group == "parameter":
            iter_pairs = [
                (entry[0], [entry])
                for entry in pairs
            ]
        else:
            iter_pairs = [("", pairs)]
        for parameter, selected in iter_pairs:
            flat_g = torch.cat([item[1].detach().reshape(-1).float() for item in selected])
            flat_u = torch.cat([item[2].detach().reshape(-1).float() for item in selected])
            flat_raw = torch.cat([item[3].detach().reshape(-1).float() for item in selected])
            flat_anchor = torch.cat([
                item[4].detach().reshape(-1).float() for item in selected
            ])
            flat_residual = torch.cat([
                item[5].detach().reshape(-1).float() for item in selected
            ])
            grad_norm = torch.linalg.vector_norm(flat_g)
            update_norm = torch.linalg.vector_norm(flat_u)
            raw_norm = torch.linalg.vector_norm(flat_raw)
            denom = torch.clamp(grad_norm * update_norm, min=eps)
            cosine = torch.dot(flat_u, flat_g) / denom
            inferred_scale = torch.clamp(
                update_norm / torch.clamp(raw_norm, min=eps), max=1.0
            )
            recorded_clip_scales = [
                item[8] for item in selected if item[8] == item[8]
            ]
            recorded_clip_scale = (
                min(recorded_clip_scales)
                if recorded_clip_scales else _safe_float(inferred_scale)
            )
            recorded_trust_scale = min(item[7] for item in selected)
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
                clip_scale=float(recorded_clip_scale),
                was_clipped=int(float(recorded_clip_scale) < 0.999999),
                n_params=int(flat_g.numel()),
                anchor_update_norm=_safe_float(
                    torch.linalg.vector_norm(flat_anchor)
                ),
                residual_update_norm=_safe_float(
                    torch.linalg.vector_norm(flat_residual)
                ),
                gate_mean=float(np.nanmean([item[6] for item in selected]))
                if any(item[6] == item[6] for item in selected) else float("nan"),
                trust_scale=float(recorded_trust_scale),
                was_trust_limited=int(recorded_trust_scale < 0.999999),
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
            "anchor_update_norm": row.anchor_update_norm,
            "residual_update_norm": row.residual_update_norm,
            "gate_mean": row.gate_mean,
            "trust_scale": row.trust_scale,
            "was_trust_limited": row.was_trust_limited,
        })
    return output
