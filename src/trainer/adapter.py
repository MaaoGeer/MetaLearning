"""Few-shot adaptation adapter for MetaOpt/SGD/Adam evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..data.task_sampler import MetaTask
from ..evaluation.adaptation_speed import SpeedResult, compute_speed
from ..evaluation.metrics import ClassificationMetrics, compute_metrics
from ..evaluation.update_analysis import UpdateTrace, summarize_update_step
from ..meta_learning.functional import functional_forward
from ..meta_learning.inner_loop import InnerLoop


@dataclass
class AdaptOutcome:
    """Full result for one support-set adaptation run."""

    speed: SpeedResult
    final_metrics: ClassificationMetrics
    final_logits: torch.Tensor = None
    final_targets: torch.Tensor = None
    update_trace: UpdateTrace = None
    diagnostics: List[dict] = None
    support_losses: List[float] = None


class FewShotAdapter:
    """Evaluate few-shot adaptation with a pluggable functional optimizer."""

    def __init__(self, model: nn.Module, device: torch.device,
                 loss_fn: Optional[nn.Module] = None) -> None:
        self.model = model
        self.device = device
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    @torch.enable_grad()
    def adapt_once(
        self,
        init_params: Dict[str, torch.Tensor],
        task: MetaTask,
        optimizer,
        adapt_names: List[str],
        n_way: int,
        max_steps: int = 200,
        target_f1_grid: Optional[List[float]] = None,
        attack_class_indices: Optional[List[int]] = None,
        collect_update_stats: bool = False,
    ) -> AdaptOutcome:
        """Adapt from shared theta0 and record query metrics at every step."""
        target_f1_grid = target_f1_grid or [0.75, 0.80, 0.85, 0.90]
        task = task.to(self.device)
        self.model.eval()

        metric_traj: Dict[str, List[float]] = {
            "accuracy": [],
            "macro_f1": [],
            "weighted_f1": [],
            "precision": [],
            "recall": [],
            "pr_auc": [],
            "attack_recall": [],
        }
        last_logits_holder: Dict[str, torch.Tensor] = {}
        diagnostics: List[dict] = []
        support_loss_by_step: Dict[int, float] = {}

        def evaluate_params(step: int, merged_params: Dict[str, torch.Tensor]) -> None:
            with torch.no_grad():
                q_logits = functional_forward(self.model, merged_params, task.query_x)
            metrics = compute_metrics(
                q_logits.detach().float().cpu(), task.query_y.cpu(),
                num_classes=n_way, attack_class_indices=attack_class_indices)
            logits_cpu = q_logits.detach().float().cpu()
            targets_cpu = task.query_y.detach().cpu()
            preds = logits_cpu.argmax(dim=1)
            pos_rate = float((preds == 1).float().mean()) if n_way == 2 else float("nan")
            normal_recall = float("nan")
            if n_way == 2:
                normal_mask = targets_cpu == 0
                if int(normal_mask.sum()) > 0:
                    normal_recall = float((preds[normal_mask] == 0).float().mean())
            diagnostics.append({
                "step": int(step),
                "logits_mean": float(logits_cpu.mean()),
                "logits_std": float(logits_cpu.std(unbiased=False)),
                "prediction_positive_rate": pos_rate,
                "accuracy": float(metrics.accuracy),
                "macro_f1": float(metrics.macro_f1 if metrics.macro_f1 is not None else metrics.f1),
                "attack_recall": (
                    float(metrics.attack_recall)
                    if metrics.attack_recall is not None else float("nan")
                ),
                "normal_recall": normal_recall,
            })
            metric_traj["accuracy"].append(metrics.accuracy)
            metric_traj["macro_f1"].append(metrics.macro_f1 if metrics.macro_f1 is not None else metrics.f1)
            metric_traj["weighted_f1"].append(
                metrics.weighted_f1 if metrics.weighted_f1 is not None else float("nan"))
            metric_traj["precision"].append(metrics.precision)
            metric_traj["recall"].append(metrics.recall)
            metric_traj["pr_auc"].append(
                float(metrics.pr_auc) if metrics.pr_auc is not None else float("nan"))
            metric_traj["attack_recall"].append(
                float(metrics.attack_recall) if metrics.attack_recall is not None else float("nan"))
            last_logits_holder["logits"] = q_logits.detach().float().cpu()

        evaluate_params(0, init_params)

        def record_fn(step: int, merged_params: Dict[str, torch.Tensor]) -> None:
            evaluate_params(step + 1, merged_params)
            if (step + 1) in support_loss_by_step and diagnostics:
                diagnostics[-1]["support_loss"] = support_loss_by_step[step + 1]

        update_trace = UpdateTrace() if collect_update_stats else None
        if collect_update_stats:
            self._adapt_with_update_trace(
                init_params, task, optimizer, adapt_names, max_steps,
                record_fn, update_trace, support_loss_by_step)
        else:
            inner = InnerLoop(self.model, optimizer, inner_steps=max_steps,
                              tbptt_steps=0, first_order=True, loss_fn=self.loss_fn)
            inner_result = inner.adapt(
                init_params, task, adapt_names=adapt_names, record_fn=record_fn)
            support_loss_by_step.update({
                idx + 1: float(value)
                for idx, value in enumerate(inner_result.support_losses)
            })

        speed = compute_speed(
            metric_traj["macro_f1"][1:], target_f1_grid, max_steps=max_steps)
        speed.metric_trajectories = metric_traj
        final_targets = task.query_y.cpu()
        final_logits = last_logits_holder["logits"]
        final_metrics = compute_metrics(
            final_logits, final_targets,
            num_classes=n_way, attack_class_indices=attack_class_indices)
        return AdaptOutcome(
            speed=speed,
            final_metrics=final_metrics,
            final_logits=final_logits,
            final_targets=final_targets,
            update_trace=update_trace,
            diagnostics=diagnostics,
            support_losses=[
                support_loss_by_step[step]
                for step in sorted(support_loss_by_step)
            ],
        )

    def _adapt_with_update_trace(
        self,
        init_params: Dict[str, torch.Tensor],
        task: MetaTask,
        optimizer,
        adapt_names: List[str],
        max_steps: int,
        record_fn,
        update_trace: UpdateTrace,
        support_loss_by_step: Dict[int, float],
    ) -> None:
        """Run first-order adaptation while recording gradient/update statistics."""
        full = dict(init_params)
        adapt_set = set(adapt_names)
        frozen = {name: param for name, param in full.items() if name not in adapt_set}
        adaptable = {name: full[name] for name in adapt_names}
        state = optimizer.init_state(adaptable)

        for step in range(max_steps):
            merged = {**frozen, **adaptable}
            logits = functional_forward(self.model, merged, task.support_x)
            loss = self.loss_fn(logits, task.support_y)
            support_loss_by_step[step + 1] = float(loss.detach().cpu())
            grads = torch.autograd.grad(
                loss, list(adaptable.values()),
                create_graph=False, retain_graph=False, allow_unused=False)
            grad_dict = {name: grad for name, grad in zip(adaptable.keys(), grads)}
            updates, state = optimizer.step(grad_dict, state)
            raw_updates = getattr(optimizer, "last_raw_updates", None)
            update_trace.rows.extend(
                summarize_update_step(step + 1, grad_dict, updates, raw_updates=raw_updates)
            )
            adaptable = {name: adaptable[name] + updates[name] for name in adaptable}
            record_fn(step, {**frozen, **adaptable})
            if step != max_steps - 1:
                adaptable = {
                    name: param.detach().clone().requires_grad_(True)
                    for name, param in adaptable.items()
                }
                state = optimizer.detach_state(state)
