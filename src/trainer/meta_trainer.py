"""Meta-training loop for the LSTM + Meta Optimizer + Few-shot mainline."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..data.task_sampler import FewShotTaskSampler
from ..evaluation.evaluator import FewShotEvaluator
from ..evaluation.metrics import ClassificationMetrics
from ..meta_learning.inner_loop import InnerLoop
from ..meta_learning.outer_loop import OuterLoop
from ..meta_optimizer.lstm_optimizer import LSTMOptimizer
from ..utils.config import Config
from ..utils.logger import get_logger
from .callbacks import CheckpointManager, EarlyStopping

logger = get_logger(__name__)


@dataclass
class TrainHistory:
    """Training curves recorded at evaluation epochs."""

    meta_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    val_auc: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)


class MetaTrainer:
    """End-to-end meta trainer.

    The base learner initialization is random and fixed. The outer optimizer updates only
    the learned optimizer parameters phi, so later MetaOpt/Adam/SGD comparisons share the
    same theta0 and differ only by the adaptation update rule.
    """

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        meta_opt: LSTMOptimizer,
        train_sampler: FewShotTaskSampler,
        val_sampler: FewShotTaskSampler,
        device: torch.device,
        adapt_names: Optional[List[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.meta_opt = meta_opt.to(device)
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.device = device
        self.adapt_names = adapt_names

        meta_cfg = cfg.meta
        train_cfg = cfg.train
        for loss_key in ("inner_loss", "query_loss"):
            if loss_key in meta_cfg:
                value = str(meta_cfg.get(loss_key)).lower()
                if value not in {"cross_entropy", "ce"}:
                    raise ValueError(f"Only cross_entropy loss is supported, got meta.{loss_key}={value}")

        self.inner_loop = InnerLoop(
            model=self.model,
            meta_opt=self.meta_opt,
            inner_steps=int(meta_cfg.inner_steps),
            tbptt_steps=int(meta_cfg.get("tbptt_steps", 0)),
            first_order=bool(meta_cfg.get("first_order", False)),
        )
        self.outer_loop = OuterLoop(self.model, self.inner_loop, adapt_names=adapt_names)

        meta_opt_params = list(self.meta_opt.parameters())
        if not meta_opt_params:
            raise ValueError("Meta optimizer has no learnable parameters.")
        self.optimizer = torch.optim.Adam(
            [{"params": meta_opt_params, "lr": float(meta_cfg.meta_optimizer_lr)}]
        )
        self._phi_group_index = 0

        self.amp_enabled = bool(cfg.device.get("amp", False)) and device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        if bool(cfg.device.get("amp", False)) and device.type != "cuda":
            logger.warning("AMP is CUDA-only; current device=%s, disabled.", device.type)
        if self.amp_enabled and not bool(meta_cfg.get("first_order", False)):
            logger.warning("AMP with second-order gradients can be numerically unstable.")

        self.grad_clip = float(train_cfg.get("grad_clip", 0.0) or 0.0)
        self.meta_batch_size = int(cfg.data.meta_batch_size)
        self.tasks_per_epoch = int(cfg.data.tasks_per_epoch)
        self.meta_epochs = int(train_cfg.meta_epochs)
        self.log_interval = int(train_cfg.get("log_interval", 20))
        self.eval_interval = int(train_cfg.get("eval_interval", 1))
        self.eval_tasks = int(train_cfg.get("eval_tasks", 50))
        self.fixed_validation_tasks = bool(train_cfg.get("fixed_validation_tasks", True))
        self.validation_task_audit_path = train_cfg.get("validation_task_audit_path", None)
        self._validation_task_pool = None

        es_cfg = train_cfg.early_stopping
        self.early_stopping = (
            EarlyStopping(patience=int(es_cfg.patience), mode=str(es_cfg.mode))
            if bool(es_cfg.enabled)
            else None
        )
        self.monitor_metric = str(es_cfg.get("metric", "f1"))

        ck_cfg = train_cfg.checkpoint
        self.ckpt = CheckpointManager(
            ckpt_dir=str(ck_cfg.dir),
            save_best=bool(ck_cfg.save_best),
            save_last=bool(ck_cfg.save_last),
        )
        self.monitor_ckpt = str(ck_cfg.get("monitor", self.monitor_metric))

        self.tb_writer = None
        tb_cfg = train_cfg.get("tensorboard", None)
        if tb_cfg is not None and bool(tb_cfg.get("enabled", False)):
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=str(tb_cfg.dir))
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not initialize TensorBoard: %s", exc)

        self.evaluator = FewShotEvaluator(
            self.model,
            self.meta_opt,
            inner_steps=int(meta_cfg.inner_steps),
            device=device,
            adapt_names=adapt_names,
        )
        self.history = TrainHistory()
        self.best_metric: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_meta_opt_state: Optional[Dict[str, torch.Tensor]] = None
        self.global_step = 0

    def _window_row_ids(self, window_ids: List[int]) -> List[List[int]]:
        dataset = getattr(self.val_sampler, "dataset", None)
        row_ids = getattr(dataset, "row_ids", None)
        if row_ids is None:
            return []
        out: List[List[int]] = []
        for wid in window_ids:
            out.append([int(x) for x in row_ids[int(wid)].tolist()])
        return out

    def _write_validation_task_audit(self, tasks: List) -> None:
        if not self.validation_task_audit_path:
            return
        rows = []
        for task_id, task in enumerate(tasks):
            rows.append({
                "task_id": int(task_id),
                "global_classes": [int(x) for x in task.global_classes],
                "attack_class": (
                    int(task.global_classes[1])
                    if len(task.global_classes) > 1 else None
                ),
                "shot": int(getattr(self.val_sampler, "k_shot", 0)),
                "support_window_ids": [int(x) for x in task.support_window_ids],
                "query_window_ids": [int(x) for x in task.query_window_ids],
                "support_row_ids": self._window_row_ids(task.support_window_ids),
                "query_row_ids": self._window_row_ids(task.query_window_ids),
                "support_labels": [int(x) for x in task.support_y.tolist()],
                "query_labels": [int(x) for x in task.query_y.tolist()],
            })
        audit_dir = os.path.dirname(str(self.validation_task_audit_path))
        if audit_dir:
            os.makedirs(audit_dir, exist_ok=True)
        with open(str(self.validation_task_audit_path), "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)
        logger.info("Saved fixed validation task audit: %s", self.validation_task_audit_path)

    def _prepare_validation_tasks(self) -> Optional[List]:
        if not self.fixed_validation_tasks:
            return None
        if self._validation_task_pool is None:
            self._validation_task_pool = [
                self.val_sampler.sample_task() for _ in range(self.eval_tasks)
            ]
            self._write_validation_task_audit(self._validation_task_pool)
            logger.info(
                "Fixed validation task pool enabled: %d tasks",
                len(self._validation_task_pool),
            )
        return self._validation_task_pool

    def _select_metric(self, metrics: ClassificationMetrics) -> float:
        value = getattr(metrics, self.monitor_metric, None)
        if value is None or value != value:
            return metrics.f1
        return float(value)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        n_updates = max(self.tasks_per_epoch // self.meta_batch_size, 1)
        loss_sum, acc_sum, support_loss_sum = 0.0, 0.0, 0.0
        nonfinite_count = 0
        clip_ratio_sum = 0.0
        t0 = time.time()

        for update_idx in range(n_updates):
            tasks = [
                task.to(self.device)
                for task in self.train_sampler.sample_batch(self.meta_batch_size)
            ]

            self.optimizer.zero_grad(set_to_none=True)
            # theta0 is fixed but participates in higher-order graphs; clear leaf grads
            # that are intentionally not stepped by the outer optimizer.
            self.model.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                result = self.outer_loop.run_meta_batch(tasks)
            meta_loss = result.meta_loss
            if not torch.isfinite(meta_loss.detach()):
                nonfinite_count += 1

            self.scaler.scale(meta_loss).backward()
            for param in self.meta_opt.parameters():
                if param.grad is not None:
                    nonfinite_count += int((~torch.isfinite(param.grad)).sum().detach().cpu())

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.meta_opt.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_sum += float(meta_loss.detach())
            acc_sum += result.query_acc
            support_loss_sum += float(result.avg_support_loss)
            clip_scales = list(getattr(self.meta_opt, "last_clip_scales", {}).values())
            clip_ratio = (
                sum(1 for value in clip_scales if float(value) < 0.999999) / len(clip_scales)
                if clip_scales else 0.0
            )
            clip_ratio_sum += clip_ratio
            self.global_step += 1

            if self.tb_writer is not None:
                self.tb_writer.add_scalar("train/meta_loss", float(meta_loss.detach()), self.global_step)
                self.tb_writer.add_scalar("train/query_acc", result.query_acc, self.global_step)
                self.tb_writer.add_scalar("train/support_loss", result.avg_support_loss, self.global_step)
                self.tb_writer.add_scalar("train/update_clip_ratio", clip_ratio, self.global_step)

            if (update_idx + 1) % self.log_interval == 0:
                lr_phi = self.optimizer.param_groups[self._phi_group_index]["lr"]
                logger.info(
                    "epoch %d | update %d/%d | meta_loss=%.4f | support_loss=%.4f "
                    "| query_acc=%.4f | lr_phi=%.2e | clip_ratio=%.4f | nonfinite=%d",
                    epoch,
                    update_idx + 1,
                    n_updates,
                    loss_sum / (update_idx + 1),
                    support_loss_sum / (update_idx + 1),
                    acc_sum / (update_idx + 1),
                    lr_phi,
                    clip_ratio_sum / (update_idx + 1),
                    nonfinite_count,
                )

        avg_loss = loss_sum / n_updates
        avg_acc = acc_sum / n_updates
        avg_support_loss = support_loss_sum / n_updates
        avg_clip_ratio = clip_ratio_sum / n_updates
        logger.info(
            "[epoch %d] train done | avg_meta_loss=%.4f | avg_support_loss=%.4f "
            "| avg_query_acc=%.4f | avg_clip_ratio=%.4f | nonfinite=%d | %.1fs",
            epoch,
            avg_loss,
            avg_support_loss,
            avg_acc,
            avg_clip_ratio,
            nonfinite_count,
            time.time() - t0,
        )
        return {
            "meta_loss": avg_loss,
            "support_loss": avg_support_loss,
            "query_acc": avg_acc,
            "clip_ratio": avg_clip_ratio,
            "nonfinite": float(nonfinite_count),
        }

    def _build_state(self, epoch: int) -> Dict:
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "meta_optimizer": self.meta_opt.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.cfg.to_dict(),
        }

    def fit(self) -> TrainHistory:
        logger.info(
            "Start meta-training: epochs=%d, tasks/epoch=%d, meta_batch=%d, device=%s",
            self.meta_epochs,
            self.tasks_per_epoch,
            self.meta_batch_size,
            self.device,
        )
        validation_tasks = self._prepare_validation_tasks()
        for epoch in range(1, self.meta_epochs + 1):
            train_stats = self.train_one_epoch(epoch)

            is_best = False
            if epoch % self.eval_interval == 0:
                if validation_tasks is not None:
                    val_metrics = self.evaluator.evaluate_tasks(
                        validation_tasks, n_way=self.val_sampler.n_way, desc=f"val@{epoch}"
                    )
                else:
                    val_metrics = self.evaluator.evaluate(
                        self.val_sampler, num_tasks=self.eval_tasks, desc=f"val@{epoch}"
                    )
                monitor_value = self._select_metric(val_metrics)

                self.history.epochs.append(epoch)
                self.history.meta_loss.append(train_stats["meta_loss"])
                self.history.train_acc.append(train_stats["query_acc"])
                self.history.val_acc.append(val_metrics.accuracy)
                self.history.val_f1.append(val_metrics.f1)
                self.history.val_auc.append(
                    val_metrics.roc_auc if val_metrics.roc_auc is not None else float("nan")
                )

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("val/accuracy", val_metrics.accuracy, epoch)
                    self.tb_writer.add_scalar("val/f1", val_metrics.f1, epoch)
                    if val_metrics.roc_auc is not None:
                        self.tb_writer.add_scalar("val/roc_auc", val_metrics.roc_auc, epoch)

                if self.best_metric is None or monitor_value > self.best_metric:
                    self.best_metric = monitor_value
                    self.best_epoch = epoch
                    self.best_model_state = copy.deepcopy(
                        {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                    )
                    self.best_meta_opt_state = copy.deepcopy(
                        {k: v.detach().cpu() for k, v in self.meta_opt.state_dict().items()}
                    )
                    is_best = True

                if self.early_stopping is not None:
                    self.early_stopping.step(monitor_value)

            self.ckpt.save(self._build_state(epoch), is_best=is_best)

            if self.early_stopping is not None and self.early_stopping.should_stop:
                logger.info("early stopping at epoch %d", epoch)
                break

        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        logger.info(
            "Meta-training finished. best %s = %.4f",
            self.monitor_metric,
            self.best_metric if self.best_metric is not None else float("nan"),
        )
        return self.history
