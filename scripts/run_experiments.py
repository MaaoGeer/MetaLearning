"""Offline temporal few-shot unknown-attack adaptation experiments.

指标:
  1. Adaptation Speed: 固定 support 适配, query 上每步 macro-F1, 达到阈值的步数
  2. Final Generalization: 相同 init/support/query 上的最终 Acc/P/R/F1/AUC/attack_recall

公平性:
  - 相同 random initialization / 相同 test tasks
  - SGD/Adam LR 仅在 validation tasks 上网格搜索
  - test tasks 不参与 LR/阈值选择
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build import build_meta_model, build_meta_optimizer, load_artifacts  # noqa: E402
from src.data.pipeline import build_pipeline, cache_key as data_cache_key  # noqa: E402
from src.data.task_sampler import MetaTask  # noqa: E402
from src.evaluation.adaptation_speed import (  # noqa: E402
    adaptation_selection_key,
    summarize_adaptation,
)
from src.evaluation.metrics import aggregate_logits, compute_metrics  # noqa: E402
from src.evaluation.task_manifest import (  # noqa: E402
    assert_manifest_split_isolation,
    load_tasks_from_manifest,
    manifest_reuse_statistics,
    read_task_manifest,
    sha256_file,
    tensor_state_sha256,
    write_task_manifest,
)
from src.evaluation.prediction_artifact import write_prediction_trajectories  # noqa: E402
from src.evaluation.update_analysis import update_rows_to_dicts  # noqa: E402
from src.meta_optimizer.handcrafted import HandcraftedOptimizer  # noqa: E402
from src.trainer.adapter import AdaptOutcome, FewShotAdapter  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.provenance import raw_data_catalog, write_provenance_receipt  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.visualization.plots import (  # noqa: E402
    plot_adaptation_curves,
    plot_confusion_matrix,
    plot_speed_bars,
)

logger = get_logger("run_experiments")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Few-shot 未知攻击适配实验")
    p.add_argument("--artifacts", default="checkpoints/meta_artifacts.pt")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--out", default="outputs/experiments")
    p.add_argument(
        "--task-manifest",
        default=None,
        help="Explicit test-task manifest. Requires a single matching compare.shots entry.",
    )
    p.add_argument(
        "--validation-task-manifest",
        default=None,
        help="Explicit validation-task manifest used for LR/stop-step selection.",
    )
    p.add_argument(
        "--test-task-manifest",
        default=None,
        help="Explicit test-task manifest. Test is never used for selection.",
    )
    p.add_argument(
        "--phase",
        choices=["validation", "test", "both"],
        default="both",
        help=(
            "Run validation-only selection, one-time test from a frozen selection "
            "receipt, or the deprecated compatibility path that runs both."
        ),
    )
    p.add_argument(
        "--selection-receipt",
        default=None,
        help="validation_selection.json produced by --phase validation; required for --phase test.",
    )
    return p.parse_args()


def _sample_tasks(sampler, n: int) -> List[MetaTask]:
    return [sampler.sample_task() for _ in range(n)]


def _manifest_protocol_check(
    manifest: dict,
    *,
    split: str,
    shot: int,
    q_query: int,
    artifact_path: str,
    artifact: dict,
    unknown_class: str,
) -> None:
    protocol = manifest["protocol"]
    if int(protocol["shot"]) != int(shot):
        raise ValueError(
            f"{split} manifest shot={protocol['shot']} does not match {shot}"
        )
    if int(protocol["q_query"]) != int(q_query):
        raise ValueError(
            f"{split} manifest q_query={protocol['q_query']} does not match {q_query}"
        )
    if str(protocol["split"]) != split:
        raise ValueError(
            f"expected {split!r} manifest, got {protocol['split']!r}"
        )
    metadata = manifest.get("metadata", {})
    if metadata.get("unknown_class", unknown_class) != unknown_class:
        raise ValueError(f"{split} manifest unknown_class does not match artifact")
    schema = int(manifest.get("schema_version", 1))
    if schema == 1:
        expected = str(manifest.get("base_checkpoint_sha256", ""))
        actual = sha256_file(artifact_path)
        if expected and expected != actual:
            raise ValueError(
                f"{split} legacy manifest artifact SHA256 mismatch"
            )
    else:
        expected_init = str(manifest.get("base_initialization_sha256", ""))
        actual_init = tensor_state_sha256(artifact["meta_init_state"])
        if expected_init and expected_init != actual_init:
            raise ValueError(
                f"{split} manifest base initialization SHA256 mismatch"
            )


def _write_run_manifest(
    path: Path,
    tasks: List[MetaTask],
    *,
    split: str,
    shot: int,
    q_query: int,
    task_seed: int,
    artifact_path: str,
    artifact: dict,
    dataset,
    unknown_class: str,
    cfg: Config,
) -> dict:
    write_task_manifest(
        path,
        tasks,
        protocol={
            "shot": int(shot),
            "q_query": int(q_query),
            "split": split,
            "task_seed": int(task_seed),
            "attack": unknown_class,
        },
        base_checkpoint_path=str(Path(artifact_path).resolve()),
        base_checkpoint_sha256=sha256_file(artifact_path),
        base_initialization_sha256=tensor_state_sha256(
            artifact["meta_init_state"]
        ),
        metadata={
            "dataset": str(cfg.data.name),
            "unknown_class": unknown_class,
            "strict_adapt_test": bool(
                cfg.data.get("strict_adapt_test", False)
            ),
        },
        dataset=dataset,
    )
    return read_task_manifest(path)


def _avg_trajectory(outcomes: List[AdaptOutcome]) -> List[float]:
    mat = np.array([
        o.speed.metric_trajectories.get("macro_f1", o.speed.f1_trajectory)
        for o in outcomes
    ], dtype=float)
    return mat.mean(axis=0).tolist()


def _avg_final_metrics(outcomes: List[AdaptOutcome]) -> Dict[str, float]:
    keys = ["accuracy", "precision", "recall", "f1"]
    out = {k: float(np.mean([getattr(o.final_metrics, k) for o in outcomes])) for k in keys}
    out["macro_f1"] = out["f1"]
    out["weighted_f1"] = float(np.mean([
        o.final_metrics.weighted_f1 for o in outcomes
        if o.final_metrics.weighted_f1 is not None
    ])) if outcomes else float("nan")
    aucs = [o.final_metrics.roc_auc for o in outcomes if o.final_metrics.roc_auc is not None]
    out["roc_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    pr_aucs = [o.final_metrics.pr_auc for o in outcomes if o.final_metrics.pr_auc is not None]
    out["pr_auc"] = float(np.mean(pr_aucs)) if pr_aucs else float("nan")
    ar = [o.final_metrics.attack_recall for o in outcomes if o.final_metrics.attack_recall is not None]
    out["attack_recall"] = float(np.mean(ar)) if ar else float("nan")
    fprs = [
        o.final_metrics.false_positive_rate
        for o in outcomes
        if o.final_metrics.false_positive_rate is not None
    ]
    out["false_positive_rate"] = float(np.mean(fprs)) if fprs else float("nan")
    brier = [
        o.final_metrics.brier_score for o in outcomes
        if o.final_metrics.brier_score is not None
    ]
    out["brier_score"] = float(np.mean(brier)) if brier else float("nan")
    ece = [
        o.final_metrics.ece for o in outcomes
        if o.final_metrics.ece is not None
    ]
    out["ece"] = float(np.mean(ece)) if ece else float("nan")
    return out


def _avg_support_loss(outcomes: List[AdaptOutcome]) -> float:
    values = [
        outcome.support_losses[-1]
        for outcome in outcomes
        if outcome.support_losses
    ]
    return float(np.mean(values)) if values else float("nan")


def _nonfinite_count(outcomes: List[AdaptOutcome]) -> int:
    count = 0
    for outcome in outcomes:
        for values in outcome.speed.metric_trajectories.values():
            arr = np.asarray(values, dtype=float)
            count += int((~np.isfinite(arr)).sum())
        for item in outcome.diagnostics or []:
            for value in item.values():
                if isinstance(value, (int, float)) and not np.isfinite(float(value)):
                    count += 1
    return count


def _update_clip_summary(update_rows: List[dict], method: str, experiment: str) -> Dict[str, float]:
    selected = [
        row for row in update_rows
        if row.get("method") == method
        and row.get("experiment") == experiment
        and row.get("group") == "all"
    ]
    if not selected:
        return {"n_update_rows": 0, "clip_ratio": float("nan")}
    clipped = sum(int(float(row.get("was_clipped", 0))) for row in selected)
    return {
        "n_update_rows": int(len(selected)),
        "clip_ratio": float(clipped / len(selected)),
    }


def _mean_metric_trajectory(outcomes: List[AdaptOutcome],
                            metric: str = "macro_f1") -> List[float]:
    """Average a recorded adaptation trajectory across tasks."""
    trajectories = []
    for outcome in outcomes:
        values = outcome.speed.metric_trajectories.get(
            metric, outcome.speed.f1_trajectory if metric == "macro_f1" else [])
        if values:
            trajectories.append(np.asarray(values, dtype=float))
    if not trajectories:
        return []
    max_len = max(len(t) for t in trajectories)
    mat = np.full((len(trajectories), max_len), np.nan, dtype=float)
    for i, traj in enumerate(trajectories):
        mat[i, :len(traj)] = traj
    return np.nanmean(mat, axis=0).tolist()


def select_validation_stop_step(outcomes: List[AdaptOutcome],
                                metric: str = "macro_f1") -> int:
    """Choose an adaptation stop step from validation tasks only."""
    mean_curve = _mean_metric_trajectory(outcomes, metric=metric)
    if not mean_curve:
        return 0
    arr = np.asarray(mean_curve, dtype=float)
    if np.all(np.isnan(arr)):
        return 0
    best = float(np.nanmax(arr))
    candidates = np.where(np.isclose(arr, best, equal_nan=False))[0]
    return int(candidates[0]) if len(candidates) else int(np.nanargmax(arr))


def _pooled_metrics(outcomes: List[AdaptOutcome], n_way: int, attack_idx: int):
    """汇总所有 test task 的 query 预测, 计算 pooled 指标 + 混淆矩阵 + per-class recall。"""
    logits = [o.final_logits for o in outcomes if o.final_logits is not None]
    targets = [o.final_targets for o in outcomes if o.final_targets is not None]
    if not logits:
        return None
    pooled_logits, pooled_targets = aggregate_logits(logits, targets)
    return compute_metrics(pooled_logits, pooled_targets,
                           num_classes=n_way, attack_class_indices=[attack_idx])


def _metrics_to_dict(metrics):
    if metrics is None:
        return None
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "macro_f1": metrics.macro_f1 if metrics.macro_f1 is not None else metrics.f1,
        "weighted_f1": (
            metrics.weighted_f1 if metrics.weighted_f1 is not None else float("nan")),
        "roc_auc": metrics.roc_auc if metrics.roc_auc is not None else float("nan"),
        "pr_auc": metrics.pr_auc if metrics.pr_auc is not None else float("nan"),
        "attack_recall": (
            metrics.attack_recall if metrics.attack_recall is not None else float("nan")),
        "false_positive_rate": (
            metrics.false_positive_rate
            if metrics.false_positive_rate is not None else float("nan")),
        "brier_score": (
            metrics.brier_score
            if metrics.brier_score is not None else float("nan")),
        "ece": metrics.ece if metrics.ece is not None else float("nan"),
        "per_class_recall": {str(k): v for k, v in metrics.per_class_recall.items()},
        "confusion_matrix": metrics.confusion.astype(int).tolist(),
    }


def run_method(
    adapter: FewShotAdapter, init_params, tasks: List[MetaTask],
    optimizer_factory, adapt_names, n_way, max_steps, target_grid, attack_idx,
    collect_update_stats: bool = False,
) -> List[AdaptOutcome]:
    outcomes = []
    for task in tasks:
        opt = optimizer_factory()
        outcomes.append(adapter.adapt_once(
            init_params, task, opt, adapt_names, n_way,
            max_steps=max_steps, target_f1_grid=target_grid,
            attack_class_indices=[attack_idx],
            collect_update_stats=collect_update_stats))
    return outcomes


def grid_search_lr(
    adapter, init_params, val_tasks, kind, lr_grid, adapt_names, n_way,
    max_steps, target_grid, attack_idx, target_f1, checkpoints,
) -> float:
    best_lr, best_key = lr_grid[0], None
    for lr in lr_grid:
        outs = run_method(adapter, init_params, val_tasks,
                          lambda lr=lr: HandcraftedOptimizer(kind=kind, lr=lr),
                          adapt_names, n_way, max_steps, target_grid, attack_idx)
        summary = summarize_adaptation(
            [o.speed for o in outs], target_f1, checkpoints)
        key = adaptation_selection_key(summary)
        logger.info(
            "  [grid][val] %s lr=%-7g reach=%.3f steps=%.2f curve_auc=%.4f final_f1=%.4f",
            kind, lr, summary["reach_rate"], summary["mean_steps"],
            summary["curve_auc_mean"], summary["final_f1_mean"])
        if best_key is None or key > best_key:
            best_key, best_lr = key, lr
    logger.info("  [grid][val] %s best lr=%g key=%s", kind, best_lr, best_key)
    return best_lr


def _grid_boundary_status(best_lr: float, grid: List[float], method: str) -> dict:
    values = sorted({float(value) for value in grid})
    # Cast numpy.bool_ explicitly so JSON contains booleans, not the strings
    # "True"/"False" (PowerShell treats every non-empty string as true).
    at_lower = bool(values) and bool(np.isclose(best_lr, values[0]))
    at_upper = bool(values) and bool(np.isclose(best_lr, values[-1]))
    if at_lower or at_upper:
        logger.warning(
            "%s validation-selected LR=%g is on the %s grid boundary; "
            "do not claim the baseline is fully tuned.",
            method,
            best_lr,
            "upper" if at_upper else "lower",
        )
    return {
        "selected_lr": float(best_lr),
        "grid": values,
        "at_lower_boundary": at_lower,
        "at_upper_boundary": at_upper,
        "fully_tuned_claim_allowed": not (at_lower or at_upper),
    }


def _write_fixed_budget_csv(all_results: Dict[str, dict], path: str) -> None:
    rows = []
    for experiment, result in all_results.items():
        for method, method_result in result["methods"].items():
            summary = method_result["adaptation_analysis"]
            oracle = method_result.get("descriptive_only_test_oracle", {})
            for step, metrics in summary["checkpoints"].items():
                row = {
                    "experiment": experiment,
                    "shot": result["shot"],
                    "method": method,
                    "step": int(step),
                    "target_f1": summary["target_f1"],
                    "reach_rate": summary["reach_rate"],
                    "mean_steps": summary["mean_steps"],
                    "curve_auc_mean": summary["curve_auc_mean"],
                    "final_f1_mean": summary["final_f1_mean"],
                    "descriptive_oracle_best_f1_mean": oracle.get(
                        "best_f1_mean", float("nan")
                    ),
                    "descriptive_post_peak_drop_mean": oracle.get(
                        "post_peak_drop_mean", float("nan")
                    ),
                }
                for metric, stats in metrics.items():
                    row[f"{metric}_mean"] = stats["mean"]
                    row[f"{metric}_std"] = stats["std"]
                rows.append(row)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _task_metric_rows(
    outcomes: List[AdaptOutcome],
    *,
    experiment: str,
    shot: int,
    method: str,
    split: str,
    selected_stop_step: int = None,
) -> List[dict]:
    rows = []
    for task_id, outcome in enumerate(outcomes):
        metrics = outcome.final_metrics.as_dict()
        row = {
            "experiment": experiment,
            "shot": int(shot),
            "method": method,
            "split": split,
            "task_id": int(task_id),
            "selected_stop_step": selected_stop_step,
        }
        for key, value in metrics.items():
            if key == "per_class_recall":
                continue
            row[key] = value
        rows.append(row)
    return rows


def _curve_rows(
    outcomes: List[AdaptOutcome],
    *,
    experiment: str,
    shot: int,
    method: str,
    split: str = "test",
) -> List[dict]:
    rows = []
    for task_id, outcome in enumerate(outcomes):
        trajectories = outcome.speed.metric_trajectories
        max_len = max((len(values) for values in trajectories.values()), default=0)
        for step in range(max_len):
            row = {
                "experiment": experiment,
                "shot": int(shot),
                "method": method,
                "split": split,
                "task_id": int(task_id),
                "step": int(step),
            }
            for metric, values in trajectories.items():
                row[metric] = values[step] if step < len(values) else float("nan")
            rows.append(row)
    return rows


def _diagnostic_rows(
    outcomes: List[AdaptOutcome],
    *,
    experiment: str,
    shot: int,
    method: str,
    unknown_class: str,
) -> List[dict]:
    rows = []
    for task_id, outcome in enumerate(outcomes):
        for item in outcome.diagnostics or []:
            row = {
                "experiment": experiment,
                "unknown_class": unknown_class,
                "shot": int(shot),
                "method": method,
                "task_id": int(task_id),
            }
            row.update(item)
            rows.append(row)
    return rows


def _task_overlap_audit(tasks: List[MetaTask], sampler, split: str) -> List[dict]:
    from src.data.leakage import check_support_query_overlap

    rows = []
    dataset = getattr(sampler, "dataset", None)
    for task_id, task in enumerate(tasks):
        overlaps = (
            check_support_query_overlap(
                task.support_window_ids, task.query_window_ids, dataset)
            if dataset is not None else []
        )
        rows.append({
            "split": split,
            "task_id": int(task_id),
            "global_classes": [int(cls) for cls in task.global_classes],
            "support_windows": int(len(task.support_window_ids)),
            "query_windows": int(len(task.query_window_ids)),
            "support_query_overlap_pairs": int(len(overlaps)),
        })
    return rows


def _dataset_audit(bundle) -> dict:
    def window_counts(ds):
        return {
            str(int(cls)): int(len(indices))
            for cls, indices in ds.class_to_indices.items()
        }

    def raw_counts(split):
        labels, counts = np.unique(split.labels, return_counts=True)
        return {str(label): int(count) for label, count in zip(labels, counts)}

    unknown_idx = bundle._adapt_class_to_idx[bundle.unknown_class]
    return {
        "known_classes": list(bundle.known_classes),
        "unknown_class": bundle.unknown_class,
        "unknown_idx": int(unknown_idx),
        "raw_class_counts": {
            "meta_train": raw_counts(bundle.loao.train),
            "meta_val": raw_counts(bundle.loao.eval),
            "test": raw_counts(bundle.loao.test),
            "unknown": raw_counts(bundle.loao.unknown),
        },
        "window_class_counts": {
            "meta_train": window_counts(bundle.meta_train_dataset),
            "meta_val": window_counts(bundle.meta_val_dataset),
            "adapt_val": window_counts(bundle.adapt_val_dataset),
            "adapt_test": window_counts(bundle.adapt_test_dataset),
        },
        "unknown_windows": {
            "adapt_val": int(len(bundle.adapt_val_dataset.class_to_indices.get(unknown_idx, []))),
            "adapt_test": int(len(bundle.adapt_test_dataset.class_to_indices.get(unknown_idx, []))),
        },
    }


def _convergence95(summary: Dict[str, object]) -> int:
    """Earliest checkpoint reaching 95% of the best checkpoint macro-F1."""
    checkpoints = summary.get("checkpoints", {})
    values = []
    for raw_step, metrics in checkpoints.items():
        macro = metrics.get("macro_f1", {})
        values.append((int(raw_step), float(macro.get("mean", float("nan")))))
    values = [(step, value) for step, value in values if value == value]
    if not values:
        return -1
    best = max(value for _, value in values)
    target = 0.95 * best
    for step, value in sorted(values):
        if value >= target:
            return int(step)
    return int(sorted(values)[-1][0])


def main() -> None:
    args = parse_args()
    art = load_artifacts(args.artifacts)
    if args.phase == "test" and not args.selection_receipt:
        raise ValueError("--phase test requires --selection-receipt")
    if args.phase != "test" and args.selection_receipt:
        raise ValueError("--selection-receipt is only valid with --phase test")
    if args.phase == "both":
        logger.warning(
            "--phase both is retained for backward compatibility. For unbiased "
            "experiments, use --phase validation and then run --phase test once "
            "with the frozen validation_selection.json receipt."
        )
    selection_input = None
    if args.selection_receipt:
        with open(args.selection_receipt, "r", encoding="utf-8") as handle:
            selection_input = json.load(handle)
        if int(selection_input.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported validation selection receipt schema")
        actual_init_hash = tensor_state_sha256(art["meta_init_state"])
        if selection_input.get("meta_init_state_sha256") != actual_init_hash:
            raise ValueError(
                "Selection receipt was produced from a different theta0"
            )
    cfg: Config = Config(art["config"])
    if args.override:
        cfg = cfg.apply_overrides(args.override)
    extra = art["extra"]
    artifact_unknown = str(extra.get("unknown_class", cfg.data.unknown_class))
    if artifact_unknown != str(cfg.data.unknown_class):
        raise ValueError(
            f"Artifact unknown_class={artifact_unknown!r} does not match "
            f"experiment unknown_class={cfg.data.unknown_class!r}. "
            "Train a separate artifact for each LOAO unknown attack.")
    artifact_horizon = int(extra.get("meta_inner_steps", cfg.meta.inner_steps))
    if artifact_horizon != int(cfg.meta.inner_steps):
        raise ValueError(
            f"Artifact training horizon={artifact_horizon} does not match "
            f"configured meta.inner_steps={cfg.meta.inner_steps}.")
    artifact_fraction = float(extra.get(
        "train_fraction", cfg.data.get("train_fraction", 1.0)))
    configured_fraction = float(cfg.data.get("train_fraction", 1.0))
    if not np.isclose(artifact_fraction, configured_fraction):
        raise ValueError(
            f"Artifact train_fraction={artifact_fraction} does not match "
            f"configured data.train_fraction={configured_fraction}.")

    seed = int(cfg.experiment.get("seed", 42))
    set_seed(seed, bool(cfg.experiment.get("deterministic", True)))
    device = resolve_device(str(cfg.device.get("prefer", "auto")))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "effective_config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2, ensure_ascii=False)

    bundle = build_pipeline(cfg, seed=seed)
    dataset_audit = _dataset_audit(bundle)
    with open(os.path.join(args.out, "dataset_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(dataset_audit, handle, indent=2, ensure_ascii=False)
    model = build_meta_model(cfg, extra["feature_dim"], extra["window_size"]).to(device)
    model.load_state_dict(art["meta_init_state"])
    init_params = {n: p for n, p in model.named_parameters()}

    meta_opt = build_meta_optimizer(cfg).to(device)
    meta_opt.load_state_dict(art["meta_opt_state"])
    meta_opt.eval()

    adapt_names = extra["adapt_names"]
    n_way = int(extra["n_way"])
    adapter = FewShotAdapter(model, device)
    attack_idx = 1 if n_way == 2 else n_way - 1

    sp_cfg = cfg.adaptation_speed
    max_steps = int(sp_cfg.get("max_steps", 200))
    target_f1 = float(sp_cfg.get("target_f1", 0.80))
    target_grid = [float(x) for x in sp_cfg.get("target_f1_grid", [0.75, 0.80, 0.85, 0.90])]
    checkpoints = [int(x) for x in sp_cfg.get(
        "checkpoints", [0, 1, 2, 5, 10, 20, 50])]
    trained_horizon = artifact_horizon
    if max_steps > trained_horizon:
        logger.warning(
            "Evaluation horizon %d exceeds MetaOpt training horizon %d; "
            "long-horizon results are extrapolation.",
            max_steps, trained_horizon)

    cmp_cfg = cfg.compare
    shots = [int(s) for s in cmp_cfg.get("shots", [1, 5])]
    n_val = int(cmp_cfg.get("val_tasks", 30))
    n_test = int(cmp_cfg.get("test_tasks", 100))
    sgd_grid = [float(x) for x in cmp_cfg.baseline_lr_grid.sgd]
    adam_grid = [float(x) for x in cmp_cfg.baseline_lr_grid.adam]
    mode = str(cfg.data.get("task_mode", "binary"))
    q_query = int(cfg.data.q_query)
    disallow_ov = bool(cfg.data.get("disallow_support_query_overlap", True))
    disallow_internal = bool(cfg.data.get("disallow_internal_overlap", True))

    if args.task_manifest and args.test_task_manifest:
        raise ValueError(
            "--task-manifest is the deprecated alias for --test-task-manifest; "
            "provide only one"
        )
    external_test_manifest = args.test_task_manifest or args.task_manifest
    external_val_manifest = args.validation_task_manifest
    if (external_test_manifest or external_val_manifest) and len(shots) != 1:
        raise ValueError(
            "explicit validation/test manifests require exactly one compare.shots value"
        )
    if not bool(cfg.data.get("strict_adapt_test", False)):
        raise ValueError("manifest-based evaluation requires strict_adapt_test=true")

    all_results: Dict[str, dict] = {}
    task_rows: List[dict] = []
    curve_rows: List[dict] = []
    update_rows: List[dict] = []
    diagnostic_rows: List[dict] = []
    overlap_rows: List[dict] = []
    prediction_records: List[dict] = []
    used_manifest_paths: List[Path] = []
    selection_output = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_metrics_used": False,
        "meta_init_state_sha256": tensor_state_sha256(art["meta_init_state"]),
        "artifact_path": str(Path(args.artifacts).resolve()),
        "experiments": {},
    }

    for exp_idx, shot in enumerate(shots, start=1):
        logger.info("==== Exp %d: %d-shot | offline temporal adaptation ====",
                    exp_idx, shot)
        experiment_name = f"exp{exp_idx}_{shot}shot"
        # P0-3: val 与 test adaptation 任务来自原始样本级 disjoint 的两个数据集。
        val_sampler = bundle.make_adaptation_sampler(
            k_shot=shot, q_query=q_query, mode=mode, n_way=n_way,
            seed=seed + exp_idx, disallow_support_query_overlap=disallow_ov,
            disallow_internal_overlap=disallow_internal, split="val")
        test_sampler = bundle.make_adaptation_sampler(
            k_shot=shot, q_query=q_query, mode=mode, n_way=n_way,
            seed=seed + 1000 + exp_idx, disallow_support_query_overlap=disallow_ov,
            disallow_internal_overlap=disallow_internal, split="test")

        manifest_dir = Path(args.out) / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        val_manifest_path = Path(
            external_val_manifest
            or manifest_dir / f"validation_{shot}shot.json"
        )
        test_manifest_path = Path(
            external_test_manifest
            or manifest_dir / f"test_{shot}shot.json"
        )

        if val_manifest_path.exists():
            val_manifest = read_task_manifest(
                val_manifest_path, verify_sha256=True
            )
            _manifest_protocol_check(
                val_manifest,
                split="val", shot=shot, q_query=q_query,
                artifact_path=args.artifacts, artifact=art,
                unknown_class=artifact_unknown,
            )
            val_tasks = load_tasks_from_manifest(
                val_manifest, bundle.adapt_val_dataset
            )
        else:
            if external_val_manifest:
                raise FileNotFoundError(external_val_manifest)
            val_tasks = _sample_tasks(val_sampler, n_val)
            val_manifest = _write_run_manifest(
                val_manifest_path, val_tasks,
                split="val", shot=shot, q_query=q_query,
                task_seed=seed + exp_idx,
                artifact_path=args.artifacts, artifact=art,
                dataset=bundle.adapt_val_dataset,
                unknown_class=artifact_unknown, cfg=cfg,
            )

        if test_manifest_path.exists():
            test_manifest = read_task_manifest(
                test_manifest_path, verify_sha256=True
            )
            _manifest_protocol_check(
                test_manifest,
                split="test", shot=shot, q_query=q_query,
                artifact_path=args.artifacts, artifact=art,
                unknown_class=artifact_unknown,
            )
            test_tasks = load_tasks_from_manifest(
                test_manifest, bundle.adapt_test_dataset
            )
        else:
            if external_test_manifest:
                raise FileNotFoundError(external_test_manifest)
            test_tasks = _sample_tasks(test_sampler, n_test)
            test_manifest = _write_run_manifest(
                test_manifest_path, test_tasks,
                split="test", shot=shot, q_query=q_query,
                task_seed=seed + 1000 + exp_idx,
                artifact_path=args.artifacts, artifact=art,
                dataset=bundle.adapt_test_dataset,
                unknown_class=artifact_unknown, cfg=cfg,
            )
        assert_manifest_split_isolation(val_manifest, test_manifest)
        used_manifest_paths.extend([val_manifest_path, test_manifest_path])
        effective_n_test = len(test_tasks)
        overlap_rows.extend(_task_overlap_audit(val_tasks, val_sampler, f"{experiment_name}_val"))
        overlap_rows.extend(_task_overlap_audit(test_tasks, test_sampler, f"{experiment_name}_test"))

        val_manifest_hash = sha256_file(val_manifest_path)
        test_manifest_hash = sha256_file(test_manifest_path)
        frozen_selection = None
        if args.phase == "test":
            frozen_selection = selection_input.get("experiments", {}).get(
                experiment_name
            )
            if frozen_selection is None:
                raise ValueError(
                    f"Selection receipt has no entry for {experiment_name}"
                )
            if frozen_selection.get("validation_manifest_sha256") != val_manifest_hash:
                raise ValueError(
                    f"Validation manifest hash mismatch for {experiment_name}"
                )
            if frozen_selection.get("test_manifest_sha256") != test_manifest_hash:
                raise ValueError(
                    f"Test manifest hash mismatch for {experiment_name}; refusing "
                    "to evaluate a different test task set"
                )
            sgd_lr = float(frozen_selection["selected_learning_rates"]["SGD"])
            adam_lr = float(frozen_selection["selected_learning_rates"]["Adam"])
            lr_boundary = frozen_selection["baseline_lr_validation"]
        else:
            sgd_lr = grid_search_lr(
                adapter, init_params, val_tasks, "sgd", sgd_grid,
                adapt_names, n_way, max_steps, target_grid, attack_idx,
                target_f1, checkpoints,
            )
            adam_lr = grid_search_lr(
                adapter, init_params, val_tasks, "adam", adam_grid,
                adapt_names, n_way, max_steps, target_grid, attack_idx,
                target_f1, checkpoints,
            )
            lr_boundary = {
                "SGD": _grid_boundary_status(sgd_lr, sgd_grid, "SGD"),
                "Adam": _grid_boundary_status(adam_lr, adam_grid, "Adam"),
            }

        factories = {
            "SGD": lambda lr=sgd_lr: HandcraftedOptimizer(kind="sgd", lr=lr),
            "Adam": lambda lr=adam_lr: HandcraftedOptimizer(kind="adam", lr=lr),
            "MetaOpt": lambda: meta_opt,
        }

        shot_result = {
            "shot": shot, "sgd_lr": sgd_lr, "adam_lr": adam_lr,
            "n_val_tasks": len(val_tasks), "n_test_tasks": effective_n_test, "methods": {},
            "baseline_lr_validation": lr_boundary,
        }
        validation_reuse = manifest_reuse_statistics(val_manifest)
        test_reuse = manifest_reuse_statistics(test_manifest)
        for split_name, reuse in (
            ("validation", validation_reuse),
            ("test", test_reuse),
        ):
            if int(reuse["raw_disjoint_task_count_greedy"]) < int(
                reuse["task_count"]
            ):
                logger.warning(
                    "%s manifest has %d sampled tasks but only %d greedily "
                    "raw-disjoint tasks. Treat sampled-task variance as "
                    "pseudo-replication, not %d independent replications.",
                    split_name,
                    reuse["task_count"],
                    reuse["raw_disjoint_task_count_greedy"],
                    reuse["task_count"],
                )
        shot_result["task_manifests"] = {
            "validation": {
                "path": str(val_manifest_path.resolve()),
                "sha256": val_manifest_hash,
                "reuse_statistics": validation_reuse,
                "independent_replication_claim_allowed": (
                    int(validation_reuse["raw_disjoint_task_count_greedy"])
                    == int(validation_reuse["task_count"])
                ),
            },
            "test": {
                "path": str(test_manifest_path.resolve()),
                "sha256": test_manifest_hash,
                "reuse_statistics": test_reuse,
                "independent_replication_claim_allowed": (
                    int(test_reuse["raw_disjoint_task_count_greedy"])
                    == int(test_reuse["task_count"])
                ),
            },
            "validation_test_raw_row_overlap": 0,
        }
        experiment_selection = {
            "shot": int(shot),
            "validation_manifest_sha256": val_manifest_hash,
            "test_manifest_sha256": test_manifest_hash,
            "selected_learning_rates": {
                "SGD": float(sgd_lr),
                "Adam": float(adam_lr),
            },
            "baseline_lr_validation": lr_boundary,
            "methods": {},
        }
        trajectories: Dict[str, List[float]] = {}
        speed_bars: Dict[str, float] = {}

        class_names = ["benign", "attack"] if n_way == 2 else [f"class{i}" for i in range(n_way)]

        for name, factory in factories.items():
            if args.phase == "test":
                method_selection = frozen_selection.get("methods", {}).get(name)
                if method_selection is None:
                    raise ValueError(
                        f"Selection receipt has no {name} entry for {experiment_name}"
                    )
                selected_stop_step = int(
                    method_selection["selected_stop_step"]
                )
                validation_mean_curve = method_selection[
                    "validation_mean_curve"
                ]
                val_outs_for_stop = []
            else:
                val_outs_for_stop = run_method(
                    adapter, init_params, val_tasks, factory, adapt_names,
                    n_way, max_steps, target_grid, attack_idx,
                )
                selected_stop_step = select_validation_stop_step(
                    val_outs_for_stop, metric="macro_f1"
                )
                validation_mean_curve = _mean_metric_trajectory(
                    val_outs_for_stop, metric="macro_f1"
                )
                validation_summary = summarize_adaptation(
                    [outcome.speed for outcome in val_outs_for_stop],
                    target_f1,
                    checkpoints,
                )
                validation_summary.pop(
                    "descriptive_only_test_oracle", None
                )
                experiment_selection["methods"][name] = {
                    "selected_stop_step": int(selected_stop_step),
                    "selection_metric": "macro_f1",
                    "validation_mean_curve": validation_mean_curve,
                    "validation_adaptation_analysis": validation_summary,
                }

            if args.phase == "validation":
                final = _avg_final_metrics(val_outs_for_stop)
                final["support_loss"] = _avg_support_loss(val_outs_for_stop)
                trajectories[name] = validation_mean_curve
                speed_bars[name] = validation_summary["mean_steps"]
                shot_result["methods"][name] = {
                    "validation_only": True,
                    "validation_selected": {
                        "selected_stop_step": int(selected_stop_step),
                        "selection_metric": "macro_f1",
                        "validation_mean_curve": validation_mean_curve,
                    },
                    "final_metrics_avg_per_task": final,
                    "adaptation_analysis": validation_summary,
                    "descriptive_only_test_oracle": {},
                }
                task_rows.extend(_task_metric_rows(
                    val_outs_for_stop,
                    experiment=experiment_name,
                    shot=shot,
                    method=name,
                    split="validation",
                ))
                curve_rows.extend(_curve_rows(
                    val_outs_for_stop,
                    experiment=experiment_name,
                    shot=shot,
                    method=name,
                    split="validation",
                ))
                for task_id, outcome in enumerate(val_outs_for_stop):
                    prediction_records.append({
                        "experiment": experiment_name,
                        "shot": int(shot),
                        "method": name,
                        "split": "validation",
                        "task_id": int(task_id),
                        "step_logits": outcome.step_logits,
                        "labels": outcome.step_targets,
                    })
                logger.info(
                    "[%d-shot][%s][validation-only] selected stop=%d | "
                    "final F1(avg)=%.4f",
                    shot,
                    name,
                    selected_stop_step,
                    final["f1"],
                )
                continue

            outs = run_method(adapter, init_params, test_tasks, factory, adapt_names,
                              n_way, max_steps, target_grid, attack_idx,
                              collect_update_stats=True)
            selected_outs = run_method(
                adapter, init_params, test_tasks, factory, adapt_names, n_way,
                selected_stop_step, target_grid, attack_idx)
            traj = _avg_trajectory(outs)
            final = _avg_final_metrics(outs)
            selected_final = _avg_final_metrics(selected_outs)
            final["support_loss"] = _avg_support_loss(outs)
            selected_final["support_loss"] = _avg_support_loss(selected_outs)
            speed_agg = summarize_adaptation(
                [o.speed for o in outs], target_f1, checkpoints)
            speed_agg["convergence95_step"] = _convergence95(speed_agg)
            selected_speed_agg = summarize_adaptation(
                [o.speed for o in selected_outs], target_f1,
                [c for c in checkpoints if c <= selected_stop_step])
            descriptive_oracle = speed_agg.pop(
                "descriptive_only_test_oracle", {}
            )
            selected_speed_agg.pop("descriptive_only_test_oracle", None)
            trajectories[name] = traj
            speed_bars[name] = speed_agg["mean_steps"]

            pooled = _pooled_metrics(outs, n_way, attack_idx)
            selected_pooled = _pooled_metrics(selected_outs, n_way, attack_idx)
            pooled_dict = None
            if pooled is not None:
                pooled_dict = _metrics_to_dict(pooled)
                try:
                    plot_confusion_matrix(
                        pooled.confusion, class_names, args.out,
                        prefix=f"exp{exp_idx}_{shot}shot_{name}")
                except Exception as exc:
                    logger.warning("混淆矩阵绘图失败(%s): %s", name, exc)

            shot_result["methods"][name] = {
                "final_metrics_avg_per_task": final,
                "final_metrics_pooled": pooled_dict,
                "validation_selected": {
                    "selected_stop_step": selected_stop_step,
                    "selection_metric": "macro_f1",
                    "validation_mean_curve": validation_mean_curve,
                    "final_metrics_avg_per_task": selected_final,
                    "final_metrics_pooled": _metrics_to_dict(selected_pooled),
                    "adaptation_analysis": selected_speed_agg,
                },
                "adaptation_speed": {
                    key: value for key, value in speed_agg.items()
                    if key not in {
                        "checkpoints", "curve_auc_mean", "curve_auc_std",
                        "final_f1_mean",
                    }
                },
                "adaptation_analysis": speed_agg,
                "descriptive_only_test_oracle": descriptive_oracle,
            }
            task_rows.extend(_task_metric_rows(
                outs, experiment=experiment_name, shot=shot, method=name, split="test"))
            task_rows.extend(_task_metric_rows(
                selected_outs, experiment=experiment_name, shot=shot, method=name,
                split="test_validation_selected", selected_stop_step=selected_stop_step))
            curve_rows.extend(_curve_rows(
                outs, experiment=experiment_name, shot=shot, method=name))
            for task_id, outcome in enumerate(outs):
                prediction_records.append({
                    "experiment": experiment_name,
                    "shot": int(shot),
                    "method": name,
                    "split": "test_fixed_horizon",
                    "task_id": int(task_id),
                    "step_logits": outcome.step_logits,
                    "labels": outcome.step_targets,
                })
            diagnostic_rows.extend(_diagnostic_rows(
                outs,
                experiment=experiment_name,
                shot=shot,
                method=name,
                unknown_class=artifact_unknown,
            ))
            for task_id, outcome in enumerate(outs):
                if outcome.update_trace is not None:
                    update_rows.extend(update_rows_to_dicts(
                        outcome.update_trace.rows,
                        experiment=experiment_name,
                        shot=shot,
                        method=name,
                        task_id=task_id,
                    ))
            clip_summary = _update_clip_summary(update_rows, name, experiment_name)
            nonfinite_count = _nonfinite_count(outs)
            shot_result["methods"][name]["update_clip_summary"] = clip_summary
            shot_result["methods"][name]["nonfinite_count"] = int(nonfinite_count)
            logger.info("[%d-shot][%s][test] F1(avg)=%.4f attack_rec=%.4f | speed@%.2f=%.1f steps",
                        shot, name, final["f1"], final.get("attack_recall", float("nan")),
                        target_f1, speed_agg["mean_steps"])
            logger.info(
                "[%d-shot][%s][audit] support_loss=%.4f fpr=%.4f clip_ratio=%.4f "
                "nonfinite=%d",
                shot,
                name,
                final.get("support_loss", float("nan")),
                final.get("false_positive_rate", float("nan")),
                clip_summary.get("clip_ratio", float("nan")),
                nonfinite_count,
            )
            logger.info("[%d-shot][%s][val-selected] stop=%d | F1(avg)=%.4f attack_rec=%.4f",
                        shot, name, selected_stop_step, selected_final["f1"],
                        selected_final.get("attack_recall", float("nan")))

        try:
            plot_adaptation_curves(trajectories, args.out, target_f1=target_f1,
                                   prefix=f"exp{exp_idx}_{shot}shot")
            plot_speed_bars(speed_bars, args.out, target_f1=target_f1,
                            prefix=f"exp{exp_idx}_{shot}shot")
        except Exception as exc:
            logger.warning("绘图失败: %s", exc)

        if args.phase != "test":
            selection_output["experiments"][experiment_name] = experiment_selection
        shot_result["phase"] = args.phase
        all_results[experiment_name] = shot_result

    out_json = os.path.join(args.out, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    if args.phase != "test":
        selection_receipt_path = os.path.join(
            args.out, "validation_selection.json"
        )
        with open(selection_receipt_path, "w", encoding="utf-8") as handle:
            json.dump(
                selection_output,
                handle,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
    else:
        selection_receipt_path = str(Path(args.selection_receipt).resolve())
    _write_fixed_budget_csv(
        all_results, os.path.join(args.out, "fixed_budget_results.csv"))
    _write_csv(task_rows, os.path.join(args.out, "task_level_results.csv"))
    _write_csv(curve_rows, os.path.join(args.out, "adaptation_curves.csv"))
    _write_csv(update_rows, os.path.join(args.out, "update_analysis.csv"))
    _write_csv(diagnostic_rows, os.path.join(args.out, "step_diagnostics.csv"))
    _write_csv(overlap_rows, os.path.join(args.out, "support_query_overlap_audit.csv"))
    _write_csv(
        [row for row in update_rows if row.get("group") != "all"],
        os.path.join(args.out, "layer_update_distribution.csv"))
    _write_csv(
        [row for row in update_rows if row.get("group") == "all"],
        os.path.join(args.out, "gradient_evolution.csv"))
    prediction_path = write_prediction_trajectories(
        os.path.join(args.out, "prediction_trajectories.npz"),
        prediction_records,
    )
    with open(
        os.path.join(args.out, "result_schema.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "schema_version": 2,
            "evaluation_phase": args.phase,
            "primary_result_fields": [
                "fixed checkpoints",
                "validation_selected",
                "adaptation_analysis.curve_auc_mean",
                "final_metrics_avg_per_task",
            ],
            "selection_allowed_splits": ["validation"],
            "test_selection_allowed": False,
            "descriptive_only_fields": [
                "descriptive_only_test_oracle",
            ],
            "prediction_trajectory_artifact": str(prediction_path),
            "step_metric_artifact": os.path.join(
                args.out, "adaptation_curves.csv"
            ),
            "step_diagnostic_artifact": os.path.join(
                args.out, "step_diagnostics.csv"
            ),
            "step_update_artifact": os.path.join(
                args.out, "update_analysis.csv"
            ),
            "selection_receipt": selection_receipt_path,
            "step_zero_included": True,
        }, handle, indent=2, ensure_ascii=False)
    training_provenance_path = Path(args.artifacts).parent / "provenance.json"
    raw_files = []
    if training_provenance_path.exists():
        raw_files = json.loads(
            training_provenance_path.read_text(encoding="utf-8")
        ).get("raw_data_files", [])
    if not raw_files:
        raw_files = raw_data_catalog(
            str(cfg.data.root),
            include_sha256=bool(
                cfg.get("provenance", {}).get("hash_raw_data", True)
            ),
        )
    write_provenance_receipt(
        os.path.join(args.out, "provenance.json"),
        config=cfg.to_dict(),
        cache_key=data_cache_key(cfg),
        raw_files=raw_files,
        artifacts={
            "meta_artifacts": args.artifacts,
            "results": out_json,
            "prediction_trajectories": prediction_path,
            "validation_selection": selection_receipt_path,
            "effective_config": os.path.join(
                args.out, "effective_config.json"
            ),
        },
        task_manifests=used_manifest_paths,
    )
    logger.info("实验完成: %s", out_json)


if __name__ == "__main__":
    main()
