"""Reproducible, read-only audit of the current fast-adaptation experiment.

The script reads existing logs/results and writes only under reports/.
It does not import the training stack or execute model inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs"
MATRIX_ROOT = LOG_ROOT / "outputs" / "fast_adaptation_matrix_20260722"
REPORT_ROOT = ROOT / "reports"
FIG_ROOT = REPORT_ROOT / "current_run_figures"
TABLE_ROOT = REPORT_ROOT / "current_run_tables"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_ROOT.mkdir(parents=True, exist_ok=True)

METHODS = ["SGD", "Adam", "MetaOpt"]
COLORS = {"SGD": "#315A7D", "Adam": "#A56A28", "MetaOpt": "#7C4D91"}
ATTACKS = ["botnet", "bruteforce", "ddos", "dos", "portscan", "webattack"]
STEPS = [0, 1, 2, 5, 10, 20]


def run_ids(path: Path) -> tuple[str, int]:
    parts = path.parts
    idx = parts.index("runs")
    return parts[idx + 1], int(parts[idx + 3].split("_", 1)[1])


def load_run_csv(name: str, usecols: list[str] | None = None) -> pd.DataFrame:
    frames = []
    pattern = f"runs/*/fraction_1/seed_*/horizon_20/evaluation/{name}"
    for path in MATRIX_ROOT.glob(pattern):
        frame = pd.read_csv(path, usecols=usecols)
        unknown, seed = run_ids(path)
        frame["unknown"] = unknown
        frame["seed"] = seed
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No run-level files found for {name}")
    return pd.concat(frames, ignore_index=True)


def compact_stats(frame: pd.DataFrame, group: str, metrics: list[str]) -> pd.DataFrame:
    parts = []
    for metric in metrics:
        stats = (
            frame.groupby(group)[metric]
            .agg(["mean", "std", "median", "min", "max"])
            .reset_index()
        )
        stats.insert(1, "metric", metric)
        parts.append(stats)
    return pd.concat(parts, ignore_index=True)


def audit_inventory() -> dict:
    all_files = [path for path in LOG_ROOT.rglob("*") if path.is_file()]
    ext_counts = Counter(path.suffix or "<none>" for path in all_files)
    artifact_types = pd.DataFrame(
        [
            {
                "extension": ext,
                "files": count,
                "bytes": sum(
                    path.stat().st_size
                    for path in all_files
                    if (path.suffix or "<none>") == ext
                ),
            }
            for ext, count in sorted(ext_counts.items())
        ]
    )
    artifact_types.to_csv(TABLE_ROOT / "artifact_type_counts.csv", index=False)

    expected_root = [
        "effective_config.json",
        "meta_artifacts.pt",
        "validation_task_pool.json",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
    ]
    expected_eval = [
        "evaluation/adaptation_curves.csv",
        "evaluation/dataset_audit.json",
        "evaluation/effective_config.json",
        "evaluation/fixed_budget_results.csv",
        "evaluation/gradient_evolution.csv",
        "evaluation/layer_update_distribution.csv",
        "evaluation/results.json",
        "evaluation/step_diagnostics.csv",
        "evaluation/support_query_overlap_audit.csv",
        "evaluation/task_level_results.csv",
        "evaluation/update_analysis.csv",
    ]
    run_rows = []
    for run_dir in sorted(MATRIX_ROOT.glob("runs/*/fraction_1/seed_*/horizon_20")):
        unknown = run_dir.parts[run_dir.parts.index("runs") + 1]
        seed = int(run_dir.parts[run_dir.parts.index("runs") + 3].split("_")[1])
        missing = [name for name in expected_root + expected_eval if not (run_dir / name).is_file()]
        pngs = list((run_dir / "evaluation").glob("*.png"))
        config_match = False
        if not missing:
            config_match = json.loads((run_dir / "effective_config.json").read_text()) == json.loads(
                (run_dir / "evaluation" / "effective_config.json").read_text()
            )
        run_rows.append(
            {
                "unknown": unknown,
                "seed": seed,
                "horizon": 20,
                "missing_required_files": ";".join(missing),
                "evaluation_png_count": len(pngs),
                "config_snapshot_match": config_match,
                "status": "complete" if not missing and len(pngs) == 20 else "partial",
            }
        )
    run_inventory = pd.DataFrame(run_rows)
    run_inventory.to_csv(TABLE_ROOT / "run_inventory.csv", index=False)

    log_rows = []
    for path in sorted((LOG_ROOT / "logs").glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        has_error = bool(re.search(r"Traceback|\bERROR\b|CalledProcessError", text))
        if "Fast-adaptation matrix complete:" in text:
            kind, status = "matrix_launcher", "complete"
        elif "实验完成:" in text:
            kind, status = "evaluation", "complete"
        elif "Meta-training finished. Artifact:" in text:
            kind, status = "meta_training", "complete"
        else:
            kind, status = "standalone_or_preflight", "partial"
        log_rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "kind": kind,
                "status": "failed" if has_error else status,
                "has_error_marker": has_error,
                "last_line": text.splitlines()[-1] if text.splitlines() else "",
            }
        )
    pd.DataFrame(log_rows).to_csv(TABLE_ROOT / "log_inventory.csv", index=False)

    return {
        "files": len(all_files),
        "bytes": sum(path.stat().st_size for path in all_files),
        "empty_files": sum(path.stat().st_size == 0 for path in all_files),
        "run_count": len(run_rows),
        "complete_runs": int((run_inventory.status == "complete").sum()),
        "root_outputs_exists": (ROOT / "outputs").exists(),
    }


def validate_traceability(matrix: pd.DataFrame) -> dict:
    keys = ["unknown", "seed", "train_fraction", "train_horizon", "shot", "method", "step"]
    duplicate_keys = int(matrix.duplicated(keys).sum())
    expected_rows = 6 * 5 * 1 * 1 * 4 * 3 * 6

    summary = pd.read_csv(MATRIX_ROOT / "summary" / "matrix_results.csv")
    summary_keys = ["unknown", "train_fraction", "train_horizon", "shot", "method", "step"]
    reconstructed = (
        matrix.groupby(summary_keys, as_index=False)
        .agg(
            n_seeds=("seed", "size"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            attack_recall_mean=("attack_recall", "mean"),
        )
    )
    merged = summary.merge(reconstructed, on=summary_keys, suffixes=("_saved", "_calc"))
    max_diff = 0.0
    for metric in [
        "accuracy_mean",
        "macro_f1_mean",
        "precision_mean",
        "recall_mean",
        "pr_auc_mean",
        "attack_recall_mean",
    ]:
        max_diff = max(
            max_diff,
            float((merged[f"{metric}_saved"] - merged[f"{metric}_calc"]).abs().max()),
        )
    return {
        "matrix_rows": len(matrix),
        "expected_matrix_rows": expected_rows,
        "duplicate_matrix_keys": duplicate_keys,
        "summary_rows": len(summary),
        "summary_max_recompute_diff": max_diff,
        "summary_all_n_seeds_5": bool((summary.n_seeds == 5).all()),
    }


def load_and_summarize() -> dict:
    matrix = pd.read_csv(MATRIX_ROOT / "matrix_results.csv")
    task = load_run_csv(
        "task_level_results.csv",
        [
            "accuracy",
            "precision",
            "recall",
            "macro_f1",
            "roc_auc",
            "pr_auc",
            "attack_recall",
            "false_positive_rate",
            "method",
            "shot",
            "split",
            "task_id",
            "selected_stop_step",
        ],
    )
    curves = load_run_csv(
        "adaptation_curves.csv",
        [
            "accuracy",
            "precision",
            "recall",
            "macro_f1",
            "pr_auc",
            "attack_recall",
            "method",
            "shot",
            "step",
            "task_id",
        ],
    )
    diagnostics = load_run_csv(
        "step_diagnostics.csv",
        [
            "method",
            "shot",
            "step",
            "task_id",
            "support_loss",
            "macro_f1",
            "prediction_positive_rate",
            "attack_recall",
            "normal_recall",
        ],
    )
    dynamics = load_run_csv(
        "gradient_evolution.csv",
        [
            "method",
            "shot",
            "step",
            "task_id",
            "grad_norm",
            "update_norm",
            "update_to_grad_ratio",
            "cosine_update_grad",
            "was_clipped",
        ],
    )

    final_task = task[task.split == "test"].copy()
    seed_final = (
        final_task.groupby(["unknown", "seed", "shot", "method"], as_index=False)[
            [
                "accuracy",
                "precision",
                "recall",
                "macro_f1",
                "roc_auc",
                "pr_auc",
                "attack_recall",
                "false_positive_rate",
            ]
        ]
        .mean()
    )
    final_metrics = [
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "roc_auc",
        "pr_auc",
        "attack_recall",
        "false_positive_rate",
    ]
    compact_stats(seed_final, "method", final_metrics).to_csv(
        TABLE_ROOT / "descriptive_stats_final.csv", index=False
    )
    (
        seed_final.groupby(["unknown", "method"], as_index=False)[final_metrics]
        .agg(["mean", "std"])
        .to_csv(TABLE_ROOT / "attack_method_final_metrics.csv")
    )
    seed_final.to_csv(
        TABLE_ROOT / "per_attack_method_seed_shot_final.csv", index=False
    )
    (
        seed_final.groupby(["unknown", "seed", "method"], as_index=False)[final_metrics]
        .mean()
        .to_csv(TABLE_ROOT / "per_attack_method_seed_final.csv", index=False)
    )

    stage_rows = []
    for step in [0, 1, 20]:
        subset = matrix[matrix.step == step]
        for method, group in subset.groupby("method"):
            stage_rows.append(
                {
                    "stage": f"step_{step}",
                    "method": method,
                    "macro_f1": group.macro_f1.mean(),
                    "accuracy": group.accuracy.mean(),
                    "precision": group.precision.mean(),
                    "recall": group.recall.mean(),
                    "pr_auc": group.pr_auc.mean(),
                    "attack_recall": group.attack_recall.mean(),
                }
            )
    for method, group in matrix[matrix.step == 20].groupby("method"):
        stage_rows.append(
            {
                "stage": "best_test_curve_descriptive_only",
                "method": method,
                "macro_f1": group.best_f1.mean(),
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "pr_auc": np.nan,
                "attack_recall": np.nan,
            }
        )
    selected = task[task.split == "test_validation_selected"]
    for method, group in selected.groupby("method"):
        stage_rows.append(
            {
                "stage": "validation_selected_stop",
                "method": method,
                "macro_f1": group.macro_f1.mean(),
                "accuracy": group.accuracy.mean(),
                "precision": group.precision.mean(),
                "recall": group.recall.mean(),
                "pr_auc": group.pr_auc.mean(),
                "attack_recall": group.attack_recall.mean(),
            }
        )
    pd.DataFrame(stage_rows).to_csv(TABLE_ROOT / "stage_summary.csv", index=False)

    delta_rows = []
    wtl_rows = []
    for step in [1, 20]:
        for metric in ["macro_f1", "pr_auc", "attack_recall"]:
            pivot = matrix[matrix.step == step].pivot(
                index=["unknown", "seed", "shot"], columns="method", values=metric
            )
            for baseline in ["SGD", "Adam"]:
                delta = pivot.MetaOpt - pivot[baseline]
                for shot, values in delta.groupby(level="shot"):
                    delta_rows.append(
                        {
                            "grain": "seed_attack_shot",
                            "step": step,
                            "metric": metric,
                            "shot": shot,
                            "baseline": baseline,
                            "mean_delta": values.mean(),
                            "relative_delta_mean": (values / pivot.loc[values.index, baseline]).mean(),
                            "median_delta": values.median(),
                            "min_delta": values.min(),
                            "max_delta": values.max(),
                        }
                    )
                    wtl_rows.append(
                        {
                            "grain": "seed_attack_shot",
                            "step": step,
                            "metric": metric,
                            "shot": shot,
                            "baseline": baseline,
                            "wins": int((values > 0).sum()),
                            "ties": int((values == 0).sum()),
                            "losses": int((values < 0).sum()),
                            "n": len(values),
                        }
                    )
    curve_pivot = matrix[matrix.step == 20].pivot(
        index=["unknown", "seed", "shot"], columns="method", values="curve_auc"
    )
    for baseline in ["SGD", "Adam"]:
        delta = curve_pivot.MetaOpt - curve_pivot[baseline]
        for shot, values in delta.groupby(level="shot"):
            delta_rows.append(
                {
                    "grain": "seed_attack_shot",
                    "step": "curve",
                    "metric": "curve_auc",
                    "shot": shot,
                    "baseline": baseline,
                    "mean_delta": values.mean(),
                    "relative_delta_mean": (
                        values / curve_pivot.loc[values.index, baseline]
                    ).mean(),
                    "median_delta": values.median(),
                    "min_delta": values.min(),
                    "max_delta": values.max(),
                }
            )
            wtl_rows.append(
                {
                    "grain": "seed_attack_shot",
                    "step": "curve",
                    "metric": "curve_auc",
                    "shot": shot,
                    "baseline": baseline,
                    "wins": int((values > 0).sum()),
                    "ties": int((values == 0).sum()),
                    "losses": int((values < 0).sum()),
                    "n": len(values),
                }
            )

    task_curve = curves.pivot(
        index=["unknown", "seed", "shot", "task_id", "step"],
        columns="method",
        values="macro_f1",
    )
    for step in [1, 20]:
        step_values = task_curve.xs(step, level="step")
        for baseline in ["SGD", "Adam"]:
            delta = step_values.MetaOpt - step_values[baseline]
            wtl_rows.append(
                {
                    "grain": "paired_task",
                    "step": step,
                    "metric": "macro_f1",
                    "shot": "all",
                    "baseline": baseline,
                    "wins": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "losses": int((delta < 0).sum()),
                    "n": len(delta),
                }
            )
            delta_rows.append(
                {
                    "grain": "paired_task",
                    "step": step,
                    "metric": "macro_f1",
                    "shot": "all",
                    "baseline": baseline,
                    "mean_delta": delta.mean(),
                    "relative_delta_mean": np.nan,
                    "median_delta": delta.median(),
                    "min_delta": delta.min(),
                    "max_delta": delta.max(),
                }
            )
    pd.DataFrame(delta_rows).to_csv(TABLE_ROOT / "paired_deltas.csv", index=False)
    pd.DataFrame(wtl_rows).to_csv(TABLE_ROOT / "win_tie_loss.csv", index=False)

    dynamics_summary = (
        dynamics.groupby(["method", "step"], as_index=False)[
            [
                "grad_norm",
                "update_norm",
                "update_to_grad_ratio",
                "cosine_update_grad",
                "was_clipped",
            ]
        ]
        .agg(["mean", "median", "std"])
    )
    dynamics_summary.to_csv(TABLE_ROOT / "dynamics_summary.csv")

    diagnostic_summary = (
        diagnostics.groupby(["method", "step"], as_index=False)[
            [
                "support_loss",
                "macro_f1",
                "prediction_positive_rate",
                "attack_recall",
                "normal_recall",
            ]
        ]
        .mean()
    )
    diagnostic_summary.to_csv(TABLE_ROOT / "diagnostic_summary.csv", index=False)

    lr_rows = []
    stop_rows = []
    confusion = {method: np.zeros((2, 2), dtype=float) for method in METHODS}
    for path in MATRIX_ROOT.glob(
        "runs/*/fraction_1/seed_*/horizon_20/evaluation/results.json"
    ):
        unknown, seed = run_ids(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for experiment, result in payload.items():
            lr_rows.append(
                {
                    "unknown": unknown,
                    "seed": seed,
                    "experiment": experiment,
                    "shot": result["shot"],
                    "sgd_lr": result["sgd_lr"],
                    "adam_lr": result["adam_lr"],
                }
            )
            for method in METHODS:
                method_result = result["methods"][method]
                stop_rows.append(
                    {
                        "unknown": unknown,
                        "seed": seed,
                        "shot": result["shot"],
                        "method": method,
                        "selected_stop_step": method_result["validation_selected"][
                            "selected_stop_step"
                        ],
                    }
                )
                confusion[method] += np.asarray(
                    method_result["final_metrics_pooled"]["confusion_matrix"], dtype=float
                )
    pd.DataFrame(lr_rows).to_csv(TABLE_ROOT / "selected_learning_rates.csv", index=False)
    pd.DataFrame(stop_rows).to_csv(TABLE_ROOT / "validation_selected_steps.csv", index=False)

    make_plots(matrix, seed_final, curves, diagnostics, dynamics, confusion)
    data_counts = summarize_data_windows()

    return {
        "matrix": matrix,
        "seed_final": seed_final,
        "curves": curves,
        "diagnostics": diagnostics,
        "dynamics": dynamics,
        "traceability": validate_traceability(matrix),
        "data_counts": data_counts,
    }


def summarize_data_windows() -> pd.DataFrame:
    rows = []
    for path in MATRIX_ROOT.glob(
        "runs/*/fraction_1/seed_*/horizon_20/evaluation/dataset_audit.json"
    ):
        unknown, seed = run_ids(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "unknown": unknown,
                "seed": seed,
                "raw_unknown_rows": sum(payload["raw_class_counts"]["unknown"].values()),
                "adapt_val_unknown_windows": payload["unknown_windows"]["adapt_val"],
                "adapt_test_unknown_windows": payload["unknown_windows"]["adapt_test"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_ROOT / "unknown_data_counts_by_seed.csv", index=False)
    summary = (
        frame.groupby("unknown", as_index=False)
        .agg(
            raw_unknown_rows=("raw_unknown_rows", "mean"),
            adapt_val_unknown_windows=("adapt_val_unknown_windows", "mean"),
            adapt_test_unknown_windows=("adapt_test_unknown_windows", "mean"),
        )
        .sort_values("adapt_test_unknown_windows")
    )
    summary.to_csv(TABLE_ROOT / "unknown_data_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(summary))
    ax.bar(
        x - 0.18,
        summary.adapt_val_unknown_windows,
        width=0.36,
        label="Adaptation validation",
        color="#729ECE",
    )
    ax.bar(
        x + 0.18,
        summary.adapt_test_unknown_windows,
        width=0.36,
        label="Adaptation test",
        color="#D18F52",
    )
    ax.set_xticks(x, summary.unknown)
    ax.set_ylabel("Independent non-overlapping windows")
    ax.set_title("Held-out attack windows available to adaptation tasks")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "unknown_window_counts.png", dpi=180)
    plt.close(fig)
    return summary


def make_plots(
    matrix: pd.DataFrame,
    seed_final: pd.DataFrame,
    curves: pd.DataFrame,
    diagnostics: pd.DataFrame,
    dynamics: pd.DataFrame,
    confusion: dict[str, np.ndarray],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    seed_curve = (
        curves.groupby(["unknown", "seed", "shot", "method", "step"], as_index=False)
        .macro_f1.mean()
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, shot in zip(axes.ravel(), [1, 3, 5, 10]):
        subset = seed_curve[seed_curve.shot == shot]
        for method in METHODS:
            group = subset[subset.method == method].groupby("step").macro_f1
            mean = group.mean()
            sem = group.sem()
            ax.plot(mean.index, mean.values, marker="o", label=method, color=COLORS[method])
            ax.fill_between(
                mean.index,
                mean.values - 1.96 * sem.values,
                mean.values + 1.96 * sem.values,
                color=COLORS[method],
                alpha=0.12,
            )
        ax.set_title(f"{shot}-shot")
        ax.set_xticks(STEPS)
        ax.set_ylim(0.3, 1.0)
    axes[1, 0].set_xlabel("Adaptation step")
    axes[1, 1].set_xlabel("Adaptation step")
    axes[0, 0].set_ylabel("Query Macro-F1")
    axes[1, 0].set_ylabel("Query Macro-F1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Adaptation curves across attacks and seeds (mean ± 95% SE)", y=1.015)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_ROOT / "adaptation_curves_macro_f1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics = ["macro_f1", "roc_auc", "pr_auc"]
    titles = ["Final Macro-F1", "Final ROC-AUC", "Final PR-AUC"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    x = np.arange(len(ATTACKS))
    width = 0.24
    for ax, metric, title in zip(axes, metrics, titles):
        for idx, method in enumerate(METHODS):
            values = (
                seed_final[seed_final.method == method]
                .groupby("unknown")[metric]
                .mean()
                .reindex(ATTACKS)
            )
            errors = (
                seed_final[seed_final.method == method]
                .groupby("unknown")[metric]
                .std()
                .reindex(ATTACKS)
            )
            ax.bar(
                x + (idx - 1) * width,
                values,
                width,
                yerr=errors,
                capsize=2,
                label=method,
                color=COLORS[method],
            )
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_title(title + " (mean ± seed/shot SD)")
        ax.set_ylim(0.5, 1.02)
    axes[-1].set_xticks(x, ATTACKS)
    axes[0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "final_metrics_by_attack.png", dpi=180)
    plt.close(fig)

    seed_summary = seed_final.groupby(["seed", "method"], as_index=False).macro_f1.mean()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for method in METHODS:
        group = seed_summary[seed_summary.method == method]
        ax.plot(
            group.seed,
            group.macro_f1,
            marker="o",
            linewidth=2,
            label=method,
            color=COLORS[method],
        )
    ax.set_xticks(sorted(seed_summary.seed.unique()))
    ax.set_ylim(0.78, 0.93)
    ax.set_xlabel("Training/evaluation seed")
    ax.set_ylabel("Final Macro-F1")
    ax.set_title("Final performance by seed (averaged over attacks and shots)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "seed_variation_final_macro_f1.png", dpi=180)
    plt.close(fig)

    final_pivot = matrix[matrix.step == 20].pivot(
        index=["unknown", "seed", "shot"], columns="method", values="macro_f1"
    )
    curve_pivot = matrix[matrix.step == 20].pivot(
        index=["unknown", "seed", "shot"], columns="method", values="curve_auc"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, pivot, title in [
        (axes[0], final_pivot, "Final Macro-F1"),
        (axes[1], curve_pivot, "Adaptation curve AUC"),
    ]:
        for baseline, marker in [("SGD", "o"), ("Adam", "s")]:
            delta = (pivot.MetaOpt - pivot[baseline]).groupby("unknown").mean().reindex(ATTACKS)
            ax.plot(
                ATTACKS,
                delta,
                marker=marker,
                linewidth=1.8,
                label=f"MetaOpt − {baseline}",
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(title + " paired delta")
        ax.set_ylabel("Absolute delta")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "metaopt_paired_deltas.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, method in zip(axes, METHODS):
        cm = confusion[method]
        normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{normalized[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if normalized[i, j] > 0.55 else "black",
                )
        ax.set_title(method)
        ax.set_xticks([0, 1], ["Benign", "Attack"])
        ax.set_yticks([0, 1], ["Benign", "Attack"])
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.04)
    fig.suptitle("Final-step pooled confusion matrices (row-normalized)")
    fig.savefig(FIG_ROOT / "confusion_matrices_final.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    dyn = dynamics.groupby(["method", "step"], as_index=False).agg(
        grad_norm=("grad_norm", "mean"),
        update_norm=("update_norm", "mean"),
        update_to_grad_ratio=("update_to_grad_ratio", "median"),
        descent_alignment=("cosine_update_grad", lambda x: -x.mean()),
    )
    diag = diagnostics.groupby(["method", "step"], as_index=False).agg(
        support_loss=("support_loss", "mean"),
        macro_f1=("macro_f1", "mean"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    panels = [
        (axes[0, 0], dyn, "update_norm", "Mean update norm"),
        (axes[0, 1], dyn, "descent_alignment", "Cosine(update, −gradient)"),
        (axes[1, 0], diag, "support_loss", "Support loss before each update"),
        (axes[1, 1], diag, "macro_f1", "Query Macro-F1 after each update"),
    ]
    for ax, frame, column, title in panels:
        for method in METHODS:
            group = frame[frame.method == method]
            ax.plot(
                group.step,
                group[column],
                label=method,
                color=COLORS[method],
                linewidth=2,
            )
        ax.set_title(title)
        ax.set_xlabel("Adaptation step")
        ax.set_xticks(STEPS)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Learning dynamics across all attacks, seeds, shots, and tasks")
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "learning_dynamics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    inventory = audit_inventory()
    analysis = load_and_summarize()
    payload = {
        "inventory": inventory,
        "traceability": analysis["traceability"],
        "figure_paths": sorted(str(path.relative_to(ROOT)) for path in FIG_ROOT.glob("*.png")),
        "table_paths": sorted(str(path.relative_to(ROOT)) for path in TABLE_ROOT.glob("*.csv")),
    }
    (REPORT_ROOT / "current_run_analysis_receipt.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
