"""绘图工具: 训练曲线、混淆矩阵、ROC、K-shot 对比。

所有函数将图保存到指定目录并返回保存路径, 使用非交互后端, 适合服务器环境。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # 无显示环境后端
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_training_curves(history, out_dir: str, prefix: str = "train") -> str:
    """绘制 meta loss / val accuracy / val f1 / val auc 曲线。"""
    _ensure_dir(out_dir)
    epochs = history.epochs if history.epochs else list(range(1, len(history.meta_loss) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(range(1, len(history.meta_loss) + 1), history.meta_loss,
                 label="meta loss", color="tab:red")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("meta loss")
    axes[0].set_title("Meta Training Loss"); axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if history.val_f1:
        axes[1].plot(epochs, history.val_acc, label="val acc", marker="o")
        axes[1].plot(epochs, history.val_f1, label="val f1", marker="s")
        if any(not np.isnan(v) for v in history.val_auc):
            axes[1].plot(epochs, history.val_auc, label="val auc", marker="^")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score")
    axes[1].set_title("Validation Metrics"); axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str],
                          out_dir: str, prefix: str = "eval",
                          normalize: bool = True) -> str:
    """绘制混淆矩阵热图。"""
    _ensure_dir(out_dir)
    cm = np.asarray(cm, dtype=np.float64)
    if normalize and cm.sum() > 0:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_disp = cm / row_sums
    else:
        cm_disp = cm

    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 3, 1.2 * len(class_names) + 2))
    im = ax.imshow(cm_disp, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    thresh = cm_disp.max() / 2.0 if cm_disp.size else 0.5
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            ax.text(j, i, f"{cm_disp[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_disp[i, j] > thresh else "black", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_confusion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curves(logits: torch.Tensor, targets: torch.Tensor,
                    class_names: Sequence[str], out_dir: str,
                    prefix: str = "eval") -> str:
    """绘制每类的 ROC 曲线 (OVR)。"""
    _ensure_dir(out_dir)
    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
    y_true = targets.cpu().numpy().astype(int)
    n_classes = probs.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(n_classes):
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        name = class_names[c] if c < len(class_names) else f"class {c}"
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (OVR)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_roc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _adaptation_step_axis(
    trajectory: Sequence[float],
    steps: Optional[Sequence[int]] = None,
) -> List[int]:
    """Return the explicit adaptation-step axis, including baseline step 0."""
    axis = list(range(len(trajectory))) if steps is None else [int(step) for step in steps]
    if len(axis) != len(trajectory):
        raise ValueError(
            f"adaptation steps/trajectory length mismatch: {len(axis)} != {len(trajectory)}"
        )
    return axis


def plot_adaptation_curves(trajectories: Dict[str, List[float]], out_dir: str,
                           target_f1: float = 0.80, prefix: str = "adapt",
                           steps: Optional[Sequence[int]] = None) -> str:
    """绘制各方法的 query-F1 随适配步数变化曲线(Adaptation Speed 可视化)。

    Args:
        trajectories: {方法名: 每步平均 F1 列表}。
        target_f1: 目标阈值(画水平参考线)。
    """
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, traj in trajectories.items():
        ax.plot(_adaptation_step_axis(traj, steps), traj, marker="", label=method)
    ax.axhline(target_f1, color="gray", linestyle="--", alpha=0.7,
               label=f"target F1={target_f1}")
    ax.set_xlabel("adaptation step"); ax.set_ylabel("query F1 (macro)")
    ax.set_title("Adaptation Speed: F1 vs Step"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_f1_vs_step.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_speed_bars(speeds: Dict[str, float], out_dir: str, target_f1: float = 0.80,
                    prefix: str = "adapt") -> str:
    """绘制各方法达到目标 F1 所需步数柱状图(越低越好)。

    Args:
        speeds: {方法名: 平均步数}。
    """
    _ensure_dir(out_dir)
    methods = list(speeds.keys())
    values = [speeds[m] for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(methods, values, color=["tab:blue", "tab:orange", "tab:green", "tab:red"][:len(methods)])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"steps to reach F1={target_f1}")
    ax.set_title("Adaptation Speed Comparison (lower = faster)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_speed_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_kshot_comparison(results: Dict[str, Dict[int, float]], out_dir: str,
                          metric_name: str = "f1", prefix: str = "ablation") -> str:
    """绘制不同方法在不同 K-shot 下的指标对比折线图。

    Args:
        results: {方法名: {k_shot: 指标值}}。
    """
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, kv in results.items():
        ks = sorted(kv.keys())
        ys = [kv[k] for k in ks]
        ax.plot(ks, ys, marker="o", label=method)
    ax.set_xlabel("K-shot"); ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs K-shot"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_{metric_name}_vs_kshot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_matrix_kshot(df, out_dir: str, metric: str = "macro_f1",
                      prefix: str = "paper_kshot") -> str:
    """Plot mean metric vs K-shot for each method from matrix summary rows."""
    _ensure_dir(out_dir)
    value_col = f"{metric}_mean"
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, sub in df.groupby("method"):
        by_shot = sub.groupby("shot")[value_col].mean().sort_index()
        ax.plot(by_shot.index, by_shot.values, marker="o", label=str(method))
    ax.set_xlabel("K-shot")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Few-shot Performance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_{metric}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_loao_heatmap(df, out_dir: str, metric: str = "macro_f1",
                      method: str = "MetaOpt", prefix: str = "paper_loao") -> str:
    """Plot unknown-attack x shot heatmap for one method."""
    _ensure_dir(out_dir)
    value_col = f"{metric}_mean"
    sub = df[df["method"] == method]
    pivot = sub.pivot_table(index="unknown", columns="shot", values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.1 * max(len(pivot.columns), 3) + 3, 0.45 * max(len(pivot), 3) + 2))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("K-shot")
    ax.set_ylabel("Unknown attack")
    ax.set_title(f"LOAO {metric.replace('_', ' ')} ({method})")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            ax.text(j, i, f"{value:.3f}" if value == value else "nan",
                    ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_{method}_{metric}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_convergence_bars(df, out_dir: str, prefix: str = "paper_convergence") -> str:
    """Plot convergence95 step by method; lower is faster."""
    _ensure_dir(out_dir)
    value_col = "convergence95_step_mean"
    grouped = df.groupby("method")[value_col].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(grouped.index.astype(str), grouped.values)
    for bar, value in zip(bars, grouped.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("steps to 95% best checkpoint")
    ax.set_title("Adaptation Convergence")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_update_scatter(df, out_dir: str, prefix: str = "paper_update_scatter") -> str:
    """Plot update norm against gradient norm for update analysis rows."""
    _ensure_dir(out_dir)
    sub = df[df["group"] == "all"] if "group" in df else df
    fig, ax = plt.subplots(figsize=(6, 5))
    for method, part in sub.groupby("method"):
        ax.scatter(part["grad_norm"], part["update_norm"], s=12, alpha=0.45, label=str(method))
    ax.set_xlabel("gradient norm")
    ax.set_ylabel("update norm")
    ax.set_title("Update vs Gradient Magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_layer_update_distribution(df, out_dir: str,
                                   prefix: str = "paper_layer_updates") -> str:
    """Plot mean update-to-gradient ratio by layer group and method."""
    _ensure_dir(out_dir)
    grouped = df.groupby(["group", "method"])["update_to_grad_ratio"].mean().unstack("method")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    grouped.plot(kind="bar", ax=ax)
    ax.set_xlabel("parameter group")
    ax.set_ylabel("mean ||delta|| / ||grad||")
    ax.set_title("Layer-wise Update Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
