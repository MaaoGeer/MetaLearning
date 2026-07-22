"""Generate paper-ready tables, figures, and a Chinese experiment section."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.plots import (  # noqa: E402
    plot_convergence_bars,
    plot_layer_update_distribution,
    plot_loao_heatmap,
    plot_matrix_kshot,
    plot_update_scatter,
)


FORBIDDEN_CLAIMS = [
    "Future Prediction",
    "Next Flow Prediction",
    "Learning Initialization",
    "Supervised Pretraining",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper experiment report.")
    parser.add_argument("--input", default="outputs/fast_adaptation_matrix")
    parser.add_argument("--out", default="outputs/paper_report")
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _step_filtered_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    # Prefer the largest reported checkpoint for final few-shot tables.
    max_step = summary["step"].max()
    return summary[summary["step"] == max_step].copy()


def _fewshot_table(summary: pd.DataFrame) -> pd.DataFrame:
    final = _step_filtered_summary(summary)
    cols = [
        "unknown", "shot", "method", "n_seeds",
        "accuracy_mean", "accuracy_std",
        "macro_f1_mean", "macro_f1_std",
        "weighted_f1_mean", "weighted_f1_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "attack_recall_mean", "attack_recall_std",
    ]
    return final[[c for c in cols if c in final.columns]].sort_values(
        [c for c in ["unknown", "shot", "method"] if c in final.columns])


def _adaptation_speed_table(summary: pd.DataFrame) -> pd.DataFrame:
    final = _step_filtered_summary(summary)
    cols = [
        "unknown", "shot", "method", "n_seeds",
        "mean_steps_mean", "mean_steps_std",
        "reach_rate_mean", "reach_rate_std",
        "curve_auc_mean", "curve_auc_std",
        "convergence95_step_mean", "convergence95_step_std",
        "post_peak_drop_mean", "post_peak_drop_std",
    ]
    return final[[c for c in cols if c in final.columns]].sort_values(
        [c for c in ["unknown", "shot", "method"] if c in final.columns])


def _significance_table(significance: pd.DataFrame) -> pd.DataFrame:
    if significance.empty:
        return significance
    sub = significance[
        (significance["metric"] == "macro_f1")
        & (significance["comparison"].isin(["MetaOpt-Adam", "MetaOpt-SGD"]))
    ].copy()
    cols = [
        "unknown", "shot", "step", "comparison", "metric", "n_paired_seeds",
        "mean_delta", "std_delta", "p_value", "ci95_low", "ci95_high",
        "probability_left_better",
    ]
    return sub[[c for c in cols if c in sub.columns]].sort_values(
        [c for c in ["unknown", "shot", "comparison", "step"] if c in sub.columns])


def _safe_best(df: pd.DataFrame, method: str, metric: str) -> float:
    col = f"{metric}_mean"
    if df.empty or col not in df or "method" not in df:
        return float("nan")
    sub = df[df["method"] == method]
    return float(sub[col].mean()) if len(sub) else float("nan")


def _section_text(summary: pd.DataFrame, significance: pd.DataFrame) -> str:
    final = _step_filtered_summary(summary)
    meta_macro = _safe_best(final, "MetaOpt", "macro_f1")
    adam_macro = _safe_best(final, "Adam", "macro_f1")
    sgd_macro = _safe_best(final, "SGD", "macro_f1")
    sig_hint = "暂无显著性结果"
    if not significance.empty and "p_value" in significance:
        sig = significance[
            (significance["comparison"] == "MetaOpt-Adam")
            & (significance["metric"] == "macro_f1")
        ]
        if len(sig):
            sig_hint = f"MetaOpt 相对 Adam 的平均 p 值为 {sig['p_value'].mean():.4f}"

    text = f"""# 实验结果与分析

## 实验任务定义

本文实验统一采用 Historical Window-assisted Current Flow Classification。模型输入为连续 Flow 构成的时间窗口，窗口内样本只包含当前 Flow 及其历史上下文；模型输出为窗口最后一个 Flow 的攻击类别。所有实验均使用单向 LSTM 作为基础分类器，并比较 SGD、Adam 与 Learned Meta Optimizer 在相同 few-shot episode 上的适应能力。

## 实验设置

所有方法共享相同的数据划分、相同随机初始化参数、相同 support/query 集合、相同 inner update step 和相同评价指标。SGD 与 Adam 的学习率只在 validation episode 上选择，test episode 不参与任何超参数选择。主要评价指标包括 Accuracy、Macro-F1、Weighted-F1、Precision、Recall 与 unknown attack recall。

## Few-shot 性能

在最终 checkpoint 汇总结果中，MetaOpt 的平均 Macro-F1 为 {meta_macro:.4f}，Adam 为 {adam_macro:.4f}，SGD 为 {sgd_macro:.4f}。该结果用于回答 learned update rule 是否能在少量支持样本下获得更好的任务适应性能。

## 适应速度与收敛性

Adaptation curve 以 inner update step 为横轴，以 query Macro-F1 为纵轴。若 MetaOpt 在更少 step 内达到相同 Macro-F1，或拥有更高 curve AUC，则说明其优势来自更有效的快速适应过程，而不是单纯最终性能波动。

## Unknown Attack 泛化

LOAO 实验逐一将一个攻击家族作为未知攻击，只在测试适应阶段暴露少量 support 样本。该设置检验 MetaOpt 是否能将从已知攻击 episode 中学到的更新策略迁移到未参与 meta-training 的攻击类型。

## 统计显著性

显著性检验采用 paired seed 对齐：同一 unknown、shot、step 与 seed 下比较两个优化器的 Macro-F1 差值。{sig_hint}。当 p 值低于 0.05 且 bootstrap 置信区间不跨 0 时，可以认为 MetaOpt 的提升具有统计支持。

## 更新规则可解释性

Update analysis 统计每一步的梯度范数、参数更新范数、更新与梯度的余弦相似度，以及不同 LSTM/Classifier 参数组的更新比例。若 MetaOpt 在不同层上呈现不同 update-to-gradient ratio，且其更新方向并非固定负梯度缩放，则说明它学习到的是任务相关的更新策略，而不是复杂形式的固定学习率 SGD。
"""
    for forbidden in FORBIDDEN_CLAIMS:
        text = text.replace(forbidden, "")
    return text


def _make_figures(summary: pd.DataFrame, raw_rows: pd.DataFrame, out_dir: Path) -> List[str]:
    fig_dir = out_dir / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    final = _step_filtered_summary(summary)
    if not final.empty:
        paths.append(plot_matrix_kshot(final, str(fig_dir), metric="macro_f1"))
        paths.append(plot_matrix_kshot(final, str(fig_dir), metric="accuracy"))
        if "unknown" in final and final["unknown"].nunique() > 0:
            paths.append(plot_loao_heatmap(final, str(fig_dir), metric="macro_f1", method="MetaOpt"))
        if "convergence95_step_mean" in final:
            paths.append(plot_convergence_bars(final, str(fig_dir)))

    update_path = out_dir.parent / "experiments" / "update_analysis.csv"
    # Matrix runs keep per-run update CSVs under runs/*; raw_rows is used when a
    # caller passes a merged update_analysis-like file in the input root.
    if not raw_rows.empty and {"grad_norm", "update_norm", "group", "method"}.issubset(raw_rows.columns):
        paths.append(plot_update_scatter(raw_rows, str(fig_dir)))
        layer = raw_rows[raw_rows["group"] != "all"]
        if len(layer):
            paths.append(plot_layer_update_distribution(layer, str(fig_dir)))
    elif update_path.exists():
        update_df = pd.read_csv(update_path)
        paths.append(plot_update_scatter(update_df, str(fig_dir)))
        layer = update_df[update_df["group"] != "all"]
        if len(layer):
            paths.append(plot_layer_update_distribution(layer, str(fig_dir)))
    return paths


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    out_dir = Path(args.out)
    table_dir = out_dir / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_csv(input_dir / "summary" / "matrix_results.csv")
    significance = _read_csv(input_dir / "significance" / "matrix_results.csv")
    raw_rows = _read_csv(input_dir / "update_analysis" / "matrix_results.csv")

    _write_table(_fewshot_table(summary), table_dir / "fewshot_performance.csv")
    _write_table(_adaptation_speed_table(summary), table_dir / "adaptation_speed.csv")
    _write_table(_significance_table(significance), table_dir / "significance_macro_f1.csv")

    figures = _make_figures(summary, raw_rows, out_dir)
    section = _section_text(summary, significance)
    section += "\n\n## 自动生成图表\n\n"
    for path in figures:
        section += f"- {path}\n"

    section_path = out_dir / "experiment_section.md"
    section_path.write_text(section, encoding="utf-8")
    print(f"Generated report: {section_path}")


if __name__ == "__main__":
    main()
