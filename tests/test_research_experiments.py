import csv
import importlib.util
import os

import pandas as pd
import torch

from src.data.task_sampler import MetaTask
from src.meta_optimizer.handcrafted import HandcraftedOptimizer
from src.models import LSTMClassifier
from src.trainer.adapter import FewShotAdapter


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    path = os.path.join(ROOT, "scripts", name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_task(feat=4, seq=5, n_way=2, k=2, q=2):
    torch.manual_seed(0)
    return MetaTask(
        support_x=torch.randn(n_way * k, seq, feat),
        support_y=torch.arange(n_way).repeat_interleave(k),
        query_x=torch.randn(n_way * q, seq, feat),
        query_y=torch.arange(n_way).repeat_interleave(q),
        global_classes=list(range(n_way)),
    )


def test_update_trace_has_layer_and_all_groups():
    model = LSTMClassifier(feature_dim=4, n_classes=2, hidden_size=6)
    adapter = FewShotAdapter(model, torch.device("cpu"))
    init_params = {name: param for name, param in model.named_parameters()}
    outcome = adapter.adapt_once(
        init_params,
        _make_task(),
        HandcraftedOptimizer(kind="sgd", lr=0.01),
        list(init_params.keys()),
        n_way=2,
        max_steps=2,
        collect_update_stats=True,
    )

    assert outcome.update_trace is not None
    groups = {row.group for row in outcome.update_trace.rows}
    assert "all" in groups
    assert any(group.startswith("lstm.") for group in groups)
    assert all(row.grad_norm >= 0 for row in outcome.update_trace.rows)
    assert all(row.update_norm >= 0 for row in outcome.update_trace.rows)


def test_significance_writes_all_paired_comparisons(tmp_path):
    mod = _load_script("run_fast_adaptation_matrix.py")
    rows = []
    for seed in [42, 52, 62]:
        for method, value in [("MetaOpt", 0.8), ("Adam", 0.7), ("SGD", 0.6)]:
            rows.append({
                "unknown": "webattack",
                "train_fraction": 1.0,
                "train_horizon": 2,
                "shot": 1,
                "step": 2,
                "seed": seed,
                "method": method,
                "macro_f1": value + seed * 0.0001,
                "accuracy": value,
                "weighted_f1": value,
                "precision": value,
                "recall": value,
                "attack_recall": value,
            })

    mod._write_significance(rows, tmp_path)
    out = tmp_path / "significance" / "matrix_results.csv"
    assert out.exists()
    comparisons = {row["comparison"] for row in csv.DictReader(out.open(encoding="utf-8"))}
    assert {"MetaOpt-Adam", "MetaOpt-SGD", "Adam-SGD"}.issubset(comparisons)


def test_generate_paper_report_outputs_tables_and_section(tmp_path):
    report = _load_script("generate_paper_report.py")
    input_dir = tmp_path / "matrix"
    summary_dir = input_dir / "summary"
    sig_dir = input_dir / "significance"
    summary_dir.mkdir(parents=True)
    sig_dir.mkdir(parents=True)

    pd.DataFrame([
        {
            "unknown": "webattack",
            "shot": 1,
            "method": "MetaOpt",
            "step": 2,
            "n_seeds": 3,
            "accuracy_mean": 0.8,
            "accuracy_std": 0.01,
            "macro_f1_mean": 0.75,
            "macro_f1_std": 0.02,
            "weighted_f1_mean": 0.76,
            "weighted_f1_std": 0.02,
            "precision_mean": 0.74,
            "precision_std": 0.02,
            "recall_mean": 0.73,
            "recall_std": 0.02,
            "attack_recall_mean": 0.72,
            "attack_recall_std": 0.02,
            "mean_steps_mean": 1.0,
            "mean_steps_std": 0.0,
            "reach_rate_mean": 1.0,
            "reach_rate_std": 0.0,
            "curve_auc_mean": 0.7,
            "curve_auc_std": 0.01,
            "convergence95_step_mean": 1.0,
            "convergence95_step_std": 0.0,
            "post_peak_drop_mean": 0.0,
            "post_peak_drop_std": 0.0,
        }
    ]).to_csv(summary_dir / "matrix_results.csv", index=False)
    pd.DataFrame([
        {
            "unknown": "webattack",
            "shot": 1,
            "step": 2,
            "comparison": "MetaOpt-Adam",
            "metric": "macro_f1",
            "n_paired_seeds": 3,
            "mean_delta": 0.1,
            "std_delta": 0.01,
            "p_value": 0.03,
            "ci95_low": 0.05,
            "ci95_high": 0.15,
            "probability_left_better": 0.99,
        }
    ]).to_csv(sig_dir / "matrix_results.csv", index=False)

    out_dir = tmp_path / "paper"
    summary = report._read_csv(summary_dir / "matrix_results.csv")
    significance = report._read_csv(sig_dir / "matrix_results.csv")
    report._write_table(report._fewshot_table(summary), out_dir / "paper_tables" / "fewshot.csv")
    section = report._section_text(summary, significance)
    (out_dir / "experiment_section.md").write_text(section, encoding="utf-8")

    assert (out_dir / "paper_tables" / "fewshot.csv").exists()
    assert "Historical Window-assisted Current Flow Classification" in section
    assert "Future Prediction" not in section
    assert "Supervised Pretraining" not in section
