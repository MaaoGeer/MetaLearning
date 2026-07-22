"""Small-scale Adam inner-loop LR diagnosis on an explicit task manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build import build_meta_model, load_artifacts  # noqa: E402
from src.data.pipeline import build_pipeline  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.task_manifest import (  # noqa: E402
    load_tasks_from_manifest,
    read_task_manifest,
    sha256_file,
)
from src.meta_learning.functional import functional_forward  # noqa: E402
from src.meta_optimizer.handcrafted import HandcraftedOptimizer  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.visualization.plots import plot_adaptation_curves  # noqa: E402


CHECKPOINTS = [0, 1, 2, 5, 10, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lrs", default="0.001,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--steps", type=int, default=20)
    return parser.parse_args()


def l2_norm(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        total += float(tensor.detach().double().pow(2).sum().cpu())
    return math.sqrt(total)


def evaluate_state(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    task,
    loss_fn: nn.Module,
) -> Tuple[float, float, float]:
    with torch.no_grad():
        support_logits = functional_forward(model, params, task.support_x)
        query_logits = functional_forward(model, params, task.query_x)
        support_loss = float(loss_fn(support_logits, task.support_y).cpu())
        query_loss = float(loss_fn(query_logits, task.query_y).cpu())
    if not torch.isfinite(query_logits).all():
        return support_loss, query_loss, float("nan")
    metrics = compute_metrics(query_logits.cpu(), task.query_y.cpu(), num_classes=2)
    return support_loss, query_loss, float(metrics.macro_f1)


def run_task(
    model: nn.Module,
    init_state: Dict[str, torch.Tensor],
    task,
    adapt_names: List[str],
    lr: float,
    steps: int,
    task_index: int,
) -> List[dict]:
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    full = OrderedDict(
        (name, value.detach().clone().to(task.support_x.device).requires_grad_(True))
        for name, value in init_state.items()
    )
    frozen = OrderedDict((name, value) for name, value in full.items() if name not in adapt_names)
    adaptable = OrderedDict((name, full[name]) for name in adapt_names)
    optimizer = HandcraftedOptimizer(kind="adam", lr=lr)
    state = optimizer.init_state(adaptable)
    support_loss, query_loss, macro_f1 = evaluate_state(
        model, {**frozen, **adaptable}, task, loss_fn
    )
    rows = [{
        "lr": lr,
        "task_index": task_index,
        "step": 0,
        "support_loss": support_loss,
        "query_loss": query_loss,
        "query_macro_f1": macro_f1,
        "gradient_norm": float("nan"),
        "parameter_update_norm": 0.0,
        "adam_exp_avg_norm": 0.0,
        "adam_exp_avg_sq_norm": 0.0,
        "none_grad_count": 0,
        "nan_grad_count": 0,
        "inf_grad_count": 0,
        "status": "ok",
    }]

    for step in range(1, steps + 1):
        merged = {**frozen, **adaptable}
        support_logits = functional_forward(model, merged, task.support_x)
        loss = loss_fn(support_logits, task.support_y)
        try:
            grads = torch.autograd.grad(
                loss,
                list(adaptable.values()),
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
            none_count = 0
        except Exception as exc:
            rows.append({
                "lr": lr,
                "task_index": task_index,
                "step": step,
                "support_loss": float(loss.detach().cpu()),
                "query_loss": float("nan"),
                "query_macro_f1": float("nan"),
                "gradient_norm": float("nan"),
                "parameter_update_norm": float("nan"),
                "adam_exp_avg_norm": float("nan"),
                "adam_exp_avg_sq_norm": float("nan"),
                "none_grad_count": len(adaptable),
                "nan_grad_count": 0,
                "inf_grad_count": 0,
                "status": f"gradient_error:{type(exc).__name__}",
            })
            break
        nan_count = sum(int(torch.isnan(grad).sum().cpu()) for grad in grads)
        inf_count = sum(int(torch.isinf(grad).sum().cpu()) for grad in grads)
        grad_dict = OrderedDict(zip(adaptable.keys(), grads))
        updates, state = optimizer.step(grad_dict, state)
        adaptable = OrderedDict(
            (name, adaptable[name] + updates[name]) for name in adaptable
        )
        support_loss, query_loss, macro_f1 = evaluate_state(
            model, {**frozen, **adaptable}, task, loss_fn
        )
        moment_norm = l2_norm(layers[0][0] for layers in state.values())
        moment_sq_norm = l2_norm(layers[0][1] for layers in state.values())
        values = [
            support_loss,
            query_loss,
            macro_f1,
            l2_norm(grads),
            l2_norm(updates.values()),
            moment_norm,
            moment_sq_norm,
        ]
        status = "ok" if all(np.isfinite(value) for value in values) else "nonfinite"
        rows.append({
            "lr": lr,
            "task_index": task_index,
            "step": step,
            "support_loss": support_loss,
            "query_loss": query_loss,
            "query_macro_f1": macro_f1,
            "gradient_norm": values[3],
            "parameter_update_norm": values[4],
            "adam_exp_avg_norm": moment_norm,
            "adam_exp_avg_sq_norm": moment_sq_norm,
            "none_grad_count": none_count,
            "nan_grad_count": nan_count,
            "inf_grad_count": inf_count,
            "status": status,
        })
        if status != "ok":
            break
        if step != steps:
            adaptable = OrderedDict(
                (name, value.detach().clone().requires_grad_(True))
                for name, value in adaptable.items()
            )
            state = optimizer.detach_state(state)
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    keys = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_rows: List[dict], lrs: List[float]) -> Tuple[List[dict], dict]:
    summary_rows: List[dict] = []
    selection_rows = []
    for lr in lrs:
        lr_rows = [row for row in raw_rows if row["lr"] == lr]
        nonfinite_count = sum(row["status"] != "ok" for row in lr_rows)
        by_step = {}
        for step in range(21):
            step_rows = [
                row for row in lr_rows
                if row["step"] == step and row["status"] == "ok"
            ]
            if not step_rows:
                continue
            item = {
                "lr": lr,
                "step": step,
                "n_tasks": len(step_rows),
                "nonfinite_rows_for_lr": nonfinite_count,
            }
            for field in [
                "query_macro_f1",
                "support_loss",
                "query_loss",
                "gradient_norm",
                "parameter_update_norm",
            ]:
                values = np.asarray([row[field] for row in step_rows], dtype=float)
                finite = values[np.isfinite(values)]
                item[f"{field}_mean"] = float(finite.mean()) if len(finite) else float("nan")
                item[f"{field}_std"] = float(finite.std(ddof=0)) if len(finite) else float("nan")
            summary_rows.append(item)
            by_step[step] = item
        early = np.asarray([
            by_step[step]["query_macro_f1_mean"]
            for step in [1, 2, 5]
            if step in by_step
        ])
        step20 = by_step.get(20, {}).get("query_macro_f1_mean", float("nan"))
        all_f1 = [
            row["query_macro_f1_mean"]
            for row in summary_rows
            if row["lr"] == lr and np.isfinite(row["query_macro_f1_mean"])
        ]
        post_peak_drop = max(all_f1) - all_f1[-1] if all_f1 else float("inf")
        selection_rows.append({
            "lr": lr,
            "early_f1_mean_steps_1_2_5": float(early.mean()) if len(early) == 3 else float("nan"),
            "step20_macro_f1": step20,
            "post_peak_drop": post_peak_drop,
            "nonfinite_rows": nonfinite_count,
            "stable": nonfinite_count == 0,
        })
    stable = [
        row for row in selection_rows
        if row["stable"] and np.isfinite(row["early_f1_mean_steps_1_2_5"])
    ]
    candidates = stable or selection_rows
    recommended = max(
        candidates,
        key=lambda row: (
            row["early_f1_mean_steps_1_2_5"],
            row["step20_macro_f1"],
            -row["post_peak_drop"],
        ),
    )
    return summary_rows, {
        "selection_rule": (
            "maximize mean macro-F1 at steps 1/2/5; tie-break by step 20, "
            "then lower post-peak drop; exclude nonfinite configurations when possible"
        ),
        "diagnostic_only_not_for_test_claims": True,
        "recommended_lr_for_next_validation_phase": recommended["lr"],
        "candidates": selection_rows,
    }


def plot_loss_curves(summary_rows: List[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for lr in sorted({row["lr"] for row in summary_rows}):
        rows = sorted(
            [row for row in summary_rows if row["lr"] == lr],
            key=lambda row: row["step"],
        )
        steps = [row["step"] for row in rows]
        axes[0].plot(steps, [row["support_loss_mean"] for row in rows], label=f"lr={lr:g}")
        axes[1].plot(steps, [row["query_loss_mean"] for row in rows], label=f"lr={lr:g}")
    axes[0].set_title("Support loss")
    axes[1].set_title("Query loss")
    for axis in axes:
        axis.set_xlabel("adaptation step")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("mean loss")
    fig.tight_layout()
    fig.savefig(out_dir / "adam_lr_loss_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    lrs = [float(value) for value in args.lrs.split(",") if value.strip()]
    if args.steps != 20:
        raise ValueError("phase-1 protocol requires exactly 20 adaptation steps")
    artifact = load_artifacts(args.artifacts)
    cfg = Config(artifact["config"])
    if not bool(cfg.data.get("strict_adapt_test", False)):
        raise ValueError("Adam tuning requires strict_adapt_test=true")
    if str(artifact["extra"].get("adaptation_scope")) != "head_only":
        raise ValueError("Adam tuning requires the current head_only artifact")
    manifest = read_task_manifest(args.task_manifest, verify_sha256=True)
    artifact_hash = sha256_file(args.artifacts)
    if artifact_hash != manifest["base_checkpoint_sha256"]:
        raise ValueError("artifact SHA256 does not match task manifest")
    seed = int(cfg.experiment.get("seed", 42))
    set_seed(seed, bool(cfg.experiment.get("deterministic", True)))
    device = resolve_device(str(cfg.device.get("prefer", "auto")))
    bundle = build_pipeline(cfg, seed=seed)
    tasks = [task.to(device) for task in load_tasks_from_manifest(
        manifest, bundle.adapt_test_dataset
    )]
    model = build_meta_model(
        cfg, artifact["extra"]["feature_dim"], artifact["extra"]["window_size"]
    ).to(device)
    model.load_state_dict(artifact["meta_init_state"])
    model.eval()
    init_state = OrderedDict(
        (name, value.detach().clone())
        for name, value in artifact["meta_init_state"].items()
    )
    adapt_names = list(artifact["extra"]["adapt_names"])

    raw_rows: List[dict] = []
    for lr in lrs:
        for task_index, task in enumerate(tasks):
            raw_rows.extend(run_task(
                model,
                init_state,
                task,
                adapt_names,
                lr,
                args.steps,
                task_index,
            ))
    summary_rows, selection = summarize(raw_rows, lrs)
    write_csv(out_dir / "adam_lr_raw.csv", raw_rows)
    write_csv(out_dir / "adam_lr_summary.csv", summary_rows)
    (out_dir / "adam_lr_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    run_config = {
        "artifacts": str(Path(args.artifacts).resolve()),
        "artifact_sha256": artifact_hash,
        "task_manifest": str(Path(args.task_manifest).resolve()),
        "task_manifest_sha256": sha256_file(args.task_manifest),
        "task_count": len(tasks),
        "lrs": lrs,
        "steps": args.steps,
        "checkpoints": CHECKPOINTS,
        "strict_adapt_test": True,
        "adapt_scope": "head_only",
        "device": str(device),
        "warning": "Diagnostic test-manifest tuning only; final LR must be selected on a validation manifest.",
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    trajectories = {}
    for lr in lrs:
        rows = sorted(
            [row for row in summary_rows if row["lr"] == lr],
            key=lambda row: row["step"],
        )
        trajectories[f"Adam lr={lr:g}"] = [
            row["query_macro_f1_mean"] for row in rows
        ]
    plot_adaptation_curves(
        trajectories,
        str(out_dir),
        target_f1=0.8,
        prefix="adam_lr_macro_f1",
        steps=list(range(args.steps + 1)),
    )
    plot_loss_curves(summary_rows, out_dir)
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
