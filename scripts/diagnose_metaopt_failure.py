"""Small MetaOpt failure diagnostics without running a full experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build import (  # noqa: E402
    build_meta_model,
    build_meta_optimizer,
    load_artifacts,
)
from src.data.pipeline import build_pipeline  # noqa: E402
from src.data.task_sampler import MetaTask  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.update_analysis import update_rows_to_dicts  # noqa: E402
from src.meta_learning.functional import functional_forward  # noqa: E402
from src.meta_learning.inner_loop import InnerLoop  # noqa: E402
from src.meta_learning.outer_loop import OuterLoop  # noqa: E402
from src.meta_optimizer.dummy import DummyMetaOptimizer  # noqa: E402
from src.meta_optimizer.handcrafted import HandcraftedOptimizer  # noqa: E402
from src.trainer.adapter import FewShotAdapter  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MetaOpt failure diagnostics")
    parser.add_argument("--artifacts", default="outputs/first_fix/botnet_head_s5.pt")
    parser.add_argument("--out", default="outputs/diagnostics/metaopt_failure")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--p0_tasks", type=int, default=30)
    parser.add_argument("--p0_val_tasks", type=int, default=10)
    parser.add_argument("--p1_steps", type=int, default=300)
    parser.add_argument("--p1_log_every", type=int, default=10)
    parser.add_argument("--shot", type=int, default=5)
    return parser.parse_args()


def _sample_tasks(sampler, n: int) -> List[MetaTask]:
    return [sampler.sample_task() for _ in range(n)]


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _outcome_summary(outcomes) -> Dict[str, float]:
    diagnostics = [o.diagnostics[-1] for o in outcomes if o.diagnostics]
    return {
        "support_loss": _mean([
            o.support_losses[-1] for o in outcomes if o.support_losses
        ]),
        "query_macro_f1": _mean([
            o.final_metrics.macro_f1 if o.final_metrics.macro_f1 is not None else o.final_metrics.f1
            for o in outcomes
        ]),
        "attack_recall": _mean([
            o.final_metrics.attack_recall
            for o in outcomes
            if o.final_metrics.attack_recall is not None
        ]),
        "false_positive_rate": _mean([
            o.final_metrics.false_positive_rate
            for o in outcomes
            if o.final_metrics.false_positive_rate is not None
        ]),
        "benign_recall": _mean([
            float(row.get("normal_recall", float("nan")))
            for row in diagnostics
        ]),
        "predicted_attack_rate": _mean([
            float(row.get("prediction_positive_rate", float("nan")))
            for row in diagnostics
        ]),
    }


def _run(adapter, init_params, tasks, factory, adapt_names, n_way, steps, attack_idx):
    outcomes = []
    for task in tasks:
        outcomes.append(adapter.adapt_once(
            init_params,
            task,
            factory(),
            adapt_names,
            n_way,
            max_steps=steps,
            attack_class_indices=[attack_idx],
            collect_update_stats=True,
        ))
    return outcomes


def _grid_sgd_lr(adapter, init_params, val_tasks, adapt_names, n_way, steps, attack_idx, grid):
    best_lr = float(grid[0])
    best_score = -float("inf")
    for lr in grid:
        outcomes = _run(
            adapter,
            init_params,
            val_tasks,
            lambda lr=float(lr): HandcraftedOptimizer(kind="sgd", lr=lr),
            adapt_names,
            n_way,
            steps,
            attack_idx,
        )
        score = _outcome_summary(outcomes)["query_macro_f1"]
        if score > best_score:
            best_score = score
            best_lr = float(lr)
    return best_lr, best_score


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_p0(cfg, art, bundle, model, meta_opt, init_params, adapter, args, device):
    seed = int(cfg.experiment.get("seed", 42))
    n_way = int(art["extra"]["n_way"])
    attack_idx = 1 if n_way == 2 else n_way - 1
    q_query = int(cfg.data.q_query)
    mode = str(cfg.data.get("task_mode", "binary"))
    disallow_ov = bool(cfg.data.get("disallow_support_query_overlap", True))
    disallow_internal = bool(cfg.data.get("disallow_internal_overlap", True))
    adapt_names = list(art["extra"]["adapt_names"])
    steps = int(cfg.meta.inner_steps)

    val_sampler = bundle.make_adaptation_sampler(
        k_shot=args.shot,
        q_query=q_query,
        mode=mode,
        n_way=n_way,
        seed=seed + 501,
        disallow_support_query_overlap=disallow_ov,
        disallow_internal_overlap=disallow_internal,
        split="val",
    )
    test_sampler = bundle.make_adaptation_sampler(
        k_shot=args.shot,
        q_query=q_query,
        mode=mode,
        n_way=n_way,
        seed=seed + 1501,
        disallow_support_query_overlap=disallow_ov,
        disallow_internal_overlap=disallow_internal,
        split="test",
    )
    val_tasks = _sample_tasks(val_sampler, args.p0_val_tasks)
    test_tasks = _sample_tasks(test_sampler, args.p0_tasks)
    sgd_grid = [float(x) for x in cfg.compare.baseline_lr_grid.sgd]
    best_sgd_lr, best_val_f1 = _grid_sgd_lr(
        adapter, init_params, val_tasks, adapt_names, n_way, steps, attack_idx, sgd_grid)

    methods = {
        "SGD": lambda: HandcraftedOptimizer(kind="sgd", lr=best_sgd_lr),
        "DummyMetaOptimizer": lambda: DummyMetaOptimizer(lr=best_sgd_lr),
        "LSTM_MetaOpt": lambda: meta_opt,
    }
    result = {
        "shot": int(args.shot),
        "inner_steps": steps,
        "selected_sgd_lr": best_sgd_lr,
        "selected_sgd_val_macro_f1": best_val_f1,
        "methods": {},
    }
    update_rows = []
    for name, factory in methods.items():
        outcomes = _run(
            adapter, init_params, test_tasks, factory, adapt_names,
            n_way, steps, attack_idx)
        result["methods"][name] = _outcome_summary(outcomes)
        for task_id, outcome in enumerate(outcomes):
            if outcome.update_trace is not None:
                update_rows.extend(update_rows_to_dicts(
                    outcome.update_trace.rows,
                    experiment="p0_dummy_compare",
                    shot=args.shot,
                    method=name,
                    task_id=task_id,
                ))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "p0_dummy_compare.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    _write_csv(os.path.join(args.out, "p0_update_analysis.csv"), update_rows)
    return result


def _eval_single_task(model, meta_opt, init_params, task, adapt_names, n_way, steps, device):
    adapter = FewShotAdapter(model, device)
    outcome = adapter.adapt_once(
        init_params,
        task,
        meta_opt,
        adapt_names,
        n_way,
        max_steps=steps,
        attack_class_indices=[1 if n_way == 2 else n_way - 1],
        collect_update_stats=True,
    )
    task_on_device = task.to(device)
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        query_loss = float(loss_fn(
            outcome.final_logits.to(device), task_on_device.query_y).detach().cpu())
    metrics = outcome.final_metrics
    diag = outcome.diagnostics[-1] if outcome.diagnostics else {}
    all_rows = [
        row for row in (outcome.update_trace.rows if outcome.update_trace else [])
        if row.group == "all"
    ]
    return {
        "support_loss": outcome.support_losses[-1] if outcome.support_losses else float("nan"),
        "query_loss": query_loss,
        "macro_f1": metrics.macro_f1 if metrics.macro_f1 is not None else metrics.f1,
        "attack_recall": metrics.attack_recall if metrics.attack_recall is not None else float("nan"),
        "benign_recall": float(diag.get("normal_recall", float("nan"))),
        "false_positive_rate": (
            metrics.false_positive_rate
            if metrics.false_positive_rate is not None else float("nan")
        ),
        "predicted_attack_rate": float(diag.get("prediction_positive_rate", float("nan"))),
        "grad_norm": _mean([row.grad_norm for row in all_rows]),
        "update_norm": _mean([row.update_norm for row in all_rows]),
        "cosine_delta_neg_grad": _mean([-row.cosine_update_grad for row in all_rows]),
        "clip_ratio": _mean([float(row.was_clipped) for row in all_rows]),
    }


def run_p1(cfg, art, bundle, model, meta_opt, init_params, args, device):
    seed = int(cfg.experiment.get("seed", 42))
    n_way = int(art["extra"]["n_way"])
    q_query = int(cfg.data.q_query)
    mode = str(cfg.data.get("task_mode", "binary"))
    disallow_ov = bool(cfg.data.get("disallow_support_query_overlap", True))
    disallow_internal = bool(cfg.data.get("disallow_internal_overlap", True))
    adapt_names = list(art["extra"]["adapt_names"])
    steps = int(cfg.meta.inner_steps)

    sampler = bundle.make_adaptation_sampler(
        k_shot=args.shot,
        q_query=q_query,
        mode=mode,
        n_way=n_way,
        seed=seed + 2601,
        disallow_support_query_overlap=disallow_ov,
        disallow_internal_overlap=disallow_internal,
        split="test",
    )
    fixed_task = sampler.sample_task().to(device)
    loss_fn = nn.CrossEntropyLoss()
    inner = InnerLoop(
        model,
        meta_opt,
        inner_steps=steps,
        tbptt_steps=int(cfg.meta.get("tbptt_steps", 0)),
        first_order=bool(cfg.meta.get("first_order", False)),
        loss_fn=loss_fn,
    )
    outer = OuterLoop(model, inner, adapt_names=adapt_names, query_loss_fn=loss_fn)
    optimizer = torch.optim.Adam(
        meta_opt.parameters(), lr=float(cfg.meta.meta_optimizer_lr))

    rows = []
    for step in range(1, args.p1_steps + 1):
        meta_opt.train()
        optimizer.zero_grad()
        result = outer.run_meta_batch([fixed_task], init_params=init_params)
        result.meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(meta_opt.parameters(), float(cfg.train.get("grad_clip", 1.0)))
        optimizer.step()
        if step == 1 or step % args.p1_log_every == 0 or step == args.p1_steps:
            meta_opt.eval()
            row = {
                "meta_step": int(step),
                "meta_query_loss": float(result.meta_loss.detach().cpu()),
            }
            row.update(_eval_single_task(
                model, meta_opt, init_params, fixed_task, adapt_names,
                n_way, steps, device))
            rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    _write_csv(os.path.join(args.out, "p1_single_task_overfit.csv"), rows)
    with open(os.path.join(args.out, "p1_single_task_overfit.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    return rows


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    art = load_artifacts(args.artifacts)
    cfg = Config(art["config"])
    if args.override:
        cfg = cfg.apply_overrides(args.override)

    seed = int(cfg.experiment.get("seed", 42))
    set_seed(seed, bool(cfg.experiment.get("deterministic", True)))
    device = resolve_device(str(cfg.device.get("prefer", "auto")))
    bundle = build_pipeline(cfg, seed=seed)
    model = build_meta_model(cfg, art["extra"]["feature_dim"], art["extra"]["window_size"]).to(device)
    model.load_state_dict(art["meta_init_state"])
    init_params = {name: param for name, param in model.named_parameters()}
    meta_opt = build_meta_optimizer(cfg).to(device)
    meta_opt.load_state_dict(art["meta_opt_state"])
    meta_opt.eval()
    adapter = FewShotAdapter(model, device)

    with open(os.path.join(args.out, "effective_config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2, ensure_ascii=False)
    p0 = run_p0(cfg, art, bundle, model, meta_opt, init_params, adapter, args, device)
    p1 = run_p1(cfg, art, bundle, model, meta_opt, init_params, args, device)
    summary = {"p0": p0, "p1_last": p1[-1] if p1 else None}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
