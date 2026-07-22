"""Validation-only diagnosis of MetaOpt scope and update behavior.

This is deliberately a diagnostic runner.  It creates fresh ``adapt_val``
manifests, never opens a test manifest, and does not train or alter MetaOpt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build import build_meta_model, build_meta_optimizer, load_artifacts  # noqa: E402
from src.data.pipeline import build_pipeline  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.task_manifest import (  # noqa: E402
    load_tasks_from_manifest, read_task_manifest, sha256_file, write_task_manifest,
)
from src.meta_learning.functional import functional_forward  # noqa: E402
from src.meta_optimizer.handcrafted import HandcraftedOptimizer  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


STEPS = 20
CHECKPOINTS = (0, 1, 2, 5, 10, 20)
TARGET = .8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="42,52,62")
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--shot", type=int, default=3)
    p.add_argument("--q-query", type=int, default=10)
    p.add_argument("--adam-head-lrs", default="0.01,0.05,0.1")
    p.add_argument("--adam-full-lrs", default="0.0005,0.001,0.005,0.01")
    return p.parse_args()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def norm(values: Iterable[torch.Tensor]) -> float:
    return math.sqrt(sum(float(x.detach().double().pow(2).sum().cpu()) for x in values))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa, bb = a.detach().reshape(-1).float(), b.detach().reshape(-1).float()
    denom = torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb)
    return float((torch.dot(aa, bb) / torch.clamp(denom, min=1e-12)).cpu())


def aggregate_cosine(updates: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]) -> float:
    a = torch.cat([updates[name].detach().reshape(-1).float() for name in updates])
    b = torch.cat([target[name].detach().reshape(-1).float() for name in updates])
    return cosine(a, b)


def evaluate(model: nn.Module, params: Mapping[str, torch.Tensor], task: Any, loss: nn.Module) -> tuple[float, float, float]:
    with torch.no_grad():
        support = functional_forward(model, params, task.support_x)
        query = functional_forward(model, params, task.query_x)
        sl, ql = float(loss(support, task.support_y).cpu()), float(loss(query, task.query_y).cpu())
    f1 = float("nan") if not torch.isfinite(query).all() else float(
        compute_metrics(query.cpu(), task.query_y.cpu(), num_classes=2).macro_f1)
    return sl, ql, f1


def fresh_manifest(out: Path, artifact_path: Path, artifact: Mapping[str, Any], cfg: Config,
                   seed: int, task_seed: int, n_tasks: int, shot: int, q_query: int) -> tuple[dict, Any, dict]:
    """Generate an independently seeded validation-only manifest."""
    bundle = build_pipeline(cfg, seed=seed)
    sampler = bundle.make_adaptation_sampler(
        k_shot=shot, q_query=q_query, mode=str(cfg.data.get("task_mode", "binary")),
        n_way=int(artifact["extra"]["n_way"]), seed=task_seed,
        disallow_support_query_overlap=bool(cfg.data.get("disallow_support_query_overlap", True)),
        disallow_internal_overlap=bool(cfg.data.get("disallow_internal_overlap", True)), split="val")
    tasks = [sampler.sample_task() for _ in range(n_tasks)]
    path = out / "manifests" / f"validation_seed_{seed}_taskseed_{task_seed}.json"
    digest = write_task_manifest(
        path, tasks,
        protocol={"shot": shot, "q_query": q_query, "n_way": int(artifact["extra"]["n_way"]),
                  "split": "val", "task_seed": task_seed,
                  "data_split_source": "adapt_val: held-out known evaluation partition plus held-out unknown validation partition",
                  "sampler": "AdaptationTaskSampler sequential RNG stream"},
        base_checkpoint_path=str(artifact_path.resolve()), base_checkpoint_sha256=sha256_file(artifact_path),
        metadata={"dataset": str(cfg.data.name), "unknown_class": str(artifact["extra"]["unknown_class"]),
                  "experiment_seed": seed, "adapt_scope_artifact": str(artifact["extra"]["adaptation_scope"]),
                  "strict_adapt_test": bool(cfg.data.get("strict_adapt_test", False)),
                  "stage": "stage3_validation_only"}, dataset=bundle.adapt_val_dataset)
    return read_task_manifest(path), bundle.adapt_val_dataset, {"path": str(path.resolve()), "sha256": digest, "task_seed": task_seed}


def scope_names(init: Mapping[str, torch.Tensor], scope: str) -> List[str]:
    if scope == "head_only":
        return [name for name in init if name.startswith("classifier.")]
    if scope == "full":
        return list(init)
    raise ValueError(scope)


def _row_base(seed: int, task_index: int, method: str, scope: str, lr: float | None, step: int) -> dict:
    return {"seed": seed, "task_index": task_index, "method": method, "scope": scope,
            "lr": "" if lr is None else lr, "step": step}


def trace_one(
    model: nn.Module, init: Mapping[str, torch.Tensor], task: Any, names: Sequence[str], method: str,
    lr: float | None, meta: nn.Module | None, seed: int, task_index: int, scope: str,
    compare_adam_lr: float = .1,
) -> tuple[List[dict], List[dict]]:
    """Record trajectory and tensor-level updates.  MetaOpt gets an Adam reference on its own gradient stream."""
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    full = OrderedDict((n, v.detach().clone().to(task.support_x.device).requires_grad_(True)) for n, v in init.items())
    selected = set(names)
    frozen = OrderedDict((n, v) for n, v in full.items() if n not in selected)
    params = OrderedDict((n, full[n]) for n in names)
    if method == "MetaOpt":
        assert meta is not None
        optimizer: Any = meta
        adam_reference = HandcraftedOptimizer("adam", lr=compare_adam_lr)
        reference_state = adam_reference.init_state(params)
    else:
        optimizer = HandcraftedOptimizer(method.lower(), lr=float(lr))
        adam_reference = reference_state = None
    state = optimizer.init_state(params)
    rows, tensor_rows = [], []
    sl, ql, f1 = evaluate(model, {**frozen, **params}, task, loss_fn)
    rows.append({**_row_base(seed, task_index, method, scope, lr, 0), "support_loss": sl, "query_loss": ql,
                 "query_macro_f1": f1, "grad_norm": float("nan"), "update_norm": 0.,
                 "update_to_parameter_norm_ratio": 0., "clip_triggered_fraction": 0.,
                 "cosine_update_negative_gradient": float("nan"), "cosine_update_adam": float("nan"), "status": "ok"})
    for step in range(1, STEPS + 1):
        loss = loss_fn(functional_forward(model, {**frozen, **params}, task.support_x), task.support_y)
        try:
            gradients = torch.autograd.grad(loss, list(params.values()), create_graph=False, retain_graph=False, allow_unused=False)
        except Exception as exc:
            rows.append({**_row_base(seed, task_index, method, scope, lr, step), "support_loss": float(loss.detach().cpu()),
                         "query_loss": float("nan"), "query_macro_f1": float("nan"), "grad_norm": float("nan"),
                         "update_norm": float("nan"), "update_to_parameter_norm_ratio": float("nan"),
                         "clip_triggered_fraction": float("nan"), "cosine_update_negative_gradient": float("nan"),
                         "cosine_update_adam": float("nan"), "status": f"gradient_error:{type(exc).__name__}"})
            break
        grads = OrderedDict(zip(params.keys(), gradients))
        reference_updates = None
        if method == "MetaOpt":
            reference_updates, reference_state = adam_reference.step(grads, reference_state)
        updates, state = optimizer.step(grads, state)
        pre_param_norm = norm(params.values())
        update_norm = norm(updates.values())
        clips = getattr(optimizer, "last_clip_scales", {}) if method == "MetaOpt" else {}
        clip_fraction = (sum(float(value) < .999999 for value in clips.values()) / len(clips)) if clips else 0.
        sl, ql, f1 = evaluate(model, {**frozen, **OrderedDict((n, params[n] + updates[n]) for n in params)}, task, loss_fn)
        status = "ok" if all(np.isfinite(v) for v in (sl, ql, f1, norm(grads.values()), update_norm)) else "nonfinite"
        rows.append({**_row_base(seed, task_index, method, scope, lr, step), "support_loss": sl, "query_loss": ql,
                     "query_macro_f1": f1, "grad_norm": norm(grads.values()), "update_norm": update_norm,
                     "update_to_parameter_norm_ratio": update_norm / max(pre_param_norm, 1e-12),
                     "clip_triggered_fraction": clip_fraction,
                     "cosine_update_negative_gradient": aggregate_cosine(updates, OrderedDict((n, -g) for n, g in grads.items())),
                     "cosine_update_adam": aggregate_cosine(updates, reference_updates) if reference_updates is not None else float("nan"),
                     "status": status})
        for name in params:
            pnorm, unorm, gnorm = norm([params[name]]), norm([updates[name]]), norm([grads[name]])
            tensor_rows.append({**_row_base(seed, task_index, method, scope, lr, step), "parameter": name,
                                "parameter_group": "classifier.weight" if name == "classifier.weight" else ("classifier.bias" if name == "classifier.bias" else "lstm"),
                                "parameter_norm": pnorm, "grad_norm": gnorm, "update_norm": unorm,
                                "update_to_parameter_norm_ratio": unorm / max(pnorm, 1e-12),
                                "clip_scale": float(clips.get(name, 1.)), "was_clipped": int(float(clips.get(name, 1.)) < .999999),
                                "cosine_update_negative_gradient": cosine(updates[name], -grads[name]),
                                "cosine_update_adam": cosine(updates[name], reference_updates[name]) if reference_updates is not None else float("nan")})
        params = OrderedDict((n, params[n] + updates[n]) for n in params)
        if status != "ok": break
        if step != STEPS:
            params = OrderedDict((n, value.detach().clone().requires_grad_(True)) for n, value in params.items())
            state = optimizer.detach_state(state)
            if method == "MetaOpt": reference_state = adam_reference.detach_state(reference_state)
    return rows, tensor_rows


def outcomes(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    groups: Dict[tuple, list] = {}
    for row in rows: groups.setdefault((row["seed"], row["task_index"], row["method"], row["scope"], str(row["lr"])), []).append(row)
    result = []
    for key, trace in groups.items():
        hit = [int(r["step"]) for r in trace if np.isfinite(float(r["query_macro_f1"])) and float(r["query_macro_f1"]) >= TARGET]
        result.append({"seed": key[0], "task_index": key[1], "method": key[2], "scope": key[3], "lr": key[4],
                       "steps_to_f1_0_8_capped_20": hit[0] if hit else STEPS, "reached_f1_0_8": bool(hit)})
    return result


def choose_adam(rows: Sequence[Mapping[str, Any]], scope: str) -> dict:
    candidates = sorted({str(r["lr"]) for r in rows if r["method"] == "Adam" and r["scope"] == scope})
    results = []
    all_outcomes = outcomes(rows)
    for lr in candidates:
        sub = [r for r in rows if r["method"] == "Adam" and r["scope"] == scope and str(r["lr"]) == lr and r["status"] == "ok"]
        by_step = {step: [float(r["query_macro_f1"]) for r in sub if int(r["step"]) == step] for step in CHECKPOINTS}
        reach = [r["steps_to_f1_0_8_capped_20"] for r in all_outcomes if r["method"] == "Adam" and r["scope"] == scope and r["lr"] == lr]
        results.append({"scope": scope, "lr": lr, "early_f1_steps_1_2_5": float(np.mean([np.mean(by_step[x]) for x in (1,2,5)])),
                        "step20_f1": float(np.mean(by_step[20])), "mean_steps_to_0_8": float(np.mean(reach)),
                        "non_ok": sum(r["status"] != "ok" for r in rows if r["method"] == "Adam" and r["scope"] == scope and str(r["lr"]) == lr)})
    stable = [r for r in results if not r["non_ok"]] or results
    return {"candidates": results, "selected": max(stable, key=lambda r: (r["early_f1_steps_1_2_5"], -r["mean_steps_to_0_8"], r["step20_f1"]))}


def aggregate(rows: Sequence[Mapping[str, Any]], scopes: Sequence[str]) -> List[dict]:
    result = []
    for scope in scopes:
        for method in ("Adam", "MetaOpt", "SGD"):
            sub = [r for r in rows if r["scope"] == scope and r["method"] == method and r["status"] == "ok"]
            if not sub: continue
            for step in CHECKPOINTS:
                values = [float(r["query_macro_f1"]) for r in sub if int(r["step"]) == step]
                if values: result.append({"scope": scope, "method": method, "step": step, "metric": "query_macro_f1", "mean": float(np.mean(values)), "std_across_tasks": float(np.std(values)), "n": len(values)})
    return result


def update_aggregate(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    fields = ("support_loss", "query_loss", "query_macro_f1", "grad_norm", "update_norm", "update_to_parameter_norm_ratio", "clip_triggered_fraction", "cosine_update_negative_gradient", "cosine_update_adam")
    output = []
    for method in ("Adam", "SGD", "MetaOpt"):
        for step in range(STEPS + 1):
            sub = [r for r in rows if r["method"] == method and int(r["step"]) == step and r["status"] == "ok"]
            if not sub: continue
            item = {"method": method, "step": step, "n_tasks": len(sub)}
            for field in fields:
                values = np.asarray([float(r[field]) for r in sub], dtype=float); values = values[np.isfinite(values)]
                item[field + "_mean"] = float(values.mean()) if len(values) else float("nan")
                item[field + "_std"] = float(values.std()) if len(values) else float("nan")
            output.append(item)
    return output


def parameter_update_aggregate(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    """Aggregate tensor-level head/LSTM updates without discarding the raw trace."""
    output = []
    for method in ("Adam", "SGD", "MetaOpt"):
        for group in sorted({str(r["parameter_group"]) for r in rows if r["method"] == method}):
            for step in range(1, STEPS + 1):
                sub = [r for r in rows if r["method"] == method and r["parameter_group"] == group and int(r["step"]) == step]
                if not sub: continue
                output.append({"method": method, "parameter_group": group, "step": step, "n": len(sub),
                               "update_norm_mean": float(np.mean([float(r["update_norm"]) for r in sub])),
                               "update_to_parameter_norm_ratio_mean": float(np.mean([float(r["update_to_parameter_norm_ratio"]) for r in sub])),
                               "was_clipped_fraction": float(np.mean([float(r["was_clipped"]) for r in sub])),
                               "cosine_update_negative_gradient_mean": float(np.mean([float(r["cosine_update_negative_gradient"]) for r in sub]))})
    return output


def plot_updates(agg: Sequence[Mapping[str, Any]], out: Path) -> None:
    specs = (("query_macro_f1_mean", "query macro-F1", "update_behavior_f1.png"), ("update_norm_mean", "update norm", "update_behavior_update_norm.png"), ("update_to_parameter_norm_ratio_mean", "update / parameter norm", "update_behavior_update_ratio.png"), ("cosine_update_negative_gradient_mean", "cosine(update, -gradient)", "update_behavior_direction.png"))
    for field, label, filename in specs:
        fig, ax = plt.subplots(figsize=(7,4))
        for method in ("Adam", "SGD", "MetaOpt"):
            sub = sorted([r for r in agg if r["method"] == method], key=lambda r: int(r["step"]))
            x = [r["step"] for r in sub]; y = [r[field] for r in sub]
            ax.plot(x, y, label=method)
        ax.set_xlabel("adaptation step"); ax.set_ylabel(label); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out / filename, dpi=160); plt.close(fig)


def plot_head_updates(agg: Sequence[Mapping[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for axis, group in zip(axes, ("classifier.weight", "classifier.bias")):
        for method in ("Adam", "SGD", "MetaOpt"):
            sub = sorted([r for r in agg if r["parameter_group"] == group and r["method"] == method], key=lambda r: int(r["step"]))
            axis.plot([r["step"] for r in sub], [r["update_norm_mean"] for r in sub], label=method)
        axis.set_title(group); axis.set_xlabel("adaptation step"); axis.set_ylabel("update norm"); axis.grid(alpha=.3); axis.legend()
    fig.tight_layout(); fig.savefig(out / "update_behavior_classifier_tensor_norms.png", dpi=160); plt.close(fig)


def plot_scope(agg: Sequence[Mapping[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for axis, scope in zip(axes, ("head_only", "full")):
        for method in ("Adam", "MetaOpt"):
            sub = sorted([r for r in agg if r["scope"] == scope and r["method"] == method], key=lambda r: int(r["step"]))
            axis.plot([r["step"] for r in sub], [r["mean"] for r in sub], label=method)
        axis.set_title(scope + " diagnostic ablation"); axis.set_xlabel("adaptation step"); axis.set_ylabel("query macro-F1"); axis.grid(alpha=.3); axis.legend()
    fig.tight_layout(); fig.savefig(out / "scope_ablation_macro_f1.png", dpi=160); plt.close(fig)


def report(out: Path, static: Mapping[str, Any], adam: Mapping[str, Any], scope_rows: Sequence[Mapping[str, Any]], behavior: Sequence[Mapping[str, Any]]) -> None:
    lookup = {(r["scope"], r["method"], int(r["step"])): r for r in scope_rows}
    head_adam5, head_meta5 = lookup[("head_only", "Adam", 5)]["mean"], lookup[("head_only", "MetaOpt", 5)]["mean"]
    full_adam5, full_meta5 = lookup[("full", "Adam", 5)]["mean"], lookup[("full", "MetaOpt", 5)]["mean"]
    meta_rows = [r for r in behavior if r["method"] == "MetaOpt" and int(r["step"]) > 0]
    clip = float(np.mean([r["clip_triggered_fraction_mean"] for r in meta_rows])) if meta_rows else float("nan")
    direction = float(np.mean([r["cosine_update_negative_gradient_mean"] for r in meta_rows])) if meta_rows else float("nan")
    if full_meta5 >= full_adam5 and head_meta5 < head_adam5 and static["training_scope"] == "full":
        decision = "A: MetaOpt 的有效性依赖全参数适应。"
    elif clip > .5 or (np.isfinite(direction) and direction < 0):
        decision = "C: head_only 失效存在可验证的 clip/尺度/方向异常；在提出最小修复并重训前，不应扩展矩阵。"
    else:
        decision = "B: 当前 artifact 在 head_only 和诊断性 full 条件下均未显示 MetaOpt 对 Adam 的可靠优势；应转入训练目标、更新尺度、任务分布和算法设计诊断。"
    lines = ["# Stage 3: MetaOpt scope and update diagnosis", "", "## Static evidence", "",
             f"- Artifact training scope: `{static['training_scope']}` for all checked seeds; saved adapted tensors: `{', '.join(static['artifact_adapt_names'])}`.",
             f"- Saved head-only parameter count: {static['artifact_adapt_parameter_count']}; full base learner count: {static['full_parameter_count']}.",
             "- The optimizer is coordinate-wise: it reshapes each named gradient tensor to coordinates and reshapes its output back to the same tensor shape. State is keyed by parameter name; no flatten/unflatten or name-order remapping is used.",
             "- State is initialized once per task and retained across adaptation steps; the functional loop detaches state between steps but does not reset it. A new task calls `init_state`, correctly resetting state.",
             "", "## Validation-only protocol", "",
             "Fresh manifests in `manifests/` use only `adapt_val`, with task seeds distinct from prior Stage 2 manifests. No test manifest or test metric was read.",
             "", "## Adam LR selection for scope ablation", "",
             f"- head_only selected `{adam['head_only']['selected']['lr']}`; full selected `{adam['full']['selected']['lr']}`. Selection maximized validation macro-F1 at steps 1/2/5, then faster F1≥0.8 and step 20.",
             "", "## Scope ablation (validation diagnostic only)", "",
             "| Scope | Adam step 5 | MetaOpt step 5 | Adam step 20 | MetaOpt step 20 |", "|---|---:|---:|---:|---:|"]
    for scope in ("head_only", "full"):
        lines.append(f"| {scope} | {lookup[(scope,'Adam',5)]['mean']:.4f} | {lookup[(scope,'MetaOpt',5)]['mean']:.4f} | {lookup[(scope,'Adam',20)]['mean']:.4f} | {lookup[(scope,'MetaOpt',20)]['mean']:.4f} |")
    lines += ["", "`full` is a counterfactual diagnostic: current MetaOpt was trained head_only, so it is not training-scope-consistent and cannot be used as a final performance conclusion.",
              "", "## Update behavior", "",
              f"- MetaOpt mean tensor clip-trigger fraction across steps 1–20: {clip:.4f}.",
              f"- MetaOpt mean cosine(update, -gradient) across steps 1–20: {direction:.4f}.",
              "", "| Step | MetaOpt F1 | Update norm | Update/parameter norm | Clip fraction | cos(update, -grad) | cos(update, Adam) |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for step in (1, 2, 5, 10, 20):
        row = next(item for item in behavior if item["method"] == "MetaOpt" and int(item["step"]) == step)
        lines.append(
            f"| {step} | {row['query_macro_f1_mean']:.4f} | {row['update_norm_mean']:.4f} | "
            f"{row['update_to_parameter_norm_ratio_mean']:.4f} | {row['clip_triggered_fraction_mean']:.4f} | "
            f"{row['cosine_update_negative_gradient_mean']:.4f} | {row['cosine_update_adam_mean']:.4f} |"
        )
    lines += [
              "- `update_behavior_raw.csv` contains every task/step; `update_behavior_per_parameter.csv` separates classifier weight/bias and every adapted tensor; aggregate curves are saved alongside it.",
              "", "## Decision", "", decision,
              "", "A head_only-specific MetaOpt retrain is not justified solely as a scope-alignment fix: these artifacts were already trained head_only. If retraining is approved later, it should test a concrete update-scale/training-objective change on validation before a fresh test manifest.",
              "Do not expand to all attacks × shots until this diagnostic yields a training-consistent MetaOpt advantage over Adam."]
    (out / "STAGE3_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args(); out = Path(args.out)
    if out.exists(): raise FileExistsError(f"refusing to overwrite: {out}")
    out.mkdir(parents=True); (out / "manifests").mkdir()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    head_grid = [float(x) for x in args.adam_head_lrs.split(",") if x.strip()]
    full_grid = [float(x) for x in args.adam_full_lrs.split(",") if x.strip()]
    root = Path(args.artifact_root); prepared = []; manifest_info = []
    static = {"training_scope": None, "artifact_adapt_names": None, "artifact_adapt_parameter_count": None, "full_parameter_count": None, "seeds": []}
    for seed in seeds:
        path = root / f"seed_{seed}" / "horizon_20" / "meta_artifacts.pt"; artifact = load_artifacts(str(path)); cfg = Config(artifact["config"])
        if not bool(cfg.data.get("strict_adapt_test", False)): raise ValueError("strict split required")
        set_seed(seed, bool(cfg.experiment.get("deterministic", True)))
        manifest, dataset, info = fresh_manifest(out, path, artifact, cfg, seed, seed + 4002, args.tasks, args.shot, args.q_query)
        model = build_meta_model(cfg, artifact["extra"]["feature_dim"], artifact["extra"]["window_size"])
        init = OrderedDict((n, v.detach().clone()) for n, v in artifact["meta_init_state"].items())
        head = scope_names(init, "head_only"); full = scope_names(init, "full")
        recorded = list(artifact["extra"]["adapt_names"])
        if recorded != head: raise ValueError(f"artifact head names mismatch for seed {seed}: {recorded} != {head}")
        static["training_scope"] = str(artifact["extra"]["adaptation_scope"]); static["artifact_adapt_names"] = recorded
        static["artifact_adapt_parameter_count"] = int(sum(init[n].numel() for n in recorded)); static["full_parameter_count"] = int(sum(v.numel() for v in init.values()))
        static["seeds"].append({"seed": seed, "artifact": str(path.resolve()), "artifact_sha256": sha256_file(path), "recorded_adapt_names": recorded, "shapes": {n:list(init[n].shape) for n in init}})
        prepared.append((seed, artifact, cfg, manifest, dataset, init, head, full)); manifest_info.append(info)
    write_json(out / "static_scope_audit.json", static); write_json(out / "validation_manifest_index.json", manifest_info)
    write_json(out / "run_protocol.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "validation_only": True, "seeds": seeds, "tasks": args.tasks, "shot": args.shot, "q_query": args.q_query, "steps": STEPS, "head_adam_grid": head_grid, "full_adam_grid": full_grid})
    device = resolve_device(str(prepared[0][2].device.get("prefer", "auto")))
    behavior_rows: List[dict] = []; behavior_tensors: List[dict] = []; ablation_raw: List[dict] = []
    # Head-only behavior: use fixed Stage-2 validation-selected baselines, but fresh Stage-3 tasks.
    for seed, artifact, cfg, manifest, dataset, init, head, full in prepared:
        model = build_meta_model(cfg, artifact["extra"]["feature_dim"], artifact["extra"]["window_size"]).to(device); model.load_state_dict(artifact["meta_init_state"]); model.eval()
        tasks = [x.to(device) for x in load_tasks_from_manifest(manifest, dataset)]
        meta = build_meta_optimizer(cfg).to(device); meta.load_state_dict(artifact["meta_opt_state"]); meta.eval()
        for p in meta.parameters(): p.requires_grad_(False)
        for method, lr in (("Adam", .1), ("SGD", .5), ("MetaOpt", None)):
            for i, task in enumerate(tasks):
                r, t = trace_one(model, init, task, head, method, lr, meta if method == "MetaOpt" else None, seed, i, "head_only")
                behavior_rows.extend(r); behavior_tensors.extend(t)
    write_csv(out / "update_behavior_raw.csv", behavior_rows); write_csv(out / "update_behavior_per_parameter.csv", behavior_tensors)
    behavior_agg = update_aggregate(behavior_rows); write_csv(out / "update_behavior_by_step.csv", behavior_agg); plot_updates(behavior_agg, out)
    tensor_agg = parameter_update_aggregate(behavior_tensors); write_csv(out / "update_behavior_per_parameter_by_step.csv", tensor_agg); plot_head_updates(tensor_agg, out)
    # Scope ablation: independently choose Adam LR for each scope on the same fresh validation tasks.
    for seed, artifact, cfg, manifest, dataset, init, head, full in prepared:
        model = build_meta_model(cfg, artifact["extra"]["feature_dim"], artifact["extra"]["window_size"]).to(device); model.load_state_dict(artifact["meta_init_state"]); model.eval()
        tasks = [x.to(device) for x in load_tasks_from_manifest(manifest, dataset)]
        meta = build_meta_optimizer(cfg).to(device); meta.load_state_dict(artifact["meta_opt_state"]); meta.eval()
        for p in meta.parameters(): p.requires_grad_(False)
        for scope, names, grid in (("head_only", head, head_grid), ("full", full, full_grid)):
            for lr in grid:
                for i, task in enumerate(tasks):
                    r, _ = trace_one(model, init, task, names, "Adam", lr, None, seed, i, scope); ablation_raw.extend(r)
            for i, task in enumerate(tasks):
                r, _ = trace_one(model, init, task, names, "MetaOpt", None, meta, seed, i, scope); ablation_raw.extend(r)
    write_csv(out / "scope_ablation_candidate_raw.csv", ablation_raw)
    selections = {scope: choose_adam(ablation_raw, scope) for scope in ("head_only", "full")}; write_json(out / "scope_ablation_adam_selection.json", selections)
    final_ablation = []
    for scope in ("head_only", "full"):
        chosen = str(selections[scope]["selected"]["lr"])
        final_ablation.extend([r for r in ablation_raw if r["scope"] == scope and (r["method"] == "MetaOpt" or (r["method"] == "Adam" and str(r["lr"]) == chosen))])
    write_csv(out / "scope_ablation_locked_raw.csv", final_ablation)
    scope_agg = aggregate(final_ablation, ("head_only", "full")); write_csv(out / "scope_ablation_summary.csv", scope_agg); plot_scope(scope_agg, out)
    report(out, static, selections, scope_agg, behavior_agg)
    print(json.dumps({"out": str(out.resolve()), "training_scope": static["training_scope"], "adam_head": selections["head_only"]["selected"], "adam_full": selections["full"]["selected"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
