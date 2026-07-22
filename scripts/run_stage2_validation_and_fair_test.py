"""Stage-2 validation tuning and locked, manifest-based fair test.

This script deliberately separates validation selection from the final test:
all candidate selection is computed from ``adapt_val`` manifests before any
``adapt_test`` task is evaluated.  It is a small bruteforce/3-shot protocol,
not a replacement for the complete experiment matrix.
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
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build import build_meta_model, build_meta_optimizer, load_artifacts  # noqa: E402
from src.data.pipeline import build_pipeline  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.task_manifest import (  # noqa: E402
    load_tasks_from_manifest,
    manifest_raw_row_ids,
    read_task_manifest,
    sha256_file,
    write_task_manifest,
)
from src.meta_learning.functional import functional_forward  # noqa: E402
from src.meta_optimizer.handcrafted import HandcraftedOptimizer  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


CHECKPOINTS = (0, 1, 2, 5, 10, 20)
STEPS = 20
TARGET_F1 = 0.8
METHODS = ("SGD", "Adam", "MetaOpt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True,
                        help=".../fast_adaptation_matrix/runs/bruteforce/fraction_1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--q-query", type=int, default=10)
    parser.add_argument("--adam-lrs", default="0.01,0.02,0.05,0.1")
    parser.add_argument("--sgd-lrs", default="0.01,0.05,0.1,0.5")
    return parser.parse_args()


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _float(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _l2(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        total += float(tensor.detach().double().pow(2).sum().cpu())
    return math.sqrt(total)


def _task_manifest(
    out: Path,
    artifact_path: Path,
    cfg: Config,
    artifact: Mapping[str, Any],
    split: str,
    seed: int,
    task_seed: int,
    tasks: int,
    shot: int,
    q_query: int,
) -> tuple[Path, dict, Any]:
    """Create one audit-complete manifest and return its source dataset."""
    bundle = build_pipeline(cfg, seed=seed)
    dataset = bundle.adapt_val_dataset if split == "val" else bundle.adapt_test_dataset
    sampler = bundle.make_adaptation_sampler(
        k_shot=shot,
        q_query=q_query,
        mode=str(cfg.data.get("task_mode", "binary")),
        n_way=int(artifact["extra"]["n_way"]),
        seed=task_seed,
        disallow_support_query_overlap=bool(cfg.data.get("disallow_support_query_overlap", True)),
        disallow_internal_overlap=bool(cfg.data.get("disallow_internal_overlap", True)),
        split=split,
    )
    sampled = [sampler.sample_task() for _ in range(tasks)]
    source = (
        "adapt_val: held-out known evaluation partition plus held-out unknown validation partition"
        if split == "val" else
        "strict adapt_test: known loao.test plus held-out unknown test partition"
    )
    manifest_path = out / "manifests" / f"{split}_seed_{seed}_taskseed_{task_seed}.json"
    digest = write_task_manifest(
        manifest_path,
        sampled,
        protocol={
            "shot": shot, "q_query": q_query, "n_way": int(artifact["extra"]["n_way"]),
            "split": split, "data_split_source": source, "task_seed": task_seed,
            "sampler": "AdaptationTaskSampler sequential RNG stream",
        },
        base_checkpoint_path=str(artifact_path.resolve()),
        base_checkpoint_sha256=sha256_file(artifact_path),
        metadata={
            "dataset": str(cfg.data.name), "unknown_class": str(artifact["extra"]["unknown_class"]),
            "experiment_seed": seed, "train_fraction": float(cfg.data.get("train_fraction", 1.0)),
            "train_horizon": int(artifact["extra"]["meta_inner_steps"]),
            "adapt_scope": str(artifact["extra"]["adaptation_scope"]),
            "strict_adapt_test": True,
        },
        dataset=dataset,
    )
    return manifest_path, read_task_manifest(manifest_path), (dataset, digest)


def _evaluate(model: nn.Module, params: Mapping[str, torch.Tensor], task: Any, loss_fn: nn.Module) -> tuple[float, float, float]:
    with torch.no_grad():
        support_logits = functional_forward(model, params, task.support_x)
        query_logits = functional_forward(model, params, task.query_x)
        support_loss = float(loss_fn(support_logits, task.support_y).cpu())
        query_loss = float(loss_fn(query_logits, task.query_y).cpu())
    f1 = float("nan") if not torch.isfinite(query_logits).all() else float(
        compute_metrics(query_logits.cpu(), task.query_y.cpu(), num_classes=2).macro_f1
    )
    return support_loss, query_loss, f1


def _run_task(
    model: nn.Module,
    init_state: Mapping[str, torch.Tensor],
    task: Any,
    adapt_names: Sequence[str],
    method: str,
    lr: float | None,
    meta_opt: nn.Module | None,
    seed: int,
    phase: str,
    task_index: int,
) -> List[dict]:
    """One task, one fresh optimizer state; step 0 is the unadapted baseline."""
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    full = OrderedDict((name, value.detach().clone().to(task.support_x.device).requires_grad_(True))
                       for name, value in init_state.items())
    frozen = OrderedDict((name, value) for name, value in full.items() if name not in adapt_names)
    adaptable = OrderedDict((name, full[name]) for name in adapt_names)
    if method == "MetaOpt":
        assert meta_opt is not None
        optimizer = meta_opt
    else:
        optimizer = HandcraftedOptimizer(kind=method.lower(), lr=float(lr))
    state = optimizer.init_state(adaptable)
    support_loss, query_loss, f1 = _evaluate(model, {**frozen, **adaptable}, task, loss_fn)
    rows = [{
        "phase": phase, "seed": seed, "task_index": task_index, "method": method,
        "lr": "" if lr is None else lr, "step": 0, "support_loss": support_loss,
        "query_loss": query_loss, "query_macro_f1": f1, "gradient_norm": float("nan"),
        "parameter_update_norm": 0.0, "adam_exp_avg_norm": 0.0,
        "adam_exp_avg_sq_norm": 0.0, "none_grad_count": 0, "nan_grad_count": 0,
        "inf_grad_count": 0, "status": "ok",
    }]
    for step in range(1, STEPS + 1):
        loss = loss_fn(functional_forward(model, {**frozen, **adaptable}, task.support_x), task.support_y)
        try:
            grads = torch.autograd.grad(loss, list(adaptable.values()), create_graph=False,
                                        retain_graph=False, allow_unused=False)
        except Exception as exc:  # retain the failure in the raw trace
            rows.append({**rows[-1], "step": step, "support_loss": float(loss.detach().cpu()),
                         "query_loss": float("nan"), "query_macro_f1": float("nan"),
                         "gradient_norm": float("nan"), "parameter_update_norm": float("nan"),
                         "status": f"gradient_error:{type(exc).__name__}"})
            break
        grad_dict = OrderedDict(zip(adaptable.keys(), grads))
        updates, state = optimizer.step(grad_dict, state)
        adaptable = OrderedDict((name, adaptable[name] + updates[name]) for name in adaptable)
        support_loss, query_loss, f1 = _evaluate(model, {**frozen, **adaptable}, task, loss_fn)
        adam_m = adam_v = 0.0
        if method == "Adam":
            adam_m = _l2(layer[0] for layers in state.values() for layer in layers)
            adam_v = _l2(layer[1] for layers in state.values() for layer in layers)
        values = (support_loss, query_loss, f1, _l2(grads), _l2(updates.values()), adam_m, adam_v)
        rows.append({
            "phase": phase, "seed": seed, "task_index": task_index, "method": method,
            "lr": "" if lr is None else lr, "step": step, "support_loss": support_loss,
            "query_loss": query_loss, "query_macro_f1": f1, "gradient_norm": values[3],
            "parameter_update_norm": values[4], "adam_exp_avg_norm": adam_m,
            "adam_exp_avg_sq_norm": adam_v, "none_grad_count": 0,
            "nan_grad_count": sum(int(torch.isnan(g).sum().cpu()) for g in grads),
            "inf_grad_count": sum(int(torch.isinf(g).sum().cpu()) for g in grads),
            "status": "ok" if all(np.isfinite(v) for v in values) else "nonfinite",
        })
        if rows[-1]["status"] != "ok":
            break
        if step != STEPS:
            adaptable = OrderedDict((name, value.detach().clone().requires_grad_(True))
                                    for name, value in adaptable.items())
            state = optimizer.detach_state(state)
    return rows


def _outcomes(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    grouped: Dict[tuple, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["phase"], row["seed"], row["method"], row["lr"], row["task_index"]), []).append(row)
    outcomes = []
    for key, trace in grouped.items():
        ordered = sorted(trace, key=lambda item: int(item["step"]))
        hits = [int(item["step"]) for item in ordered if np.isfinite(_float(item["query_macro_f1"])) and _float(item["query_macro_f1"]) >= TARGET_F1]
        outcomes.append({"phase": key[0], "seed": key[1], "method": key[2], "lr": key[3],
                         "task_index": key[4], "reach_0_8_step": hits[0] if hits else STEPS,
                         "reached_0_8": bool(hits), "final_status": ordered[-1]["status"]})
    return outcomes


def _validation_selection(rows: Sequence[Mapping[str, Any]], method: str) -> List[dict]:
    candidates = sorted({str(row["lr"]) for row in rows if row["method"] == method})
    out = []
    outcome_rows = _outcomes(rows)
    for candidate in candidates:
        items = [row for row in rows if row["method"] == method and str(row["lr"]) == candidate and row["status"] == "ok"]
        lookup = {int(row["step"]): [] for row in items}
        for row in items:
            lookup[int(row["step"])].append(_float(row["query_macro_f1"]))
        early = [float(np.mean(lookup[step])) for step in (1, 2, 5) if lookup.get(step)]
        reaches = [row["reach_0_8_step"] for row in outcome_rows if row["method"] == method and str(row["lr"]) == candidate]
        out.append({
            "method": method, "lr": candidate,
            "early_f1_mean_steps_1_2_5": float(np.mean(early)) if len(early) == 3 else float("nan"),
            "step1_macro_f1": float(np.mean(lookup.get(1, [np.nan]))),
            "step2_macro_f1": float(np.mean(lookup.get(2, [np.nan]))),
            "step5_macro_f1": float(np.mean(lookup.get(5, [np.nan]))),
            "step20_macro_f1": float(np.mean(lookup.get(20, [np.nan]))),
            "mean_steps_to_0_8_capped_at_20": float(np.mean(reaches)) if reaches else float("nan"),
            "reach_rate_0_8": float(np.mean([row["reached_0_8"] for row in outcome_rows if row["method"] == method and str(row["lr"]) == candidate])),
            "non_ok_rows": sum(row["status"] != "ok" for row in rows if row["method"] == method and str(row["lr"]) == candidate),
        })
    return out


def _choose(selection: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    stable = [row for row in selection if int(row["non_ok_rows"]) == 0]
    pool = stable or list(selection)
    return max(pool, key=lambda row: (
        _float(row["early_f1_mean_steps_1_2_5"]),
        -_float(row["mean_steps_to_0_8_capped_at_20"]),
        _float(row["step20_macro_f1"]),
    ))


def _summary(rows: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]]) -> tuple[List[dict], List[dict]]:
    seed_rows: List[dict] = []
    aggregate: List[dict] = []
    for method in METHODS:
        for seed in sorted({int(row["seed"]) for row in rows}):
            selected = [row for row in rows if row["method"] == method and int(row["seed"]) == seed and row["status"] == "ok"]
            for step in CHECKPOINTS:
                vals = [_float(row["query_macro_f1"]) for row in selected if int(row["step"]) == step]
                seed_rows.append({"method": method, "seed": seed, "step": step, "n_tasks": len(vals),
                                  "macro_f1_mean": float(np.mean(vals)), "macro_f1_std_across_tasks": float(np.std(vals, ddof=0))})
            reach = [row["reach_0_8_step"] for row in outcomes if row["method"] == method and int(row["seed"]) == seed]
            seed_rows.append({"method": method, "seed": seed, "step": "reach_0_8", "n_tasks": len(reach),
                              "macro_f1_mean": float(np.mean(reach)), "macro_f1_std_across_tasks": float(np.std(reach, ddof=0))})
        for step in CHECKPOINTS:
            vals = [row["macro_f1_mean"] for row in seed_rows if row["method"] == method and row["step"] == step]
            aggregate.append({"method": method, "metric": "macro_f1", "step": step,
                              "mean_across_seeds": float(np.mean(vals)), "std_across_seeds": float(np.std(vals, ddof=1)), "n_seeds": len(vals)})
        reach = [row["macro_f1_mean"] for row in seed_rows if row["method"] == method and row["step"] == "reach_0_8"]
        aggregate.append({"method": method, "metric": "steps_to_0_8_capped_at_20", "step": "reach_0_8",
                          "mean_across_seeds": float(np.mean(reach)), "std_across_seeds": float(np.std(reach, ddof=1)), "n_seeds": len(reach)})
    return seed_rows, aggregate


def _bootstrap_ci(values: np.ndarray, seed: int = 20260710) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _paired_stats(rows: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]]) -> List[dict]:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    result = []
    for baseline in ("Adam", "SGD"):
        for metric, step in (("step5_macro_f1", 5), ("steps_to_0_8", None)):
            if step is not None:
                values = {(int(row["seed"]), int(row["task_index"]), row["method"]): _float(row["query_macro_f1"])
                          for row in rows if int(row["step"]) == step and row["status"] == "ok"}
            else:
                values = {(int(row["seed"]), int(row["task_index"]), row["method"]): _float(row["reach_0_8_step"])
                          for row in outcomes}
            keys = sorted({key[:2] for key in values if key[2] == baseline} & {key[:2] for key in values if key[2] == "MetaOpt"})
            # Positive means the baseline is better: higher F1, fewer steps.
            diffs = np.asarray([
                values[(seed, task, baseline)] - values[(seed, task, "MetaOpt")]
                if metric == "step5_macro_f1" else
                values[(seed, task, "MetaOpt")] - values[(seed, task, baseline)]
                for seed, task in keys
            ], dtype=float)
            p = 1.0
            if wilcoxon is not None and len(diffs) and not np.allclose(diffs, 0):
                try:
                    p = float(wilcoxon(diffs, alternative="two-sided", zero_method="wilcox").pvalue)
                except ValueError:
                    p = float("nan")
            lo, hi = _bootstrap_ci(diffs)
            result.append({"comparison": f"{baseline} vs MetaOpt", "metric": metric,
                           "effect_definition": "positive = baseline better", "n_paired_tasks": len(diffs),
                           "mean_effect": float(np.mean(diffs)) if len(diffs) else float("nan"),
                           "bootstrap_95ci_low": lo, "bootstrap_95ci_high": hi,
                           "wilcoxon_two_sided_p": p})
    return result


def _plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        means, stds = [], []
        for step in range(STEPS + 1):
            seed_means = []
            for seed in sorted({int(row["seed"]) for row in rows}):
                values = [_float(row["query_macro_f1"]) for row in rows
                          if row["method"] == method and int(row["seed"]) == seed and int(row["step"]) == step and row["status"] == "ok"]
                if values:
                    seed_means.append(float(np.mean(values)))
            means.append(float(np.mean(seed_means)))
            stds.append(float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0)
        x = np.arange(STEPS + 1)
        y = np.asarray(means)
        err = np.asarray(stds)
        ax.plot(x, y, label=method)
        ax.fill_between(x, y - err, y + err, alpha=.18)
    ax.set_xlabel("adaptation step")
    ax.set_ylabel("query macro-F1")
    ax.set_xlim(0, STEPS)
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _report(out: Path, manifests: Sequence[Mapping[str, Any]], locked: Mapping[str, Any], aggregate: Sequence[Mapping[str, Any]], stats: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Stage 2: validation tuning and fair test", "",
             "## Protocol", "",
             "- Scope: `bruteforce / fraction_1 / 3-shot / q=10/class / horizon=20 / head_only`.",
             "- All artifacts require `strict_adapt_test=true`; no full-scope result is used.",
             "- Validation manifests were generated from `adapt_val`; test manifests were generated only from strict `adapt_test`.",
             "- `manifest_isolation_audit.json` records raw-row-ID intersections; each validation/test pair is required to have an empty intersection.",
             "- Locked hyperparameters were written before test evaluation. Test metrics were not used for selection.", "",
             "## Locked hyperparameters", "",
             f"- Adam inner LR: `{locked['Adam']['lr']}` (validation-selected)",
             f"- SGD inner LR: `{locked['SGD']['lr']}` (validation-selected grid)",
             "- MetaOpt: artifact weights and inference-time algorithm unchanged; no separate inference LR/clip tuned.", "",
             "## Test mean ± std across the three seed means", "",
             "| Method | Step 1 | Step 2 | Step 5 | Step 10 | Step 20 | Steps to F1≥0.8 (capped at 20) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    by = {(row["method"], str(row["step"])): row for row in aggregate}
    for method in METHODS:
        cells = []
        for step in (1, 2, 5, 10, 20, "reach_0_8"):
            row = by[(method, str(step))]
            cells.append(f"{row['mean_across_seeds']:.4f} ± {row['std_across_seeds']:.4f}")
        lines.append("| " + method + " | " + " | ".join(cells) + " |")
    lines += ["", "## Paired task statistics", "",
              "Positive effects favor the named baseline; confidence intervals are nonparametric bootstrap intervals over paired tasks.", "",
              "| Comparison | Metric | Mean effect | 95% CI | Wilcoxon p | n |",
              "|---|---|---:|---:|---:|---:|"]
    for row in stats:
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_effect']:.4f} | [{row['bootstrap_95ci_low']:.4f}, {row['bootstrap_95ci_high']:.4f}] | {row['wilcoxon_two_sided_p']:.4g} | {row['n_paired_tasks']} |")
    adam_early = np.mean([by[("Adam", str(step))]["mean_across_seeds"] for step in (1, 2, 5)])
    meta_early = np.mean([by[("MetaOpt", str(step))]["mean_across_seeds"] for step in (1, 2, 5)])
    lines += ["", "## Conclusion", ""]
    if meta_early < adam_early:
        lines.append("在当前 head-only、严格独立测试协议下，MetaOpt 尚未优于经 validation 调优的 Adam baseline。")
    else:
        lines.append("在当前 head-only、严格独立测试协议下，MetaOpt 的早期适应不低于经 validation 调优的 Adam baseline；是否领先应结合表中的不确定性与配对统计解释。")
    lines += ["", "This stage does not compare old `full + 5-shot` results with current `head_only + 3-shot` results, so it cannot support a claim that the new method is better than the old experiment.",
              "", "## Next step", "",
              "If MetaOpt is tied or behind, diagnose MetaOpt training and update-rule behavior before expanding to the attack × shot matrix. If it leads robustly, extend the locked protocol to the remaining attacks and shot counts."]
    (out / "STAGE2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing stage-2 output: {out}")
    out.mkdir(parents=True)
    (out / "manifests").mkdir()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    adam_lrs = [float(value) for value in args.adam_lrs.split(",") if value.strip()]
    sgd_lrs = [float(value) for value in args.sgd_lrs.split(",") if value.strip()]
    artifact_root = Path(args.artifact_root)
    validation: List[tuple] = []
    test: List[tuple] = []
    audits = []
    for seed in seeds:
        artifact_path = artifact_root / f"seed_{seed}" / "horizon_20" / "meta_artifacts.pt"
        artifact = load_artifacts(str(artifact_path))
        cfg = Config(artifact["config"])
        if not bool(cfg.data.get("strict_adapt_test", False)) or str(artifact["extra"].get("adaptation_scope")) != "head_only":
            raise ValueError(f"artifact does not satisfy strict head_only protocol: {artifact_path}")
        set_seed(seed, bool(cfg.experiment.get("deterministic", True)))
        val_path, val_manifest, (val_dataset, val_digest) = _task_manifest(out, artifact_path, cfg, artifact, "val", seed, seed + 2002, args.tasks, args.shot, args.q_query)
        test_path, test_manifest, (test_dataset, test_digest) = _task_manifest(out, artifact_path, cfg, artifact, "test", seed, seed + 3002, args.tasks, args.shot, args.q_query)
        val_rows, test_rows = manifest_raw_row_ids(val_manifest), manifest_raw_row_ids(test_manifest)
        if val_rows & test_rows:
            raise RuntimeError(f"validation/test raw-row overlap for seed {seed}: {len(val_rows & test_rows)}")
        validation.append((seed, artifact_path, artifact, cfg, val_path, val_manifest, val_dataset, val_digest))
        test.append((seed, artifact_path, artifact, cfg, test_path, test_manifest, test_dataset, test_digest))
        audits.append({"seed": seed, "validation_manifest": str(val_path.resolve()), "test_manifest": str(test_path.resolve()),
                       "validation_manifest_sha256": val_digest, "test_manifest_sha256": test_digest,
                       "validation_raw_row_ids": len(val_rows), "test_raw_row_ids": len(test_rows),
                       "raw_row_id_intersection": len(val_rows & test_rows), "is_disjoint": True})
    _json(out / "manifest_isolation_audit.json", audits)
    _csv(out / "manifest_isolation_audit.csv", audits)
    _json(out / "run_protocol.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "seeds": seeds,
          "tasks_per_manifest": args.tasks, "shot": args.shot, "q_query": args.q_query, "steps": STEPS,
          "target_f1": TARGET_F1, "adam_candidates": adam_lrs, "sgd_candidates": sgd_lrs,
          "selection_rule": "maximize mean macro-F1 at steps 1/2/5; then fewer capped steps to 0.8; then step20; reject nonfinite when possible"})

    device = resolve_device(str(validation[0][3].device.get("prefer", "auto")))
    validation_rows: List[dict] = []
    for seed, artifact_path, artifact, cfg, _, manifest, dataset, _ in validation:
        set_seed(seed, bool(cfg.experiment.get("deterministic", True)))
        model = build_meta_model(cfg, artifact["extra"]["feature_dim"], artifact["extra"]["window_size"]).to(device)
        model.load_state_dict(artifact["meta_init_state"]); model.eval()
        init = OrderedDict((name, value.detach().clone()) for name, value in artifact["meta_init_state"].items())
        tasks = [item.to(device) for item in load_tasks_from_manifest(manifest, dataset)]
        names = list(artifact["extra"]["adapt_names"])
        for method, candidates in (("Adam", adam_lrs), ("SGD", sgd_lrs)):
            for lr in candidates:
                for index, task in enumerate(tasks):
                    validation_rows.extend(_run_task(model, init, task, names, method, lr, None, seed, "validation", index))
    _csv(out / "validation_raw_trajectories.csv", validation_rows)
    adam_selection = _validation_selection(validation_rows, "Adam")
    sgd_selection = _validation_selection(validation_rows, "SGD")
    selection = adam_selection + sgd_selection
    _csv(out / "validation_hyperparameter_selection.csv", selection)
    locked = {"Adam": dict(_choose(adam_selection)), "SGD": dict(_choose(sgd_selection)),
              "MetaOpt": {"algorithm": "artifact checkpoint; unchanged inference-time update rule", "tuned": False}}
    _json(out / "locked_hyperparameters.json", locked)

    test_rows: List[dict] = []
    for seed, artifact_path, artifact, cfg, _, manifest, dataset, _ in test:
        set_seed(seed, bool(cfg.experiment.get("deterministic", True)))
        model = build_meta_model(cfg, artifact["extra"]["feature_dim"], artifact["extra"]["window_size"]).to(device)
        model.load_state_dict(artifact["meta_init_state"]); model.eval()
        init = OrderedDict((name, value.detach().clone()) for name, value in artifact["meta_init_state"].items())
        tasks = [item.to(device) for item in load_tasks_from_manifest(manifest, dataset)]
        names = list(artifact["extra"]["adapt_names"])
        meta = build_meta_optimizer(cfg).to(device); meta.load_state_dict(artifact["meta_opt_state"]); meta.eval()
        for parameter in meta.parameters(): parameter.requires_grad_(False)
        for method in METHODS:
            lr = None if method == "MetaOpt" else float(locked[method]["lr"])
            for index, task in enumerate(tasks):
                test_rows.extend(_run_task(model, init, task, names, method, lr, meta if method == "MetaOpt" else None, seed, "test", index))
    _csv(out / "test_raw_trajectories.csv", test_rows)
    outcomes = _outcomes(test_rows)
    _csv(out / "test_task_outcomes.csv", outcomes)
    seed_summary, aggregate = _summary(test_rows, outcomes)
    _csv(out / "test_seed_summary.csv", seed_summary)
    _csv(out / "test_aggregate_summary.csv", aggregate)
    stats = _paired_stats(test_rows, outcomes)
    _csv(out / "paired_statistics.csv", stats)
    _plot(test_rows, out / "test_macro_f1_mean_std.png")
    _report(out, audits, locked, aggregate, stats)
    print(json.dumps({"out": str(out.resolve()), "locked": locked, "test_tasks": len(outcomes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
