"""Small-scale Adam inner-loop LR diagnosis on an explicit task manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

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
    SCHEMA_VERSION,
    dataset_fingerprint,
    load_tasks_from_manifest,
    read_task_manifest,
    sha256_file,
    tensor_state_sha256,
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
    parser.add_argument(
        "--split",
        choices=["validation"],
        default="validation",
        help="This entry point is validation-only; test is deliberately unsupported.",
    )
    parser.add_argument("--expected-attack")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-shot", type=int)
    return parser.parse_args()


def _normalise_validation_split(value: Any, *, source: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{source} split is missing")
    normalised = str(value).strip().lower()
    if normalised in {"val", "validation"}:
        return "validation"
    if normalised == "test":
        raise ValueError(f"{source} declares test; Adam tuning accepts validation only")
    raise ValueError(f"{source} has unknown split={value!r}")


def _read_required_sidecar(manifest_path: str | Path) -> dict:
    source = Path(manifest_path)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"required manifest sidecar is missing: {sidecar}")
    tokens = sidecar.read_text(encoding="utf-8").split()
    if len(tokens) < 2:
        raise ValueError("manifest sidecar is malformed")
    fields: Dict[str, str] = {}
    for token in tokens[2:]:
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    required = {"split", "schema", "dataset_fingerprint"}
    missing = required - set(fields)
    if missing:
        raise ValueError(
            f"manifest sidecar missing required fields: {sorted(missing)}"
        )
    actual_manifest_hash = sha256_file(source)
    if tokens[0].lower() != actual_manifest_hash:
        raise ValueError(
            "manifest sidecar hash mismatch: "
            f"expected={tokens[0]} actual={actual_manifest_hash}"
        )
    if tokens[1] != source.name:
        raise ValueError(
            "manifest sidecar filename mismatch: "
            f"expected={source.name!r} declared={tokens[1]!r}"
        )
    return {
        "path": str(sidecar.resolve()),
        "sha256": sha256_file(sidecar),
        "declared_manifest_sha256": tokens[0].lower(),
        "declared_filename": tokens[1],
        **fields,
    }


def _validate_expected_identity(
    manifest: Mapping[str, Any],
    artifact: Mapping[str, Any],
    cfg: Config,
    *,
    expected_attack: str | None,
    expected_seed: int | None,
    expected_shot: int | None,
) -> dict:
    protocol = manifest.get("protocol")
    metadata = manifest.get("metadata")
    if not isinstance(protocol, Mapping):
        raise ValueError("manifest protocol is missing or invalid")
    if not isinstance(metadata, Mapping):
        raise ValueError("manifest metadata is missing or invalid")

    artifact_attack = str(artifact["extra"]["unknown_class"])
    config_attack = str(cfg.data.unknown_class)
    requested_attack = expected_attack or artifact_attack
    declared_attack = str(protocol.get("attack", "")).strip()
    metadata_attack = str(metadata.get("unknown_class", "")).strip()
    if not declared_attack or not metadata_attack:
        raise ValueError("manifest attack metadata is missing")
    if len({requested_attack, artifact_attack, config_attack,
            declared_attack, metadata_attack}) != 1:
        raise ValueError(
            "attack mismatch among request/artifact/config/manifest: "
            f"{requested_attack!r}/{artifact_attack!r}/{config_attack!r}/"
            f"{declared_attack!r}/{metadata_attack!r}"
        )

    config_seed = int(cfg.experiment.get("seed", 42))
    requested_seed = config_seed if expected_seed is None else int(expected_seed)
    if "experiment_seed" not in metadata:
        raise ValueError("manifest metadata.experiment_seed is missing")
    declared_seed = int(metadata["experiment_seed"])
    if requested_seed != config_seed or declared_seed != config_seed:
        raise ValueError(
            "seed mismatch among request/config/manifest: "
            f"{requested_seed}/{config_seed}/{declared_seed}"
        )

    config_shot = int(cfg.data.k_shot)
    requested_shot = config_shot if expected_shot is None else int(expected_shot)
    if "shot" not in protocol:
        raise ValueError("manifest protocol.shot is missing")
    if "task_seed" not in protocol:
        raise ValueError("manifest protocol.task_seed is missing")
    declared_shot = int(protocol["shot"])
    declared_task_seed = int(protocol["task_seed"])
    if requested_shot != config_shot or declared_shot != config_shot:
        raise ValueError(
            "shot mismatch among request/config/manifest: "
            f"{requested_shot}/{config_shot}/{declared_shot}"
        )
    for task_index, task in enumerate(manifest.get("tasks", [])):
        if _normalise_validation_split(
            task.get("split"), source=f"task {task_index}"
        ) != "validation":
            raise AssertionError("unreachable split normalisation result")
        if str(task.get("attack", "")) != declared_attack:
            raise ValueError(f"task {task_index} attack mismatch")
        if int(task.get("shot", -1)) != declared_shot:
            raise ValueError(f"task {task_index} shot mismatch")
        if int(task.get("task_seed", -1)) != declared_task_seed:
            raise ValueError(f"task {task_index} task seed mismatch")
    return {
        "attack": declared_attack,
        "seed": declared_seed,
        "shot": declared_shot,
    }


def validate_validation_inputs(
    *,
    artifact_path: str | Path,
    manifest_path: str | Path,
    artifact: Mapping[str, Any],
    cfg: Config,
    bundle: Any,
    requested_split: str = "validation",
    expected_attack: str | None = None,
    expected_seed: int | None = None,
    expected_shot: int | None = None,
) -> tuple[list, dict]:
    """Bind a verified validation manifest to adapt_val_dataset only."""
    effective_requested_split = _normalise_validation_split(
        requested_split, source="requested"
    )
    sidecar = _read_required_sidecar(manifest_path)
    sidecar_split = _normalise_validation_split(
        sidecar["split"], source="manifest sidecar"
    )
    manifest = read_task_manifest(manifest_path, verify_sha256=True)
    schema = int(manifest.get("schema_version", -1))
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Adam validation requires manifest schema {SCHEMA_VERSION}, got {schema}"
        )
    if int(sidecar["schema"]) != schema:
        raise ValueError("manifest and sidecar schema mismatch")
    declared_split = _normalise_validation_split(
        manifest.get("protocol", {}).get("split"), source="manifest"
    )
    if len({effective_requested_split, sidecar_split, declared_split}) != 1:
        raise ValueError(
            "requested/manifest/sidecar split mismatch: "
            f"{effective_requested_split}/{declared_split}/{sidecar_split}"
        )

    identity = _validate_expected_identity(
        manifest,
        artifact,
        cfg,
        expected_attack=expected_attack,
        expected_seed=expected_seed,
        expected_shot=expected_shot,
    )
    artifact_hash = sha256_file(artifact_path)
    if artifact_hash != str(manifest.get("base_checkpoint_sha256", "")).lower():
        raise ValueError("artifact/checkpoint SHA256 does not match task manifest")
    theta0_hash = tensor_state_sha256(artifact["meta_init_state"])
    if theta0_hash != str(manifest.get("base_initialization_sha256", "")).lower():
        raise ValueError("theta0 SHA256 does not match task manifest")

    metadata = manifest["metadata"]
    if str(metadata.get("dataset_role", "")) != "adapt_val_dataset":
        raise ValueError(
            "validation manifest must declare dataset_role=adapt_val_dataset"
        )
    declared_fingerprint = str(metadata.get("dataset_fingerprint", "")).lower()
    if not declared_fingerprint:
        raise ValueError("manifest metadata.dataset_fingerprint is missing")
    if str(sidecar["dataset_fingerprint"]).lower() != declared_fingerprint:
        raise ValueError("manifest and sidecar dataset fingerprint mismatch")

    # This is deliberately the only adaptation dataset property accessed here.
    dataset = bundle.adapt_val_dataset
    effective_fingerprint = dataset_fingerprint(dataset)
    if effective_fingerprint != declared_fingerprint:
        raise ValueError(
            "validation dataset fingerprint mismatch: "
            f"expected={declared_fingerprint} actual={effective_fingerprint}"
        )
    tasks = load_tasks_from_manifest(manifest, dataset)
    if not tasks:
        raise ValueError("validation manifest contains no tasks")
    return tasks, {
        "requested_split": effective_requested_split,
        "manifest_declared_split": declared_split,
        "sidecar_declared_split": sidecar_split,
        "effective_split": "validation",
        "effective_dataset": "adapt_val_dataset",
        "effective_dataset_fingerprint": effective_fingerprint,
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_schema_version": schema,
        "sidecar_path": sidecar["path"],
        "sidecar_sha256": sidecar["sha256"],
        "sidecar_declared_manifest_sha256": sidecar[
            "declared_manifest_sha256"
        ],
        "artifact_path": str(Path(artifact_path).resolve()),
        "artifact_sha256": artifact_hash,
        "checkpoint_sha256": artifact_hash,
        "theta0_sha256": theta0_hash,
        "attack": identity["attack"],
        "seed": identity["seed"],
        "shot": identity["shot"],
        "task_count": len(tasks),
        "test_dataset_accessed": False,
        "metaopt_ran": False,
    }


def source_state_receipt(repo_root: Path) -> dict:
    tracked_sources = [
        "scripts/generate_eval_task_manifest.py",
        "scripts/run_experiments.py",
        "scripts/run_metaopt_minimal_experiments.ps1",
        "scripts/tune_adam_manifest.py",
        "src/data/pipeline.py",
        "src/data/task_sampler.py",
        "src/evaluation/task_manifest.py",
        "src/meta_optimizer/handcrafted.py",
    ]
    file_hashes = {
        path: sha256_file(repo_root / path)
        for path in tracked_sources
    }
    digest = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "git_commit": commit,
        "dirty_worktree": bool(status),
        "git_status_short": status,
        "source_state_sha256": digest,
        "source_file_sha256": file_hashes,
    }


def build_adam_validation_receipt(
    *,
    validation_receipt: Mapping[str, Any],
    source_state: Mapping[str, Any],
    selection: Mapping[str, Any],
    raw_rows: List[dict],
    lrs: List[float],
    steps: int,
    device: str,
    started_at: datetime,
    finished_at: datetime,
    elapsed_seconds: float,
    argv: List[str],
) -> dict:
    """Build the machine-readable receipt without performing any evaluation."""
    return {
        **dict(validation_receipt),
        **dict(source_state),
        "receipt_schema_version": 2,
        "candidate_lrs": list(lrs),
        "selected_lr": selection["selected_lr"],
        "boundary_hit": bool(selection["boundary_hit"]),
        "selection_rule": selection["selection_rule"],
        "selected_candidate": selection["selected_candidate"],
        "steps": int(steps),
        "checkpoints": CHECKPOINTS,
        "nonfinite_count": int(sum(row["status"] != "ok" for row in raw_rows)),
        "strict_adapt_test": True,
        "adapt_scope": "head_only",
        "device": str(device),
        "validation_started_at_utc": started_at.isoformat(),
        "validation_finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": float(elapsed_seconds),
        "test_dataset_accessed": False,
        "metaopt_ran": False,
        "command_line_arguments": list(argv),
    }


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


def summarize(
    raw_rows: List[dict],
    lrs: List[float],
    *,
    steps: int,
) -> Tuple[List[dict], dict]:
    summary_rows: List[dict] = []
    selection_rows = []
    for lr in lrs:
        lr_rows = [row for row in raw_rows if row["lr"] == lr]
        nonfinite_count = sum(row["status"] != "ok" for row in lr_rows)
        by_step = {}
        for step in range(steps + 1):
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
        final_f1 = by_step.get(steps, {}).get(
            "query_macro_f1_mean", float("nan")
        )
        ordered_steps = [
            step for step in range(steps + 1)
            if step in by_step
            and np.isfinite(by_step[step]["query_macro_f1_mean"])
        ]
        if len(ordered_steps) == steps + 1 and steps > 0:
            curve_values = np.asarray(
                [by_step[step]["query_macro_f1_mean"] for step in ordered_steps],
                dtype=float,
            )
            curve_axis = np.asarray(ordered_steps, dtype=float)
            curve_auc = float(
                np.sum(
                    0.5 * (curve_values[:-1] + curve_values[1:])
                    * np.diff(curve_axis)
                ) / steps
            )
        else:
            curve_auc = float("nan")
        selectable_steps = [step for step in ordered_steps if step > 0]
        selected_stop_step = (
            max(
                selectable_steps,
                key=lambda step: (
                    by_step[step]["query_macro_f1_mean"],
                    -step,
                ),
            )
            if selectable_steps else -1
        )
        selected_stop_f1 = (
            by_step[selected_stop_step]["query_macro_f1_mean"]
            if selected_stop_step >= 0 else float("nan")
        )
        all_f1 = [
            row["query_macro_f1_mean"]
            for row in summary_rows
            if row["lr"] == lr and np.isfinite(row["query_macro_f1_mean"])
        ]
        post_peak_drop = max(all_f1) - all_f1[-1] if all_f1 else float("inf")
        selection_rows.append({
            "lr": lr,
            "early_f1_mean_steps_1_2_5": float(early.mean()) if len(early) == 3 else float("nan"),
            "step1_macro_f1": by_step.get(1, {}).get(
                "query_macro_f1_mean", float("nan")
            ),
            "curve_auc_step0_to_final": curve_auc,
            "validation_selected_step": selected_stop_step,
            "validation_selected_macro_f1": selected_stop_f1,
            "final_step": steps,
            "final_macro_f1": final_f1,
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
            row["final_macro_f1"],
            -row["post_peak_drop"],
        ),
    )
    return summary_rows, {
        "selection_rule": (
            "maximize mean macro-F1 at steps 1/2/5; tie-break by step 20, "
            "then lower post-peak drop; exclude nonfinite configurations when possible"
        ),
        "diagnostic_only_not_for_test_claims": True,
        "selected_lr": recommended["lr"],
        "boundary_hit": (
            recommended["lr"] == min(lrs) or recommended["lr"] == max(lrs)
        ),
        "selected_candidate": recommended,
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
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    lrs = [float(value) for value in args.lrs.split(",") if value.strip()]
    if not lrs or len(lrs) != len(set(lrs)):
        raise ValueError("candidate LR list must be non-empty and contain no duplicates")
    if any(not np.isfinite(value) or value <= 0 for value in lrs):
        raise ValueError("every candidate LR must be positive and finite")
    if args.steps != 20:
        raise ValueError("phase-1 protocol requires exactly 20 adaptation steps")
    artifact = load_artifacts(args.artifacts)
    cfg = Config(artifact["config"])
    if not bool(cfg.data.get("strict_adapt_test", False)):
        raise ValueError("Adam tuning requires strict_adapt_test=true")
    if str(artifact["extra"].get("adaptation_scope")) != "head_only":
        raise ValueError("Adam tuning requires the current head_only artifact")
    seed = int(cfg.experiment.get("seed", 42))
    set_seed(seed, bool(cfg.experiment.get("deterministic", True)))
    bundle = build_pipeline(
        cfg,
        seed=seed,
        adaptation_dataset_role="validation",
    )
    tasks, validation_receipt = validate_validation_inputs(
        artifact_path=args.artifacts,
        manifest_path=args.task_manifest,
        artifact=artifact,
        cfg=cfg,
        bundle=bundle,
        requested_split=args.split,
        expected_attack=args.expected_attack,
        expected_seed=args.expected_seed,
        expected_shot=args.expected_shot,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    device = resolve_device(str(cfg.device.get("prefer", "auto")))
    tasks = [task.to(device) for task in tasks]
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
    summary_rows, selection = summarize(raw_rows, lrs, steps=args.steps)
    write_csv(out_dir / "adam_lr_raw.csv", raw_rows)
    write_csv(out_dir / "adam_lr_summary.csv", summary_rows)
    (out_dir / "adam_lr_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    finished_at = datetime.now(timezone.utc)
    repo_root = Path(__file__).resolve().parents[1]
    source_state = source_state_receipt(repo_root)
    run_config = build_adam_validation_receipt(
        validation_receipt=validation_receipt,
        source_state=source_state,
        selection=selection,
        raw_rows=raw_rows,
        lrs=lrs,
        steps=args.steps,
        device=str(device),
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=time.perf_counter() - started_clock,
        argv=list(sys.argv),
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "adam_validation_receipt.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
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
    print(json.dumps(run_config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
