"""Generate an explicit, hash-verifiable adaptation evaluation task manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build import build_meta_model, load_artifacts, task_n_way  # noqa: E402
from src.data.pipeline import build_pipeline  # noqa: E402
from src.evaluation.task_manifest import (  # noqa: E402
    dataset_fingerprint,
    manifest_reuse_statistics,
    read_task_manifest,
    sha256_file,
    tensor_state_sha256,
    write_task_manifest,
)
from src.utils.config import Config, load_config  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    assert_execution_gate,
    write_completion_receipt,
)
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifacts")
    source.add_argument("--config")
    parser.add_argument("--dataset")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument(
        "--base-checkpoint-path",
        help=(
            "Future E0 artifact path for config-derived validation manifests. "
            "The manifest binds theta0; the later artifact must match it."
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--task-seed", type=int, required=True)
    parser.add_argument(
        "--split",
        choices=["val", "validation", "test"],
        required=True,
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-source-state")
    return parser.parse_args()


def _normalise_split(value: str) -> tuple[str, str, str]:
    if value in {"val", "validation"}:
        return "validation", "val", "adapt_val_dataset"
    if value == "test":
        return "test", "test", "adapt_test_dataset"
    raise ValueError(f"unsupported manifest split={value!r}")


def _load_manifest_context(args: argparse.Namespace, role: str):
    if args.artifacts:
        artifact = load_artifacts(args.artifacts)
        cfg = Config(artifact["config"])
        if args.override:
            cfg = cfg.apply_overrides(args.override)
        artifact_path = str(Path(args.artifacts).resolve())
        artifact_hash = sha256_file(args.artifacts)
        theta0 = artifact["meta_init_state"]
        extra = artifact["extra"]
    else:
        if not args.dataset:
            raise ValueError("--config mode requires --dataset")
        if not args.base_checkpoint_path:
            raise ValueError(
                "--config mode requires --base-checkpoint-path"
            )
        cfg = load_config(args.config)
        cfg = cfg.merge(load_config(args.dataset).to_dict())
        if args.override:
            cfg = cfg.apply_overrides(args.override)
        seed = int(cfg.experiment.get("seed", 42))
        set_seed(seed, bool(cfg.experiment.get("deterministic", True)))
        basis_bundle = build_pipeline(
            cfg, seed=seed, adaptation_dataset_role=role
        )
        model = build_meta_model(
            cfg, basis_bundle.feature_dim, basis_bundle.window_size
        )
        theta0 = {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        }
        extra = {
            "feature_dim": basis_bundle.feature_dim,
            "window_size": basis_bundle.window_size,
            "n_way": task_n_way(cfg),
            "unknown_class": str(cfg.data.unknown_class),
            "adaptation_scope": str(cfg.meta.get("adapt_scope", "")),
            "meta_inner_steps": int(cfg.meta.inner_steps),
        }
        artifact_path = str(Path(args.base_checkpoint_path).resolve())
        artifact_hash = (
            sha256_file(artifact_path)
            if Path(artifact_path).is_file() else ""
        )
        return cfg, basis_bundle, theta0, extra, artifact_path, artifact_hash

    seed = int(cfg.experiment.get("seed", 42))
    set_seed(seed, bool(cfg.experiment.get("deterministic", True)))
    bundle = build_pipeline(
        cfg, seed=seed, adaptation_dataset_role=role
    )
    return cfg, bundle, theta0, extra, artifact_path, artifact_hash


def run(args: argparse.Namespace) -> dict:
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    gate = assert_execution_gate(
        args.expected_commit,
        expected_source_state_sha256=args.expected_source_state,
    )
    destination = Path(args.out)
    if destination.exists() or destination.with_suffix(destination.suffix + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {destination}")
    requested_split, protocol_split, effective_dataset = _normalise_split(
        args.split
    )
    role = requested_split
    cfg, bundle, theta0, extra, artifact_path, artifact_hash = (
        _load_manifest_context(args, role)
    )
    if not bool(cfg.data.get("strict_adapt_test", False)):
        raise ValueError("manifest generation requires strict_adapt_test=true")
    if str(cfg.meta.get("adapt_scope", "")) != "head_only":
        raise ValueError("phase-1 manifest generation requires meta.adapt_scope=head_only")
    seed = int(cfg.experiment.get("seed", 42))
    q_query = int(cfg.data.q_query)
    sampler = bundle.make_adaptation_sampler(
        k_shot=args.shot,
        q_query=q_query,
        mode=str(cfg.data.get("task_mode", "binary")),
        n_way=int(extra["n_way"]),
        seed=args.task_seed,
        disallow_support_query_overlap=bool(
            cfg.data.get("disallow_support_query_overlap", True)
        ),
        disallow_internal_overlap=bool(cfg.data.get("disallow_internal_overlap", True)),
        split=protocol_split,
    )
    tasks = [sampler.sample_task() for _ in range(args.tasks)]
    split_dataset = (
        bundle.adapt_val_dataset
        if requested_split == "validation"
        else bundle.adapt_test_dataset
    )
    split_source = (
        "adapt_val (held-out known eval partition + held-out unknown validation partition)"
        if requested_split == "validation" else
        "strict adapt_test (known loao.test + held-out unknown test partition)"
    )
    protocol = {
        "shot": int(args.shot),
        "q_query": q_query,
        "n_way": int(extra["n_way"]),
        "split": protocol_split,
        "data_split_source": split_source,
        "task_seed": int(args.task_seed),
        "attack": str(extra["unknown_class"]),
        "sampler": "AdaptationTaskSampler sequential RNG stream",
    }
    metadata = {
        "dataset": str(cfg.data.name),
        "unknown_class": str(extra["unknown_class"]),
        "experiment_seed": seed,
        "train_fraction": float(cfg.data.get("train_fraction", 1.0)),
        "train_horizon": int(extra["meta_inner_steps"]),
        "adapt_scope": str(extra["adaptation_scope"]),
        "strict_adapt_test": True,
        "disallow_support_query_overlap": bool(
            cfg.data.get("disallow_support_query_overlap", True)
        ),
        "disallow_internal_overlap": bool(cfg.data.get("disallow_internal_overlap", True)),
    }
    digest = write_task_manifest(
        destination,
        tasks,
        protocol=protocol,
        base_checkpoint_path=artifact_path,
        base_checkpoint_sha256=artifact_hash,
        metadata=metadata,
        dataset=split_dataset,
        base_initialization_sha256=tensor_state_sha256(theta0),
    )
    reuse = manifest_reuse_statistics(read_task_manifest(destination))
    config_path = destination.with_name(destination.stem + "_effective_config.json")
    config_path.write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    receipt_path = destination.with_name(
        destination.stem + "_generation_receipt.json"
    )
    details = {
        "requested_split": requested_split,
        "effective_split": requested_split,
        "dataset_role": role,
        "effective_dataset": effective_dataset,
        "effective_dataset_fingerprint": dataset_fingerprint(split_dataset),
        "effective_config_path": str(config_path.resolve()),
        "effective_config_sha256": sha256_file(config_path),
        "manifest_path": str(destination.resolve()),
        "manifest_sha256": digest,
        "sidecar_path": str(sidecar.resolve()),
        "sidecar_sha256": sha256_file(sidecar),
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_hash or None,
        "theta0_sha256": tensor_state_sha256(theta0),
        "task_count": len(tasks),
        "task_seed": args.task_seed,
        "reuse_statistics": reuse,
        "validation_dataset_constructed": bool(
            bundle.adaptation_validation_constructed
        ),
        "test_dataset_constructed": bool(
            bundle.adaptation_test_constructed
        ),
        "validation_dataset_accessed": requested_split == "validation",
        "test_dataset_accessed": requested_split == "test",
        "metaopt_training_ran": False,
        "command_line": list(sys.argv),
    }
    finished_at = datetime.now(timezone.utc)
    receipt = write_completion_receipt(
        receipt_path,
        stage=f"generate_{requested_split}_manifest",
        status="success",
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        duration_seconds=time.perf_counter() - started_clock,
        exit_code=0,
        output_paths=[destination, sidecar, config_path, receipt_path],
        required_artifacts=[destination, sidecar, config_path],
        execution_state=gate,
        details=details,
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return receipt


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    try:
        run(args)
    except KeyboardInterrupt:
        write_completion_receipt(
            Path(args.out).with_name(
                Path(args.out).stem + "_generation_interrupted.json"
            ),
            stage="generate_manifest",
            status="interrupted",
            started_at_utc=started_at.isoformat(),
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            duration_seconds=time.perf_counter() - started_clock,
            exit_code=130,
            output_paths=[args.out],
            details={"command_line": list(sys.argv)},
        )
        raise
    except Exception:
        write_completion_receipt(
            Path(args.out).with_name(
                Path(args.out).stem + "_generation_failed.json"
            ),
            stage="generate_manifest",
            status="failed",
            started_at_utc=started_at.isoformat(),
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            duration_seconds=time.perf_counter() - started_clock,
            exit_code=1,
            output_paths=[args.out],
            details={"command_line": list(sys.argv)},
        )
        raise


if __name__ == "__main__":
    main()
