"""Generate an explicit, hash-verifiable adaptation evaluation task manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build import load_artifacts  # noqa: E402
from src.data.pipeline import build_pipeline  # noqa: E402
from src.evaluation.task_manifest import (  # noqa: E402
    manifest_reuse_statistics,
    read_task_manifest,
    sha256_file,
    tensor_state_sha256,
    write_task_manifest,
)
from src.utils.config import Config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--task-seed", type=int, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = Path(args.out)
    if destination.exists() or destination.with_suffix(destination.suffix + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {destination}")
    artifact = load_artifacts(args.artifacts)
    cfg = Config(artifact["config"])
    if not bool(cfg.data.get("strict_adapt_test", False)):
        raise ValueError("manifest generation requires strict_adapt_test=true")
    if str(cfg.meta.get("adapt_scope", "")) != "head_only":
        raise ValueError("phase-1 manifest generation requires meta.adapt_scope=head_only")
    seed = int(cfg.experiment.get("seed", 42))
    set_seed(seed, bool(cfg.experiment.get("deterministic", True)))
    bundle = build_pipeline(cfg, seed=seed)
    q_query = int(cfg.data.q_query)
    sampler = bundle.make_adaptation_sampler(
        k_shot=args.shot,
        q_query=q_query,
        mode=str(cfg.data.get("task_mode", "binary")),
        n_way=int(artifact["extra"]["n_way"]),
        seed=args.task_seed,
        disallow_support_query_overlap=bool(
            cfg.data.get("disallow_support_query_overlap", True)
        ),
        disallow_internal_overlap=bool(cfg.data.get("disallow_internal_overlap", True)),
        split=args.split,
    )
    tasks = [sampler.sample_task() for _ in range(args.tasks)]
    artifact_hash = sha256_file(args.artifacts)
    split_dataset = (
        bundle.adapt_val_dataset if args.split == "val" else bundle.adapt_test_dataset
    )
    split_source = (
        "adapt_val (held-out known eval partition + held-out unknown validation partition)"
        if args.split == "val" else
        "strict adapt_test (known loao.test + held-out unknown test partition)"
    )
    protocol = {
        "shot": int(args.shot),
        "q_query": q_query,
        "n_way": int(artifact["extra"]["n_way"]),
        "split": args.split,
        "data_split_source": split_source,
        "task_seed": int(args.task_seed),
        "attack": str(artifact["extra"]["unknown_class"]),
        "sampler": "AdaptationTaskSampler sequential RNG stream",
    }
    metadata = {
        "dataset": str(cfg.data.name),
        "unknown_class": str(artifact["extra"]["unknown_class"]),
        "experiment_seed": seed,
        "train_fraction": float(cfg.data.get("train_fraction", 1.0)),
        "train_horizon": int(artifact["extra"]["meta_inner_steps"]),
        "adapt_scope": str(artifact["extra"]["adaptation_scope"]),
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
        base_checkpoint_path=str(Path(args.artifacts).resolve()),
        base_checkpoint_sha256=artifact_hash,
        metadata=metadata,
        dataset=split_dataset,
        base_initialization_sha256=tensor_state_sha256(
            artifact["meta_init_state"]
        ),
    )
    reuse = manifest_reuse_statistics(read_task_manifest(destination))
    config_path = destination.with_name(destination.stem + "_effective_config.json")
    config_path.write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(destination.resolve()),
                "manifest_sha256": digest,
                "artifact_sha256": artifact_hash,
                "task_count": len(tasks),
                "task_seed": args.task_seed,
                "reuse_statistics": reuse,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
