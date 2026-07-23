"""Verify MetaOpt artifact/checkpoint lineage and adaptation scope."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build import (  # noqa: E402
    build_meta_model,
    load_artifacts,
    resolve_adapt_names,
)
from src.evaluation.task_manifest import (  # noqa: E402
    sha256_file,
    tensor_state_sha256,
)
from src.trainer.callbacks import CheckpointManager  # noqa: E402
from src.utils.config import Config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--best", required=True)
    parser.add_argument("--last", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _max_abs_diff(left: dict, right: dict) -> float:
    if set(left) != set(right):
        return float("inf")
    return max(
        (
            float((left[name].detach().cpu() - right[name].detach().cpu()).abs().max())
            for name in left
        ),
        default=0.0,
    )


def main() -> None:
    args = parse_args()
    artifact = load_artifacts(args.artifacts)
    best = CheckpointManager.load(args.best, map_location="cpu")
    last = CheckpointManager.load(args.last, map_location="cpu")
    cfg = Config(artifact["config"])
    extra = artifact["extra"]
    model = build_meta_model(
        cfg, int(extra["feature_dim"]), int(extra["window_size"])
    )
    model.load_state_dict(artifact["meta_init_state"])
    adapt_names = resolve_adapt_names(model, cfg)
    lookup = dict(model.named_parameters())
    parameter_count = sum(lookup[name].numel() for name in adapt_names)
    artifact_state = artifact["meta_opt_state"]
    best_state = best["meta_optimizer"]
    last_state = last["meta_optimizer"]
    best_diff = _max_abs_diff(artifact_state, best_state)
    receipt = {
        "schema_version": 1,
        "paths": {
            "meta_artifacts": str(Path(args.artifacts).resolve()),
            "best": str(Path(args.best).resolve()),
            "last": str(Path(args.last).resolve()),
        },
        "file_sha256": {
            "meta_artifacts": sha256_file(args.artifacts),
            "best": sha256_file(args.best),
            "last": sha256_file(args.last),
        },
        "tensor_state_sha256": {
            "artifact_meta_optimizer": tensor_state_sha256(artifact_state),
            "best_meta_optimizer": tensor_state_sha256(best_state),
            "last_meta_optimizer": tensor_state_sha256(last_state),
            "artifact_meta_init": tensor_state_sha256(
                artifact["meta_init_state"]
            ),
        },
        "artifact_vs_best_max_abs_diff": best_diff,
        "artifact_matches_best": bool(best_diff == 0.0),
        "artifact_vs_last_max_abs_diff": _max_abs_diff(
            artifact_state, last_state
        ),
        "scope": {
            "effective_config": str(cfg.meta.get("adapt_scope", "")),
            "artifact_extra": str(extra.get("adaptation_scope", "")),
            "adapt_parameter_names": adapt_names,
            "tensor_count": len(adapt_names),
            "parameter_count": int(parameter_count),
        },
    }
    receipt["scope"]["matches_head_only_2_tensors_66_params"] = bool(
        receipt["scope"]["effective_config"] == "head_only"
        and receipt["scope"]["artifact_extra"] == "head_only"
        and len(adapt_names) == 2
        and parameter_count == 66
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    if not receipt["artifact_matches_best"]:
        raise SystemExit("artifact MetaOpt state does not match best checkpoint")
    if not receipt["scope"]["matches_head_only_2_tensors_66_params"]:
        raise SystemExit("artifact adaptation scope is not head_only/2 tensors/66 params")


if __name__ == "__main__":
    main()

