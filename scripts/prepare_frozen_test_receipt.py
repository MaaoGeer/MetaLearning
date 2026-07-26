"""Freeze validation-selected state and test manifests without running test."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.task_manifest import sha256_file  # noqa: E402
from src.utils.experiment_protocol import (  # noqa: E402
    ATTACKS,
    SEEDS,
    VARIANT_OVERRIDES,
    formal_run_relative,
    test_manifest_relative,
    validation_manifest_relative,
)
from src.utils.provenance import (  # noqa: E402
    assert_execution_gate,
    canonical_sha256,
    runtime_environment,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-source-state")
    parser.add_argument(
        "--selected-variant",
        choices=list(VARIANT_OVERRIDES),
        required=True,
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    state = assert_execution_gate(
        args.expected_commit,
        expected_source_state_sha256=args.expected_source_state,
    )
    root = Path(args.root)
    gate_path = root / "validation_gate_receipt.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    if not gate.get("validation_gate_passed"):
        raise ValueError("validation gate did not pass")
    if gate.get("selected_variant") != args.selected_variant:
        raise ValueError("selected variant differs from validation gate")
    variants = tuple(dict.fromkeys(("E0_final_only", args.selected_variant)))
    cells = []
    for attack in ATTACKS:
        for seed in SEEDS:
            val_manifest = root / validation_manifest_relative(attack, seed)
            test_manifest = root / test_manifest_relative(attack, seed)
            test_generation = test_manifest.with_name(
                test_manifest.stem + "_generation_receipt.json"
            )
            generation = json.loads(
                test_generation.read_text(encoding="utf-8-sig")
            )
            required_generation = {
                "effective_split": "test",
                "dataset_role": "test",
                "effective_dataset": "adapt_test_dataset",
                "validation_dataset_constructed": False,
                "validation_dataset_accessed": False,
                "test_dataset_constructed": True,
                "test_dataset_accessed": True,
                "metaopt_training_ran": False,
            }
            for key, expected in required_generation.items():
                if generation.get(key) != expected:
                    raise ValueError(
                        f"test manifest generation receipt {key} mismatch: "
                        f"{test_generation}"
                    )
            test_hash = sha256_file(test_manifest)
            for variant in variants:
                run_dir = root / formal_run_relative(variant, attack, seed)
                selection_path = (
                    run_dir / "validation" / "validation_selection.json"
                )
                selection = json.loads(
                    selection_path.read_text(encoding="utf-8-sig")
                )
                if selection.get("test_metrics_used") is not False:
                    raise ValueError("validation selection used test metrics")
                frozen_selection = copy.deepcopy(selection)
                frozen_selection["frozen"] = True
                frozen_selection["selection_source"] = "validation_only"
                frozen_selection["test_metrics_used"] = False
                for experiment in frozen_selection["experiments"].values():
                    experiment["test_manifest_sha256"] = test_hash
                frozen_path = (
                    root / "frozen_test" / "selections" / variant / attack
                    / f"seed_{seed}" / "validation_selection_frozen.json"
                )
                if frozen_path.exists():
                    raise FileExistsError(
                        f"refusing to overwrite frozen selection: {frozen_path}"
                    )
                frozen_path.parent.mkdir(parents=True, exist_ok=True)
                frozen_path.write_text(
                    json.dumps(
                        frozen_selection, indent=2, ensure_ascii=False
                    ) + "\n",
                    encoding="utf-8",
                )
                effective_config = run_dir / "effective_config.json"
                artifact = run_dir / "meta_artifacts.pt"
                best = run_dir / "checkpoints" / "best.pt"
                provenance_path = run_dir / "provenance.json"
                provenance = json.loads(
                    provenance_path.read_text(encoding="utf-8-sig")
                )
                cells.append({
                    "variant": variant,
                    "attack": attack,
                    "seed": seed,
                    "effective_config_path": str(effective_config.resolve()),
                    "effective_config_sha256": sha256_file(effective_config),
                    "artifact_path": str(artifact.resolve()),
                    "artifact_sha256": sha256_file(artifact),
                    "checkpoint_path": str(best.resolve()),
                    "checkpoint_sha256": sha256_file(best),
                    "theta0_sha256": selection["meta_init_state_sha256"],
                    "validation_manifest_path": str(val_manifest.resolve()),
                    "validation_manifest_sha256": sha256_file(val_manifest),
                    "test_manifest_path": str(test_manifest.resolve()),
                    "test_manifest_sha256": test_hash,
                    "dataset_cache_key": provenance.get("cache_key"),
                    "dataset_fingerprint": generation[
                        "effective_dataset_fingerprint"
                    ],
                    "selection_path": str(frozen_path.resolve()),
                    "selection_sha256": sha256_file(frozen_path),
                    "selected_learning_rates": next(iter(
                        frozen_selection["experiments"].values()
                    ))["selected_learning_rates"],
                    "selected_stop_steps": {
                        name: details["selected_stop_step"]
                        for name, details in next(iter(
                            frozen_selection["experiments"].values()
                        ))["methods"].items()
                    },
                })
    receipt = {
        "receipt_schema_version": 1,
        "selected_variant": args.selected_variant,
        "validation_gate_passed": True,
        "git_commit": state["git_commit"],
        "parent_commit": state["parent_commit"],
        "worktree_clean": state["worktree_clean"],
        "source_state_sha256": state["source_state_sha256"],
        "tracked_source_file_sha256": state[
            "tracked_source_file_sha256"
        ],
        "validation_gate_path": str(gate_path.resolve()),
        "validation_gate_sha256": sha256_file(gate_path),
        "curve_auc_definition": (
            "trapezoidal area of the per-task Macro-F1 trajectory over "
            "steps 0..20, divided by 20, then averaged across manifest tasks"
        ),
        "metric_schema_version": 2,
        "prediction_artifact_schema_version": 1,
        "environment": runtime_environment(),
        "cells": cells,
        "cells_sha256": canonical_sha256({"cells": cells}),
        "test_started": False,
        "test_metrics_accessed": False,
        "metaopt_training_ran": False,
    }
    destination = root / "frozen_test" / "frozen_experiment_receipt.json"
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen experiment receipt: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return receipt


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
