"""Pre-registered execution plan for the MetaOpt remediation rerun.

This module is intentionally data-free.  It is the single source of truth used
by the PowerShell runner and CPU tests to expand protocol stages without
constructing a dataset, loading a checkpoint, or touching CUDA.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional


ATTACKS = ("ddos", "botnet")
SEEDS = (42, 62)
SHOT = 5
HORIZON = 20
VALIDATION_TASKS = 30
TEST_TASKS = 100

KNOWN_CLASSES = {
    "ddos": [
        "botnet", "bruteforce", "dos", "heartbleed", "infiltration",
        "portscan", "webattack",
    ],
    "botnet": [
        "bruteforce", "ddos", "dos", "heartbleed", "infiltration",
        "portscan", "webattack",
    ],
}

COMMON_OVERRIDES = (
    "data.k_shot=5",
    "meta.inner_steps=20",
    "train.early_stopping.enabled=true",
    "train.early_stopping.patience=4",
    "train.validation.checkpoints=[1,2,5,10,20]",
    "adaptation_speed.max_steps=20",
    "adaptation_speed.checkpoints=[0,1,2,5,10,20]",
)

VARIANT_OVERRIDES: Mapping[str, tuple[str, ...]] = {
    "E0_final_only": (
        "meta.query_objective.mode=final_only",
        "meta.random_horizon.enabled=false",
        "meta.mixed_shot.enabled=false",
        "meta_optimizer.update_mode=learned_delta",
    ),
    "E1_multistep": (
        "meta.query_objective.mode=multi_step",
        "meta.query_objective.supervised_steps=[1,2,5,10,20]",
        "meta.query_objective.weighting=early_heavy",
        "meta.query_objective.early_heavy_power=0.5",
        "meta.random_horizon.enabled=false",
        "meta.mixed_shot.enabled=false",
        "meta_optimizer.update_mode=learned_delta",
        "train.validation.selection_metric=curve_auc",
    ),
    "E2_multistep_random_horizon": (
        "meta.query_objective.mode=multi_step",
        "meta.query_objective.supervised_steps=[1,2,5,10,20]",
        "meta.query_objective.weighting=early_heavy",
        "meta.query_objective.early_heavy_power=0.5",
        "meta.random_horizon.enabled=true",
        "meta.random_horizon.min_steps=1",
        "meta.random_horizon.max_steps=20",
        "meta.mixed_shot.enabled=false",
        "meta_optimizer.update_mode=learned_delta",
        "train.validation.selection_metric=curve_auc",
    ),
    "E3_sgd_residual": (
        "meta.query_objective.mode=multi_step",
        "meta.query_objective.supervised_steps=[1,2,5,10,20]",
        "meta.query_objective.weighting=early_heavy",
        "meta.query_objective.early_heavy_power=0.5",
        "meta.random_horizon.enabled=false",
        "meta.mixed_shot.enabled=false",
        "meta_optimizer.update_mode=sgd_residual",
        "meta_optimizer.anchor_lr=0.1",
        "meta_optimizer.learnable_anchor_lr=false",
        "meta_optimizer.residual_enabled=true",
        "meta_optimizer.residual_zero_init=true",
        "meta_optimizer.gate_init=0.01",
        "meta_optimizer.learnable_gate=true",
        "meta_optimizer.trust_region_factor=2.0",
        "train.validation.selection_metric=curve_auc",
    ),
}


@dataclass(frozen=True)
class ProtocolJob:
    stage: str
    kind: str
    variant: Optional[str]
    attack: Optional[str]
    seed: Optional[int]
    shot: int
    horizon: int
    output_relative: str
    manifest_relative: Optional[str]
    training_epochs: Optional[int]
    overrides: tuple[str, ...]


def _cell_overrides(attack: str, seed: int) -> tuple[str, ...]:
    known = json.dumps(KNOWN_CLASSES[attack], separators=(",", ":"))
    return (
        f"experiment.seed={seed}",
        f"data.unknown_class={attack}",
        f"data.known_classes={known}",
        *COMMON_OVERRIDES,
    )


def validation_manifest_relative(attack: str, seed: int) -> str:
    return f"validation/manifests/{attack}/seed_{seed}/validation_5shot.json"


def test_manifest_relative(attack: str, seed: int) -> str:
    return f"frozen_test/manifests/{attack}/seed_{seed}/test_5shot.json"


def formal_run_relative(variant: str, attack: str, seed: int) -> str:
    return (
        f"validation/{variant}/{attack}/seed_{seed}/"
        f"horizon_{HORIZON}"
    )


def stage_jobs(
    stage: str,
    *,
    selected_variant: Optional[str] = None,
) -> list[ProtocolJob]:
    """Expand a protocol stage without accessing data or experiment artifacts."""
    if stage == "Preflight":
        return []
    if stage == "Smoke":
        attack, seed, variant = "ddos", 42, "E0_final_only"
        return [ProtocolJob(
            stage=stage,
            kind="smoke_validation_run",
            variant=variant,
            attack=attack,
            seed=seed,
            shot=SHOT,
            horizon=HORIZON,
            output_relative=(
                f"smoke/{variant}/{attack}/seed_{seed}/horizon_{HORIZON}"
            ),
            manifest_relative=(
                f"smoke/manifests/{attack}/seed_{seed}/validation_5shot.json"
            ),
            training_epochs=1,
            overrides=(
                *_cell_overrides(attack, seed),
                *VARIANT_OVERRIDES[variant],
                "train.meta_epochs=1",
            ),
        )]
    if stage == "PrepareValidation":
        return [
            ProtocolJob(
                stage=stage,
                kind="validation_manifest",
                variant=None,
                attack=attack,
                seed=seed,
                shot=SHOT,
                horizon=HORIZON,
                output_relative=validation_manifest_relative(attack, seed),
                manifest_relative=validation_manifest_relative(attack, seed),
                training_epochs=None,
                overrides=_cell_overrides(attack, seed),
            )
            for attack in ATTACKS
            for seed in SEEDS
        ]
    if stage == "RunValidation":
        return [
            ProtocolJob(
                stage=stage,
                kind="formal_validation_run",
                variant=variant,
                attack=attack,
                seed=seed,
                shot=SHOT,
                horizon=HORIZON,
                output_relative=formal_run_relative(variant, attack, seed),
                manifest_relative=validation_manifest_relative(attack, seed),
                training_epochs=10,
                overrides=(
                    *_cell_overrides(attack, seed),
                    *VARIANT_OVERRIDES[variant],
                    "train.meta_epochs=10",
                ),
            )
            for variant in VARIANT_OVERRIDES
            for attack in ATTACKS
            for seed in SEEDS
        ]
    if stage == "AuditValidation":
        return [ProtocolJob(
            stage=stage,
            kind="validation_audit",
            variant=None,
            attack=None,
            seed=None,
            shot=SHOT,
            horizon=HORIZON,
            output_relative="validation/audit",
            manifest_relative=None,
            training_epochs=None,
            overrides=(),
        )]
    if stage == "PrepareFrozenTest":
        return [
            ProtocolJob(
                stage=stage,
                kind="test_manifest",
                variant=None,
                attack=attack,
                seed=seed,
                shot=SHOT,
                horizon=HORIZON,
                output_relative=test_manifest_relative(attack, seed),
                manifest_relative=test_manifest_relative(attack, seed),
                training_epochs=None,
                overrides=_cell_overrides(attack, seed),
            )
            for attack in ATTACKS
            for seed in SEEDS
        ]
    if stage == "FrozenTest":
        if selected_variant not in VARIANT_OVERRIDES:
            raise ValueError(
                "FrozenTest requires a selected pre-registered variant"
            )
        variants = tuple(dict.fromkeys(("E0_final_only", selected_variant)))
        return [
            ProtocolJob(
                stage=stage,
                kind="frozen_test_run",
                variant=variant,
                attack=attack,
                seed=seed,
                shot=SHOT,
                horizon=HORIZON,
                output_relative=(
                    f"frozen_test/results/{variant}/{attack}/seed_{seed}/"
                    f"horizon_{HORIZON}"
                ),
                manifest_relative=test_manifest_relative(attack, seed),
                training_epochs=None,
                overrides=(),
            )
            for variant in variants
            for attack in ATTACKS
            for seed in SEEDS
        ]
    raise ValueError(f"unsupported protocol stage={stage!r}")


def serialise_plan(
    stage: str,
    root: str | Path,
    expected_commit: str,
    *,
    selected_variant: Optional[str] = None,
) -> dict:
    jobs = stage_jobs(stage, selected_variant=selected_variant)
    return {
        "plan_schema_version": 1,
        "stage": stage,
        "paper_runner": True,
        "deprecated_phase_both_used": False,
        "expected_commit": expected_commit,
        "root": str(Path(root)),
        "job_count": len(jobs),
        "jobs": [asdict(job) for job in jobs],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "Preflight", "Smoke", "PrepareValidation", "RunValidation",
            "AuditValidation", "PrepareFrozenTest", "FrozenTest",
        ],
        required=True,
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--selected-variant")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(json.dumps(serialise_plan(
        args.stage,
        args.root,
        args.expected_commit,
        selected_variant=args.selected_variant,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
