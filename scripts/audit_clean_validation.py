"""Audit only the 16-run clean validation namespace and apply registered gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.task_manifest import (  # noqa: E402
    manifest_reuse_statistics,
    read_task_manifest,
    sha256_file,
)
from src.utils.experiment_protocol import (  # noqa: E402
    ATTACKS,
    HORIZON,
    SEEDS,
    VARIANT_OVERRIDES,
    formal_run_relative,
    validation_manifest_relative,
)
from src.utils.provenance import assert_execution_gate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-source-state")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty audit table: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _macro_final(method: dict) -> float:
    final = method["final_metrics_avg_per_task"]
    return float(final.get("macro_f1", final["f1"]))


def _checkpoint(method: dict, step: int) -> float:
    return float(
        method["adaptation_analysis"]["checkpoints"][str(step)][
            "macro_f1"
        ]["mean"]
    )


def _curve_auc(method: dict) -> float:
    return float(method["adaptation_analysis"]["curve_auc_mean"])


def _dynamics(path: Path) -> dict:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("method") == "MetaOpt" and row.get("group") == "all"
        ]
    if not rows:
        raise ValueError(f"missing aggregate MetaOpt dynamics: {path}")
    ratios = []
    clipped = 0
    nonfinite = 0
    for row in rows:
        clipped += int(float(row["was_clipped"]))
        for field in (
            "grad_norm", "update_norm", "update_to_grad_ratio",
            "raw_update_norm",
        ):
            value = float(row[field])
            nonfinite += int(not math.isfinite(value))
        ratio = float(row["update_to_grad_ratio"])
        if math.isfinite(ratio) and ratio > 0:
            ratios.append(ratio)
    ratio_span = max(ratios) / min(ratios) if ratios else math.inf
    return {
        "clip_ratio": clipped / len(rows),
        "update_grad_ratio_span": ratio_span,
        "nonfinite_dynamics": nonfinite,
        "update_row_count": len(rows),
    }


def _load_cell(root: Path, variant: str, attack: str, seed: int) -> dict:
    run_dir = root / formal_run_relative(variant, attack, seed)
    result_path = run_dir / "validation" / "results.json"
    completion_path = run_dir / "validation" / "completion_receipt.json"
    provenance_path = run_dir / "validation" / "provenance.json"
    artifact_path = run_dir / "meta_artifacts.pt"
    for required in (result_path, completion_path, provenance_path, artifact_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    completion = json.loads(completion_path.read_text(encoding="utf-8-sig"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    if not completion.get("success"):
        raise ValueError(f"run completion is not successful: {completion_path}")
    if completion.get("git_commit") != provenance.get("git_commit"):
        raise ValueError(f"completion/provenance commit mismatch: {run_dir}")
    for key in (
        "worktree_clean",
        "validation_dataset_constructed",
        "validation_dataset_accessed",
    ):
        if not completion.get(key):
            raise ValueError(f"{key}=false in {completion_path}")
    for key in ("test_dataset_constructed", "test_dataset_accessed"):
        if completion.get(key):
            raise ValueError(f"{key}=true in {completion_path}")
    if completion.get("dataset_role") != "validation":
        raise ValueError(f"wrong dataset role in {completion_path}")
    results = json.loads(result_path.read_text(encoding="utf-8-sig"))
    if set(results) != {"exp1_5shot"}:
        raise ValueError(f"unexpected experiment keys in {result_path}")
    experiment = results["exp1_5shot"]
    methods = experiment["methods"]
    dynamics = _dynamics(run_dir / "validation" / "update_analysis.csv")
    meta = methods["MetaOpt"]
    sgd = methods["SGD"]
    adam = methods["Adam"]
    return {
        "Variant": variant,
        "Attack": attack,
        "Seed": seed,
        "MetaStep1MacroF1": _checkpoint(meta, 1),
        "SGDStep1MacroF1": _checkpoint(sgd, 1),
        "AdamStep1MacroF1": _checkpoint(adam, 1),
        "MetaMinusSGDStep1": _checkpoint(meta, 1) - _checkpoint(sgd, 1),
        "MetaCurveAUC": _curve_auc(meta),
        "SGDCurveAUC": _curve_auc(sgd),
        "AdamCurveAUC": _curve_auc(adam),
        "MetaMinusSGDCurveAUC": _curve_auc(meta) - _curve_auc(sgd),
        "MetaFinalMacroF1": _macro_final(meta),
        "SGDFinalMacroF1": _macro_final(sgd),
        "AdamFinalMacroF1": _macro_final(adam),
        "MetaNonfiniteCount": int(meta["nonfinite_count"]),
        "ClipRatio": dynamics["clip_ratio"],
        "UpdateGradRatioSpan": dynamics["update_grad_ratio_span"],
        "NonfiniteDynamics": dynamics["nonfinite_dynamics"],
        "UpdateRowCount": dynamics["update_row_count"],
        "Theta0Sha256": completion["theta0_sha256"],
        "CheckpointSha256": completion["checkpoint_sha256"],
        "ManifestSha256": completion["manifest_sha256"][0],
        "DatasetFingerprint": completion["dataset_fingerprint"],
        "GitCommit": completion["git_commit"],
        "WorktreeClean": bool(completion["worktree_clean"]),
        "ResultsSha256": sha256_file(result_path),
    }


def _all_close(values: list[float], atol: float = 1e-12) -> bool:
    return bool(values) and max(values) - min(values) <= atol


def run(args: argparse.Namespace) -> dict:
    state = assert_execution_gate(
        args.expected_commit,
        expected_source_state_sha256=args.expected_source_state,
    )
    root = Path(args.root)
    expected_leaf = f"metaopt_remediation_clean_{args.expected_commit[:7]}"
    if root.name != expected_leaf:
        raise ValueError(f"audit root must be named {expected_leaf}")
    rows = [
        _load_cell(root, variant, attack, seed)
        for variant in VARIANT_OVERRIDES
        for attack in ATTACKS
        for seed in SEEDS
    ]
    if len(rows) != 16:
        raise AssertionError(f"expected 16 rows, got {len(rows)}")
    keys = {(r["Variant"], r["Attack"], r["Seed"]) for r in rows}
    if len(keys) != 16:
        raise ValueError("duplicate validation experiment key")
    for row in rows:
        if row["GitCommit"] != args.expected_commit or not row["WorktreeClean"]:
            raise ValueError("validation run provenance is not clean/exact")

    consistency = []
    for attack in ATTACKS:
        for seed in SEEDS:
            group = [
                row for row in rows
                if row["Attack"] == attack and row["Seed"] == seed
            ]
            checks = {
                "theta0": len({row["Theta0Sha256"] for row in group}) == 1,
                "manifest": len({row["ManifestSha256"] for row in group}) == 1,
                "dataset": len({row["DatasetFingerprint"] for row in group}) == 1,
                "sgd_step1": _all_close(
                    [row["SGDStep1MacroF1"] for row in group]
                ),
                "sgd_curve": _all_close(
                    [row["SGDCurveAUC"] for row in group]
                ),
                "sgd_final": _all_close(
                    [row["SGDFinalMacroF1"] for row in group]
                ),
            }
            consistency.append({
                "Attack": attack,
                "Seed": seed,
                **{name: bool(value) for name, value in checks.items()},
                "Passed": all(checks.values()),
            })
    if not all(row["Passed"] for row in consistency):
        raise ValueError("cross-variant fairness consistency check failed")

    e0_by_cell = {
        (row["Attack"], row["Seed"]): row
        for row in rows if row["Variant"] == "E0_final_only"
    }
    scorecard = []
    for row in rows:
        e0 = e0_by_cell[(row["Attack"], row["Seed"])]
        step1_gain = row["MetaStep1MacroF1"] - e0["MetaStep1MacroF1"]
        final_delta = row["MetaFinalMacroF1"] - e0["MetaFinalMacroF1"]
        scorecard.append({
            **row,
            "MetaMinusE0Step1": step1_gain,
            "MetaMinusE0Final": final_delta,
            "PassStep1Gain": (
                row["Variant"] != "E0_final_only" and step1_gain >= 0.08
            ),
            "PassStep1Gap": row["MetaMinusSGDStep1"] >= -0.05,
            "PassCurveGap": row["MetaMinusSGDCurveAUC"] >= -0.02,
            "PassFinalRegression": final_delta >= -0.01,
            "PassFinite": (
                row["MetaNonfiniteCount"] == 0
                and row["NonfiniteDynamics"] == 0
            ),
            "PassClipRatio": row["ClipRatio"] < 0.05,
            "PassDynamicsSpan": row["UpdateGradRatioSpan"] <= 100.0,
        })
    gate_fields = (
        "PassStep1Gain", "PassStep1Gap", "PassCurveGap",
        "PassFinalRegression", "PassFinite", "PassClipRatio",
        "PassDynamicsSpan",
    )
    variant_gate = []
    for variant in VARIANT_OVERRIDES:
        group = [row for row in scorecard if row["Variant"] == variant]
        passed = (
            variant != "E0_final_only"
            and all(all(bool(row[field]) for field in gate_fields) for row in group)
            and all(
                row["MetaMinusE0Step1"] >= 0.08
                for row in group
            )
        )
        variant_gate.append({
            "Variant": variant,
            "MeanMetaStep1": mean(row["MetaStep1MacroF1"] for row in group),
            "MeanStep1GapVsSGD": mean(
                row["MetaMinusSGDStep1"] for row in group
            ),
            "MeanMetaCurveAUC": mean(row["MetaCurveAUC"] for row in group),
            "MeanCurveGapVsSGD": mean(
                row["MetaMinusSGDCurveAUC"] for row in group
            ),
            "MeanMetaFinalMacroF1": mean(
                row["MetaFinalMacroF1"] for row in group
            ),
            "AllFourCellsPassed": passed,
        })
    passing = [row for row in variant_gate if row["AllFourCellsPassed"]]
    selected = (
        max(passing, key=lambda row: row["MeanMetaCurveAUC"])["Variant"]
        if passing else None
    )

    independence = []
    for seed in SEEDS:
        manifest_path = root / validation_manifest_relative("botnet", seed)
        manifest = read_task_manifest(manifest_path, verify_sha256=True)
        stats = manifest_reuse_statistics(manifest)
        independence.append({
            "Attack": "botnet",
            "Seed": seed,
            "Split": "validation",
            "ManifestSha256": sha256_file(manifest_path),
            **stats,
            "IndependentClaimAllowed": (
                stats["raw_disjoint_task_count_greedy"]
                == stats["task_count"]
            ),
        })

    _write_csv(root / "validation_scorecard.csv", scorecard)
    _write_csv(root / "validation_summary.csv", variant_gate)
    _write_csv(root / "botnet_task_independence.csv", independence)
    _write_csv(root / "validation_fairness_consistency.csv", consistency)
    receipt = {
        "receipt_schema_version": 1,
        "git_commit": state["git_commit"],
        "worktree_clean": state["worktree_clean"],
        "source_state_sha256": state["source_state_sha256"],
        "source_state_algorithm": state["source_state_algorithm"],
        "source_state_commit": state["source_state_commit"],
        "tracked_source_file_sha256": state[
            "tracked_source_file_sha256"
        ],
        "tracked_source_git_blob_oid": state[
            "tracked_source_git_blob_oid"
        ],
        "worktree_source_state_sha256": state[
            "worktree_source_state_sha256"
        ],
        "worktree_source_state_algorithm": state[
            "worktree_source_state_algorithm"
        ],
        "run_count": len(rows),
        "expected_run_count": 16,
        "all_runs_complete": True,
        "validation_only": True,
        "test_dataset_accessed": False,
        "test_results_read": False,
        "validation_gate_passed": selected is not None,
        "selected_variant": selected,
        "selection_rule": (
            "among variants passing all registered gates in all four "
            "attack/seed cells, select highest mean validation Curve AUC"
        ),
        "variant_gates": variant_gate,
        "fairness_consistency_passed": True,
    }
    (root / "validation_gate_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return receipt


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
