"""Strict LSTM-only LOAO fast-adaptation experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.pipeline import _load_clean, build_pipeline  # noqa: E402
from src.utils.config import Config, load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("fast_adaptation_matrix")


def _csv_list(raw: str, cast=str) -> List:
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict unknown x shot x seed x horizon LSTM MetaOpt/Adam/SGD matrix")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", default="configs/datasets/cicids2017.yaml")
    parser.add_argument("--out", default="outputs/fast_adaptation_matrix")
    parser.add_argument("--unknowns", default="botnet,bruteforce,ddos,dos")
    parser.add_argument("--shots", default="1,3,5,10")
    parser.add_argument("--seeds", default="42,52,62,72,82")
    parser.add_argument("--train-horizons", default="5,10,20")
    parser.add_argument("--train-fractions", default="1.0")
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def _unknown_matrix(cfg: Config, mode: str) -> Tuple[List[str], Dict[str, List[str]]]:
    configured_unknown = str(cfg.data.unknown_class)
    configured_known = [str(value) for value in cfg.data.known_classes]
    if mode == "configured":
        return [configured_unknown], {configured_unknown: configured_known}
    if mode != "all":
        unknowns = _csv_list(mode)
        loaded = _load_clean(cfg)
        attacks = sorted({
            str(label) for label in loaded.df["label"].unique()
            if str(label) not in {"benign", "normal"}
        })
        universe = attacks or sorted(set(configured_known + [configured_unknown]))
        return unknowns, {
            unknown: [attack for attack in universe if attack != unknown]
            for unknown in unknowns
        }
    loaded = _load_clean(cfg)
    attacks = sorted({
        str(label) for label in loaded.df["label"].unique()
        if str(label) not in {"benign", "normal"}
    })
    return attacks, {
        unknown: [attack for attack in attacks if attack != unknown]
        for unknown in attacks
    }


def _run(command: List[str], dry_run: bool) -> None:
    logger.info("RUN %s", subprocess.list2cmdline(command))
    if not dry_run:
        subprocess.run(command, check=True)


def _flatten_results(
    results_path: Path,
    unknown: str,
    seed: int,
    horizon: int,
    train_fraction: float,
) -> List[dict]:
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for experiment, result in payload.items():
        for method, method_result in result["methods"].items():
            analysis = method_result["adaptation_analysis"]
            base = {
                "unknown": unknown,
                "seed": seed,
                "train_horizon": horizon,
                "train_fraction": train_fraction,
                "shot": int(result["shot"]),
                "method": method,
                "experiment": experiment,
                "target_f1": analysis["target_f1"],
                "reach_rate": analysis["reach_rate"],
                "mean_steps": analysis["mean_steps"],
                "curve_auc": analysis["curve_auc_mean"],
                "best_f1": analysis["best_f1_mean"],
                "final_f1": analysis["final_f1_mean"],
                "post_peak_drop": analysis["post_peak_drop_mean"],
                "convergence95_step": analysis.get("convergence95_step", -1),
            }
            for step, metrics in analysis["checkpoints"].items():
                row = dict(base, step=int(step))
                for metric, stats in metrics.items():
                    row[metric] = stats["mean"]
                    row[f"{metric}_std_tasks"] = stats["std"]
                rows.append(row)
    return rows


def _flatten_update_rows(
    update_path: Path,
    unknown: str,
    seed: int,
    horizon: int,
    train_fraction: float,
) -> List[dict]:
    if not update_path.exists():
        return []
    rows: List[dict] = []
    with update_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row.update({
                "unknown": unknown,
                "seed": seed,
                "train_horizon": horizon,
                "train_fraction": train_fraction,
            })
            rows.append(row)
    return rows


def _write_rows(rows: List[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "matrix_results.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    if rows:
        keys = sorted({key for row in rows for key in row})
        with (out_dir / "matrix_results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def _audit_unknown_windows(
    cfg: Config,
    unknowns: List[str],
    known_map: Dict[str, List[str]],
    shots: List[int],
    out_dir: Path,
) -> Tuple[List[str], List[dict]]:
    rows: List[dict] = []
    passed: List[str] = []
    q_query = int(cfg.data.q_query)
    need = max(shots) + q_query if shots else int(cfg.data.k_shot) + q_query
    for unknown in unknowns:
        audit_cfg = cfg.apply_overrides([
            f"data.unknown_class={unknown}",
            "data.known_classes=" + json.dumps(known_map[unknown]),
        ])
        row = {
            "unknown": unknown,
            "required_windows": int(need),
            "status": "fail",
            "reason": "",
        }
        try:
            bundle = build_pipeline(audit_cfg, seed=int(audit_cfg.experiment.get("seed", 42)))
            unknown_idx = bundle._adapt_class_to_idx[bundle.unknown_class]
            val_windows = len(bundle.adapt_val_dataset.class_to_indices.get(unknown_idx, []))
            test_windows = len(bundle.adapt_test_dataset.class_to_indices.get(unknown_idx, []))
            row.update({
                "unknown_idx": int(unknown_idx),
                "adapt_val_unknown_windows": int(val_windows),
                "adapt_test_unknown_windows": int(test_windows),
                "raw_unknown_rows": int(len(bundle.loao.unknown)),
                "known_classes": list(bundle.known_classes),
            })
            if val_windows >= need and test_windows >= need:
                row["status"] = "pass"
                passed.append(unknown)
            else:
                row["reason"] = (
                    f"unknown windows below required {need}: "
                    f"val={val_windows}, test={test_windows}"
                )
        except Exception as exc:
            row["reason"] = str(exc)
        rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "window_sufficiency_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    if rows:
        keys = sorted({key for row in rows for key in row})
        with (out_dir / "window_sufficiency_audit.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    skipped = [row for row in rows if row["status"] != "pass"]
    for row in skipped:
        logger.warning("Skipping unknown=%s: %s", row["unknown"], row["reason"])
    return passed, rows


def _write_summary(rows: List[dict], out_dir: Path) -> None:
    import numpy as np

    groups: Dict[tuple, List[dict]] = {}
    for row in rows:
        key = (
            row["unknown"], row["train_fraction"], row["train_horizon"],
            row["shot"], row["method"], row["step"],
        )
        groups.setdefault(key, []).append(row)

    summary = []
    metrics = (
        "reach_rate", "mean_steps", "curve_auc", "best_f1", "final_f1",
        "post_peak_drop", "convergence95_step", "accuracy", "macro_f1",
        "weighted_f1", "precision", "recall", "pr_auc", "attack_recall",
    )
    for key, values in groups.items():
        item = dict(zip(
            ("unknown", "train_fraction", "train_horizon", "shot", "method", "step"),
            key,
        ))
        item["n_seeds"] = len(values)
        for metric in metrics:
            arr = np.asarray([float(value.get(metric, np.nan)) for value in values], dtype=float)
            item[f"{metric}_mean"] = float(np.nanmean(arr))
            item[f"{metric}_std"] = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
        summary.append(item)
    _write_rows(summary, out_dir / "summary")


def _write_significance(rows: List[dict], out_dir: Path) -> None:
    """Paired t-test and bootstrap deltas across matched training seeds."""
    import numpy as np
    from scipy import stats

    comparisons = [("MetaOpt", "Adam"), ("MetaOpt", "SGD"), ("Adam", "SGD")]
    metrics = ["macro_f1", "accuracy", "weighted_f1", "precision", "recall", "attack_recall"]

    indexed: Dict[tuple, Dict[str, dict]] = {}
    for row in rows:
        key = (
            row["unknown"], row["train_fraction"], row["train_horizon"],
            row["shot"], row["step"], row["seed"],
        )
        indexed.setdefault(key, {})[row["method"]] = row

    grouped: Dict[tuple, List[Tuple[float, float]]] = {}
    for key, methods in indexed.items():
        for left, right in comparisons:
            if left not in methods or right not in methods:
                continue
            for metric in metrics:
                grouped.setdefault(key[:5] + (left, right, metric), []).append((
                    float(methods[left].get(metric, np.nan)),
                    float(methods[right].get(metric, np.nan)),
                ))

    rng = np.random.default_rng(2026)
    output = []
    for key, pairs in grouped.items():
        values = np.asarray(pairs, dtype=float)
        values = values[~np.isnan(values).any(axis=1)]
        if not len(values):
            continue
        deltas = values[:, 0] - values[:, 1]
        draws = []
        for _ in range(10000):
            sample = rng.choice(deltas, size=len(deltas), replace=True)
            draws.append(float(sample.mean()))
        if len(deltas) > 1:
            t_stat, p_value = stats.ttest_rel(values[:, 0], values[:, 1], nan_policy="omit")
            t_stat = float(t_stat) if t_stat == t_stat else float("nan")
            p_value = float(p_value) if p_value == p_value else float("nan")
        else:
            t_stat, p_value = float("nan"), float("nan")
        output.append({
            "unknown": key[0],
            "train_fraction": key[1],
            "train_horizon": key[2],
            "shot": key[3],
            "step": key[4],
            "comparison": f"{key[5]}-{key[6]}",
            "left_method": key[5],
            "right_method": key[6],
            "metric": key[7],
            "n_paired_seeds": int(len(deltas)),
            "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "t_statistic": t_stat,
            "p_value": p_value,
            "ci95_low": float(np.percentile(draws, 2.5)),
            "ci95_high": float(np.percentile(draws, 97.5)),
            "probability_left_better": float(np.mean(np.asarray(draws) > 0.0)),
        })
    _write_rows(output, out_dir / "significance")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        cfg = cfg.merge(load_config(args.dataset).to_dict())
    if args.override:
        cfg = cfg.apply_overrides(args.override)

    shots = [1] if args.quick else _csv_list(args.shots, int)
    seeds = [0] if args.quick else _csv_list(args.seeds, int)
    horizons = [5] if args.quick else _csv_list(args.train_horizons, int)
    train_fractions = [1.0] if args.quick else _csv_list(args.train_fractions, float)
    eval_steps = 5 if args.quick else int(args.eval_steps)
    unknowns, known_map = _unknown_matrix(cfg, args.unknowns)
    if args.quick:
        unknowns = unknowns[:1]

    root = Path(args.out)
    unknowns, window_audit = _audit_unknown_windows(cfg, unknowns, known_map, shots, root)
    if not unknowns:
        raise ValueError(
            "No unknown classes passed window sufficiency audit. "
            f"Audit written to {root / 'window_sufficiency_audit.json'}"
        )
    all_rows: List[dict] = []
    all_update_rows: List[dict] = []
    for unknown in unknowns:
        for seed in seeds:
            for train_fraction in train_fractions:
                for horizon in horizons:
                    fraction_name = f"{train_fraction:.6f}".rstrip("0").rstrip(".")
                    run_dir = (
                        root / "runs" / unknown / f"fraction_{fraction_name}"
                        / f"seed_{seed}" / f"horizon_{horizon}"
                    )
                    artifact = run_dir / "meta_artifacts.pt"
                    results_path = run_dir / "evaluation" / "results.json"
                    common_overrides = [
                        f"data.unknown_class={unknown}",
                        "data.known_classes=" + json.dumps(known_map[unknown]),
                        f"data.train_fraction={train_fraction}",
                        "model.arch=lstm",
                        f"experiment.seed={seed}",
                        f"meta.inner_steps={horizon}",
                    ] + list(args.override)

                    if not (args.skip_existing and artifact.exists()):
                        _run([
                            sys.executable, "train_meta.py",
                            "--config", args.config,
                            "--dataset", args.dataset,
                            "--out", str(artifact),
                            "--override", *common_overrides,
                        ], args.dry_run)

                    evaluation_overrides = common_overrides + [
                        "compare.shots=" + json.dumps(shots),
                        f"adaptation_speed.max_steps={eval_steps}",
                        "adaptation_speed.checkpoints=" + json.dumps(
                            sorted({
                                step
                                for step in [0, 1, 2, 5, 10, 20, eval_steps]
                                if step <= eval_steps
                            })),
                    ]
                    if not (args.skip_existing and results_path.exists()):
                        _run([
                            sys.executable, "scripts/run_experiments.py",
                            "--artifacts", str(artifact),
                            "--out", str(run_dir / "evaluation"),
                            "--override", *evaluation_overrides,
                        ], args.dry_run)

                    if not args.dry_run and results_path.exists():
                        all_rows.extend(_flatten_results(
                            results_path, unknown, seed, horizon, train_fraction))
                        all_update_rows.extend(_flatten_update_rows(
                            run_dir / "evaluation" / "update_analysis.csv",
                            unknown, seed, horizon, train_fraction))
                        _write_rows(all_rows, root)
                        _write_rows(all_update_rows, root / "update_analysis")

    if not args.dry_run:
        _write_rows(all_rows, root)
        _write_rows(all_update_rows, root / "update_analysis")
        _write_summary(all_rows, root)
        _write_significance(all_rows, root)
        logger.info("Fast-adaptation matrix complete: %s", root)


if __name__ == "__main__":
    main()
