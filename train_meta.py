"""Meta-training entry point for LSTM + Meta Optimizer + Few-shot NIDS.

Research mainline:
    Raw flow -> temporal window -> single-direction LSTM -> last hidden state
    -> classifier -> support loss -> learned optimizer update -> query loss.

There is deliberately no supervised pretraining stage. The saved artifact contains
one shared random initialization and the learned optimizer, so evaluation can compare
MetaOpt, Adam, and SGD from the exact same starting point and episodes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

from src.build import (
    build_meta_model,
    build_meta_optimizer,
    resolve_adapt_names,
    save_artifacts,
    task_n_way,
)
from src.data.pipeline import build_pipeline
from src.trainer.meta_trainer import MetaTrainer
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.visualization.plots import plot_training_curves

logger = get_logger("train_meta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM Meta Optimizer meta-training")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", default="configs/datasets/cicids2017.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--out", default="checkpoints/meta_artifacts.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg = cfg.merge(load_config(args.dataset).to_dict())
    if args.override:
        cfg = cfg.apply_overrides(args.override)

    artifact_dir = os.path.dirname(args.out) or "."
    os.makedirs(artifact_dir, exist_ok=True)
    if cfg.train.get("validation_task_audit_path", None) is None:
        cfg.train.validation_task_audit_path = os.path.join(
            artifact_dir, "validation_task_pool.json")
    if (
        str(cfg.train.checkpoint.get("dir", "checkpoints")) == "checkpoints"
        and os.path.normpath(artifact_dir) != os.path.normpath("checkpoints")
    ):
        cfg.train.checkpoint.dir = os.path.join(artifact_dir, "checkpoints")
        logger.info("Using run-local checkpoint dir: %s", cfg.train.checkpoint.dir)
    with open(os.path.join(artifact_dir, "effective_config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2, ensure_ascii=False)

    seed = int(cfg.experiment.get("seed", 42))
    set_seed(seed, bool(cfg.experiment.get("deterministic", True)))
    device = resolve_device(str(cfg.device.get("prefer", "auto")))
    logger.info("Device: %s | seed=%d | arch=%s", device, seed, cfg.model.arch)

    bundle = build_pipeline(cfg, seed=seed)
    meta_model = build_meta_model(cfg, bundle.feature_dim, bundle.window_size)
    adapt_names = resolve_adapt_names(meta_model, cfg)
    meta_opt = build_meta_optimizer(cfg)

    # Shared random initialization. It is frozen during meta-training and reused by
    # MetaOpt, Adam, and SGD in evaluation.
    meta_init_state = copy.deepcopy({
        k: v.detach().cpu() for k, v in meta_model.state_dict().items()
    })

    trainer = MetaTrainer(
        cfg=cfg,
        model=meta_model,
        meta_opt=meta_opt,
        train_sampler=bundle.meta_train_sampler,
        val_sampler=bundle.meta_val_sampler,
        device=device,
        adapt_names=adapt_names,
    )
    history = trainer.fit()

    if trainer.best_meta_opt_state is not None:
        meta_opt.load_state_dict(trainer.best_meta_opt_state)

    save_artifacts(
        path=args.out,
        meta_init_state=meta_init_state,
        meta_opt_state=meta_opt.state_dict(),
        cfg=cfg,
        extra={
            "feature_dim": bundle.feature_dim,
            "window_size": bundle.window_size,
            "n_way": task_n_way(cfg),
            "adapt_names": adapt_names,
            "known_classes": bundle.known_classes,
            "unknown_class": bundle.unknown_class,
            "seed": seed,
            "meta_inner_steps": int(cfg.meta.inner_steps),
            "best_meta_epoch": trainer.best_epoch,
            "best_meta_metric": trainer.best_metric,
            "artifact_selection": "validation_best",
            "initialization": "random_shared",
            "adaptation_scope": str(cfg.meta.get("adapt_scope", "full")),
            "train_fraction": float(cfg.data.get("train_fraction", 1.0)),
        },
    )

    try:
        fig_dir = str(cfg.output.get("figures_dir", "outputs/figures"))
        path = plot_training_curves(history, fig_dir, prefix="meta")
        logger.info("Training curves: %s", path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to plot training curves: %s", exc)

    logger.info("Meta-training finished. Artifact: %s", args.out)


if __name__ == "__main__":
    main()
