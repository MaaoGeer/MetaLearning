"""CPU-only guards for split-isolated protocol execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import generate_eval_task_manifest, run_experiments
from src.data import pipeline
from src.data.dataset import IntrusionDataset
from src.data.loao import LOAOResult, SplitArrays
from src.data.task_sampler import MetaTask
from src.utils.config import Config
from src.utils.experiment_protocol import (
    VARIANT_OVERRIDES,
    serialise_plan,
    stage_jobs,
)
from src.utils.provenance import (
    assert_execution_gate,
    build_completion_receipt,
)


def _split(offset: int) -> SplitArrays:
    count = 24
    return SplitArrays(
        features=np.arange(count * 2, dtype=np.float32).reshape(count, 2),
        labels=np.array(["benign"] * 12 + ["ddos"] * 12, dtype=object),
        order=np.arange(count, dtype=np.float64) + offset,
        row_id=np.arange(count, dtype=np.int64) + offset * 100,
        segment_id=np.zeros(count, dtype=np.int64) + offset,
    )


def _window_dataset(offset: float = 0.0) -> IntrusionDataset:
    features = (
        np.arange(16 * 2 * 2, dtype=np.float32).reshape(16, 2, 2) + offset
    )
    labels = np.array([0] * 8 + [1] * 8, dtype=np.int64)
    rows = (
        np.arange(16 * 2, dtype=np.int64).reshape(16, 2)
        + int(offset * 100)
    )
    return IntrusionDataset(
        features,
        labels,
        sequence_length=2,
        row_ids=rows,
        segment_id=np.arange(16, dtype=np.int64),
        order_start=np.arange(16, dtype=np.float64),
        order_end=np.arange(16, dtype=np.float64),
    )


def _pipeline_cfg() -> Config:
    return Config({
        "data": {
            "name": "sentinel",
            "root": "unused",
            "window_size": 2,
            "stride": 1,
            "known_classes": ["benign"],
            "unknown_class": "ddos",
            "include_benign": True,
            "eval_ratio": 0.2,
            "test_ratio": 0.2,
            "split_mode": "temporal",
            "split_granularity": "per_class_temporal",
            "max_per_class": 0,
            "train_fraction": 1.0,
            "standardize": True,
            "strict_adapt_test": True,
            "adapt_val_ratio": 0.5,
            "k_shot": 1,
            "q_query": 1,
            "task_mode": "binary",
            "n_way": 2,
            "binary_pair_mode": "benign_vs_attack",
            "window_label_strategy": "last",
        }
    })


def _patch_pipeline(monkeypatch):
    splits = {
        "train": _split(1),
        "eval": _split(2),
        "test": _split(3),
        "unknown": _split(4),
    }
    loao = LOAOResult(
        feature_columns=["f0", "f1"],
        known_classes=["benign"],
        known_class_to_idx={"benign": 0},
        unknown_class="ddos",
        train=splits["train"],
        eval=splits["eval"],
        test=splits["test"],
        unknown=splits["unknown"],
        standardizer=None,
    )
    calls = {"meta_windows": 0, "adapt": []}
    monkeypatch.setattr(pipeline, "_load_clean", lambda cfg: object())
    monkeypatch.setattr(pipeline, "build_loao", lambda *a, **k: loao)
    monkeypatch.setattr(pipeline, "audit_pipeline_splits", lambda value: None)

    def build_meta(*args, **kwargs):
        calls["meta_windows"] += 1
        return _window_dataset()

    monkeypatch.setattr(pipeline, "build_windowed_dataset", build_meta)
    monkeypatch.setattr(pipeline, "make_meta_sampler", lambda *a, **k: object())

    def build_adapt(eval_split, unknown_split, *args, **kwargs):
        calls["adapt"].append((eval_split, unknown_split))
        offset = 1.0 if len(calls["adapt"]) == 1 else 2.0
        return _window_dataset(offset)

    monkeypatch.setattr(pipeline, "_build_adapt_dataset", build_adapt)
    return calls


def test_meta_training_role_constructs_no_adaptation_dataset(monkeypatch):
    calls = _patch_pipeline(monkeypatch)
    bundle = pipeline.build_pipeline(
        _pipeline_cfg(), adaptation_dataset_role="none"
    )
    assert calls["meta_windows"] == 2
    assert calls["adapt"] == []
    assert bundle.adaptation_validation_constructed is False
    assert bundle.adaptation_test_constructed is False
    with pytest.raises(RuntimeError, match="adapt_test_dataset"):
        _ = bundle.adapt_test_dataset


@pytest.mark.parametrize(
    ("role", "validation_constructed", "test_constructed"),
    [
        ("validation", True, False),
        ("test", False, True),
    ],
)
def test_split_role_constructs_only_requested_adaptation_dataset(
    monkeypatch, role, validation_constructed, test_constructed
):
    calls = _patch_pipeline(monkeypatch)
    bundle = pipeline.build_pipeline(
        _pipeline_cfg(), adaptation_dataset_role=role
    )
    assert calls["meta_windows"] == 0
    assert len(calls["adapt"]) == 1
    assert bundle.adaptation_validation_constructed is validation_constructed
    assert bundle.adaptation_test_constructed is test_constructed
    if role == "validation":
        with pytest.raises(RuntimeError, match="adapt_test_dataset"):
            _ = bundle.adapt_test_dataset
    else:
        with pytest.raises(RuntimeError, match="adapt_val_dataset"):
            _ = bundle.adapt_val_dataset


def test_pipeline_role_is_required_and_unknown_role_fails(monkeypatch):
    _patch_pipeline(monkeypatch)
    with pytest.raises(TypeError):
        pipeline.build_pipeline(_pipeline_cfg())
    with pytest.raises(ValueError, match="adaptation_dataset_role"):
        pipeline.build_pipeline(
            _pipeline_cfg(), adaptation_dataset_role="ambiguous"
        )


class _ManifestSampler:
    def __init__(self, dataset: IntrusionDataset):
        self.dataset = dataset

    def sample_task(self) -> MetaTask:
        return MetaTask(
            support_x=self.dataset.features[[0, 8]],
            support_y=torch.tensor([0, 1]),
            query_x=self.dataset.features[[1, 9]],
            query_y=torch.tensor([0, 1]),
            global_classes=[0, 1],
            support_window_ids=[0, 8],
            query_window_ids=[1, 9],
            shot=1,
        )


class _ManifestBundle:
    feature_dim = 2
    window_size = 2

    def __init__(self, role: str):
        self.role = role
        self.validation = _window_dataset(0.0)
        self.test = _window_dataset(10.0)
        self.validation_accesses = 0
        self.test_accesses = 0
        self.adaptation_validation_constructed = role == "validation"
        self.adaptation_test_constructed = role == "test"

    @property
    def adapt_val_dataset(self):
        self.validation_accesses += 1
        if self.role != "validation":
            raise AssertionError("validation dataset accessed in test role")
        return self.validation

    @property
    def adapt_test_dataset(self):
        self.test_accesses += 1
        if self.role != "test":
            raise AssertionError("test dataset accessed in validation role")
        return self.test

    def make_adaptation_sampler(self, *, split: str, **kwargs):
        if split == "val":
            return _ManifestSampler(self.adapt_val_dataset)
        if split == "test":
            return _ManifestSampler(self.adapt_test_dataset)
        raise AssertionError(split)


@pytest.mark.parametrize(
    ("requested", "expected_role", "other_access"),
    [("validation", "validation", "test_accesses"), ("test", "test", "validation_accesses")],
)
def test_manifest_generator_uses_only_requested_role(
    tmp_path, monkeypatch, requested, expected_role, other_access
):
    bundle = _ManifestBundle(expected_role)
    cfg = Config({
        "data": {
            "name": "sentinel",
            "strict_adapt_test": True,
            "q_query": 1,
            "task_mode": "binary",
            "unknown_class": "ddos",
            "train_fraction": 1.0,
        },
        "meta": {"adapt_scope": "head_only", "inner_steps": 20},
        "experiment": {"seed": 42},
    })
    theta0 = {
        "classifier.weight": torch.zeros(2, 2),
        "classifier.bias": torch.zeros(2),
    }
    roles = []

    def context(args, role):
        roles.append(role)
        return (
            cfg, bundle, theta0,
            {
                "n_way": 2,
                "unknown_class": "ddos",
                "adaptation_scope": "head_only",
                "meta_inner_steps": 20,
            },
            str(tmp_path / "future.pt"),
            "",
        )

    monkeypatch.setattr(
        generate_eval_task_manifest, "_load_manifest_context", context
    )
    monkeypatch.setattr(
        generate_eval_task_manifest,
        "assert_execution_gate",
        lambda *a, **k: {
            "git_commit": "a" * 40,
            "parent_commit": "b" * 40,
            "worktree_clean": True,
            "source_state_sha256": "c" * 64,
            "tracked_source_file_sha256": {},
        },
    )
    args = argparse.Namespace(
        artifacts=None,
        config="base.yaml",
        dataset="dataset.yaml",
        override=[],
        base_checkpoint_path=str(tmp_path / "future.pt"),
        out=str(tmp_path / f"{requested}.json"),
        shot=1,
        tasks=1,
        task_seed=43,
        split=requested,
        expected_commit="a" * 40,
        expected_source_state=None,
    )
    receipt = generate_eval_task_manifest.run(args)
    assert roles == [expected_role]
    assert getattr(bundle, other_access) == 0
    assert receipt["dataset_role"] == expected_role
    assert receipt[f"{other_access.removesuffix('_accesses')}_dataset_accessed"] is False
    assert receipt[
        f"{other_access.removesuffix('_accesses')}_dataset_constructed"
    ] is False


def _phase_args(phase: str, **kwargs) -> argparse.Namespace:
    values = {
        "phase": phase,
        "task_manifest": None,
        "validation_task_manifest": None,
        "test_task_manifest": None,
        "selection_receipt": None,
    }
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_validation_phase_rejects_every_test_manifest_argument():
    for field in ("task_manifest", "test_task_manifest"):
        with pytest.raises(ValueError, match="rejects"):
            run_experiments.validate_phase_arguments(
                _phase_args("validation", **{field: "test.json"})
            )
    assert run_experiments.validate_phase_arguments(
        _phase_args("validation", validation_task_manifest="val.json")
    ) == "validation"


def test_test_phase_rejects_validation_manifest_and_requires_frozen_inputs():
    with pytest.raises(ValueError, match="rejects"):
        run_experiments.validate_phase_arguments(_phase_args(
            "test",
            validation_task_manifest="val.json",
            test_task_manifest="test.json",
            selection_receipt="selection.json",
        ))
    with pytest.raises(ValueError, match="requires"):
        run_experiments.validate_phase_arguments(_phase_args("test"))


def test_protocol_stage_expansion_is_exact_and_never_uses_phase_both():
    assert len(stage_jobs("Smoke")) == 1
    smoke = stage_jobs("Smoke")[0]
    assert (smoke.variant, smoke.attack, smoke.seed) == (
        "E0_final_only", "ddos", 42
    )
    assert smoke.training_epochs == 1
    assert "train.meta_epochs=1" in smoke.overrides
    assert "train.meta_epochs=10" not in smoke.overrides
    assert len(stage_jobs("PrepareValidation")) == 4
    jobs = stage_jobs("RunValidation")
    assert len(jobs) == 16
    assert len({
        (job.variant, job.attack, job.seed) for job in jobs
    }) == 16
    assert all(job.kind == "formal_validation_run" for job in jobs)
    assert all("train.meta_epochs=10" in job.overrides for job in jobs)
    plan = serialise_plan(
        "RunValidation", "outputs/sentinel", "a" * 40
    )
    assert plan["deprecated_phase_both_used"] is False


def test_preregistered_variant_definitions_are_not_duplicated_in_runner():
    runner = Path(
        "scripts/run_metaopt_minimal_experiments.ps1"
    ).read_text(encoding="utf-8")
    assert "--phase\", \"both" not in runner
    assert "meta.query_objective.supervised_steps" not in runner
    assert "meta_optimizer.trust_region_factor" not in runner
    assert set(VARIANT_OVERRIDES) == {
        "E0_final_only",
        "E1_multistep",
        "E2_multistep_random_horizon",
        "E3_sgd_residual",
    }


def _git(tmp_path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=tmp_path, text=True
    ).strip()


def test_dirty_and_wrong_commit_execution_gates_fail(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "cpu-test@example.invalid")
    _git(tmp_path, "config", "user.name", "CPU Test")
    source = tmp_path / "entry.py"
    source.write_text("print('clean')\n", encoding="utf-8")
    _git(tmp_path, "add", "entry.py")
    _git(tmp_path, "commit", "-m", "initial")
    head = _git(tmp_path, "rev-parse", "HEAD")
    assert_execution_gate(
        head, repo_root=tmp_path, source_paths=("entry.py",)
    )
    with pytest.raises(RuntimeError, match="commit"):
        assert_execution_gate(
            "0" * 40, repo_root=tmp_path, source_paths=("entry.py",)
        )
    source.write_text("print('dirty')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        assert_execution_gate(
            head, repo_root=tmp_path, source_paths=("entry.py",)
        )


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [("success", 0), ("failed", 1), ("interrupted", 130)],
)
def test_completion_receipt_fields_for_every_terminal_status(
    tmp_path, status, exit_code
):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("complete\n", encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat()
    receipt = build_completion_receipt(
        stage="cpu_test",
        status=status,
        started_at_utc=now,
        finished_at_utc=now,
        duration_seconds=0.1,
        exit_code=exit_code,
        output_paths=[tmp_path],
        required_artifacts=[artifact] if status == "success" else [],
        execution_state={
            "git_commit": "a" * 40,
            "parent_commit": "b" * 40,
            "worktree_clean": True,
            "source_state_sha256": "c" * 64,
            "tracked_source_file_sha256": {"entry.py": "d" * 64},
        },
        details={
            "dataset_role": "validation",
            "test_dataset_constructed": False,
            "test_dataset_accessed": False,
            "metaopt_training_ran": False,
        },
    )
    required = {
        "completion_status", "git_commit", "parent_commit",
        "worktree_clean", "source_state_sha256",
        "tracked_source_file_sha256", "environment", "started_at_utc",
        "finished_at_utc", "duration_seconds", "exit_code",
        "required_artifacts", "output_paths", "dataset_role",
        "test_dataset_constructed", "test_dataset_accessed",
        "metaopt_training_ran",
    }
    assert required <= receipt.keys()


def test_success_receipt_requires_complete_artifacts(tmp_path):
    now = datetime.now(timezone.utc).isoformat()
    with pytest.raises(ValueError, match="missing"):
        build_completion_receipt(
            stage="cpu_test",
            status="success",
            started_at_utc=now,
            finished_at_utc=now,
            duration_seconds=0,
            exit_code=0,
            output_paths=[tmp_path],
            required_artifacts=[tmp_path / "missing"],
        )
