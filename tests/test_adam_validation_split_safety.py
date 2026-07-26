import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import tune_adam_manifest
from src.data.dataset import IntrusionDataset
from src.data.task_sampler import MetaTask
from src.evaluation.task_manifest import (
    sha256_file,
    tensor_state_sha256,
    write_task_manifest,
)
from src.utils.config import Config


class SentinelBundle:
    def __init__(self, validation: IntrusionDataset, test: IntrusionDataset):
        self._validation = validation
        self._test = test
        self.validation_accesses = 0
        self.test_accesses = 0

    @property
    def adapt_val_dataset(self):
        self.validation_accesses += 1
        return self._validation

    @property
    def adapt_test_dataset(self):
        self.test_accesses += 1
        raise AssertionError("Adam validation must never access adapt_test_dataset")


def _dataset(offset: float) -> IntrusionDataset:
    features = (
        np.arange(12 * 2 * 2, dtype=np.float32).reshape(12, 2, 2) + offset
    )
    labels = np.array([0] * 6 + [1] * 6, dtype=np.int64)
    row_ids = (
        np.arange(12 * 2, dtype=np.int64).reshape(12, 2)
        + int(offset * 1000)
    )
    return IntrusionDataset(
        features,
        labels,
        sequence_length=2,
        row_ids=row_ids,
        segment_id=np.arange(12, dtype=np.int64) + int(offset * 100),
        order_start=np.arange(12, dtype=np.float64) + offset,
        order_end=np.arange(12, dtype=np.float64) + offset,
    )


def _task(dataset: IntrusionDataset) -> MetaTask:
    return MetaTask(
        support_x=dataset.features[[0, 6]],
        support_y=torch.tensor([0, 1]),
        query_x=dataset.features[[1, 7]],
        query_y=torch.tensor([0, 1]),
        global_classes=[0, 1],
        support_window_ids=[0, 6],
        query_window_ids=[1, 7],
        shot=1,
    )


def _case(tmp_path: Path, *, split: str = "val", manifest_dataset=None):
    validation = _dataset(0.0)
    test = _dataset(10.0)
    dataset = validation if manifest_dataset is None else manifest_dataset
    artifact_path = tmp_path / "artifact.pt"
    artifact_path.write_bytes(b"sentinel artifact")
    artifact = {
        "meta_init_state": {
            "classifier.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "classifier.bias": torch.zeros(2),
        },
        "extra": {"unknown_class": "ddos"},
    }
    cfg = Config({
        "data": {"unknown_class": "ddos", "k_shot": 1},
        "experiment": {"seed": 62},
    })
    manifest_path = tmp_path / f"{split}.json"
    write_task_manifest(
        manifest_path,
        [_task(dataset)],
        protocol={
            "shot": 1,
            "q_query": 1,
            "split": split,
            "task_seed": 63,
            "attack": "ddos",
        },
        base_checkpoint_path=str(artifact_path),
        base_checkpoint_sha256=sha256_file(artifact_path),
        base_initialization_sha256=tensor_state_sha256(
            artifact["meta_init_state"]
        ),
        metadata={
            "unknown_class": "ddos",
            "experiment_seed": 62,
        },
        dataset=dataset,
    )
    return {
        "validation": validation,
        "test": test,
        "bundle": SentinelBundle(validation, test),
        "artifact_path": artifact_path,
        "artifact": artifact,
        "cfg": cfg,
        "manifest_path": manifest_path,
    }


def _refresh_sidecar(manifest_path: Path, *, split=None, fingerprint=None):
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256_file(manifest_path)
    declared_split = (
        payload.get("protocol", {}).get("split", "")
        if split is None else split
    )
    declared_fingerprint = (
        payload.get("metadata", {}).get("dataset_fingerprint", "")
        if fingerprint is None else fingerprint
    )
    manifest_path.with_suffix(".json.sha256").write_text(
        f"{digest}  {manifest_path.name}  split={declared_split}  "
        f"schema={payload.get('schema_version')}  "
        f"dataset_fingerprint={declared_fingerprint}\n",
        encoding="utf-8",
    )


def _validate(case):
    return tune_adam_manifest.validate_validation_inputs(
        artifact_path=case["artifact_path"],
        manifest_path=case["manifest_path"],
        artifact=case["artifact"],
        cfg=case["cfg"],
        bundle=case["bundle"],
        requested_split="validation",
        expected_attack="ddos",
        expected_seed=62,
        expected_shot=1,
    )


def test_validation_manifest_binds_only_adapt_val_dataset(tmp_path):
    case = _case(tmp_path)
    tasks, receipt = _validate(case)
    assert len(tasks) == 1
    assert receipt["effective_split"] == "validation"
    assert receipt["effective_dataset"] == "adapt_val_dataset"
    assert case["bundle"].validation_accesses == 1
    assert case["bundle"].test_accesses == 0


def test_test_manifest_is_rejected_before_dataset_access(tmp_path):
    case = _case(tmp_path, split="test", manifest_dataset=_dataset(10.0))
    with pytest.raises(ValueError, match="test"):
        _validate(case)
    assert case["bundle"].validation_accesses == 0
    assert case["bundle"].test_accesses == 0


def test_validation_manifest_cannot_bind_test_dataset(tmp_path):
    test = _dataset(10.0)
    case = _case(tmp_path, split="val", manifest_dataset=test)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _validate(case)
    assert case["bundle"].test_accesses == 0


@pytest.mark.parametrize("invalid_split", [None, "mystery"])
def test_missing_or_unknown_manifest_split_is_rejected(tmp_path, invalid_split):
    case = _case(tmp_path)
    path = case["manifest_path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if invalid_split is None:
        payload["protocol"].pop("split")
    else:
        payload["protocol"]["split"] = invalid_split
    path.write_text(json.dumps(payload), encoding="utf-8")
    _refresh_sidecar(path, split="val")
    with pytest.raises(ValueError, match="split"):
        _validate(case)


def test_manifest_sidecar_split_mismatch_is_rejected(tmp_path):
    case = _case(tmp_path)
    _refresh_sidecar(case["manifest_path"], split="test")
    with pytest.raises(ValueError, match="test|mismatch"):
        _validate(case)
    assert case["bundle"].validation_accesses == 0


def test_dataset_fingerprint_mismatch_is_rejected(tmp_path):
    case = _case(tmp_path)
    path = case["manifest_path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    wrong = hashlib.sha256(b"wrong validation dataset").hexdigest()
    payload["metadata"]["dataset_fingerprint"] = wrong
    path.write_text(json.dumps(payload), encoding="utf-8")
    _refresh_sidecar(path, fingerprint=wrong)
    with pytest.raises(ValueError, match="dataset fingerprint mismatch"):
        _validate(case)


def test_out_of_range_window_id_is_rejected(tmp_path):
    case = _case(tmp_path)
    path = case["manifest_path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tasks"][0].pop("task_sha256")
    payload["tasks"][0]["query_window_ids"][0] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    _refresh_sidecar(path)
    with pytest.raises(ValueError, match="invalid window IDs"):
        _validate(case)


def test_adam_receipt_contains_required_provenance_fields(tmp_path):
    case = _case(tmp_path)
    _, validated = _validate(case)
    selection = {
        "selected_lr": 0.3,
        "boundary_hit": False,
        "selection_rule": "sentinel validation rule",
        "selected_candidate": {"lr": 0.3},
    }
    now = datetime.now(timezone.utc)
    receipt = tune_adam_manifest.build_adam_validation_receipt(
        validation_receipt=validated,
        source_state={
            "git_commit": "a" * 40,
            "dirty_worktree": False,
            "git_status_short": [],
            "source_state_sha256": "b" * 64,
            "source_file_sha256": {},
        },
        selection=selection,
        raw_rows=[{"status": "ok"}],
        lrs=[0.1, 0.3, 0.5, 1.0],
        steps=20,
        device="cpu",
        started_at=now,
        finished_at=now,
        elapsed_seconds=0.1,
        argv=["tune_adam_manifest.py"],
    )
    required = {
        "requested_split", "manifest_declared_split", "effective_split",
        "effective_dataset", "effective_dataset_fingerprint", "manifest_path",
        "manifest_sha256", "sidecar_path", "sidecar_sha256", "attack", "seed",
        "shot", "theta0_sha256", "artifact_sha256", "checkpoint_sha256",
        "candidate_lrs", "selected_lr", "boundary_hit", "git_commit",
        "dirty_worktree", "source_state_sha256", "validation_started_at_utc",
        "validation_finished_at_utc", "test_dataset_accessed", "metaopt_ran",
        "command_line_arguments",
    }
    assert required <= receipt.keys()
    assert receipt["effective_split"] == "validation"
    assert receipt["effective_dataset"] == "adapt_val_dataset"
    assert receipt["test_dataset_accessed"] is False
    assert receipt["metaopt_ran"] is False


def test_adam_entrypoint_has_no_metaopt_train_or_evaluation_calls():
    source = inspect.getsource(tune_adam_manifest)
    forbidden = (
        "build_meta_optimizer(",
        "MetaTrainer(",
        "FewShotEvaluator(",
        "metaopt.evaluate(",
    )
    assert all(token not in source for token in forbidden)


def test_validation_entry_never_reads_test_dataset(tmp_path):
    case = _case(tmp_path)
    _validate(case)
    assert case["bundle"].test_accesses == 0
