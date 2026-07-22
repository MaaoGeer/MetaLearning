import hashlib
import json

import numpy as np
import torch

from src.data.dataset import IntrusionDataset
from src.data.task_sampler import MetaTask
from src.evaluation.task_manifest import (
    load_tasks_from_manifest,
    manifest_raw_row_ids,
    read_task_manifest,
    write_task_manifest,
)
from src.visualization.plots import _adaptation_step_axis


def test_adaptation_axis_keeps_baseline_at_step_zero():
    trajectory = [0.31, 0.52, 0.74]
    assert _adaptation_step_axis(trajectory) == [0, 1, 2]
    assert _adaptation_step_axis(trajectory, steps=[0, 2, 5]) == [0, 2, 5]


def test_task_manifest_round_trip_and_sha256(tmp_path):
    features = np.arange(8 * 2 * 3, dtype=np.float32).reshape(8, 2, 3)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    dataset = IntrusionDataset(features, labels, sequence_length=2)
    task = MetaTask(
        support_x=dataset.features[[0, 4]],
        support_y=torch.tensor([0, 1]),
        query_x=dataset.features[[1, 5]],
        query_y=torch.tensor([0, 1]),
        global_classes=[0, 1],
        support_window_ids=[0, 4],
        query_window_ids=[1, 5],
    )
    manifest_path = tmp_path / "tasks.json"
    digest = write_task_manifest(
        manifest_path,
        [task],
        protocol={
            "shot": 1,
            "q_query": 1,
            "split": "test",
            "task_seed": 1044,
        },
        base_checkpoint_path="artifact.pt",
        base_checkpoint_sha256="a" * 64,
        dataset=dataset,
    )

    assert digest == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert (tmp_path / "tasks.json.sha256").read_text(encoding="utf-8").split()[0] == digest
    payload = read_task_manifest(manifest_path)
    loaded = load_tasks_from_manifest(payload, dataset)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["task_count"] == 1
    assert len(loaded) == 1
    assert loaded[0].support_window_ids == [0, 4]
    assert loaded[0].query_window_ids == [1, 5]
    assert torch.equal(loaded[0].support_x, task.support_x)
    assert torch.equal(loaded[0].query_y, task.query_y)
    assert manifest_raw_row_ids(payload) == set()
