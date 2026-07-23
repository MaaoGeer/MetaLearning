"""Serializable, hash-verifiable evaluation task manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from ..data.dataset import IntrusionDataset
from ..data.task_sampler import MetaTask


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in deterministic order."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _window_provenance(
    dataset: Optional[IntrusionDataset],
    window_ids: Sequence[int],
) -> List[dict]:
    if dataset is None:
        return []
    records = []
    for window_id in window_ids:
        index = int(window_id)
        row_ids = (
            [int(value) for value in np.asarray(dataset.row_ids[index]).reshape(-1).tolist()]
            if dataset.row_ids is not None else []
        )
        records.append({
            "local_window_id": index,
            "row_start": int(dataset.row_start[index]) if dataset.row_start is not None else None,
            "row_end": int(dataset.row_end[index]) if dataset.row_end is not None else None,
            "segment_id": int(dataset.segment_id[index]) if dataset.segment_id is not None else None,
            "capture_or_time_block": (
                int(dataset.segment_id[index])
                if dataset.segment_id is not None else None
            ),
            "order_start": (
                float(dataset.order_start[index])
                if dataset.order_start is not None else None
            ),
            "order_end": (
                float(dataset.order_end[index])
                if dataset.order_end is not None else None
            ),
            "raw_row_ids": row_ids,
            "raw_row_ids_sha256": _canonical_sha256({"raw_row_ids": row_ids}),
            "window_tensor_sha256": _tensor_sha256(dataset.features[index]),
        })
    return records


def _task_record(
    task: MetaTask,
    task_index: int,
    protocol: Mapping[str, Any],
    dataset: Optional[IntrusionDataset] = None,
) -> dict:
    record = {
        "task_id": int(task_index),
        "task_index": int(task_index),
        "task_seed": int(protocol["task_seed"]),
        "rng_sequence_index": int(task_index),
        "shot": int(protocol["shot"]),
        "q_query": int(protocol["q_query"]),
        "split": str(protocol["split"]),
        "attack": str(protocol.get("attack", "")),
        "global_classes": [int(value) for value in task.global_classes],
        "support_window_ids": [int(value) for value in task.support_window_ids],
        "query_window_ids": [int(value) for value in task.query_window_ids],
        "support_labels": [int(value) for value in task.support_y.detach().cpu().tolist()],
        "query_labels": [int(value) for value in task.query_y.detach().cpu().tolist()],
        "support_tensor_sha256": _tensor_sha256(task.support_x),
        "query_tensor_sha256": _tensor_sha256(task.query_x),
        "shot_observed": int(task.shot or protocol["shot"]),
    }
    if dataset is not None:
        record["support_window_provenance"] = _window_provenance(
            dataset, task.support_window_ids
        )
        record["query_window_provenance"] = _window_provenance(
            dataset, task.query_window_ids
        )
    record["task_sha256"] = _canonical_sha256(record)
    return record


def write_task_manifest(
    path: str | Path,
    tasks: Sequence[MetaTask],
    *,
    protocol: Mapping[str, Any],
    base_checkpoint_path: str,
    base_checkpoint_sha256: str,
    metadata: Optional[Mapping[str, Any]] = None,
    dataset: Optional[IntrusionDataset] = None,
    base_initialization_sha256: Optional[str] = None,
) -> str:
    """Write a manifest and a sidecar containing the manifest file's SHA256."""
    required = {"shot", "q_query", "split", "task_seed"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"task manifest protocol missing fields: {sorted(missing)}")
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_checkpoint_path": str(base_checkpoint_path),
        "base_checkpoint_sha256": str(base_checkpoint_sha256),
        "base_initialization_sha256": str(base_initialization_sha256 or ""),
        "protocol": dict(protocol),
        "metadata": dict(metadata or {}),
        "task_count": int(len(tasks)),
        "tasks": [
            _task_record(task, task_index, protocol, dataset=dataset)
            for task_index, task in enumerate(tasks)
        ],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8"
    )
    return digest


def read_task_manifest(path: str | Path, verify_sha256: bool = True) -> dict:
    source = Path(path)
    if verify_sha256:
        sidecar = source.with_suffix(source.suffix + ".sha256")
        if sidecar.exists():
            expected = sidecar.read_text(encoding="utf-8").split()[0]
            actual = sha256_file(source)
            if actual != expected:
                raise ValueError(
                    f"task manifest SHA256 mismatch: expected={expected} actual={actual}"
                )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported task manifest schema={payload.get('schema_version')!r}"
        )
    if int(payload.get("task_count", -1)) != len(payload.get("tasks", [])):
        raise ValueError("task manifest task_count does not match tasks length")
    return payload


def load_tasks_from_manifest(
    manifest: Mapping[str, Any],
    dataset: IntrusionDataset,
) -> List[MetaTask]:
    """Rebuild tasks from explicit window IDs without invoking an RNG sampler."""
    tasks: List[MetaTask] = []
    for expected_index, record in enumerate(manifest.get("tasks", [])):
        recorded_task_hash = record.get("task_sha256")
        if recorded_task_hash:
            unhashed = {key: value for key, value in record.items() if key != "task_sha256"}
            if _canonical_sha256(unhashed) != recorded_task_hash:
                raise ValueError(f"task {expected_index} task_sha256 mismatch")
        task_index = int(record.get("task_id", record["task_index"]))
        if task_index != expected_index:
            raise ValueError(
                f"task manifest index mismatch: expected={expected_index} got={task_index}"
            )
        support_ids = [int(value) for value in record["support_window_ids"]]
        query_ids = [int(value) for value in record["query_window_ids"]]
        all_ids = support_ids + query_ids
        if not all_ids or min(all_ids) < 0 or max(all_ids) >= len(dataset):
            raise ValueError(f"task {task_index} contains invalid window IDs")
        if set(support_ids) & set(query_ids):
            raise ValueError(f"task {task_index} has identical support/query window IDs")
        support_y = torch.tensor(record["support_labels"], dtype=torch.long)
        query_y = torch.tensor(record["query_labels"], dtype=torch.long)
        if len(support_y) != len(support_ids) or len(query_y) != len(query_ids):
            raise ValueError(f"task {task_index} label/window count mismatch")
        task = MetaTask(
            support_x=dataset.features[support_ids],
            support_y=support_y,
            query_x=dataset.features[query_ids],
            query_y=query_y,
            global_classes=[int(value) for value in record["global_classes"]],
            support_window_ids=support_ids,
            query_window_ids=query_ids,
            shot=(
                int(record.get(
                    "shot", manifest.get("protocol", {}).get("shot", 0)
                )) or None
            ),
        )
        expected_support_hash = record.get("support_tensor_sha256")
        expected_query_hash = record.get("query_tensor_sha256")
        if expected_support_hash and _tensor_sha256(task.support_x) != expected_support_hash:
            raise ValueError(f"task {task_index} support tensor hash mismatch")
        if expected_query_hash and _tensor_sha256(task.query_x) != expected_query_hash:
            raise ValueError(f"task {task_index} query tensor hash mismatch")
        for key, ids in (
            ("support_window_provenance", support_ids),
            ("query_window_provenance", query_ids),
        ):
            provenance = record.get(key, [])
            if not provenance:
                continue
            if len(provenance) != len(ids):
                raise ValueError(f"task {task_index} {key} length mismatch")
            for window, window_id in zip(provenance, ids):
                if int(window["local_window_id"]) != window_id:
                    raise ValueError(f"task {task_index} {key} local window mismatch")
                if dataset.row_ids is not None and window.get("raw_row_ids") is not None:
                    actual_rows = [
                        int(value)
                        for value in np.asarray(dataset.row_ids[window_id]).reshape(-1).tolist()
                    ]
                    if actual_rows != [int(value) for value in window["raw_row_ids"]]:
                        raise ValueError(f"task {task_index} {key} raw row provenance mismatch")
        tasks.append(task)
    return tasks


def manifest_raw_row_ids(manifest: Mapping[str, Any]) -> set[int]:
    """Return raw row IDs recorded in a manifest, for cross-split isolation audits."""
    rows: set[int] = set()
    for task in manifest.get("tasks", []):
        for key in ("support_window_provenance", "query_window_provenance"):
            for window in task.get(key, []):
                rows.update(int(value) for value in window.get("raw_row_ids", []))
    return rows


def manifest_reuse_statistics(manifest: Mapping[str, Any]) -> Dict[str, float | int]:
    """Quantify task/window reuse and a conservative disjoint-task count."""
    tasks = list(manifest.get("tasks", []))
    occurrences: List[tuple[str, int]] = []
    task_row_sets: List[set[int]] = []
    for task in tasks:
        rows: set[int] = set()
        for key in ("support_window_provenance", "query_window_provenance"):
            for window in task.get(key, []):
                occurrences.append((
                    str(task.get("split", manifest.get("protocol", {}).get("split", ""))),
                    int(window["local_window_id"]),
                ))
                rows.update(int(value) for value in window.get("raw_row_ids", []))
        task_row_sets.append(rows)
    unique_windows = len(set(occurrences))
    total_occurrences = len(occurrences)
    seen_rows: set[int] = set()
    disjoint_tasks = 0
    for rows in sorted(task_row_sets, key=lambda values: (-len(values), sorted(values))):
        if rows and not (rows & seen_rows):
            disjoint_tasks += 1
            seen_rows.update(rows)
    if not any(task_row_sets):
        disjoint_tasks = len({
            str(task.get("task_sha256", "")) for task in tasks
            if task.get("task_sha256")
        })
    return {
        "task_count": len(tasks),
        "unique_task_hashes": len({
            str(task.get("task_sha256", "")) for task in tasks
            if task.get("task_sha256")
        }),
        "window_occurrences": total_occurrences,
        "unique_windows": unique_windows,
        "window_reuse_rate": (
            1.0 - unique_windows / total_occurrences
            if total_occurrences else 0.0
        ),
        "raw_disjoint_task_count_greedy": disjoint_tasks,
    }


def assert_manifest_split_isolation(
    validation_manifest: Mapping[str, Any],
    test_manifest: Mapping[str, Any],
) -> None:
    """Reject validation/test manifests that share split labels or raw rows."""
    val_split = str(validation_manifest.get("protocol", {}).get("split", ""))
    test_split = str(test_manifest.get("protocol", {}).get("split", ""))
    if val_split != "val" or test_split != "test":
        raise ValueError(
            f"expected val/test manifests, got {val_split!r}/{test_split!r}"
        )
    overlap = manifest_raw_row_ids(validation_manifest) & manifest_raw_row_ids(
        test_manifest
    )
    if overlap:
        raise ValueError(
            f"validation/test manifests share {len(overlap)} raw rows"
        )
