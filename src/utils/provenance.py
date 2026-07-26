"""Run provenance helpers for reproducible, non-overwriting experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


RAW_DATA_SUFFIXES = {
    ".csv", ".parquet", ".json", ".jsonl", ".txt", ".arff", ".npz", ".npy"
}

DEFAULT_EXECUTION_SOURCE_PATHS = (
    "configs/base.yaml",
    "configs/datasets/cicids2017.yaml",
    "train_meta.py",
    "scripts/audit_clean_validation.py",
    "scripts/generate_eval_task_manifest.py",
    "scripts/prepare_frozen_test_receipt.py",
    "scripts/run_experiments.py",
    "scripts/run_metaopt_minimal_experiments.ps1",
    "src/build.py",
    "src/data/loao.py",
    "src/data/pipeline.py",
    "src/data/task_builder.py",
    "src/data/task_sampler.py",
    "src/evaluation/adaptation_speed.py",
    "src/evaluation/metrics.py",
    "src/evaluation/task_manifest.py",
    "src/meta_optimizer/lstm_optimizer.py",
    "src/meta_optimizer/preprocess.py",
    "src/trainer/adapter.py",
    "src/trainer/meta_trainer.py",
    "src/utils/experiment_protocol.py",
    "src/utils/provenance.py",
)

COMMIT_SOURCE_STATE_ALGORITHM = {
    "name": "git-commit-blob-path-content-sha256",
    "version": 1,
    "entry_encoding": "canonical-json(path -> {git_blob_oid, blob_sha256})",
    "cross_platform": True,
    "gating": True,
}
WORKTREE_SOURCE_STATE_ALGORITHM = {
    "name": "worktree-path-bytes-sha256",
    "version": 1,
    "entry_encoding": "canonical-json(path -> worktree_byte_sha256)",
    "cross_platform": False,
    "gating": False,
    "diagnostic_only": True,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_id(config: Mapping, length: int = 12) -> str:
    return canonical_sha256(config)[: int(length)]


def git_commit(repo_root: str | Path = ".") -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_output(repo_root: str | Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _git_bytes(repo_root: str | Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(repo_root),
        stderr=subprocess.DEVNULL,
    )


def git_worktree_state(repo_root: str | Path = ".") -> dict:
    """Return commit and tracked/untracked cleanliness without counting ignored outputs."""
    root = Path(_git_output(repo_root, "rev-parse", "--show-toplevel"))
    commit = _git_output(root, "rev-parse", "HEAD")
    try:
        parent = _git_output(root, "rev-parse", "HEAD^")
    except subprocess.CalledProcessError:
        parent = None
    status_text = _git_output(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    status = status_text.splitlines() if status_text else []
    return {
        "repository_root": str(root.resolve()),
        "git_commit": commit,
        "parent_commit": parent,
        "git_status_porcelain": status,
        "worktree_clean": not status,
    }


def tracked_source_state(
    repo_root: str | Path = ".",
    source_paths: Sequence[str] = DEFAULT_EXECUTION_SOURCE_PATHS,
    *,
    commit: str = "HEAD",
) -> dict:
    """Hash path-addressed Git blobs from one commit, independent of checkout EOLs."""
    root = Path(_git_output(repo_root, "rev-parse", "--show-toplevel"))
    resolved_commit = _git_output(root, "rev-parse", commit)
    entries = {}
    file_sha256 = {}
    blob_oids = {}
    for raw_path in sorted(set(source_paths)):
        relative = Path(raw_path).as_posix()
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"source path must be repository-relative: {raw_path}")
        try:
            blob_oid = _git_output(
                root, "rev-parse", f"{resolved_commit}:{relative}"
            )
            object_type = _git_output(root, "cat-file", "-t", blob_oid)
        except subprocess.CalledProcessError as exc:
            raise FileNotFoundError(
                f"tracked source is absent from commit {resolved_commit}: {relative}"
            ) from exc
        if object_type != "blob":
            raise ValueError(
                f"execution source is not a Git blob: {relative} ({object_type})"
            )
        blob = _git_bytes(root, "cat-file", "blob", blob_oid)
        blob_sha256 = hashlib.sha256(blob).hexdigest()
        entries[relative] = {
            "git_blob_oid": blob_oid,
            "blob_sha256": blob_sha256,
        }
        file_sha256[relative] = blob_sha256
        blob_oids[relative] = blob_oid
    return {
        "source_state_sha256": canonical_sha256(entries),
        "source_state_algorithm": dict(COMMIT_SOURCE_STATE_ALGORITHM),
        "source_state_commit": resolved_commit,
        "tracked_source_file_sha256": file_sha256,
        "tracked_source_git_blob_oid": blob_oids,
        "tracked_source_entries": entries,
    }


def worktree_source_state(
    repo_root: str | Path = ".",
    source_paths: Sequence[str] = DEFAULT_EXECUTION_SOURCE_PATHS,
) -> dict:
    """Return checkout-byte hashes for diagnostics; never use this for gating."""
    root = Path(_git_output(repo_root, "rev-parse", "--show-toplevel"))
    rows = {}
    for raw_path in sorted(set(source_paths)):
        relative = Path(raw_path).as_posix()
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"worktree source file is missing: {relative}")
        rows[relative] = sha256_file(source)
    return {
        "worktree_source_state_sha256": canonical_sha256(rows),
        "worktree_source_state_algorithm": dict(
            WORKTREE_SOURCE_STATE_ALGORITHM
        ),
        "worktree_source_file_sha256": rows,
    }


def assert_execution_gate(
    expected_commit: str,
    *,
    repo_root: str | Path = ".",
    source_paths: Sequence[str] = DEFAULT_EXECUTION_SOURCE_PATHS,
    expected_source_state_sha256: Optional[str] = None,
) -> dict:
    """Reject dirty or wrong-commit formal experiment launches."""
    state = git_worktree_state(repo_root)
    if state["git_commit"] != str(expected_commit):
        raise RuntimeError(
            "execution gate rejected commit: "
            f"expected={expected_commit} actual={state['git_commit']}"
        )
    if not state["worktree_clean"]:
        raise RuntimeError(
            "execution gate rejected dirty worktree: "
            f"{state['git_status_porcelain']}"
        )
    source = tracked_source_state(
        state["repository_root"],
        source_paths=source_paths,
        commit=state["git_commit"],
    )
    if (
        expected_source_state_sha256
        and source["source_state_sha256"] != expected_source_state_sha256
    ):
        raise RuntimeError(
            "execution gate rejected source-state hash: "
            f"expected={expected_source_state_sha256} "
            f"actual={source['source_state_sha256']}"
        )
    diagnostic = worktree_source_state(
        state["repository_root"], source_paths=source_paths
    )
    return {**state, **source, **diagnostic}


def runtime_environment() -> dict:
    payload = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "command_line": list(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        payload.update({
            "pytorch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        })
        if torch.cuda.is_available():
            payload["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        payload.update({
            "pytorch_version": None,
            "torch_cuda_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
        })
    return payload


def build_completion_receipt(
    *,
    stage: str,
    status: str,
    started_at_utc: str,
    finished_at_utc: str,
    duration_seconds: float,
    exit_code: int,
    output_paths: Sequence[str | Path],
    required_artifacts: Sequence[str | Path] = (),
    execution_state: Optional[Mapping[str, Any]] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build a complete success/failure/interruption receipt."""
    if status not in {"success", "failed", "interrupted"}:
        raise ValueError(f"unsupported completion status={status!r}")
    required_rows = []
    missing = []
    for raw_path in required_artifacts:
        path = Path(raw_path)
        exists = path.is_file()
        required_rows.append({
            "path": str(path.resolve()),
            "exists": exists,
            "sha256": sha256_file(path) if exists else None,
        })
        if not exists:
            missing.append(str(path))
    if status == "success" and (exit_code != 0 or missing):
        raise ValueError(
            "success receipt requires exit_code=0 and all artifacts; "
            f"missing={missing}"
        )
    state = dict(execution_state or {})
    payload = {
        "receipt_schema_version": 1,
        "stage": str(stage),
        "completion_status": status,
        "success": status == "success",
        "started_at_utc": str(started_at_utc),
        "finished_at_utc": str(finished_at_utc),
        "duration_seconds": float(duration_seconds),
        "exit_code": int(exit_code),
        "git_commit": state.get("git_commit"),
        "parent_commit": state.get("parent_commit"),
        "worktree_clean": state.get("worktree_clean"),
        "source_state_sha256": state.get("source_state_sha256"),
        "source_state_algorithm": state.get("source_state_algorithm"),
        "source_state_commit": state.get("source_state_commit"),
        "tracked_source_file_sha256": state.get(
            "tracked_source_file_sha256", {}
        ),
        "tracked_source_git_blob_oid": state.get(
            "tracked_source_git_blob_oid", {}
        ),
        "worktree_source_state_sha256": state.get(
            "worktree_source_state_sha256"
        ),
        "worktree_source_state_algorithm": state.get(
            "worktree_source_state_algorithm"
        ),
        "worktree_source_file_sha256": state.get(
            "worktree_source_file_sha256", {}
        ),
        "environment": runtime_environment(),
        "output_paths": [str(Path(path).resolve()) for path in output_paths],
        "required_artifacts": required_rows,
        **dict(details or {}),
    }
    return payload


def write_completion_receipt(
    path: str | Path,
    **kwargs: Any,
) -> dict:
    payload = build_completion_receipt(**kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def raw_data_catalog(
    root: str | Path,
    *,
    include_sha256: bool = True,
) -> list[dict]:
    """Record size/mtime and, by default, content hashes for raw data files."""
    source = Path(root)
    if not source.exists():
        return []
    files = [
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in RAW_DATA_SUFFIXES
    ]
    rows = []
    for path in sorted(files):
        stat = path.stat()
        rows.append({
            "path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": sha256_file(path) if include_sha256 else None,
        })
    return rows


def write_provenance_receipt(
    path: str | Path,
    *,
    config: Mapping,
    cache_key: Mapping,
    raw_files: Iterable[Mapping],
    artifacts: Mapping[str, str | Path],
    task_manifests: Iterable[str | Path] = (),
    repo_root: str | Path = ".",
    dataset_role: Optional[str] = None,
    dataset_fingerprint: Optional[str] = None,
    split_access: Optional[Mapping[str, bool]] = None,
    source_paths: Sequence[str] = DEFAULT_EXECUTION_SOURCE_PATHS,
    execution: Optional[Mapping[str, Any]] = None,
) -> dict:
    artifact_rows = {}
    for name, artifact_path in artifacts.items():
        source = Path(artifact_path)
        artifact_rows[name] = {
            "path": str(source.resolve()),
            "sha256": sha256_file(source) if source.exists() else None,
        }
    manifest_rows = []
    for manifest_path in task_manifests:
        source = Path(manifest_path)
        manifest_rows.append({
            "path": str(source.resolve()),
            "sha256": sha256_file(source) if source.exists() else None,
        })
    worktree = git_worktree_state(repo_root)
    source_state = tracked_source_state(
        worktree["repository_root"],
        source_paths=source_paths,
        commit=worktree["git_commit"],
    )
    worktree_diagnostic = worktree_source_state(
        worktree["repository_root"], source_paths=source_paths
    )
    payload = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **worktree,
        **source_state,
        **worktree_diagnostic,
        "config_id": config_id(config),
        "effective_config_sha256": canonical_sha256(config),
        "cache_key": dict(cache_key),
        "dataset_role": dataset_role,
        "dataset_fingerprint": dataset_fingerprint,
        "split_access": dict(split_access or {}),
        "environment": runtime_environment(),
        "execution": dict(execution or {}),
        "raw_data_files": [dict(row) for row in raw_files],
        "artifacts": artifact_rows,
        "task_manifests": manifest_rows,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload
