"""Run provenance helpers for reproducible, non-overwriting experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional


RAW_DATA_SUFFIXES = {
    ".csv", ".parquet", ".json", ".jsonl", ".txt", ".arff", ".npz", ".npy"
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
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
        "config_id": config_id(config),
        "effective_config_sha256": canonical_sha256(config),
        "cache_key": dict(cache_key),
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

