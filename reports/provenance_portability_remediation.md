# Provenance source-state portability remediation

## Scope

This change is limited to provenance hashing, receipt metadata, CPU tests, and
documentation. It does not alter model code, MetaOpt behavior, configuration,
data processing, training objectives, experiment grids, or evaluation gates.

Parent baseline:
`35b2ee2266d81a0d406801110e085e7b9271eb64`.

## Confirmed root cause

The previous `tracked_source_state()` hashed raw bytes from the checked-out
worktree. The same clean commit therefore produced different values under
different Windows line-ending policies:

- development checkout source-state:
  `26157080f989fa764a8b14293a3aaca47baafe81f308870eb774b3d29c6fb36d`;
- server checkout with `core.autocrlf=true`:
  `04a7a7ca0944f1e594afeef965356000fb50c41cb6fbb75b851799314158cb67`;
- commit-blob/EOL-normalized reconciliation:
  `52aa1bafcd03c84c9c423f23ae481247c4bfc79224379748892ac9ab8e50222c`.

All 20 server mismatches were proven to be line-ending-only. No source content
or commit mismatch was found. The experiment output root did not exist.

## Portable gating algorithm

Formal source-state now uses
`git-commit-blob-path-content-sha256`, version 1:

1. resolve the exact gating commit;
2. sort the declared repository-relative source paths;
3. resolve each `commit:path` to its Git blob OID;
4. read the blob directly from the Git object database;
5. compute SHA-256 of the blob content;
6. construct the mapping
   `path -> {git_blob_oid, blob_sha256}`;
7. hash its canonical JSON representation with SHA-256.

Both the path and blob identity/content participate in the hash. Checkout
bytes, CRLF conversion, `core.autocrlf`, and operating system behavior do not.

The existing commit and clean-worktree checks remain mandatory and execute
before source-state acceptance. A dirty worktree is still rejected.

## Diagnostic worktree state

Raw checked-out file bytes are retained under separate fields:

- `worktree_source_state_sha256`;
- `worktree_source_file_sha256`;
- `worktree_source_state_algorithm`.

Their algorithm is `worktree-path-bytes-sha256`, version 1, explicitly marked:

```text
cross_platform=false
gating=false
diagnostic_only=true
```

These values may differ across LF/CRLF checkouts and cannot authorize an
experiment.

## Receipt changes

Completion, training, manifest, evaluation, validation-audit, frozen-test,
PowerShell stage/run, and Adam validation provenance can now record:

- `source_state_sha256`;
- `source_state_algorithm`;
- `source_state_commit`;
- `tracked_source_file_sha256` (Git blob content);
- `tracked_source_git_blob_oid`;
- diagnostic `worktree_source_state_sha256`;
- diagnostic `worktree_source_state_algorithm`;
- diagnostic `worktree_source_file_sha256`.

Compatibility names used by existing Adam receipt readers are retained, but
now refer to commit-blob SHA-256 values.

## CPU evidence

Tests cover:

- identical commit source-state for LF and CRLF worktree representations;
- differing diagnostic worktree hashes for those representations;
- dirty-worktree rejection;
- source-state change after a committed blob-content change;
- historical commit selection reproducing the original source-state;
- algorithm and gating/diagnostic markers in receipts;
- Adam receipt propagation of the new fields.

No GPU, dataset pipeline, checkpoint, validation, or test was run while
performing this remediation.

