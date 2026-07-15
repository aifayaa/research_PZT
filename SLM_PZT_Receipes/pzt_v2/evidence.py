from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "pzt-evidence/v1"


class EvidenceValidationError(RuntimeError):
    """Raised when evidence is missing, stale, failed, or has been modified."""


@dataclass(frozen=True)
class ArtifactDigest:
    path: str
    sha256: str
    size_bytes: int
    kind: str = "file"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactDigest":
        return cls(
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            size_bytes=int(data["size_bytes"]),
            kind=str(data.get("kind", "file")),
        )


@dataclass(frozen=True)
class EvidenceManifest:
    schema_version: str
    run_id: str
    stage: str
    verdict: str
    code_commit: str
    code_dirty: bool
    python_version: str
    platform: str
    command: List[str]
    configuration: Dict[str, Any]
    configuration_sha256: str
    inputs: List[ArtifactDigest]
    upstream_manifests: List[ArtifactDigest]
    outputs: List[ArtifactDigest]
    metrics: Dict[str, Any]
    thresholds: Dict[str, Any]
    failure_reasons: List[str]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceManifest":
        return cls(
            schema_version=str(data["schema_version"]),
            run_id=str(data["run_id"]),
            stage=str(data["stage"]),
            verdict=str(data["verdict"]),
            code_commit=str(data["code_commit"]),
            code_dirty=bool(data.get("code_dirty", False)),
            python_version=str(data["python_version"]),
            platform=str(data["platform"]),
            command=[str(item) for item in data.get("command", [])],
            configuration=dict(data.get("configuration") or {}),
            configuration_sha256=str(data["configuration_sha256"]),
            inputs=[ArtifactDigest.from_dict(item) for item in data.get("inputs", [])],
            upstream_manifests=[ArtifactDigest.from_dict(item) for item in data.get("upstream_manifests", [])],
            outputs=[ArtifactDigest.from_dict(item) for item in data.get("outputs", [])],
            metrics=dict(data.get("metrics") or {}),
            thresholds=dict(data.get("thresholds") or {}),
            failure_reasons=[str(item) for item in data.get("failure_reasons", [])],
            created_at=str(data["created_at"]),
        )


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_digest(path: Path, *, root: Path | None = None) -> ArtifactDigest:
    path = path.resolve()
    if not path.exists():
        raise EvidenceValidationError(f"artifact does not exist: {path}")
    display_path = _display_path(path, root=root)
    if path.is_dir():
        digest = hashlib.sha256()
        total_size = 0
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            relative = item.relative_to(path).as_posix()
            item_hash, item_size = _hash_file(item)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(item_hash.encode("ascii"))
            digest.update(b"\0")
            total_size += item_size
        return ArtifactDigest(display_path, digest.hexdigest(), total_size, kind="directory")
    item_hash, item_size = _hash_file(path)
    return ArtifactDigest(display_path, item_hash, item_size)


def create_manifest(
    *,
    run_id: str,
    stage: str,
    inputs: Sequence[Path],
    outputs: Sequence[Path],
    configuration: Mapping[str, Any],
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    failure_reasons: Iterable[str] = (),
    upstream_manifests: Sequence[Path] = (),
    root: Path | None = None,
    command: Sequence[str] | None = None,
) -> EvidenceManifest:
    root = (root or Path.cwd()).resolve()
    failures = list(failure_reasons)
    for upstream_path in upstream_manifests:
        upstream = load_manifest(upstream_path)
        if upstream.verdict != "PASS":
            failures.append(f"upstream_not_passed:{upstream.stage}:{upstream.verdict}")
        try:
            verify_manifest(upstream_path, root=root, require_pass=True)
        except EvidenceValidationError as exc:
            failures.append(f"invalid_upstream:{upstream.stage}:{exc}")
    commit, dirty = git_state(root)
    config = dict(configuration)
    return EvidenceManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        stage=stage,
        verdict="FAIL" if failures else "PASS",
        code_commit=commit,
        code_dirty=dirty,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        command=list(command if command is not None else sys.argv),
        configuration=config,
        configuration_sha256=canonical_json_sha256(config),
        inputs=[artifact_digest(path, root=root) for path in inputs],
        upstream_manifests=[artifact_digest(path, root=root) for path in upstream_manifests],
        outputs=[artifact_digest(path, root=root) for path in outputs],
        metrics=dict(metrics),
        thresholds=dict(thresholds),
        failure_reasons=failures,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def write_manifest(manifest: EvidenceManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_manifest(path: Path) -> EvidenceManifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvidenceValidationError(f"cannot load manifest {path}: {exc}") from exc
    manifest = EvidenceManifest.from_dict(data)
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise EvidenceValidationError(
            f"unsupported manifest schema {manifest.schema_version!r}; expected {MANIFEST_SCHEMA_VERSION!r}"
        )
    return manifest


def verify_manifest(path: Path, *, root: Path | None = None, require_pass: bool = True) -> EvidenceManifest:
    root = (root or Path.cwd()).resolve()
    manifest = load_manifest(path)
    if require_pass and manifest.verdict != "PASS":
        raise EvidenceValidationError(
            f"manifest {manifest.stage} has verdict {manifest.verdict}: {manifest.failure_reasons}"
        )
    if canonical_json_sha256(manifest.configuration) != manifest.configuration_sha256:
        raise EvidenceValidationError(f"configuration hash mismatch for stage {manifest.stage}")
    for artifact in [*manifest.inputs, *manifest.upstream_manifests, *manifest.outputs]:
        actual = artifact_digest(_resolve_artifact_path(artifact.path, root), root=root)
        if actual.sha256 != artifact.sha256 or actual.size_bytes != artifact.size_bytes:
            raise EvidenceValidationError(f"artifact hash mismatch: {artifact.path}")
    return manifest


def git_state(root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except Exception:
        return "unknown", True


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _display_path(path: Path, *, root: Path | None) -> str:
    if root is not None:
        try:
            return path.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return str(path)


def _resolve_artifact_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
