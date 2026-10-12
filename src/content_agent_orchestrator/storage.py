"""User-owned storage, integrity checks, checkpoints and execution locks."""

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .contracts import Artifact, ExecutionError, PipelineError


def digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


class UserStorage:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()

    def path(self, relative: str | Path) -> Path:
        """Reject symlink escapes and traversal before writing or deleting."""
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise PipelineError("Storage paths must be relative without traversal")
        candidate = self.root
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise PipelineError(
                    f"Storage paths must not contain symlinks: {relative}"
                )
        if not candidate.resolve().is_relative_to(self.root):
            raise PipelineError(f"Path escapes the user directory: {relative}")
        return candidate

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in ("artifacts/raw_footages", "temps", "outputs", "state/locks"):
            self.path(name).mkdir(parents=True, exist_ok=True, mode=0o700)
        database = self.path("state/executions.sqlite")
        # Refuse any database symlink, including links to another local file.
        if database.is_symlink():
            raise PipelineError("The execution database must not be a symlink")
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    specification TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    node_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    outputs TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, node_id)
                );
                CREATE INDEX IF NOT EXISTS node_cache ON nodes(fingerprint, status);
                """
            )
        if os.name != "nt":
            database.chmod(0o600)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path("state/executions.sqlite"), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def run_lock(self, run_id: str) -> Iterator[None]:
        lock_path = self.path(f"state/locks/{run_id}.lock")
        if lock_path.is_symlink():
            raise PipelineError("Execution lock must not be a symlink")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(descriptor, "r+b") as stream:
            if os.name == "nt":
                import msvcrt

                if lock_path.stat().st_size == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise ExecutionError("This execution is already active") from exc
            else:
                import fcntl

                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise ExecutionError("This execution is already active") from exc
            try:
                yield
            finally:
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def serialize_artifacts(self, artifacts: dict[str, Artifact]) -> str:
        return json.dumps(
            {
                name: {
                    "path": str(artifact.path.relative_to(self.root)),
                    "kind": artifact.kind,
                    "sha256": artifact.sha256,
                    "size": artifact.size,
                }
                for name, artifact in artifacts.items()
            },
            sort_keys=True,
        )

    def verified_artifacts(
        self, encoded: str, expected_kinds: dict[str, str]
    ) -> dict[str, Artifact] | None:
        try:
            records = json.loads(encoded)
            if not isinstance(records, dict) or set(records) != set(expected_kinds):
                return None
            artifacts = {}
            for name, record in records.items():
                path = self.path(record["path"])
                if path.is_symlink() or not path.is_file():
                    return None
                digest, size = digest_file(path)
                if (
                    record["kind"] != expected_kinds[name]
                    or digest != record["sha256"]
                    or size != record["size"]
                ):
                    return None
                artifacts[name] = Artifact(path, record["kind"], digest, size)
            return artifacts
        except (OSError, TypeError, ValueError, KeyError):
            return None
