"""Explicit enrollment and transactional, immutable registry snapshots."""

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from .contracts import ExecutionError, PipelineError
from .identity_data import (
    RECOGNIZER,
    embedding_rows,
    embedding_summary,
    validate_snapshot,
    vector,
)
from .pipeline import canonical_json
from .storage import UserStorage, digest_file


class IdentityRegistry:
    def __init__(self, storage: UserStorage):
        self.storage = storage
        storage.initialize()
        self.database = storage.path("state/identities.sqlite")
        self.database.touch(mode=0o600, exist_ok=True)
        if os.name != "nt":
            self.database.chmod(0o600)
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    registry_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    recognizer TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS identities (
                    id TEXT PRIMARY KEY, alias TEXT UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS face_references (
                    identity_id TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
                    vector TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                    track_id TEXT NOT NULL, frame_index INTEGER NOT NULL,
                    UNIQUE(identity_id, source_sha256, track_id, frame_index)
                );
            """)
            connection.execute(
                "INSERT OR IGNORE INTO metadata VALUES (1, ?, 0, ?)",
                (str(uuid.uuid4()), canonical_json(RECOGNIZER)),
            )
            if connection.execute("SELECT recognizer FROM metadata").fetchone()[
                0
            ] != canonical_json(RECOGNIZER):
                raise ExecutionError("Registry uses an incompatible recognition model")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        # Recheck storage guards on every operation, not just construction.
        connection = sqlite3.connect(
            self.storage.path("state/identities.sqlite"), timeout=10
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA secure_delete = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def list_identities(self) -> dict:
        with self.connect() as connection:
            connection.execute("BEGIN")
            metadata = connection.execute(
                "SELECT registry_id, revision FROM metadata"
            ).fetchone()
            items = connection.execute(
                "SELECT i.id AS identity_id, i.alias, count(r.vector) AS reference_count FROM identities i LEFT JOIN face_references r ON i.id = r.identity_id GROUP BY i.id ORDER BY i.id"
            ).fetchall()
            return dict(metadata) | {"identities": [dict(item) for item in items]}

    def enroll(
        self,
        frames_path,
        summary_path,
        track_id,
        *,
        alias=None,
        identity_id=None,
        min_similarity=0.65,
    ):
        if (
            type(min_similarity) not in (int, float)
            or not np.isfinite(min_similarity)
            or not 0.5 <= min_similarity <= 1
        ):
            raise PipelineError("Enrollment similarity must be finite and in [0.5, 1]")
        if alias is not None and (
            not isinstance(alias, str) or not alias.strip() or len(alias) > 128
        ):
            raise PipelineError("Alias must be nonempty and at most 128 characters")
        alias = alias.strip() if alias is not None else None
        if identity_id is not None and alias is not None:
            raise PipelineError("Use either an existing identity ID or a new alias")
        summary = embedding_summary(summary_path, frames_path)
        references = []
        first = None
        count = 0
        # Validate the complete artifact; reject mixed tracks instead of taking an average.
        for row in embedding_rows(frames_path, summary):
            for face in row["faces"]:
                if face["track_id"] != track_id:
                    continue
                if face["association"] == "ambiguous":
                    raise ExecutionError(
                        "Cannot enroll an ambiguously associated track"
                    )
                if face["embedding_status"] != "usable":
                    continue
                features = vector(face["embedding"])
                if first is None:
                    first = features
                if (
                    any(
                        float(features @ reference[1]) < min_similarity
                        for reference in references
                    )
                    or float(features @ first) < min_similarity
                ):
                    raise ExecutionError(
                        "Track contains incompatible references; select a coherent track"
                    )
                reference = (row["frame_index"], features)
                if len(references) < 16:
                    references.append(reference)
                else:
                    references[-1] = reference
                count += 1
        if count < 3:
            raise ExecutionError(
                "Enrollment requires at least three usable observations of this track"
            )
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if identity_id is None:
                identity_id = str(uuid.uuid4())
                try:
                    connection.execute(
                        "INSERT INTO identities (id, alias) VALUES (?, ?)",
                        (identity_id, alias),
                    )
                except sqlite3.IntegrityError as exc:
                    raise PipelineError(
                        "Alias already exists; specify its identity ID to add references"
                    ) from exc
            elif (
                connection.execute(
                    "SELECT id FROM identities WHERE id = ?", (identity_id,)
                ).fetchone()
                is None
            ):
                raise PipelineError("Identity does not exist")
            existing = connection.execute(
                "SELECT vector FROM face_references WHERE identity_id = ?",
                (identity_id,),
            ).fetchall()
            old_vectors = [vector(json.loads(item[0])) for item in existing]
            # Check every observation against the existing identity, including unsaved observations.
            for row in embedding_rows(frames_path, summary):
                for face in row["faces"]:
                    if (
                        face["track_id"] == track_id
                        and face["embedding_status"] == "usable"
                    ):
                        features = vector(face["embedding"])
                        if any(
                            float(features @ old) < min_similarity
                            for old in old_vectors
                        ):
                            raise ExecutionError(
                                "References are incompatible with the existing identity"
                            )
            inserted = 0
            for frame_index, features in references:
                inserted += connection.execute(
                    "INSERT OR IGNORE INTO face_references VALUES (?, ?, ?, ?, ?)",
                    (
                        identity_id,
                        canonical_json(features.tolist()),
                        summary["source"]["sha256"],
                        track_id,
                        frame_index,
                    ),
                ).rowcount
            total = len(existing) + inserted
            if total > 32:
                raise PipelineError(
                    "Identity reference limit is 32; no references were added"
                )
            if inserted:
                connection.execute("UPDATE metadata SET revision = revision + 1")
            if summary["file_sha256"]["frames"] != digest_file(frames_path)[0]:
                raise ExecutionError("Embeddings changed during enrollment")
            revision = connection.execute("SELECT revision FROM metadata").fetchone()[0]
        return {
            "identity_id": identity_id,
            "reference_count": total,
            "added_references": inserted,
            "revision": revision,
        }

    def snapshot(self):
        # Snapshot writing and deletion use the same SQLite writer lock: a deletion
        # cannot be followed by a stale snapshot published by a concurrent reader.
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            metadata = connection.execute(
                "SELECT registry_id, revision FROM metadata"
            ).fetchone()
            identities = []
            for item in connection.execute(
                "SELECT id, alias FROM identities ORDER BY id"
            ):
                references = [
                    json.loads(row[0])
                    for row in connection.execute(
                        "SELECT vector FROM face_references WHERE identity_id = ? ORDER BY source_sha256, track_id, frame_index",
                        (item["id"],),
                    )
                ]
                identities.append(
                    {
                        "identity_id": item["id"],
                        "alias": item["alias"],
                        "references": references,
                    }
                )
            snapshot = validate_snapshot(
                {
                    "schema_version": 1,
                    **dict(metadata),
                    "recognizer": RECOGNIZER,
                    "identities": identities,
                }
            )
            content = canonical_json(snapshot).encode()
            digest = hashlib.sha256(content).hexdigest()
            directory = self.storage.path("artifacts/identity_registry")
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = self.storage.path(f"artifacts/identity_registry/{digest}.json")
            if not path.exists() or digest_file(path)[0] != digest:
                temporary = self.storage.path(
                    f"artifacts/identity_registry/{uuid.uuid4().hex}.tmp"
                )
                try:
                    with temporary.open("xb") as stream:
                        if os.name != "nt":
                            os.fchmod(stream.fileno(), 0o600)
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            elif os.name != "nt":
                path.chmod(0o600)
            return path

    def delete(self, identity_id):
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not connection.execute(
                "DELETE FROM identities WHERE id = ?", (identity_id,)
            ).rowcount:
                raise PipelineError("Identity does not exist")
            connection.execute("UPDATE metadata SET revision = revision + 1")
            # Old snapshots contain biometric references even after a DB deletion.
            for relative, pattern in (
                ("artifacts/identity_registry", "*.json"),
                ("artifacts/identity_registry", "*.tmp"),
                ("artifacts/virtual_registry", "*.zip"),
                ("artifacts/virtual_registry", "*.tmp"),
            ):
                directory = self.storage.path(relative)
                if directory.exists():
                    for name in directory.glob(pattern):
                        path = self.storage.path(name.relative_to(self.storage.root))
                        # Reserved snapshot directories can contain old references
                        # or A→B links. Purge malformed snapshots as well.
                        path.unlink()
        return {
            "deleted_identity_id": identity_id,
            "deletion_scope": "registry_and_managed_snapshots",
            "retained": "original_videos, embedding_artifacts, virtual_candidates_and_portrait_exports, temporaries, reports, external_snapshot_copies_and_backups",
        }
