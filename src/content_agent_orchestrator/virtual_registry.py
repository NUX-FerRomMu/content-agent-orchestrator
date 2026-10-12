"""Generate reviewable B candidates, commit immutable A→B mappings and snapshots."""

import hashlib
import io
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path

from .contracts import ExecutionError, PipelineError
from .face_data import read_json
from .identity_registry import IdentityRegistry
from .pipeline import canonical_json
from .storage import UserStorage, digest_file
from .virtual_catalog import MAX_CATALOG_BYTES
from .virtual_face import (
    appearance_profile,
    portrait_prompt,
    read_portrait,
    source_separation,
    validate_portrait,
)


def private_write(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        if os.name != "nt":
            os.fchmod(stream.fileno(), 0o600)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


class VirtualRegistry:
    def __init__(self, storage: UserStorage):
        self.storage = storage
        self.identities = IdentityRegistry(storage)
        with self.identities.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS virtual_metadata (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1), revision INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO virtual_metadata VALUES (1, 0);
                CREATE TABLE IF NOT EXISTS virtual_faces (
                    identity_id TEXT PRIMARY KEY REFERENCES identities(id) ON DELETE CASCADE,
                    virtual_id TEXT UNIQUE NOT NULL, profile TEXT NOT NULL,
                    portrait BLOB NOT NULL, portrait_sha256 TEXT NOT NULL,
                    validation TEXT NOT NULL, generator TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TRIGGER IF NOT EXISTS virtual_face_deletion AFTER DELETE ON virtual_faces
                BEGIN UPDATE virtual_metadata SET revision = revision + 1; END;
            """)

    @staticmethod
    def _source(connection, identity_id):
        try:
            if str(uuid.UUID(identity_id)) != identity_id:
                raise ValueError
        except (ValueError, TypeError, AttributeError) as exc:
            raise PipelineError("Identity must be a canonical UUID") from exc
        if (
            connection.execute(
                "SELECT id FROM identities WHERE id = ?", (identity_id,)
            ).fetchone()
            is None
        ):
            raise PipelineError("Source identity does not exist")
        return dict(
            connection.execute("SELECT registry_id, revision FROM metadata").fetchone()
        )

    @staticmethod
    def _references(connection):
        return [
            json.loads(item[0])
            for item in connection.execute("SELECT vector FROM face_references")
        ]

    def create(self, identity_id, profile, generator, *, cache="reuse") -> dict:
        if cache not in ("reuse", "fresh"):
            raise PipelineError("Virtual generation cache must be reuse or fresh")
        profile = appearance_profile(profile)
        with self.identities.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            source = self._source(connection, identity_id)
            existing = connection.execute(
                "SELECT * FROM virtual_faces WHERE identity_id = ?", (identity_id,)
            ).fetchone()
            if existing is not None:
                if json.loads(existing["profile"]) != profile:
                    raise PipelineError(
                        "This identity already has an immutable virtual face with another profile"
                    )
                source_separation(
                    validate_portrait(existing["portrait"])["embedding"],
                    self._references(connection),
                )
                return self._export(existing) | {"status": "approved", "reused": True}
        request = {
            "schema_version": 1,
            "registry_id": source["registry_id"],
            "identity_id": identity_id,
            "profile": profile,
            "generator": generator.descriptor,
            "prompt_version": 1,
        }
        candidate_id = hashlib.sha256(canonical_json(request).encode()).hexdigest()
        work = self.storage.path(f"temps/virtual_generation/{candidate_id}")
        destination = self.storage.path(f"artifacts/virtual_candidates/{candidate_id}")
        with self.storage.run_lock(f"virtual-{candidate_id}"):
            if cache == "fresh" and work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True, exist_ok=True, mode=0o700)
            portrait = self.storage.path(
                work.relative_to(self.storage.root) / "portrait.png"
            )
            checkpoint = self.storage.path(
                work.relative_to(self.storage.root) / "generation.json"
            )
            if destination.exists():
                saved = read_json(
                    self.storage.path(
                        destination.relative_to(self.storage.root) / "candidate.json"
                    )
                )
                if saved.get("request") != request:
                    raise ExecutionError(
                        "Candidate manifest does not match its request"
                    )
                content = read_portrait(
                    self.storage.path(
                        destination.relative_to(self.storage.root) / "portrait.png"
                    )
                )
                if hashlib.sha256(content).hexdigest() != self._candidate_digest(saved):
                    raise ExecutionError("Candidate portrait checksum mismatch")
            elif checkpoint.exists():
                saved = read_json(checkpoint)
                content = read_portrait(portrait)
                if saved != {
                    "request": request,
                    "portrait_sha256": hashlib.sha256(content).hexdigest(),
                }:
                    raise ExecutionError(
                        "Generation cache checksum or request mismatch; inspect retained temporaries"
                    )
            else:
                prompt_builder = getattr(generator, "prompt", portrait_prompt)
                content = generator.generate(prompt_builder(profile), work)
                # An incomplete image or missing journal never counts as reusable generation.
                portrait.unlink(missing_ok=True)
                private_write(portrait, content)
                checkpoint.unlink(missing_ok=True)
                private_write(
                    checkpoint,
                    canonical_json(
                        {
                            "request": request,
                            "portrait_sha256": hashlib.sha256(content).hexdigest(),
                        }
                    ).encode(),
                )
            validation = validate_portrait(content)
            with self.identities.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = self._source(connection, identity_id)
                if current["registry_id"] != source["registry_id"]:
                    raise ExecutionError("Source registry changed during generation")
                similarity = source_separation(
                    validation["embedding"], self._references(connection)
                )
                if not destination.exists():
                    staging = self.storage.path(
                        f"artifacts/virtual_candidates/.{uuid.uuid4().hex}"
                    )
                    staging.mkdir(parents=True, mode=0o700)
                    try:
                        private_write(staging / "portrait.png", content)
                        private_write(
                            staging / "candidate.json",
                            canonical_json(
                                {
                                    "candidate_id": candidate_id,
                                    "request": request,
                                    "validation": validation,
                                    "max_source_similarity": similarity,
                                }
                            ).encode(),
                        )
                        destination.parent.mkdir(
                            parents=True, exist_ok=True, mode=0o700
                        )
                        staging.replace(destination)
                    finally:
                        if staging.exists():
                            shutil.rmtree(staging)
            shutil.rmtree(work)
        return {
            "status": "needs_visual_review",
            "candidate_id": candidate_id,
            "portrait_path": str(destination / "portrait.png"),
            "identity_id": identity_id,
            "profile": profile,
            "max_source_similarity": similarity,
            "required_review": "Clearly adult, fictional generation, no eyewear or eccentric facial traits, gender/skin match the source profile",
        }

    def approve(self, identity_id, candidate_id, *, reviewed=False) -> dict:
        if reviewed is not True:
            raise PipelineError(
                "Review the candidate image and pass --reviewed before assigning B"
            )
        if (
            not isinstance(candidate_id, str)
            or len(candidate_id) != 64
            or any(character not in "0123456789abcdef" for character in candidate_id)
        ):
            raise PipelineError("Invalid virtual candidate ID")
        directory = self.storage.path(f"artifacts/virtual_candidates/{candidate_id}")
        candidate = read_json(
            self.storage.path(
                directory.relative_to(self.storage.root) / "candidate.json"
            )
        )
        request = candidate.get("request")
        if (
            not isinstance(request, dict)
            or hashlib.sha256(canonical_json(request).encode()).hexdigest()
            != candidate_id
            or request.get("identity_id") != identity_id
            or request.get("prompt_version") != 1
        ):
            raise ExecutionError(
                "Candidate does not belong to this source and generation request"
            )
        if (
            set(request)
            != {
                "schema_version",
                "registry_id",
                "identity_id",
                "profile",
                "generator",
                "prompt_version",
            }
            or request["schema_version"] != 1
            or not isinstance(request["generator"], dict)
        ):
            raise ExecutionError("Invalid candidate generation request")
        profile = appearance_profile(request["profile"])
        content = read_portrait(
            self.storage.path(directory.relative_to(self.storage.root) / "portrait.png")
        )
        if hashlib.sha256(content).hexdigest() != self._candidate_digest(candidate):
            raise ExecutionError("Candidate portrait checksum mismatch")
        validation = validate_portrait(content)
        with self.identities.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            source = self._source(connection, identity_id)
            if request["registry_id"] != source["registry_id"]:
                raise ExecutionError("Candidate belongs to another source registry")
            source_separation(validation["embedding"], self._references(connection))
            existing = connection.execute(
                "SELECT * FROM virtual_faces WHERE identity_id = ?", (identity_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["portrait_sha256"] != validation["portrait_sha256"]
                    or json.loads(existing["profile"]) != profile
                ):
                    raise PipelineError(
                        "Source already has a different immutable virtual face; no replacement performed"
                    )
                return self._export(existing) | {"status": "approved", "reused": True}
            virtual_id = str(uuid.uuid4())
            connection.execute(
                "INSERT INTO virtual_faces (identity_id, virtual_id, profile, portrait, portrait_sha256, validation, generator) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    identity_id,
                    virtual_id,
                    canonical_json(profile),
                    content,
                    validation["portrait_sha256"],
                    canonical_json(validation),
                    canonical_json(request["generator"]),
                ),
            )
            connection.execute("UPDATE virtual_metadata SET revision = revision + 1")
            item = connection.execute(
                "SELECT * FROM virtual_faces WHERE identity_id = ?", (identity_id,)
            ).fetchone()
            result = self._export(item)
        return result | {"status": "approved", "reused": False}

    def _export(self, item):
        if hashlib.sha256(item["portrait"]).hexdigest() != item["portrait_sha256"]:
            raise ExecutionError("Stored virtual portrait checksum mismatch")
        directory = self.storage.path(f"artifacts/virtual_faces/{item['virtual_id']}")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        portrait = self.storage.path(
            directory.relative_to(self.storage.root) / "portrait.png"
        )
        if portrait.exists():
            if digest_file(portrait)[0] != item["portrait_sha256"]:
                raise ExecutionError(
                    "Exported virtual portrait checksum mismatch; no overwrite performed"
                )
        else:
            private_write(portrait, item["portrait"])
        return {
            "identity_id": item["identity_id"],
            "virtual_id": item["virtual_id"],
            "profile": json.loads(item["profile"]),
            "portrait_sha256": item["portrait_sha256"],
            "portrait_path": str(portrait),
        }

    @staticmethod
    def _candidate_digest(candidate):
        validation = candidate.get("validation")
        if not isinstance(validation, dict) or not isinstance(
            validation.get("portrait_sha256"), str
        ):
            raise ExecutionError("Invalid candidate validation metadata")
        return validation["portrait_sha256"]

    def list_faces(self) -> dict:
        with self.identities.connect() as connection:
            connection.execute("BEGIN")
            revision = connection.execute(
                "SELECT revision FROM virtual_metadata"
            ).fetchone()[0]
            faces = [
                {
                    "identity_id": item["identity_id"],
                    "virtual_id": item["virtual_id"],
                    "profile": json.loads(item["profile"]),
                    "portrait_sha256": item["portrait_sha256"],
                }
                for item in connection.execute(
                    "SELECT identity_id, virtual_id, profile, portrait_sha256 FROM virtual_faces ORDER BY identity_id"
                )
            ]
        return {"revision": revision, "faces": faces}

    def snapshot(self) -> Path:
        with self.identities.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            metadata = dict(
                connection.execute(
                    "SELECT registry_id, revision AS source_revision FROM metadata"
                ).fetchone()
            )
            revision = connection.execute(
                "SELECT revision FROM virtual_metadata"
            ).fetchone()[0]
            references = self._references(connection)
            entries, portraits = [], {}
            total_bytes = 0
            for item in connection.execute(
                "SELECT * FROM virtual_faces ORDER BY identity_id"
            ):
                validation = validate_portrait(item["portrait"])
                if (
                    hashlib.sha256(item["portrait"]).hexdigest()
                    != item["portrait_sha256"]
                ):
                    raise ExecutionError("Stored virtual portrait checksum mismatch")
                source_separation(validation["embedding"], references)
                name = f"portraits/{item['virtual_id']}.png"
                entries.append(
                    {
                        "identity_id": item["identity_id"],
                        "virtual_id": item["virtual_id"],
                        "profile": json.loads(item["profile"]),
                        "portrait_sha256": item["portrait_sha256"],
                        "portrait_file": name,
                        "reviewed": True,
                    }
                )
                portraits[name] = item["portrait"]
                total_bytes += len(item["portrait"]) + 256
                if total_bytes > MAX_CATALOG_BYTES:
                    raise ExecutionError("Virtual catalog exceeds the 128 MiB limit")
            manifest = {
                "schema_version": 1,
                **metadata,
                "revision": revision,
                "faces": entries,
            }
            if (
                total_bytes + len(canonical_json(manifest).encode()) + 256
                > MAX_CATALOG_BYTES
            ):
                raise ExecutionError("Virtual catalog exceeds the 128 MiB limit")
            # Store image bytes in the input artifact: caches must hash B's pixels,
            # not mutable file paths hidden inside a metadata-only snapshot.
            buffer = io.BytesIO()
            with zipfile.ZipFile(
                buffer, "w", compression=zipfile.ZIP_STORED
            ) as archive:
                archive.writestr(
                    zipfile.ZipInfo("manifest.json"), canonical_json(manifest)
                )
                for name, content in portraits.items():
                    archive.writestr(zipfile.ZipInfo(name), content)
            content = buffer.getvalue()
            digest = hashlib.sha256(content).hexdigest()
            directory = self.storage.path("artifacts/virtual_registry")
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = self.storage.path(f"artifacts/virtual_registry/{digest}.zip")
            if not path.exists():
                temporary = self.storage.path(
                    f"artifacts/virtual_registry/.{uuid.uuid4().hex}.tmp"
                )
                try:
                    private_write(temporary, content)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            elif digest_file(path)[0] != digest:
                raise ExecutionError("Virtual catalog checksum mismatch")
            return path
