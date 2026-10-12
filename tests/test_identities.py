import copy
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np

from content_agent_orchestrator.contracts import (
    Artifact,
    ExecutionError,
    ModuleContext,
    PipelineError,
)
from content_agent_orchestrator.face_data import iter_rows, read_json, write_json
from content_agent_orchestrator.identity_data import (
    RECOGNIZER,
    embedding_rows,
    embedding_summary,
    validate_snapshot,
)
from content_agent_orchestrator.identity_registry import IdentityRegistry
from content_agent_orchestrator.modules import (
    FaceEmbedding,
    IdentityReport,
    IdentityResolution,
)
from content_agent_orchestrator.storage import UserStorage, digest_file


def features(index):
    values = [0.0] * 128
    values[index] = 1.0
    return values


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.storage = UserStorage(self.root / "user")
        self.registry = IdentityRegistry(self.storage)
        self.result = self.root / "result"
        self.result.mkdir()
        self.context = ModuleContext(self.root, self.result)

    def embeddings(self, vectors=None, *, edit=None, source="a"):
        if vectors is None:
            vectors = [features(0)] * 4
        frames = self.root / "embeddings.jsonl"
        rows = []
        for index, value in enumerate(vectors):
            face = {
                "bbox": [20, 20, 80, 80],
                "landmarks": [[40, 40], [80, 40], [60, 60], [40, 80], [80, 80]],
                "score": 0.95,
                "track_id": "track-000001",
                "association": "new" if index == 0 else "matched",
                "embedding_status": "usable",
                "embedding": copy.deepcopy(value),
            }
            rows.append(
                {
                    "frame_index": index,
                    "presentation_timestamp": index * 10,
                    "presentation_time": str(index / 10),
                    "faces": [face],
                }
            )
        if edit:
            edit(rows)
        frames.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        summary = self.root / "summary.json"
        write_json(
            summary,
            {
                "schema_version": 1,
                "source": {"sha256": source * 64, "size_bytes": 10},
                "video": {
                    "width": 200,
                    "height": 200,
                    "frame_count": len(rows),
                    "time_base": "1/100",
                },
                "recognizer": copy.deepcopy(RECOGNIZER),
                "counts": {
                    "frame_count": len(rows),
                    "face_detection_count": sum(len(row["faces"]) for row in rows),
                    "usable_embedding_count": sum(
                        face["embedding_status"] == "usable"
                        for row in rows
                        for face in row["faces"]
                    ),
                },
                "file_sha256": {"frames": digest_file(frames)[0]},
            },
        )
        return frames, summary

    def enroll(self, **kwargs):
        frames, summary = self.embeddings()
        return self.registry.enroll(frames, summary, "track-000001", **kwargs)

    def resolve(self, frames, summary, *, snapshot=None, parameters=None):
        snapshot = snapshot or self.registry.snapshot()
        inputs = {}
        for name, path, kind in (
            ("frames", frames, "face.embeddings"),
            ("summary", summary, "face.embedding-summary"),
            ("registry", snapshot, "identity.registry-snapshot"),
        ):
            digest, size = digest_file(path)
            inputs[name] = Artifact(path, kind, digest, size)
        outputs = IdentityResolution().execute(inputs, parameters or {}, self.context)
        return (
            read_json(outputs["summary"]),
            list(iter_rows(outputs["frames"])),
            list(iter_rows(outputs["tracks"])),
        )

    def test_enrollment_stable_id_multiple_references_and_idempotence(self):
        first = self.enroll(alias="persona")
        self.assertEqual(first["reference_count"], 4)
        second = self.enroll(identity_id=first["identity_id"])
        self.assertEqual(second["identity_id"], first["identity_id"])
        self.assertEqual(second["added_references"], 0)
        self.assertEqual(second["revision"], first["revision"])
        frames, summary = self.embeddings(source="b")
        added = self.registry.enroll(
            frames, summary, "track-000001", identity_id=first["identity_id"]
        )
        self.assertEqual(added["reference_count"], 8)
        listed = self.registry.list_identities()
        self.assertEqual(
            listed["identities"],
            [
                {
                    "identity_id": first["identity_id"],
                    "alias": "persona",
                    "reference_count": 8,
                }
            ],
        )
        self.assertNotIn("vector", json.dumps(listed))

    def test_snapshot_is_immutable_deterministic_and_revisioned(self):
        before = self.registry.snapshot()
        original = before.read_bytes()
        self.assertEqual(before, self.registry.snapshot())
        self.enroll()
        after = self.registry.snapshot()
        self.assertNotEqual(after, before)
        self.assertEqual(before.read_bytes(), original)
        self.assertEqual(read_json(after)["revision"], 1)
        reloaded = IdentityRegistry(self.storage)
        self.assertEqual(reloaded.snapshot(), after)

    def test_mixed_track_is_rejected_without_registry_mutation(self):
        frames, summary = self.embeddings([features(0)] * 3 + [features(1)])
        with self.assertRaisesRegex(ExecutionError, "incompatible references"):
            self.registry.enroll(frames, summary, "track-000001", alias="mixed")
        self.assertEqual(self.registry.list_identities()["revision"], 0)
        self.assertEqual(self.registry.list_identities()["identities"], [])

    def test_incompatible_addition_rolls_back(self):
        first = self.enroll()
        frames, summary = self.embeddings([features(1)] * 4, source="b")
        with self.assertRaisesRegex(ExecutionError, "existing identity"):
            self.registry.enroll(
                frames, summary, "track-000001", identity_id=first["identity_id"]
            )
        self.assertEqual(self.registry.list_identities()["revision"], 1)
        self.assertEqual(
            self.registry.list_identities()["identities"][0]["reference_count"], 4
        )

    def test_incompatible_unsaved_observation_also_rejected(self):
        first = self.enroll()
        # Every new observation is coherent with the first, but one is not
        # coherent with the registered identity. It is beyond the first 16.
        base = np.zeros(128)
        base[0] = 0.8
        base[1] = 0.6
        outlier = np.zeros(128)
        outlier[0] = 0.5
        outlier[1] = np.sqrt(0.75)
        frames, summary = self.embeddings(
            [base.tolist()] * 18 + [outlier.tolist()] + [base.tolist()], source="b"
        )
        with self.assertRaisesRegex(ExecutionError, "existing identity"):
            self.registry.enroll(
                frames, summary, "track-000001", identity_id=first["identity_id"]
            )
        self.assertEqual(self.registry.list_identities()["revision"], 1)

    def test_enrollment_rejects_short_and_ambiguous_tracks(self):
        frames, summary = self.embeddings([features(0)] * 2)
        with self.assertRaisesRegex(ExecutionError, "three usable"):
            self.registry.enroll(frames, summary, "track-000001")
        frames, summary = self.embeddings(
            edit=lambda rows: rows[2]["faces"][0].update(association="ambiguous")
        )
        with self.assertRaisesRegex(ExecutionError, "ambiguously"):
            self.registry.enroll(frames, summary, "track-000001")

    def test_duplicate_alias_and_nonexistent_target_do_not_change_registry(self):
        self.enroll(alias="same")
        with self.assertRaisesRegex(PipelineError, "Alias already"):
            self.enroll(alias="same")
        with self.assertRaisesRegex(PipelineError, "does not exist"):
            self.enroll(identity_id=str(uuid.uuid4()))
        self.assertEqual(len(self.registry.list_identities()["identities"]), 1)
        self.assertEqual(self.registry.list_identities()["revision"], 1)

    def test_delete_cascades_references_and_removes_owned_snapshots(self):
        empty = self.registry.snapshot()
        identity = self.enroll()
        populated = self.registry.snapshot()
        external = self.root / "external.json"
        external.write_bytes(populated.read_bytes())
        result = self.registry.delete(identity["identity_id"])
        self.assertEqual(result["deletion_scope"], "registry_and_managed_snapshots")
        self.assertFalse(empty.exists())
        self.assertFalse(populated.exists())
        self.assertTrue(external.exists())
        with self.registry.connect() as connection:
            self.assertEqual(
                connection.execute("SELECT count(*) FROM face_references").fetchone()[
                    0
                ],
                0,
            )
            self.assertEqual(
                connection.execute("PRAGMA secure_delete").fetchone()[0], 1
            )
        self.assertEqual(read_json(self.registry.snapshot())["identities"], [])
        self.assertEqual(self.registry.list_identities()["revision"], 2)

    def test_corrupt_snapshot_does_not_prevent_deletion(self):
        identity = self.enroll()
        snapshot = self.registry.snapshot()
        snapshot.write_text("corrupt but potentially sensitive")
        self.registry.delete(identity["identity_id"])
        self.assertFalse(snapshot.exists())

    def test_reference_limit_rolls_back_the_complete_addition(self):
        frames, summary = self.embeddings([features(0)] * 16)
        first = self.registry.enroll(frames, summary, "track-000001")
        frames, summary = self.embeddings([features(0)] * 16, source="b")
        self.registry.enroll(
            frames, summary, "track-000001", identity_id=first["identity_id"]
        )
        frames, summary = self.embeddings([features(0)] * 3, source="c")
        with self.assertRaisesRegex(PipelineError, "limit is 32"):
            self.registry.enroll(
                frames, summary, "track-000001", identity_id=first["identity_id"]
            )
        self.assertEqual(self.registry.list_identities()["revision"], 2)
        self.assertEqual(
            self.registry.list_identities()["identities"][0]["reference_count"], 32
        )

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_database_and_snapshot_are_private(self):
        self.enroll()
        self.assertEqual(self.registry.database.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.registry.snapshot().stat().st_mode & 0o777, 0o600)

    def test_registry_and_snapshot_symlinks_are_rejected(self):
        target = self.root / "outside"
        target.write_text("untouched")
        self.registry.database.unlink()
        self.registry.database.symlink_to(target)
        with self.assertRaisesRegex(PipelineError, "symlink"):
            IdentityRegistry(self.storage)
        self.assertEqual(target.read_text(), "untouched")

    def test_recognition_distinguishes_known_and_unknown_observations(self):
        identity = self.enroll(alias="owner")
        frames, summary = self.embeddings([features(0), features(1), features(0)])
        _, rows, tracks = self.resolve(frames, summary)
        states = [row["faces"][0]["identity"] for row in rows]
        self.assertEqual(
            [state["status"] for state in states], ["known", "unknown", "known"]
        )
        self.assertEqual(states[0]["identity_id"], identity["identity_id"])
        self.assertNotIn("identity_id", states[1])
        self.assertEqual(tracks[0]["status"], "uncertain")
        self.assertNotIn('"embedding":', json.dumps(rows))

    def test_identical_references_in_two_identities_remain_ambiguous(self):
        self.enroll(alias="one")
        self.enroll(alias="two")
        frames, summary = self.embeddings()
        _, rows, _ = self.resolve(frames, summary)
        identity = rows[0]["faces"][0]["identity"]
        self.assertEqual(identity["status"], "ambiguous")
        self.assertEqual(len(identity["candidates"]), 2)
        self.assertNotIn("identity_id", identity)

    def test_track_association_ambiguity_prevents_identity_assignment(self):
        self.enroll()
        frames, summary = self.embeddings(
            edit=lambda rows: rows[1]["faces"][0].update(association="ambiguous")
        )
        _, rows, _ = self.resolve(frames, summary)
        self.assertEqual(rows[1]["faces"][0]["identity"]["reason"], "track_association")
        self.assertNotIn("identity_id", rows[1]["faces"][0]["identity"])

    def test_track_can_change_person_without_propagating_identity(self):
        first = self.enroll()
        frames, summary = self.embeddings([features(1)] * 4, source="b")
        second = self.registry.enroll(frames, summary, "track-000001")
        frames, summary = self.embeddings(
            [features(0), features(0), features(1), features(1)]
        )
        result, rows, tracks = self.resolve(frames, summary)
        self.assertEqual(tracks[0]["status"], "mixed")
        self.assertEqual(result["counts"]["mixed_track_count"], 1)
        self.assertEqual(
            tracks[0]["identity_changes"],
            [
                {
                    "frame_index": 2,
                    "presentation_timestamp": 20,
                    "from": first["identity_id"],
                    "to": second["identity_id"],
                }
            ],
        )
        self.assertEqual(
            rows[2]["faces"][0]["identity"]["identity_id"], second["identity_id"]
        )

    def test_empty_registry_and_unusable_observations(self):
        def unusable(rows):
            rows[1]["faces"][0].pop("embedding")
            rows[1]["faces"][0]["embedding_status"] = "small_face"

        frames, summary = self.embeddings(edit=unusable)
        _, rows, _ = self.resolve(frames, summary)
        self.assertEqual(rows[0]["faces"][0]["identity"]["status"], "unknown")
        self.assertEqual(rows[1]["faces"][0]["identity"]["status"], "unusable")

    def test_wrong_summary_hash_and_model_are_rejected(self):
        frames, summary = self.embeddings()
        frames.write_text(frames.read_text() + "\n")
        with self.assertRaisesRegex(ExecutionError, "do not belong"):
            embedding_summary(summary, frames)
        frames, summary = self.embeddings()
        data = read_json(summary)
        data["recognizer"]["model_sha256"] = "b" * 64
        write_json(summary, data)
        with self.assertRaisesRegex(ExecutionError, "Incompatible"):
            embedding_summary(summary, frames)

    def test_invalid_embeddings_counts_and_duplicate_tracks_are_rejected(self):
        def duplicate(rows):
            rows[0]["faces"].append(copy.deepcopy(rows[0]["faces"][0]))

        for edit in (
            lambda rows: rows[0]["faces"][0]["embedding"].__setitem__(0, 2.0),
            lambda rows: rows[0]["faces"][0]["embedding"].__setitem__(0, float("nan")),
            duplicate,
            lambda rows: rows[0].update(frame_index=2),
        ):
            with self.subTest(edit=edit):
                frames, summary = self.embeddings(edit=edit)
                with self.assertRaises(ExecutionError):
                    list(embedding_rows(frames, embedding_summary(summary, frames)))
        frames, summary = self.embeddings()
        data = read_json(summary)
        data["counts"]["usable_embedding_count"] = 0
        write_json(summary, data)
        with self.assertRaisesRegex(ExecutionError, "summary"):
            self.resolve(frames, summary)

    def test_incompatible_snapshot_rejected(self):
        self.enroll()
        snapshot = read_json(self.registry.snapshot())
        snapshot["recognizer"]["preprocessing"] = "different"
        with self.assertRaises(ExecutionError):
            validate_snapshot(snapshot)

    def test_invalid_module_parameters(self):
        for module, parameters in (
            (FaceEmbedding(), {"min_face_pixels": True}),
            (FaceEmbedding(), {"min_confidence": float("nan")}),
            (IdentityResolution(), {"ambiguity_margin": 0}),
            (IdentityResolution(), {"min_similarity": float("inf")}),
            (IdentityResolution(), {"unknown": 1}),
            (IdentityReport(), {"anything": 1}),
        ):
            with (
                self.subTest(module=module.name, parameters=parameters),
                self.assertRaises(PipelineError),
            ):
                module.validate_parameters(parameters)
