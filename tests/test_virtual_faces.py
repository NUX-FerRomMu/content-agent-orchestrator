import base64
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock, patch

import cv2
import numpy as np

from content_agent_orchestrator.contracts import ExecutionError, PipelineError
from content_agent_orchestrator.face_data import bundled_yunet, read_json, write_json
from content_agent_orchestrator.identity_data import RECOGNIZER, bundled_sface
from content_agent_orchestrator.modules.face_embed import FaceEmbedding
from content_agent_orchestrator.portrait_generators import (
    CommandPortraitGenerator,
    NoRedirect,
    OpenAIPortraitGenerator,
)
from content_agent_orchestrator.storage import UserStorage, digest_file
from content_agent_orchestrator.virtual_catalog import read_catalog
from content_agent_orchestrator.virtual_face import (
    appearance_profile,
    png_image,
    portrait_prompt,
    source_separation,
    validate_portrait,
)
from content_agent_orchestrator.virtual_registry import VirtualRegistry

PROFILE = {
    "gender": "woman",
    "skin_tone": 6,
    "undertone": "warm",
    "age_years": 35,
    "features": "Subtle eyeliner and natural makeup",
}
FIXTURE = Path(__file__).parent / "fixtures/virtual_adult.png"


class FixtureGenerator:
    descriptor: ClassVar = {
        "backend": "command",
        "model": "fictional-test-fixture",
        "seed": 1,
        "argv": ["fixture-generator"],
    }

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, directory):
        self.calls += 1
        return FIXTURE.read_bytes()


def enroll_reference(registry, directory, embedding, *, source="a", alias=None):
    frames = directory / "enrollment.jsonl"
    rows = [
        {
            "frame_index": i,
            "presentation_timestamp": i,
            "presentation_time": str(i / 10),
            "faces": [
                {
                    "track_id": "track-000001",
                    "association": "matched",
                    "bbox": [20, 20, 80, 80],
                    "landmarks": [[40, 40], [80, 40], [60, 60], [40, 80], [80, 80]],
                    "score": 0.95,
                    "embedding_status": "usable",
                    "embedding": embedding,
                }
            ],
        }
        for i in range(3)
    ]
    frames.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = directory / "enrollment_summary.json"
    write_json(
        summary,
        {
            "schema_version": 1,
            "source": {"sha256": source * 64, "size_bytes": 10},
            "video": {
                "width": 200,
                "height": 200,
                "frame_count": 3,
                "time_base": "1/10",
            },
            "recognizer": RECOGNIZER,
            "counts": {
                "frame_count": 3,
                "face_detection_count": 3,
                "usable_embedding_count": 3,
            },
            "file_sha256": {"frames": digest_file(frames)[0]},
        },
    )
    return registry.enroll(frames, summary, "track-000001", alias=alias)["identity_id"]


def astronaut_embedding():
    image = cv2.imread(str(FIXTURE.parent / "astronaut.png"))
    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN_create(
        str(bundled_yunet()), "", (width, height), 0.85, 0.3, 100
    )
    face = detector.detect(image)[1][0]
    recognizer = cv2.FaceRecognizerSF_create(str(bundled_sface()), "")
    return FaceEmbedding._embedding(
        recognizer,
        image,
        {
            "bbox": face[:4].tolist(),
            "landmarks": face[4:14].reshape(5, 2).tolist(),
            "score": float(face[14]),
        },
    )


class VirtualRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_embedding = astronaut_embedding()
        cls.validation = validate_portrait(FIXTURE.read_bytes())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.storage = UserStorage(self.root / "user")
        self.registry = VirtualRegistry(self.storage)
        self.identity_id = enroll_reference(
            self.registry.identities, self.root, self.source_embedding
        )
        self.generator = FixtureGenerator()

    def candidate(self):
        return self.registry.create(self.identity_id, PROFILE, self.generator)

    def approve(self):
        candidate = self.candidate()
        return self.registry.approve(
            self.identity_id, candidate["candidate_id"], reviewed=True
        )

    def test_generated_fixture_is_usable_and_separate_from_source(self):
        self.assertEqual(
            self.validation["portrait_sha256"],
            "6c075392f5b27ba0f612f8e3de39400a89eb22a7727686e170490f05cf31a1c4",
        )
        self.assertEqual(
            (self.validation["width"], self.validation["height"]), (1254, 1254)
        )
        self.assertLess(
            source_separation(self.validation["embedding"], [self.source_embedding]),
            0.3,
        )

    def test_candidate_is_not_assigned_before_visual_review(self):
        candidate = self.candidate()
        self.assertEqual(candidate["status"], "needs_visual_review")
        self.assertEqual(self.registry.list_faces()["faces"], [])
        self.assertNotIn('"embedding"', json.dumps(candidate))
        with self.assertRaisesRegex(PipelineError, "--reviewed"):
            self.registry.approve(self.identity_id, candidate["candidate_id"])
        self.assertEqual(self.registry.list_faces()["faces"], [])

    def test_repeated_generation_and_approval_keep_one_face_and_id(self):
        first = self.candidate()
        second = self.candidate()
        self.assertEqual(first["candidate_id"], second["candidate_id"])
        self.assertEqual(self.generator.calls, 1)
        approved = self.registry.approve(
            self.identity_id, first["candidate_id"], reviewed=True
        )
        again = self.registry.approve(
            self.identity_id, first["candidate_id"], reviewed=True
        )
        self.assertTrue(again["reused"])
        self.assertEqual(again["virtual_id"], approved["virtual_id"])
        reused = self.registry.create(
            self.identity_id, PROFILE, OpenAIPortraitGenerator(), cache="fresh"
        )
        self.assertEqual(reused["portrait_sha256"], approved["portrait_sha256"])
        self.assertEqual(reused["virtual_id"], approved["virtual_id"])
        self.assertEqual(self.registry.list_faces()["revision"], 1)

    def test_profile_change_does_not_replace_approved_face(self):
        approved = self.approve()
        with self.assertRaisesRegex(PipelineError, "immutable"):
            self.registry.create(
                self.identity_id,
                PROFILE | {"features": "another makeup"},
                self.generator,
            )
        self.assertEqual(self.generator.calls, 1)
        self.assertEqual(
            self.registry.list_faces()["faces"][0]["virtual_id"], approved["virtual_id"]
        )

    def test_completed_generation_cache_survives_validation_failure(self):
        with (
            patch(
                "content_agent_orchestrator.virtual_registry.validate_portrait",
                side_effect=ExecutionError("validation interrupted"),
            ),
            self.assertRaisesRegex(ExecutionError, "interrupted"),
        ):
            self.candidate()
        self.assertEqual(self.generator.calls, 1)
        self.assertEqual(
            len(
                list(
                    (self.storage.root / "temps/virtual_generation").glob(
                        "*/generation.json"
                    )
                )
            ),
            1,
        )
        candidate = self.candidate()
        self.assertEqual(self.generator.calls, 1)
        self.assertTrue(Path(candidate["portrait_path"]).exists())
        self.assertEqual(
            list((self.storage.root / "temps/virtual_generation").iterdir()), []
        )

    def test_fresh_only_discards_this_jobs_failed_temporaries(self):
        unrelated = self.storage.path("temps/other-job")
        unrelated.mkdir()
        (unrelated / "keep").write_text("keep")
        with (
            patch(
                "content_agent_orchestrator.virtual_registry.validate_portrait",
                side_effect=ExecutionError("interrupted"),
            ),
            self.assertRaises(ExecutionError),
        ):
            self.candidate()
        self.registry.create(self.identity_id, PROFILE, self.generator, cache="fresh")
        self.assertEqual(self.generator.calls, 2)
        self.assertEqual((unrelated / "keep").read_text(), "keep")

    def test_tampered_candidate_cannot_be_approved(self):
        candidate = self.candidate()
        Path(candidate["portrait_path"]).write_bytes(b"bad")
        with self.assertRaises(ExecutionError):
            self.registry.approve(
                self.identity_id, candidate["candidate_id"], reviewed=True
            )
        self.assertEqual(self.registry.list_faces()["faces"], [])

    def test_source_like_candidate_is_rejected(self):
        enroll_reference(
            self.registry.identities,
            self.root,
            self.validation["embedding"],
            source="b",
        )
        with self.assertRaisesRegex(ExecutionError, "too similar"):
            self.candidate()
        self.assertEqual(self.registry.list_faces()["faces"], [])

    def test_new_source_references_are_checked_again_before_approval_and_snapshot(self):
        candidate = self.candidate()
        enroll_reference(
            self.registry.identities,
            self.root,
            self.validation["embedding"],
            source="b",
        )
        with self.assertRaisesRegex(ExecutionError, "too similar"):
            self.registry.approve(
                self.identity_id, candidate["candidate_id"], reviewed=True
            )
        self.assertEqual(self.registry.list_faces()["faces"], [])

    def test_snapshot_revalidates_stored_pixels_against_current_sources(self):
        self.approve()
        enroll_reference(
            self.registry.identities,
            self.root,
            self.validation["embedding"],
            source="b",
        )
        with self.assertRaisesRegex(ExecutionError, "too similar"):
            self.registry.snapshot()

    def test_catalog_is_deterministic_revisioned_and_contains_no_vectors(self):
        before = self.registry.snapshot()
        self.approve()
        after = self.registry.snapshot()
        self.assertNotEqual(before, after)
        self.assertEqual(after, self.registry.snapshot())
        manifest, portraits = read_catalog(after)
        self.assertEqual(manifest["revision"], 1)
        self.assertEqual(manifest["source_revision"], 1)
        self.assertEqual(len(portraits), 1)
        self.assertNotIn('"embedding"', json.dumps(manifest))
        self.assertNotIn('"references"', json.dumps(manifest))
        self.assertEqual(next(iter(portraits.values())), FIXTURE.read_bytes())

    def test_candidate_for_a_changed_profile_cannot_replace_approved_mapping(self):
        first = self.candidate()
        other = self.registry.create(
            self.identity_id, PROFILE | {"features": "defined eyeliner"}, self.generator
        )
        approved = self.registry.approve(
            self.identity_id, first["candidate_id"], reviewed=True
        )
        with self.assertRaisesRegex(PipelineError, "different immutable"):
            self.registry.approve(
                self.identity_id, other["candidate_id"], reviewed=True
            )
        self.assertEqual(
            self.registry.list_faces()["faces"][0]["virtual_id"], approved["virtual_id"]
        )
        self.assertEqual(self.registry.list_faces()["revision"], 1)

    def test_corrupt_validation_metadata_fails_without_assigning_candidate(self):
        candidate = self.candidate()
        path = Path(candidate["portrait_path"]).parent / "candidate.json"
        data = read_json(path)
        data["validation"] = []
        write_json(path, data)
        with self.assertRaisesRegex(ExecutionError, "validation metadata"):
            self.registry.approve(
                self.identity_id, candidate["candidate_id"], reviewed=True
            )
        self.assertEqual(self.registry.list_faces()["faces"], [])

    def test_identity_deletion_cascades_virtual_face_and_purges_catalogs(self):
        approved = self.approve()
        catalog = self.registry.snapshot()
        self.registry.identities.delete(self.identity_id)
        self.assertEqual(self.registry.list_faces()["faces"], [])
        self.assertFalse(catalog.exists())
        self.assertEqual(self.registry.list_faces()["revision"], 2)
        self.assertTrue(Path(approved["portrait_path"]).exists())

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_candidates_portrait_exports_and_catalogs_are_private(self):
        candidate = self.candidate()
        approved = self.registry.approve(
            self.identity_id, candidate["candidate_id"], reviewed=True
        )
        for path in (
            Path(candidate["portrait_path"]),
            Path(approved["portrait_path"]),
            self.registry.snapshot(),
        ):
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_symlink_cannot_replace_exported_portrait(self):
        approved = self.approve()
        path = Path(approved["portrait_path"])
        path.unlink()
        target = self.root / "outside"
        target.write_text("untouched")
        path.symlink_to(target)
        with self.assertRaisesRegex(PipelineError, "symlink"):
            self.candidate()
        self.assertEqual(target.read_text(), "untouched")


class GeneratorTests(unittest.TestCase):
    def test_profiles_require_explicit_attributes_and_adult_age(self):
        for value in (
            {},
            PROFILE | {"gender": "infer"},
            PROFILE | {"skin_tone": True},
            PROFILE | {"age_years": 12},
            PROFILE | {"glasses": True},
        ):
            with self.subTest(value=value), self.assertRaises(PipelineError):
                appearance_profile(value)
        prompt = portrait_prompt(PROFILE)
        self.assertIn("no glasses", prompt)
        self.assertIn("never a child", prompt)
        self.assertIn("Subtle eyeliner", prompt)

    def test_png_validation_rejects_invalid_large_and_transparent_images(self):
        data = bytearray(FIXTURE.read_bytes())
        data[16:24] = struct.pack(">II", 100000, 100000)
        for content in (b"not png", bytes(data)):
            with self.assertRaises(ExecutionError):
                png_image(content)
        transparent = np.zeros((256, 256, 4), np.uint8)
        _, content = cv2.imencode(".png", transparent)
        with self.assertRaisesRegex(ExecutionError, "opaque"):
            png_image(content.tobytes())

    def test_empty_portrait_has_no_usable_face(self):
        _, content = cv2.imencode(".png", np.zeros((512, 512, 3), np.uint8))
        with self.assertRaisesRegex(ExecutionError, "exactly one"):
            validate_portrait(content.tobytes())

    def test_cloud_request_is_prompt_only_and_returns_png(self):
        response = Mock()
        response.read.return_value = json.dumps(
            {"data": [{"b64_json": base64.b64encode(FIXTURE.read_bytes()).decode()}]}
        ).encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-key"}, clear=True),
            patch("urllib.request.build_opener", return_value=opener),
        ):
            content = OpenAIPortraitGenerator().generate(
                portrait_prompt(PROFILE), Path("unused")
            )
        self.assertEqual(content, FIXTURE.read_bytes())
        request = opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(
            set(payload),
            {"model", "size", "quality", "output_format", "background", "prompt", "n"},
        )
        self.assertEqual(
            request.full_url, "https://api.openai.com/v1/images/generations"
        )
        self.assertNotIn("fake-test-key", request.data.decode())
        self.assertIsNone(
            NoRedirect().redirect_request(
                None, None, 307, "", None, "https://example.com"
            )
        )

    def test_cloud_errors_do_not_print_secrets_or_response_body(self):
        opener = Mock()
        opener.open.side_effect = urllib.error.HTTPError(
            "https://api.openai.com",
            400,
            "private-body",
            {},
            io.BytesIO(b"fake-test-key"),
        )
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-key"}, clear=True),
            patch("urllib.request.build_opener", return_value=opener),
            self.assertRaisesRegex(ExecutionError, "HTTP error 400") as caught,
        ):
            OpenAIPortraitGenerator().generate("fictional adult", Path("unused"))
        self.assertNotIn("private-body", str(caught.exception))
        self.assertNotIn("fake-test-key", str(caught.exception))
        self.assertEqual(opener.open.call_count, 1)

    def test_cloud_generation_requires_key_and_rejects_url_responses(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ExecutionError, "OPENAI_API_KEY"),
        ):
            OpenAIPortraitGenerator().generate("fictional adult", Path("unused"))
        response = contextlib.nullcontext(
            Mock(
                read=Mock(
                    return_value=b'{"data":[{"url":"https://untrusted.example/photo"}]}'
                )
            )
        )
        opener = Mock(open=Mock(return_value=response))
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-key"}, clear=True),
            patch("urllib.request.build_opener", return_value=opener),
            self.assertRaisesRegex(ExecutionError, "base64"),
        ):
            OpenAIPortraitGenerator().generate("fictional adult", Path("unused"))

    def test_local_command_contract_generates_png_without_shell(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "generator.py"
            script.write_text(
                "import json,sys\nfrom pathlib import Path\nr=json.load(sys.stdin)\nassert r['seed']==42\nassert r['size']==[1024,1024]\nassert 'never a child' in r['prompt']\nPath(r['output_path']).write_bytes(Path(sys.argv[1]).read_bytes())\n"
            )
            generator = CommandPortraitGenerator(
                {
                    "argv": [sys.executable, str(script), str(FIXTURE)],
                    "model": "test-model-v1",
                    "seed": 42,
                }
            )
            self.assertEqual(
                generator.generate(portrait_prompt(PROFILE), root), FIXTURE.read_bytes()
            )

    def test_local_command_failure_timeout_and_bad_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = CommandPortraitGenerator(
                {
                    "argv": [sys.executable, "-c", "raise SystemExit(7)"],
                    "model": "fixture",
                    "seed": 1,
                }
            )
            with self.assertRaisesRegex(ExecutionError, "code 7"):
                generator.generate("fictional adult", root)
            with (
                patch(
                    "subprocess.run",
                    side_effect=subprocess.TimeoutExpired(["generator"], 1),
                ),
                self.assertRaisesRegex(ExecutionError, "timed out"),
            ):
                generator.generate("fictional adult", root)
        for configuration in (
            {},
            {"argv": "sh", "model": "m", "seed": 1},
            {"argv": ["run"], "model": "m", "seed": True},
        ):
            with (
                self.subTest(configuration=configuration),
                self.assertRaises(PipelineError),
            ):
                CommandPortraitGenerator(configuration)

    def test_catalog_rejects_traversal_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "manifest.json",
                    json.dumps(
                        {
                            "schema_version": 1,
                            "faces": [],
                            "registry_id": "bad",
                            "revision": 0,
                            "source_revision": 0,
                        }
                    ),
                )
                archive.writestr("../outside.png", FIXTURE.read_bytes())
            with self.assertRaises(ExecutionError):
                read_catalog(path)
