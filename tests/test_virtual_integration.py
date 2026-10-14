import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
from test_virtual_faces import FIXTURE, PROFILE

from content_agent_orchestrator.cli import main, virtual_assignment_definition
from content_agent_orchestrator.contracts import (
    Artifact,
    ExecutionError,
    ModuleContext,
    PipelineError,
)
from content_agent_orchestrator.face_data import iter_rows, read_json
from content_agent_orchestrator.modules import builtin_registry
from content_agent_orchestrator.pipeline import Pipeline
from content_agent_orchestrator.storage import UserStorage, digest_file
from content_agent_orchestrator.virtual_catalog import read_catalog


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required"
)
class VirtualIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.user = self.root / "user"
        self.video = self.root / "source.mkv"
        image = cv2.imread(str(FIXTURE.parent / "astronaut.png"))
        self.make_video(image)
        self.profile = self.root / "profile.json"
        self.profile.write_text(json.dumps(PROFILE))
        self.command = self.root / "command.json"
        self.script = self.root / "generator.py"
        self.script.write_text(
            "import json,sys\nfrom pathlib import Path\nr=json.load(sys.stdin)\nassert set(r)=={'prompt','size','seed','output_path'}\nPath(r['output_path']).write_bytes(Path(sys.argv[1]).read_bytes())\n"
        )
        self.command.write_text(
            json.dumps(
                {
                    "argv": [sys.executable, str(self.script), str(FIXTURE.resolve())],
                    "model": "fictional-test-fixture",
                    "seed": 42,
                }
            )
        )
        result = self.cli("recognize-faces", str(self.video))
        paths = result["private_intermediates"]
        enrolled = self.cli(
            "identity",
            "register",
            "--embeddings",
            paths["frames"],
            "--summary",
            paths["summary"],
            "--track",
            "track-000001",
            "--alias",
            "source-owner",
        )
        self.identity_id = enrolled["identity_id"]

    def make_video(self, image):
        h, w = image.shape[:2]
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-video_size",
                f"{w}x{h}",
                "-framerate",
                "6",
                "-i",
                "pipe:0",
                "-c:v",
                "ffv1",
                "-y",
                str(self.video),
            ],
            input=image.tobytes() * 3,
            check=True,
            capture_output=True,
            timeout=30,
        )

    def cli(self, *arguments, expected=0):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            result = main(["--root", str(self.user), *arguments])
        self.assertEqual(result, expected, error.getvalue())
        self.assertNotIn('"embedding":', output.getvalue() + error.getvalue())
        return json.loads(output.getvalue()) if expected == 0 else error.getvalue()

    def create(self):
        return self.cli(
            "virtual",
            "create",
            self.identity_id,
            "--profile",
            str(self.profile),
            "--backend",
            "command",
            "--generator",
            str(self.command),
        )

    def approve(self):
        candidate = self.create()
        approved = self.cli(
            "virtual",
            "approve",
            self.identity_id,
            candidate["candidate_id"],
            "--reviewed",
        )
        return approved

    def test_cli_creation_review_assignment_and_catalog_cache(self):
        before = self.cli("assign-virtual-faces", str(self.video))
        report = read_json(Path(before["outputs"]["report"]["path"]))
        self.assertEqual(report["counts"]["missing_target_observation_count"], 3)
        candidate = self.create()
        self.cli(
            "virtual",
            "approve",
            self.identity_id,
            candidate["candidate_id"],
            expected=1,
        )
        self.assertEqual(self.cli("virtual", "list")["faces"], [])
        approved = self.cli(
            "virtual",
            "approve",
            self.identity_id,
            candidate["candidate_id"],
            "--reviewed",
        )
        after = self.cli("assign-virtual-faces", str(self.video))
        self.assertEqual(
            after["cache_hits"], ["inspect", "detect", "track", "embed", "resolve"]
        )
        report = read_json(Path(after["outputs"]["report"]["path"]))
        self.assertEqual(report["counts"]["ready_observation_count"], 3)
        self.assertFalse(report["anonymization_performed"])
        self.assertFalse(report["face_swap_performed"])
        for row in iter_rows(Path(after["outputs"]["frames"]["path"])):
            face = row["faces"][0]
            self.assertEqual(face["virtual"]["virtual_id"], approved["virtual_id"])
            self.assertEqual(
                face["virtual"]["portrait_sha256"], approved["portrait_sha256"]
            )
        manifest, portraits = read_catalog(Path(after["outputs"]["targets"]["path"]))
        self.assertEqual(len(manifest["faces"]), 1)
        self.assertEqual(next(iter(portraits.values())), FIXTURE.read_bytes())
        cached = self.cli("assign-virtual-faces", str(self.video))
        self.assertEqual(
            cached["cache_hits"],
            ["inspect", "detect", "track", "embed", "resolve", "virtual", "report"],
        )
        reused = self.create()
        self.assertEqual(reused["virtual_id"], approved["virtual_id"])
        self.assertTrue(reused["reused"])

    def test_unknown_person_and_no_face_do_not_receive_fallback_target(self):
        self.approve()
        hopper = cv2.imread(str(FIXTURE.parent / "grace_hopper.jpg"))
        self.make_video(hopper)
        result = self.cli("assign-virtual-faces", str(self.video))
        report = read_json(Path(result["outputs"]["report"]["path"]))
        self.assertEqual(report["counts"]["unresolved_source_observation_count"], 3)
        manifest, _ = read_catalog(Path(result["outputs"]["targets"]["path"]))
        self.assertEqual(manifest["faces"], [])
        for row in iter_rows(Path(result["outputs"]["frames"]["path"])):
            self.assertEqual(
                row["faces"][0]["virtual"],
                {"status": "unresolved_source", "reason": "unknown"},
            )
        self.make_video(np.zeros_like(hopper))
        empty = self.cli("assign-virtual-faces", str(self.video))
        self.assertEqual(
            read_json(Path(empty["outputs"]["report"]["path"]))["counts"][
                "selected_virtual_face_count"
            ],
            0,
        )

    def test_catalog_from_another_revision_fails_without_final_output(self):
        self.approve()
        catalog = Path(self.cli("virtual", "snapshot")["path"])
        manifest, portraits = read_catalog(catalog)
        manifest["source_revision"] += 1
        wrong = self.root / "wrong.zip"
        with zipfile.ZipFile(wrong, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for name, content in portraits.items():
                archive.writestr(name, content)
        error = self.cli(
            "assign-virtual-faces", str(self.video), "--catalog", str(wrong), expected=1
        )
        self.assertIn("same source registry revision", error)

    def test_deletion_removes_mapping_and_default_new_run_uses_empty_registry(self):
        self.approve()
        before = self.cli("assign-virtual-faces", str(self.video))
        self.cli("identity", "delete", self.identity_id)
        self.assertFalse(Path(before["registry_snapshot"]).exists())
        self.assertFalse(Path(before["virtual_catalog"]).exists())
        self.assertEqual(self.cli("virtual", "list")["faces"], [])
        after = self.cli("assign-virtual-faces", str(self.video))
        report = read_json(Path(after["outputs"]["report"]["path"]))
        self.assertEqual(report["counts"]["ready_observation_count"], 0)
        self.assertEqual(report["counts"]["unresolved_source_observation_count"], 3)

    def test_dag_example_and_modules_reject_invalid_configuration(self):
        modules = builtin_registry()
        definition = virtual_assignment_definition()
        pipeline = Pipeline.parse(definition, modules)
        self.assertEqual(pipeline.final, "report")
        example = (
            Path(__file__).resolve().parents[1] / "examples/assign-virtual-faces.json"
        )
        self.assertEqual(Pipeline.load(example, modules).definition, definition)
        for name in ("virtual.resolve", "virtual.report"):
            with self.assertRaises(PipelineError):
                modules[name].validate_parameters({"unknown": True})

    def test_catalog_rejects_unexpected_vectors_and_tampered_portrait(self):
        self.approve()
        manifest, portraits = read_catalog(
            Path(self.cli("virtual", "snapshot")["path"])
        )
        wrong = self.root / "wrong.zip"
        manifest["faces"][0]["embedding"] = [0] * 128
        with zipfile.ZipFile(wrong, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for name, content in portraits.items():
                archive.writestr(name, content)
        with self.assertRaises(ExecutionError):
            read_catalog(wrong)

        manifest["faces"][0].pop("embedding")
        with zipfile.ZipFile(wrong, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for name in portraits:
                archive.writestr(name, b"wrong image")
        with self.assertRaises(ExecutionError):
            read_catalog(wrong)

    def test_ambiguous_and_inconsistent_observations_cannot_select_target(self):
        self.approve()
        result = self.cli("assign-virtual-faces", str(self.video))
        storage = UserStorage(self.user)
        modules = builtin_registry()
        with storage.connect() as connection:
            encoded = connection.execute(
                "SELECT outputs FROM nodes WHERE run_id = ? AND node_id = 'resolve'",
                (result["run_id"],),
            ).fetchone()[0]
        original = storage.verified_artifacts(
            encoded, dict(modules["identity.resolve"].output_kinds)
        )
        catalog = Path(result["virtual_catalog"])
        frames = self.root / "observations.jsonl"
        summary_path = self.root / "summary.json"
        output = self.root / "direct-result"
        output.mkdir()
        for invalid in ("quality", "association", "assigned_ambiguous"):
            with self.subTest(invalid=invalid):
                rows = list(iter_rows(original["frames"].path))
                if invalid == "quality":
                    rows[0]["faces"][0]["embedding_status"] = "low_confidence"
                elif invalid == "association":
                    rows[0]["faces"][0]["association"] = "ambiguous"
                else:
                    rows[0]["faces"][0]["identity"]["status"] = "ambiguous"
                frames.write_text("".join(json.dumps(row) + "\n" for row in rows))
                summary = read_json(original["summary"].path)
                summary["file_sha256"]["frames"] = digest_file(frames)[0]
                summary_path.write_text(json.dumps(summary))
                inputs = {
                    name: Artifact(path, kind, *digest_file(path))
                    for name, path, kind in (
                        ("frames", frames, "identity.observations"),
                        ("summary", summary_path, "identity.resolution-summary"),
                        ("catalog", catalog, "virtual.catalog"),
                    )
                }
                with self.assertRaises(ExecutionError):
                    modules["virtual.resolve"].execute(
                        inputs, {}, ModuleContext(self.root, output)
                    )
        rows = list(iter_rows(original["frames"].path))
        first = rows[0]["faces"][0]
        # An ambiguous observation may list a best candidate; it still cannot bind B.
        first["identity"] = {
            "status": "ambiguous",
            "candidates": [{"identity_id": self.identity_id, "score": 0.99}],
        }
        frames.write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary = read_json(original["summary"].path)
        summary["file_sha256"]["frames"] = digest_file(frames)[0]
        summary["counts"]["known_observation_count"] = 2
        summary["counts"]["ambiguous_observation_count"] = 1
        summary_path.write_text(json.dumps(summary))
        inputs = {
            name: Artifact(path, kind, *digest_file(path))
            for name, path, kind in (
                ("frames", frames, "identity.observations"),
                ("summary", summary_path, "identity.resolution-summary"),
                ("catalog", catalog, "virtual.catalog"),
            )
        }
        outputs = modules["virtual.resolve"].execute(
            inputs, {}, ModuleContext(self.root, output)
        )
        first = next(iter_rows(outputs["frames"]))["faces"][0]
        self.assertEqual(
            first["virtual"], {"status": "unresolved_source", "reason": "ambiguous"}
        )
