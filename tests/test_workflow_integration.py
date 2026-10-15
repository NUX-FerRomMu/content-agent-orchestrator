import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import test_render_pipeline as rendering

from content_agent_orchestrator.cli import main
from content_agent_orchestrator.storage import digest_file
from content_agent_orchestrator.virtual_registry import VirtualRegistry


@unittest.skipUnless(rendering.HAS_RUNTIME, "Optional swap runtime and FFmpeg required")
class GuidedVideoTests(unittest.TestCase):
    def setUp(self):
        # Reuse native media/identity fixtures without inheriting their test suite.
        self.fixture = rendering.RenderPipelineTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.folder = self.fixture.root
        self.storage = self.fixture.storage
        self.job = self.folder / "clip.job.json"
        factory = patch(
            "content_agent_orchestrator.modules.face_swap.InSwapper",
            side_effect=lambda *a, **k: rendering.FixtureSwapper(self.fixture),
        )
        factory.start()
        self.addCleanup(factory.stop)

    def cli(self, *args, expected=0):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(["--root", str(self.storage.root), *map(str, args)])
        self.assertEqual(code, expected, error.getvalue())
        return json.loads(output.getvalue()) if code == 0 else error.getvalue()

    def create(self, *args):
        return self.cli(
            "workflow",
            "create",
            "--output",
            self.job,
            "--swap-model",
            self.fixture.swap_model,
            "--arcface-model",
            self.fixture.arcface,
            "--chunk-frames",
            "2",
            "--codec",
            "ffv1",
            *args,
        )

    def test_interactive_build_run_inspect_and_repeat_reuses_all_ten_nodes(self):
        original = digest_file(self.fixture.source)
        with patch("builtins.input", side_effect=[str(self.fixture.source), "", ""]):
            built = self.create("--interactive")
        self.assertEqual(built["final"], "export")
        self.assertEqual(len(built["nodes"]), 10)
        result = self.cli("workflow", "run", self.job)
        self.assertEqual(set(result["outputs"]), {"video", "report", "audit"})
        report = json.loads(Path(result["outputs"]["report"]["path"]).read_text())
        self.assertEqual(report["counts"], {"validated": 5, "redacted": 0})
        self.assertEqual(digest_file(self.fixture.source), original)
        inspected = self.cli("runs", "show", result["run_id"])
        self.assertEqual(inspected["status"], "succeeded")
        self.assertEqual(len(inspected["nodes"]), 10)
        repeated = self.cli("workflow", "run", self.job)
        self.assertEqual(len(repeated["cache_hits"]), 10)
        self.assertNotEqual(result["run_id"], repeated["run_id"])

    def test_interrupted_run_resumes_from_db_without_job_or_new_snapshots(self):
        self.create("--video", self.fixture.source)
        self.fixture.interrupt_at = 3
        self.cli("workflow", "run", self.job, expected=130)
        listed = self.cli("runs", "list")["runs"]
        run_id = listed[0]["id"]
        self.assertNotEqual(listed[0]["status"], "succeeded")
        state = self.cli("runs", "show", run_id)
        original_catalog = state["inputs"]["catalog"]
        self.job.unlink()
        self.fixture.interrupt_at = None
        with patch.object(
            VirtualRegistry,
            "snapshot",
            side_effect=AssertionError("Must reuse original catalog"),
        ):
            result = self.cli("runs", "resume", run_id)
        self.assertEqual(result["run_id"], run_id)
        self.assertEqual(
            self.fixture.swap_calls, 6
        )  # 2 cached + 3 remaining; failed third attempt was retried.
        self.assertEqual(
            self.cli("runs", "show", run_id)["inputs"]["catalog"], original_catalog
        )
        self.assertIn(
            "already completed", self.cli("runs", "resume", run_id, expected=1)
        )

    def test_track_listing_and_enrollment_from_run_verify_private_artifacts(self):
        initial = self.cli("runs", "list")["runs"][0]["id"]
        tracks = self.cli("identity", "tracks", "--run", initial)["tracks"]
        self.assertEqual(tracks[0]["track_id"], "track-000001")
        self.assertEqual(tracks[0]["usable_observations"], 5)
        enrolled = self.cli(
            "identity",
            "register",
            "--from-run",
            initial,
            "--track",
            tracks[0]["track_id"],
            "--identity-id",
            self.fixture.identity_id,
        )
        self.assertEqual(enrolled["identity_id"], self.fixture.identity_id)
        self.fixture.seed_frames.write_text("tampered")
        self.assertIn(
            "No intact", self.cli("identity", "tracks", "--run", initial, expected=1)
        )

    def test_workflow_cancellation_existing_jobs_and_missing_approval_are_safe(self):
        with patch("builtins.input", side_effect=EOFError):
            error = self.cli(
                "workflow", "create", "--interactive", "--output", self.job, expected=1
            )
        self.assertIn("cancelled", error)
        self.assertFalse(self.job.exists())
        self.create("--video", self.fixture.source)
        before = self.job.read_bytes()
        self.assertIn(
            "already exists",
            self.cli(
                "workflow",
                "create",
                "--output",
                self.job,
                "--video",
                self.fixture.source,
                expected=1,
            ),
        )
        self.assertEqual(self.job.read_bytes(), before)
        with patch.object(VirtualRegistry, "list_faces", return_value={"faces": []}):
            error = self.cli(
                "workflow",
                "create",
                "--output",
                self.folder / "no-approval.json",
                "--video",
                self.fixture.source,
                "--swap-model",
                self.fixture.swap_model,
                "--arcface-model",
                self.fixture.arcface,
                expected=1,
            )
        self.assertIn("No approved B", error)
        self.assertFalse((self.folder / "no-approval.json").exists())
