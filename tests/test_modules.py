import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from content_agent_orchestrator.contracts import Artifact, ExecutionError, ModuleContext
from content_agent_orchestrator.modules import ReportExport, VideoInspection
from content_agent_orchestrator.storage import digest_file


class ModuleFailureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.result = self.work / "result"
        self.result.mkdir()
        self.context = ModuleContext(self.work, self.result)

    def test_missing_ffprobe_reports_actionable_error(self):
        with (
            patch(
                "content_agent_orchestrator.modules.video.shutil.which",
                return_value=None,
            ),
            self.assertRaisesRegex(ExecutionError, "install FFmpeg"),
        ):
            VideoInspection().execute({}, {}, self.context)

    def test_probe_timeout_leaves_work_files(self):
        destination = self.work / "probe.json"
        with (
            patch(
                "content_agent_orchestrator.modules.video.subprocess.run",
                side_effect=subprocess.TimeoutExpired("ffprobe", 1),
            ),
            self.assertRaisesRegex(ExecutionError, "timeout"),
        ):
            VideoInspection._probe(["ffprobe"], destination, 1)
        self.assertTrue(destination.is_file())
        self.assertTrue(destination.with_suffix(".stderr").is_file())

    def test_decode_errors_fail_even_with_zero_probe_exit_status(self):
        def fake_probe(command, **kwargs):
            kwargs["stderr"].write(b"Error decoding a frame")
            kwargs["stderr"].flush()
            return subprocess.CompletedProcess(command, 0)

        with (
            patch(
                "content_agent_orchestrator.modules.video.subprocess.run",
                side_effect=fake_probe,
            ),
            self.assertRaisesRegex(ExecutionError, "Error decoding"),
        ):
            VideoInspection._probe(["ffprobe"], self.work / "probe.json", 1)

    def test_invalid_manifest_is_not_exported(self):
        source = self.work / "invalid.json"
        source.write_text(
            json.dumps({"schema_version": 1, "timing": {"decoded_frame_count": 0}}),
            encoding="utf-8",
        )
        digest, size = digest_file(source)
        with self.assertRaisesRegex(ExecutionError, "invalid inspection"):
            ReportExport().execute(
                {"inspection": Artifact(source, "video.manifest", digest, size)},
                {},
                self.context,
            )
        self.assertEqual(list(self.result.iterdir()), [])
