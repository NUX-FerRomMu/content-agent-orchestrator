import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from content_agent_orchestrator.cli import inspection_definition, main
from content_agent_orchestrator.modules import builtin_registry
from content_agent_orchestrator.pipeline import Pipeline
from content_agent_orchestrator.runner import Runner
from content_agent_orchestrator.storage import UserStorage, digest_file
from content_agent_orchestrator.workflow import input_records, save_document


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.storage = UserStorage(self.folder / "user")

    def cli(self, *args, expected=0):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(["--root", str(self.storage.root), *map(str, args)])
        self.assertEqual(code, expected, error.getvalue())
        return json.loads(output.getvalue()) if code == 0 else error.getvalue()

    def test_init_and_private_verified_import_are_idempotent_and_preserve_source(self):
        self.cli("init")
        source = self.folder / "original footage.MP4"
        source.write_bytes(b"private input")
        first = self.cli("artifacts", "import", source)
        second = self.cli("artifacts", "import", source)
        self.assertEqual(first, second)
        imported = Path(first["path"])
        self.assertEqual(imported.read_bytes(), source.read_bytes())
        self.assertEqual(imported.parent, self.storage.path("artifacts/raw_footages"))
        if os.name != "nt":
            self.assertEqual(imported.stat().st_mode & 0o777, 0o600)
        imported.write_bytes(b"changed")
        self.assertIn("preserved", self.cli("artifacts", "import", source, expected=1))
        self.assertEqual(imported.read_bytes(), b"changed")
        self.assertEqual(source.read_bytes(), b"private input")

    def test_import_rejects_state_temp_and_symlink_escape(self):
        self.storage.initialize()
        for area in ("temps", "state"):
            video = self.storage.path(f"{area}/private.mp4")
            video.write_bytes(b"input")
            self.assertIn(
                "Cannot import", self.cli("artifacts", "import", video, expected=1)
            )
        source = self.folder / "video.mp4"
        source.write_bytes(b"input")
        raw = self.storage.path("artifacts/raw_footages")
        raw.rmdir()
        outside = self.folder / "outside"
        outside.mkdir()
        raw.symlink_to(outside, target_is_directory=True)
        self.assertIn("symlinks", self.cli("artifacts", "import", source, expected=1))
        self.assertEqual(list(outside.iterdir()), [])

    def test_profile_interactive_and_noninteractive_match_no_automatic_review(self):
        guided, manual = self.folder / "guided.json", self.folder / "manual.json"
        with patch(
            "builtins.input", side_effect=["bad", "man", "11", "4", "", "", "eyeliner"]
        ):
            result = self.cli("profile", "create", "--interactive", "--output", guided)
        self.cli(
            "profile",
            "create",
            "--gender",
            "man",
            "--skin-tone",
            "4",
            "--features",
            "eyeliner",
            "--output",
            manual,
        )
        self.assertEqual(guided.read_bytes(), manual.read_bytes())
        self.assertNotIn("reviewed", result["profile"])
        before = guided.read_bytes()
        self.cli(
            "profile",
            "create",
            "--gender",
            "woman",
            "--skin-tone",
            "8",
            "--output",
            guided,
            expected=1,
        )
        self.assertEqual(guided.read_bytes(), before)

    def test_profile_requires_declared_appearance_and_cancel_writes_nothing(self):
        profile = self.folder / "profile.json"
        self.assertIn(
            "gender", self.cli("profile", "create", "--output", profile, expected=1)
        )
        with patch("builtins.input", side_effect=EOFError):
            self.assertIn(
                "cancelled",
                self.cli(
                    "profile",
                    "create",
                    "--interactive",
                    "--output",
                    profile,
                    expected=1,
                ),
            )
        self.assertFalse(profile.exists())

    def test_job_rejects_changed_inputs_and_different_storage_before_runner(self):
        video = self.folder / "video.mp4"
        video.write_bytes(b"original")
        job = self.folder / "clip.job.json"
        save_document(
            job,
            {
                "schema_version": 1,
                "storage_root": str(self.storage.root),
                "pipeline": inspection_definition(),
                "inputs": input_records({"video": video}, self.storage),
            },
        )
        video.write_bytes(b"changed")
        with patch.object(Runner, "run") as runner:
            self.assertIn("input changed", self.cli("workflow", "run", job, expected=1))
            runner.assert_not_called()
        video.write_bytes(b"original")
        document = json.loads(job.read_text())
        document["storage_root"] = str(self.folder / "elsewhere")
        job.write_text(json.dumps(document))
        self.assertIn(
            "storage root differs", self.cli("workflow", "run", job, expected=1)
        )
        self.assertFalse((self.folder / "elsewhere").exists())

    def test_saved_job_binds_absolute_inputs_independent_of_working_directory(self):
        video = self.folder / "video.mp4"
        video.write_bytes(b"original")
        job = self.folder / "job.json"
        save_document(
            job,
            {
                "schema_version": 1,
                "storage_root": str(self.storage.root),
                "pipeline": inspection_definition(),
                "inputs": input_records({"video": video}, self.storage),
            },
        )
        with patch.object(
            Runner, "run", side_effect=ValueError("reached runner")
        ) as runner:
            self.assertIn(
                "reached runner", self.cli("workflow", "run", job, expected=1)
            )
        self.assertEqual(runner.call_args.args[1], {"video": video})
        self.assertIsInstance(runner.call_args.args[0], Pipeline)
        self.assertEqual(
            digest_file(video)[0],
            json.loads(job.read_text())["inputs"]["video"]["sha256"],
        )

    def test_runs_unknown_invalid_and_limit_are_handled(self):
        self.assertEqual(self.cli("runs", "list")["runs"], [])
        self.assertIn("Unknown", self.cli("runs", "show", "a" * 32, expected=1))
        self.assertIn(
            "32 lowercase", self.cli("runs", "resume", "../escape", expected=1)
        )
        self.assertIn("Limit", self.cli("runs", "list", "--limit", "0", expected=1))
        self.assertIn(
            "--from-run",
            self.cli("identity", "register", "--track", "track-000001", expected=1),
        )

    def test_doctor_reports_missing_tools_without_downloading_or_creating_storage(self):
        with patch(
            "content_agent_orchestrator.diagnostics.shutil.which", return_value=None
        ):
            output, error = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                code = main(["--root", str(self.storage.root), "doctor", "--models"])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertFalse(result["ready"])
        self.assertFalse(
            next(item for item in result["checks"] if item["name"] == "ffmpeg")["ready"]
        )
        self.assertFalse(self.storage.root.exists())

    def test_module_inventory_includes_single_final_requirement_and_contracts(self):
        modules = {item["name"]: item for item in self.cli("modules")["modules"]}
        self.assertFalse(modules["face.swap"]["can_be_final"])
        self.assertTrue(modules["video.export"]["can_be_final"])
        self.assertEqual(set(modules), set(builtin_registry()))
        self.assertFalse(self.storage.root.exists())
