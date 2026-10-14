import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from content_agent_orchestrator.cli import main
from content_agent_orchestrator.contracts import ExecutionError
from content_agent_orchestrator.identity_registry import IdentityRegistry
from content_agent_orchestrator.runner import Runner
from content_agent_orchestrator.storage import UserStorage
from content_agent_orchestrator.virtual_registry import VirtualRegistry


class CliResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "user"

    def cli(self, command, *arguments):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(
                [
                    "--root",
                    str(self.root),
                    command,
                    "source.mp4",
                    "--resume",
                    "interrupted-run",
                    *arguments,
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(output.getvalue(), "")
        return error.getvalue()

    def test_missing_original_inputs_are_rejected_before_registry_or_runner_access(
        self,
    ):
        for command, arguments, required in (
            ("recognize-faces", [], "--snapshot"),
            ("assign-virtual-faces", [], "--snapshot"),
            (
                "assign-virtual-faces",
                ["--snapshot", "original.json"],
                "--catalog",
            ),
            (
                "assign-virtual-faces",
                ["--catalog", "original.zip"],
                "--snapshot",
            ),
        ):
            with (
                self.subTest(command=command, arguments=arguments),
                patch("content_agent_orchestrator.cli.IdentityRegistry") as identities,
                patch("content_agent_orchestrator.cli.VirtualRegistry") as virtual,
                patch.object(Runner, "run") as run,
            ):
                error = self.cli(command, *arguments)
                self.assertIn(f"original {required}", error)
                identities.assert_not_called()
                virtual.assert_not_called()
                run.assert_not_called()
                self.assertFalse(self.root.exists())

    def test_explicit_original_inputs_reach_runner_without_creating_new_snapshots(self):
        storage = UserStorage(self.root)
        snapshot = IdentityRegistry(storage).snapshot()
        catalog = VirtualRegistry(storage).snapshot()
        for command in ("recognize-faces", "assign-virtual-faces"):
            arguments = ["--snapshot", str(snapshot)]
            if command == "assign-virtual-faces":
                arguments += ["--catalog", str(catalog)]
            with (
                self.subTest(command=command),
                patch.object(IdentityRegistry, "snapshot") as create_snapshot,
                patch.object(VirtualRegistry, "snapshot") as create_catalog,
                patch.object(
                    Runner,
                    "run",
                    side_effect=ExecutionError("original resume attempted"),
                ) as run,
            ):
                self.assertIn(
                    "original resume attempted", self.cli(command, *arguments)
                )
                create_snapshot.assert_not_called()
                create_catalog.assert_not_called()
                inputs = run.call_args.args[1]
                self.assertEqual(inputs["registry"], snapshot)
                if command == "assign-virtual-faces":
                    self.assertEqual(inputs["catalog"], catalog)
                self.assertEqual(run.call_args.kwargs["resume"], "interrupted-run")
