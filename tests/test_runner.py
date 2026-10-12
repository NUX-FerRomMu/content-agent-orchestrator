import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from content_agent_orchestrator.contracts import ExecutionError, PipelineError
from content_agent_orchestrator.pipeline import Pipeline
from content_agent_orchestrator.runner import RunFailed, Runner
from content_agent_orchestrator.storage import UserStorage, digest_file


class TextModule:
    def __init__(self):
        self.name = "text.copy"
        self.version = "1"
        self.input_kinds = {"source": "text"}
        self.output_kinds = {"text": "text"}
        self.calls = []
        self.fail_label = None
        self.saw_cache = []
        self.bad_output = False
        self.mutate_source = False

    def validate_parameters(self, parameters):
        if set(parameters) != {"label"} or not isinstance(parameters["label"], str):
            raise PipelineError("Expected a string label")

    def execute(self, inputs, parameters, context):
        label = parameters["label"]
        self.calls.append(label)
        checkpoint = context.work_dir / "checkpoint.txt"
        self.saw_cache.append((label, checkpoint.exists()))
        checkpoint.write_text("recoverable state", encoding="utf-8")
        if label == self.fail_label:
            self.fail_label = None
            raise ExecutionError("simulated transient failure")
        if self.mutate_source:
            inputs["source"].path.write_text("modified", encoding="utf-8")
        result = context.result_dir / "text.txt"
        result.write_text(inputs["source"].path.read_text() + label, encoding="utf-8")
        return {"text": checkpoint if self.bad_output else result}


class JoinModule:
    def __init__(self):
        self.name = "text.join"
        self.version = "1"
        self.input_kinds = {"left": "text", "right": "text"}
        self.output_kinds = {"text": "text"}
        self.calls = 0

    def validate_parameters(self, parameters):
        if parameters:
            raise PipelineError("No join parameters")

    def execute(self, inputs, parameters, context):
        self.calls += 1
        result = context.result_dir / "joined.txt"
        result.write_text(
            inputs["left"].path.read_text() + "|" + inputs["right"].path.read_text(),
            encoding="utf-8",
        )
        return {"text": result}


def text_node(node_id, label, source=None):
    reference = (
        {"external": "source"} if source is None else {"node": source, "output": "text"}
    )
    return {
        "id": node_id,
        "module": "text.copy",
        "inputs": {"source": reference},
        "parameters": {"label": label},
    }


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "user"
        self.storage = UserStorage(self.root)
        self.source = Path(self.temporary.name) / "original.txt"
        self.source.write_text("original", encoding="utf-8")
        self.module = TextModule()
        self.join = JoinModule()
        self.registry = {self.module.name: self.module, self.join.name: self.join}
        self.runner = Runner(self.storage, self.registry)
        self.definition = {
            "nodes": [text_node("first", "A"), text_node("last", "B", "first")],
            "final": "last",
        }

    def run_pipeline(self, **kwargs):
        return self.runner.run(
            Pipeline.parse(self.definition, self.registry),
            {"source": self.source},
            **kwargs,
        )

    def records(self):
        with self.storage.connect() as connection:
            return connection.execute("SELECT * FROM nodes ORDER BY node_id").fetchall()

    def test_success_routes_intermediates_and_publishes_only_final(self):
        original = digest_file(self.source)
        result = self.run_pipeline()
        self.assertEqual(result.outputs["text"].path.read_text(), "originalAB")
        self.assertEqual(
            result.outputs["text"].path.parent, self.root / "outputs" / result.run_id
        )
        self.assertEqual(digest_file(self.source), original)
        self.assertEqual(list((self.root / "temps" / result.run_id).iterdir()), [])
        first = next(
            record for record in self.records() if record["node_id"] == "first"
        )
        self.assertTrue(
            json.loads(first["outputs"])["text"]["path"].startswith("artifacts/runs/")
        )
        self.assertEqual(
            [record["status"] for record in self.records()], ["succeeded", "succeeded"]
        )

    def test_diamond_executes_both_branches_before_single_sink(self):
        self.definition = {
            "nodes": [
                {
                    "id": "join",
                    "module": "text.join",
                    "inputs": {
                        "left": {"node": "left", "output": "text"},
                        "right": {"node": "right", "output": "text"},
                    },
                },
                text_node("right", "R", "first"),
                text_node("first", "A"),
                text_node("left", "L", "first"),
            ],
            "final": "join",
        }
        result = self.run_pipeline()
        self.assertEqual(
            result.outputs["text"].path.read_text(), "originalAL|originalAR"
        )
        self.assertEqual(self.module.calls[0], "A")
        self.assertEqual(set(self.module.calls[1:]), {"R", "L"})
        self.assertEqual(self.join.calls, 1)

    def test_reuses_verified_results_across_runs(self):
        first = self.run_pipeline()
        second = self.run_pipeline()
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(second.cache_hits, ("first", "last"))
        self.assertEqual(self.module.calls, ["A", "B"])
        self.assertEqual(second.outputs["text"].sha256, first.outputs["text"].sha256)

    def test_input_parameters_and_version_invalidate_cache(self):
        self.run_pipeline()
        self.source.write_text("different", encoding="utf-8")
        self.assertEqual(self.run_pipeline().cache_hits, ())
        self.definition["nodes"][0]["parameters"]["label"] = "changed"
        self.assertEqual(self.run_pipeline().cache_hits, ())
        self.module.version = "2"
        self.assertEqual(self.run_pipeline().cache_hits, ())

    def test_corrupt_or_missing_cached_final_is_recomputed(self):
        first = self.run_pipeline()
        first.outputs["text"].path.write_text("corrupt", encoding="utf-8")
        second = self.run_pipeline()
        self.assertEqual(second.cache_hits, ("first",))
        self.assertEqual(second.outputs["text"].path.read_text(), "originalAB")
        second.outputs["text"].path.unlink()
        self.assertEqual(self.run_pipeline().cache_hits, ("first",))

    def test_failure_keeps_temporaries_and_resume_recovers(self):
        self.module.fail_label = "B"
        with self.assertRaises(RunFailed) as caught:
            self.run_pipeline()
        run_id = caught.exception.run_id
        self.assertFalse((self.root / "outputs" / run_id).exists())
        self.assertTrue(
            (self.root / "temps" / run_id / "last" / "checkpoint.txt").is_file()
        )
        result = self.run_pipeline(resume=run_id)
        self.assertEqual(result.run_id, run_id)
        self.assertEqual(result.cache_hits, ("first",))
        self.assertEqual(self.module.saw_cache[-1], ("B", True))
        self.assertEqual(list((self.root / "temps" / run_id).iterdir()), [])

    def test_fresh_resume_discards_only_its_own_work(self):
        self.module.fail_label = "B"
        with self.assertRaises(RunFailed) as caught:
            self.run_pipeline()
        unrelated = self.root / "temps" / "unrelated" / "keep.txt"
        unrelated.parent.mkdir()
        unrelated.write_text("preserve", encoding="utf-8")
        result = self.run_pipeline(resume=caught.exception.run_id, cache="fresh")
        self.assertEqual(result.cache_hits, ())
        self.assertEqual(self.module.saw_cache[-1], ("B", False))
        self.assertEqual(unrelated.read_text(), "preserve")

    def test_changed_input_cannot_reuse_failed_module_temporaries(self):
        self.module.fail_label = "B"
        with self.assertRaises(RunFailed) as caught:
            self.run_pipeline()
        self.source.write_text("new input", encoding="utf-8")
        result = self.run_pipeline(resume=caught.exception.run_id)
        self.assertEqual(self.module.saw_cache[-1], ("B", False))
        self.assertEqual(result.outputs["text"].path.read_text(), "new inputAB")

    def test_recovers_final_rename_without_checkpoint_commit(self):
        result = self.run_pipeline()
        original_mtime = result.outputs["text"].path.stat().st_mtime_ns
        # State left by a process killed between rename and checkpoint commit.
        with self.storage.connect() as connection:
            connection.execute(
                "UPDATE runs SET status='running' WHERE id=?", (result.run_id,)
            )
            connection.execute(
                "UPDATE nodes SET status='running', outputs=NULL WHERE run_id=? AND node_id='last'",
                (result.run_id,),
            )
        recovered = self.run_pipeline(resume=result.run_id)
        self.assertEqual(recovered.outputs["text"].path.read_text(), "originalAB")
        self.assertEqual(
            recovered.outputs["text"].path.stat().st_mtime_ns, original_mtime
        )

    def test_does_not_overwrite_an_unrecognized_final_destination(self):
        self.module.fail_label = "B"
        with self.assertRaises(RunFailed) as caught:
            self.run_pipeline()
        destination = self.root / "outputs" / caught.exception.run_id
        destination.mkdir()
        existing = destination / "user.txt"
        existing.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(RunFailed, "refusing to overwrite"):
            self.run_pipeline(resume=caught.exception.run_id)
        self.assertEqual(existing.read_text(), "preserve")

    def assert_recovered_publication_survives_failure(self, table, condition):
        result = self.run_pipeline()
        directory = result.outputs["text"].path.parent
        original_files = {path.name: path.read_bytes() for path in directory.iterdir()}
        with self.storage.connect() as connection:
            connection.execute(
                "UPDATE runs SET status='running' WHERE id=?", (result.run_id,)
            )
            connection.execute(
                "UPDATE nodes SET status='running', outputs=NULL WHERE run_id=? AND node_id='last'",
                (result.run_id,),
            )
            connection.execute(
                f"CREATE TRIGGER fail_success BEFORE UPDATE OF status ON {table} "
                f"WHEN {condition} BEGIN SELECT RAISE(FAIL, 'simulated database failure'); END"
            )
        try:
            with self.assertRaisesRegex(RunFailed, "simulated database failure"):
                self.run_pipeline(resume=result.run_id)
            files = (
                {path.name: path.read_bytes() for path in directory.iterdir()}
                if directory.exists()
                else {}
            )
            self.assertEqual(files, original_files)
            self.assertTrue((self.root / "temps" / result.run_id / "last").is_dir())
        finally:
            with self.storage.connect() as connection:
                connection.execute("DROP TRIGGER fail_success")
        recovered = self.run_pipeline(resume=result.run_id)
        self.assertEqual(
            recovered.outputs["text"].path.read_bytes(), original_files["text.txt"]
        )

    def test_recovered_output_survives_checkpoint_failure(self):
        self.assert_recovered_publication_survives_failure(
            "nodes", "NEW.node_id='last' AND NEW.status='succeeded'"
        )

    def test_recovered_output_survives_run_status_failure(self):
        self.assert_recovered_publication_survives_failure(
            "runs", "NEW.status='succeeded'"
        )

    def test_new_output_is_removed_on_checkpoint_failure(self):
        self.storage.initialize()
        with self.storage.connect() as connection:
            connection.execute(
                "CREATE TRIGGER fail_success BEFORE UPDATE OF status ON nodes "
                "WHEN NEW.node_id='last' AND NEW.status='succeeded' "
                "BEGIN SELECT RAISE(FAIL, 'simulated database failure'); END"
            )
        with self.assertRaisesRegex(RunFailed, "simulated database failure") as caught:
            self.run_pipeline()
        self.assertFalse((self.root / "outputs" / caught.exception.run_id).exists())
        self.assertTrue(
            (
                self.root
                / "temps"
                / caught.exception.run_id
                / "last"
                / "result"
                / "text.txt"
            ).is_file()
        )

    def test_resume_rejects_changed_configuration_and_completed_run(self):
        completed = self.run_pipeline()
        with self.assertRaisesRegex(PipelineError, "already completed"):
            self.run_pipeline(resume=completed.run_id)
        self.module.fail_label = "B"
        with self.assertRaises(RunFailed) as caught:
            self.run_pipeline(cache="fresh")
        self.definition["nodes"][0]["parameters"]["label"] = "new"
        with self.assertRaisesRegex(PipelineError, "same pipeline"):
            self.run_pipeline(resume=caught.exception.run_id)

    def test_module_cannot_publish_workspace_file(self):
        self.module.bad_output = True
        with self.assertRaisesRegex(RunFailed, "inside result_dir") as caught:
            self.run_pipeline()
        self.assertFalse((self.root / "outputs" / caught.exception.run_id).exists())

    def test_detects_module_modifying_input(self):
        self.module.mutate_source = True
        with self.assertRaisesRegex(RunFailed, "input changed"):
            self.run_pipeline()
        self.assertEqual(list((self.root / "outputs").iterdir()), [])

    def test_copy_failure_does_not_publish_partial_final(self):
        real_copy = shutil.copyfile

        def failing_copy(source, destination, *args, **kwargs):
            if Path(source).read_text() == "originalAB":
                Path(destination).write_text("partial", encoding="utf-8")
                raise OSError("disk full")
            return real_copy(source, destination, *args, **kwargs)

        with (
            patch(
                "content_agent_orchestrator.runner.shutil.copyfile",
                side_effect=failing_copy,
            ),
            self.assertRaisesRegex(RunFailed, "disk full") as caught,
        ):
            self.run_pipeline()
        self.assertFalse((self.root / "outputs" / caught.exception.run_id).exists())
        self.assertTrue(
            list(
                (self.root / "temps" / caught.exception.run_id / "last").glob(
                    "publication-*"
                )
            )
        )

    def test_missing_inputs_do_not_create_execution_state(self):
        with self.assertRaises(PipelineError):
            self.runner.run(Pipeline.parse(self.definition, self.registry), {})
        self.assertFalse(self.root.exists())

    def test_rejects_cycles_and_disconnected_cycles(self):
        definitions = [
            {
                "nodes": [
                    text_node("a", "A", "b"),
                    text_node("b", "B", "a"),
                    text_node("end", "C", "a"),
                ],
                "final": "end",
            },
            {
                "nodes": [
                    text_node("a", "A", "b"),
                    text_node("b", "B", "a"),
                    text_node("end", "C"),
                ],
                "final": "end",
            },
        ]
        for definition in definitions:
            with (
                self.subTest(definition=definition),
                self.assertRaisesRegex(PipelineError, "cycle"),
            ):
                Pipeline.parse(definition, self.registry)

    def test_user_changes_to_pipeline_are_not_overwritten(self):
        definition = copy.deepcopy(self.definition)
        self.run_pipeline()
        self.assertEqual(self.definition, definition)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "user"
        self.storage = UserStorage(self.root)

    def test_rejects_traversal_and_absolute_paths(self):
        for relative in ("../escape", "/tmp/escape", "temps/../../escape"):
            with self.subTest(relative=relative), self.assertRaises(PipelineError):
                self.storage.path(relative)

    def test_rejects_symlink_storage_without_touching_target(self):
        self.root.mkdir()
        target = Path(self.temporary.name) / "outside"
        target.mkdir()
        try:
            (self.root / "temps").symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest("Symlink creation not available")
        with self.assertRaisesRegex(PipelineError, "symlink"):
            self.storage.initialize()
        self.assertEqual(list(target.iterdir()), [])

    def test_execution_lock_excludes_other_runner_and_releases(self):
        self.storage.initialize()
        other = UserStorage(self.root)
        run_id = "a" * 32
        with (
            self.storage.run_lock(run_id),
            self.assertRaisesRegex(ExecutionError, "already active"),
            other.run_lock(run_id),
        ):
            self.fail("Second lock should not have been acquired")
        with other.run_lock(run_id):
            pass
