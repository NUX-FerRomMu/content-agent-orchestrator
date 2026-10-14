import copy
import unittest

from content_agent_orchestrator.contracts import PipelineError
from content_agent_orchestrator.modules import builtin_registry
from content_agent_orchestrator.pipeline import Pipeline


def inspection_node(node_id="inspect", external="video"):
    return {
        "id": node_id,
        "module": "video.inspect",
        "inputs": {"video": {"external": external}},
    }


def report_node(source="inspect"):
    return {
        "id": "export",
        "module": "report.export",
        "inputs": {"inspection": {"node": source, "output": "manifest"}},
    }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.registry = builtin_registry()
        self.definition = {
            "nodes": [report_node(), inspection_node()],
            "final": "export",
        }

    def test_orders_dependencies_and_does_not_mutate_definition(self):
        original = copy.deepcopy(self.definition)
        pipeline = Pipeline.parse(self.definition, self.registry)
        self.assertEqual([node.id for node in pipeline.nodes], ["inspect", "export"])
        self.assertEqual(self.definition, original)

    def test_rejects_multiple_terminal_nodes(self):
        self.definition["nodes"].append(inspection_node("unused"))
        with self.assertRaisesRegex(PipelineError, "single declared final"):
            Pipeline.parse(self.definition, self.registry)

    def test_unvalidated_swap_cannot_be_declared_as_the_final_node(self):
        from content_agent_orchestrator.cli import video_anonymization_definition

        definition = video_anonymization_definition()
        definition["nodes"] = definition["nodes"][:7]
        definition["final"] = "swap"
        with self.assertRaisesRegex(PipelineError, "face.swap cannot be final"):
            Pipeline.parse(definition, self.registry)

    def test_rejects_unknown_dependency(self):
        self.definition["nodes"][0] = report_node("absent")
        with self.assertRaisesRegex(PipelineError, "Unknown dependency"):
            Pipeline.parse(self.definition, self.registry)

    def test_rejects_wrong_output_type(self):
        self.definition["nodes"][0]["inputs"]["inspection"]["output"] = "timestamps"
        with self.assertRaisesRegex(PipelineError, "Incompatible"):
            Pipeline.parse(self.definition, self.registry)

    def test_rejects_malformed_configuration(self):
        variants = [None, [], {}, {"nodes": [], "final": "export"}]
        for field, value in (
            ("id", "../escape"),
            ("module", ["video.inspect"]),
            ("inputs", []),
            ("parameters", {"timeout_seconds": True}),
            ("parameters", {"unexpected": 1}),
        ):
            definition = copy.deepcopy(self.definition)
            definition["nodes"][1][field] = value
            variants.append(definition)
        for definition in variants:
            with self.subTest(definition=definition), self.assertRaises(PipelineError):
                Pipeline.parse(definition, self.registry)

    def test_rejects_duplicate_nodes_and_unknown_final(self):
        self.definition["nodes"].append(inspection_node())
        with self.assertRaisesRegex(PipelineError, "unique"):
            Pipeline.parse(self.definition, self.registry)
        self.definition["nodes"].pop()
        self.definition["final"] = "missing"
        with self.assertRaisesRegex(PipelineError, "does not exist"):
            Pipeline.parse(self.definition, self.registry)
