import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from content_agent_orchestrator.contracts import ExecutionError
from content_agent_orchestrator.swap_engine import ARCFACE_POINTS, InSwapper, alignment

HAS_ONNX = bool(
    importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime")
)


def tiny_models(root):
    """Untrained ONNX fixtures with real adapter signatures; NOT InSwapper weights."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    swap = root / "untrained-swap.onnx"
    encoder = root / "untrained-arcface.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["target"], ["output"])],
        "test-only-passthrough",
        [
            helper.make_tensor_value_info(
                "target", TensorProto.FLOAT, [1, 3, 128, 128]
            ),
            helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, 512]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 128, 128])],
        [numpy_helper.from_array(np.eye(512, dtype=np.float32), name="emap")],
    )
    onnx.save(
        helper.make_model(
            graph, ir_version=10, opset_imports=[helper.make_opsetid("", 13)]
        ),
        swap,
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["embedding"],
                value=numpy_helper.from_array(np.ones((1, 512), np.float32)),
            )
        ],
        "test-only-constant-embedding",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 112, 112])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 512])],
    )
    onnx.save(
        helper.make_model(
            graph, ir_version=10, opset_imports=[helper.make_opsetid("", 13)]
        ),
        encoder,
    )
    return swap, encoder


class AlignmentTests(unittest.TestCase):
    def test_similarity_alignment_is_deterministic_and_handles_rotation(self):
        source = ARCFACE_POINTS @ np.array([[0, 2], [-2, 0]]) + [140, 30]
        transform = alignment(source, 112)
        actual = np.column_stack((source, np.ones(5))) @ transform.T
        np.testing.assert_allclose(actual, ARCFACE_POINTS, atol=1e-4)
        np.testing.assert_array_equal(transform, alignment(source, 112))
        shifted = (
            np.column_stack((ARCFACE_POINTS, np.ones(5)))
            @ alignment(ARCFACE_POINTS, 128).T
        )
        np.testing.assert_allclose(shifted, ARCFACE_POINTS + [8, 0], atol=1e-4)

    def test_degenerate_landmarks_are_rejected(self):
        with self.assertRaises(ExecutionError):
            alignment(np.zeros((5, 2)), 112)


@unittest.skipUnless(HAS_ONNX, "Optional swap runtime required")
class NativeSwapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.swap, self.encoder = tiny_models(self.root)

    def test_real_onnx_runtime_loads_alignment_embedding_map_and_prediction(self):
        from test_virtual_faces import FIXTURE

        engine = InSwapper(self.swap, self.encoder)
        engine.prepare_target("B", FIXTURE.read_bytes())
        image = np.full((256, 256, 3), 96, np.uint8)
        result = engine.replace(
            image, {"landmarks": (ARCFACE_POINTS + [50, 50]).tolist()}, "B"
        )
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(engine.latents["B"].shape, (1, 512))
        self.assertAlmostEqual(float(np.linalg.norm(engine.latents["B"])), 1)

    def test_cuda_is_explicit_and_never_silently_uses_cpu(self):
        with (
            patch(
                "onnxruntime.get_available_providers",
                return_value=["CPUExecutionProvider"],
            ),
            self.assertRaisesRegex(ExecutionError, "CUDA ONNX provider unavailable"),
        ):
            InSwapper(self.swap, self.encoder, device="cuda")

    def test_backend_exception_becomes_a_frame_failure_without_modifying_input(self):
        from test_virtual_faces import FIXTURE

        engine = InSwapper(self.swap, self.encoder)
        engine.prepare_target("B", FIXTURE.read_bytes())
        image = np.full((256, 256, 3), 96, np.uint8)
        original = image.copy()
        with (
            patch.object(
                engine.swap, "run", side_effect=Exception("backend-specific failure")
            ),
            self.assertRaisesRegex(ExecutionError, "ONNX inference failed"),
        ):
            engine.replace(
                image, {"landmarks": (ARCFACE_POINTS + [50, 50]).tolist()}, "B"
            )
        np.testing.assert_array_equal(image, original)

    def test_external_weight_paths_are_rejected_before_onnx_runtime_load(self):
        import onnx

        model = onnx.load(self.swap)
        weight = model.graph.initializer[-1]
        weight.ClearField("raw_data")
        weight.data_location = onnx.TensorProto.EXTERNAL
        entry = weight.external_data.add()
        entry.key, entry.value = "location", "../../outside"
        onnx.save(model, self.swap)
        with patch("onnxruntime.InferenceSession") as session:
            with self.assertRaisesRegex(ExecutionError, "internally"):
                InSwapper(self.swap, self.encoder)
            session.assert_not_called()

    def test_external_constant_tensor_is_also_rejected(self):
        import onnx

        model = onnx.load(self.encoder)
        tensor = model.graph.node[0].attribute[0].t
        tensor.ClearField("raw_data")
        tensor.data_location = onnx.TensorProto.EXTERNAL
        entry = tensor.external_data.add()
        entry.key, entry.value = "location", "../../outside"
        onnx.save(model, self.encoder)
        with self.assertRaisesRegex(ExecutionError, "internally"):
            InSwapper(self.swap, self.encoder)
