import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_generation_models import prepared_fixture
from test_virtual_faces import FIXTURE, PROFILE

from content_agent_orchestrator.cli import main
from content_agent_orchestrator.contracts import ExecutionError, PipelineError
from content_agent_orchestrator.sdxl_generator import (
    NEGATIVE_PROMPT,
    SDXLPortraitGenerator,
    sdxl_prompt,
)


class TestOutOfMemory(RuntimeError):
    pass


class SDXLTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        prepared_fixture(self.model)
        self.pipeline = Mock()
        self.pipeline.tokenizer = self.pipeline.tokenizer_2 = Mock(
            return_value={"input_ids": [1] * 60}
        )
        self.pipeline.tokenizer.model_max_length = 77
        image = Mock()
        image.convert.return_value = image
        image.save.side_effect = lambda path, **kwargs: path.write_bytes(
            FIXTURE.read_bytes()
        )
        self.pipeline.return_value = SimpleNamespace(images=[image])
        self.pipeline_class = Mock(from_pretrained=Mock(return_value=self.pipeline))
        self.scheduler = Mock()
        self.torch = SimpleNamespace(
            float16="float16",
            float32="float32",
            cuda=SimpleNamespace(
                is_available=lambda: True,
                empty_cache=Mock(),
                OutOfMemoryError=TestOutOfMemory,
            ),
            Generator=Mock(return_value=Mock()),
            inference_mode=contextlib.nullcontext,
        )
        self.runtime = patch.object(
            SDXLPortraitGenerator,
            "_runtime",
            return_value=(self.torch, self.pipeline_class, self.scheduler),
        )
        self.runtime.start()
        self.addCleanup(self.runtime.stop)

    def test_inference_is_offline_safetensors_seeded_and_offloaded(self):
        generator = SDXLPortraitGenerator(self.model, seed=123)
        content = generator.generate(sdxl_prompt(PROFILE), self.root)
        self.assertEqual(content, FIXTURE.read_bytes())
        options = self.pipeline_class.from_pretrained.call_args.kwargs
        self.assertTrue(options["use_safetensors"])
        self.assertTrue(options["local_files_only"])
        self.assertEqual(options["torch_dtype"], "float16")
        self.pipeline.enable_model_cpu_offload.assert_called_once()
        self.pipeline.enable_vae_tiling.assert_called_once()
        self.pipeline.enable_vae_slicing.assert_called_once()
        call = self.pipeline.call_args.kwargs
        self.assertEqual(call["negative_prompt"], NEGATIVE_PROMPT)
        self.assertEqual(
            (call["width"], call["height"], call["num_images_per_prompt"]),
            (1024, 1024, 1),
        )
        self.torch.Generator.return_value.manual_seed.assert_called_once_with(123)
        self.torch.cuda.empty_cache.assert_called_once()

    def test_cpu_fallback_is_explicit_and_uses_float32(self):
        SDXLPortraitGenerator(self.model, device="cpu").generate(
            sdxl_prompt(PROFILE), self.root
        )
        self.assertEqual(
            self.pipeline_class.from_pretrained.call_args.kwargs["torch_dtype"],
            "float32",
        )
        self.pipeline.to.assert_called_once_with("cpu")
        self.pipeline.enable_model_cpu_offload.assert_not_called()

    def test_missing_cuda_and_out_of_memory_fail_without_approved_candidate(self):
        self.torch.cuda.is_available = lambda: False
        with self.assertRaisesRegex(ExecutionError, "NVIDIA CUDA"):
            SDXLPortraitGenerator(self.model).generate(sdxl_prompt(PROFILE), self.root)
        self.pipeline_class.from_pretrained.assert_not_called()
        self.torch.cuda.is_available = lambda: True
        self.pipeline.side_effect = TestOutOfMemory()
        with self.assertRaisesRegex(ExecutionError, "out of GPU memory"):
            SDXLPortraitGenerator(self.model).generate(sdxl_prompt(PROFILE), self.root)
        self.torch.cuda.empty_cache.assert_called_once()

    def test_prompt_overflow_is_rejected_instead_of_silent_truncation(self):
        self.pipeline.tokenizer.return_value = {"input_ids": list(range(100))}
        with self.assertRaisesRegex(PipelineError, "token budget"):
            SDXLPortraitGenerator(self.model).generate(sdxl_prompt(PROFILE), self.root)
        self.pipeline.assert_not_called()

    def test_changed_model_is_rejected_before_runtime_loading(self):
        generator = SDXLPortraitGenerator(self.model)
        path = self.model / "generation-model.json"
        data = json.loads(path.read_text())
        data["revision"] = "b" * 40
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ExecutionError, "changed"):
            generator.generate(sdxl_prompt(PROFILE), self.root)
        self.pipeline_class.from_pretrained.assert_not_called()

    def test_required_profile_attributes_precede_cosmetic_preferences(self):
        prompt = sdxl_prompt(PROFILE)
        self.assertIn("adult woman", prompt)
        self.assertIn("age 35", prompt)
        self.assertIn("medium deep skin", prompt)
        self.assertLess(prompt.index("frontal"), prompt.index("Subtle eyeliner"))
        self.assertIn("child", NEGATIVE_PROMPT)
        self.assertIn("glasses", NEGATIVE_PROMPT)

    def test_cli_rejects_removed_openai_backend(self):
        with (
            contextlib.redirect_stderr(__import__("io").StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            main(
                [
                    "virtual",
                    "create",
                    "source",
                    "--profile",
                    "unused.json",
                    "--backend",
                    "openai",
                ]
            )
        self.assertEqual(caught.exception.code, 2)
