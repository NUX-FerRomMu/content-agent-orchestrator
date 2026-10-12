"""Local SDXL portrait inference with prepared safetensors and bounded prompts."""

from pathlib import Path

from .contracts import ExecutionError, PipelineError
from .generation_models import model_manifest, verify_model
from .virtual_face import SKIN_TONES, appearance_profile, read_portrait

NEGATIVE_PROMPT = (
    "child, teen, glasses, sunglasses, celebrity, multiple faces, tattoo, piercing, "
    "scar, mole, profile view, tilted head, occlusion, open mouth, blurry, deformed, "
    "asymmetric eyes, cropped head, text, watermark"
)


def sdxl_prompt(profile: dict) -> str:
    values = appearance_profile(profile)
    # Put required identity/appearance/framing attributes before optional cosmetics.
    # Token count is checked against BOTH CLIP tokenizers before any inference.
    return (
        f"Studio photo of one fictional adult {values['gender']}, age {values['age_years']}, "
        f"{SKIN_TONES[values['skin_tone'] - 1]} skin, {values['undertone']} undertone, "
        "ordinary face, frontal head and shoulders, entire head visible, hair back, "
        "neutral expression, closed mouth, gray background, even light, sharp realistic skin. "
        f"{values['features']}"
    )


class SDXLPortraitGenerator:
    def __init__(
        self,
        model_dir: Path,
        *,
        seed=42,
        steps=30,
        guidance=5.0,
        device="cuda",
        offload=True,
    ):
        for name, value, lower, upper in (
            ("seed", seed, 0, 2**32 - 1),
            ("steps", steps, 10, 100),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise PipelineError(f"Invalid SDXL {name}")
        if (
            type(guidance) not in (int, float)
            or not 1 <= guidance <= 12
            or device not in ("cuda", "cpu")
            or type(offload) is not bool
        ):
            raise PipelineError("Invalid SDXL guidance, device or offload setting")
        self.model_dir = model_dir.expanduser().absolute()
        manifest_path = self.model_dir / "generation-model.json"
        manifest, digest = (
            model_manifest(self.model_dir) if manifest_path.exists() else ({}, None)
        )
        self.descriptor = {
            "backend": "sdxl",
            "repository": manifest.get("repository"),
            "revision": manifest.get("revision"),
            "manifest_sha256": digest,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "device": device,
            "offload": offload,
            "prompt_format": "sdxl-portrait-v1",
            "size": [1024, 1024],
        }

    @staticmethod
    def prompt(profile):
        return sdxl_prompt(profile)

    @staticmethod
    def _runtime():
        try:
            import torch
            from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
        except (ImportError, RuntimeError, OSError):
            raise ExecutionError(
                "Local SDXL runtime is not installed; see the optional generation dependencies in docs/virtual-identities.md"
            ) from None
        return torch, StableDiffusionXLPipeline, DPMSolverMultistepScheduler

    @staticmethod
    def _check_tokens(pipeline, prompt):
        for tokenizer in (pipeline.tokenizer, pipeline.tokenizer_2):
            for text in (prompt, NEGATIVE_PROMPT):
                ids = tokenizer(text, truncation=False)["input_ids"]
                if len(ids) > tokenizer.model_max_length:
                    raise PipelineError(
                        "SDXL prompt exceeds the CLIP token budget; shorten cosmetic preferences"
                    )

    def generate(self, prompt: str, directory: Path) -> bytes:
        manifest = verify_model(self.model_dir)
        _, digest = model_manifest(self.model_dir)
        if (
            digest != self.descriptor["manifest_sha256"]
            or manifest["revision"] != self.descriptor["revision"]
        ):
            raise ExecutionError(
                "Prepared model changed after the generation request was created"
            )
        torch, pipeline_class, scheduler_class = self._runtime()
        device = self.descriptor["device"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ExecutionError(
                "SDXL requires an available NVIDIA CUDA GPU; --device cpu is an explicit slow fallback"
            )
        pipeline = None
        try:
            dtype = torch.float16 if device == "cuda" else torch.float32
            # local_files_only forbids downloads during generation. Never load pickle weights.
            pipeline = pipeline_class.from_pretrained(
                str(self.model_dir),
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
                add_watermarker=False,
            )
            self._check_tokens(pipeline, prompt)
            pipeline.scheduler = scheduler_class.from_config(
                pipeline.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
            pipeline.enable_vae_tiling()
            pipeline.enable_vae_slicing()
            if device == "cuda" and self.descriptor["offload"]:
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(device)
            pipeline.set_progress_bar_config(disable=True)
            generator = torch.Generator(device="cpu").manual_seed(
                self.descriptor["seed"]
            )
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=1024,
                    height=1024,
                    num_images_per_prompt=1,
                    num_inference_steps=self.descriptor["steps"],
                    guidance_scale=self.descriptor["guidance"],
                    generator=generator,
                )
            if len(result.images) != 1:
                raise ExecutionError("SDXL must produce exactly one portrait")
            path = directory / "sdxl.png"
            if path.is_symlink():
                raise ExecutionError("SDXL output path must not be a symlink")
            result.images[0].convert("RGB").save(path, format="PNG")
            content = read_portrait(path)
            verify_model(self.model_dir)
            return content
        except torch.cuda.OutOfMemoryError:
            raise ExecutionError(
                "SDXL ran out of GPU memory; enable CPU offload and close other GPU workloads"
            ) from None
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            if isinstance(exc, (PipelineError, ExecutionError)):
                raise
            raise ExecutionError(
                "Local SDXL inference failed; check prepared model compatibility and generation dependencies"
            ) from None
        finally:
            if pipeline is not None:
                del pipeline
            if device == "cuda":
                torch.cuda.empty_cache()
