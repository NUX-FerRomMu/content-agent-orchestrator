"""Prompt-only generators; source videos and embeddings are never sent."""

import subprocess
from pathlib import Path

from .contracts import ExecutionError, PipelineError
from .pipeline import canonical_json
from .virtual_face import read_portrait


class CommandPortraitGenerator:
    def __init__(self, configuration: dict, *, timeout=600):
        if not isinstance(configuration, dict) or set(configuration) != {
            "argv",
            "model",
            "seed",
        }:
            raise PipelineError(
                "Local generator configuration needs argv, model and seed"
            )
        argv, model, seed = (configuration[key] for key in ("argv", "model", "seed"))
        if (
            not isinstance(argv, list)
            or not argv
            or any(
                not isinstance(item, str) or not item or "\x00" in item for item in argv
            )
        ):
            raise PipelineError("Generator argv must be a nonempty list of strings")
        if (
            not isinstance(model, str)
            or not model.strip()
            or len(model) > 256
            or type(seed) is not int
            or not 0 <= seed < 2**32
        ):
            raise PipelineError(
                "Local generator needs a model identifier and uint32 seed"
            )
        self.descriptor = {"backend": "command", **configuration}
        self.timeout = timeout

    def generate(self, prompt: str, directory: Path) -> bytes:
        output = directory / "generated.png"
        if output.is_symlink():
            raise ExecutionError("Generator output must not be a symlink")
        output.unlink(missing_ok=True)
        payload = {
            "prompt": prompt,
            "size": [1024, 1024],
            "seed": self.descriptor["seed"],
            "output_path": str(output),
        }
        try:
            process = subprocess.run(
                self.descriptor["argv"],
                input=canonical_json(payload).encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise ExecutionError(
                "Local image generation timed out; temporaries retained"
            ) from None
        except OSError:
            raise ExecutionError("Cannot start local image generator") from None
        if process.returncode:
            raise ExecutionError(
                f"Local image generator exited with code {process.returncode}"
            )
        return read_portrait(output)
