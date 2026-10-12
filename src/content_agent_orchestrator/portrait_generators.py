"""Prompt-only generators; source videos and embeddings are never sent."""

import base64
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .contracts import ExecutionError, PipelineError
from .pipeline import canonical_json
from .virtual_face import MAX_PNG_BYTES, read_portrait


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


class OpenAIPortraitGenerator:
    def __init__(self, model="gpt-image-1", *, timeout=300):
        if model not in ("gpt-image-1", "gpt-image-1.5", "gpt-image-1-mini"):
            raise PipelineError("Unsupported image generation model")
        self.descriptor = {
            "backend": "openai",
            "model": model,
            "size": "1024x1024",
            "quality": "high",
            "output_format": "png",
            "background": "opaque",
        }
        self.timeout = timeout

    def generate(self, prompt: str, directory: Path) -> bytes:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ExecutionError(
                "Set OPENAI_API_KEY in the environment for cloud generation"
            )
        request = urllib.request.Request(
            "https://api.openai.com/v1/images/generations",
            data=canonical_json(
                {
                    key: value
                    for key, value in self.descriptor.items()
                    if key != "backend"
                }
                | {"prompt": prompt, "n": 1}
            ).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            opener = urllib.request.build_opener(NoRedirect())
            with opener.open(request, timeout=self.timeout) as response:
                encoded = response.read(24 * 1024 * 1024 + 1)
            if len(encoded) > 24 * 1024 * 1024:
                raise ExecutionError("Image provider response exceeds the size limit")
            payload = json.loads(encoded)
            content = base64.b64decode(payload["data"][0]["b64_json"], validate=True)
            if len(content) > MAX_PNG_BYTES:
                raise ExecutionError("Generated portrait exceeds the size limit")
            return content
        except urllib.error.HTTPError as exc:
            # Never echo provider bodies or request headers; they can contain secrets.
            raise ExecutionError(
                f"Image generation HTTP error {exc.code}; no automatic retry"
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ExecutionError(
                "Image generation transport failed; a request may already have been billed"
            ) from None
        except (ValueError, TypeError, KeyError, IndexError):
            raise ExecutionError(
                "Image provider did not return a valid base64 PNG"
            ) from None


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
