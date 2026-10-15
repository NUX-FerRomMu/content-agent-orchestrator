"""Read-only local readiness checks; never download weights or run inference."""

import importlib.metadata
import shutil
import sys

from .contracts import ExecutionError
from .face_data import YUNET_SHA256, bundled_yunet
from .identity_data import SFACE_SHA256, bundled_sface
from .storage import UserStorage, digest_file
from .swap_models import SWAP_MODELS, prepared_swap_paths


def diagnose(
    storage: UserStorage, device="cpu", generation=False, models=False
) -> dict:
    """Report missing tools, runtime/provider support and optional model integrity."""
    checks = []

    def add(name, ready, detail):
        checks.append({"name": name, "ready": bool(ready), "detail": detail})

    add("python", sys.version_info >= (3, 12), sys.version.split()[0])
    for command in ("ffmpeg", "ffprobe"):
        path = shutil.which(command)
        add(
            command,
            path is not None,
            path or "Install FFmpeg and add its bin directory to PATH",
        )
    for package in ("numpy", "opencv-python-headless", "av", "onnx"):
        try:
            add(package, True, importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            add(package, False, "uv sync --locked --group swap (or --group swap-cuda)")
    try:
        import onnxruntime as ort

        ort.disable_telemetry_events()
        providers = ort.get_available_providers()
        required = (
            "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        )
        add(
            "onnxruntime",
            required in providers,
            {
                "required": required,
                "available": providers,
                "note": "Availability does not validate GPU inference; the swap engine also checks the actual session provider",
            },
        )
    except (ImportError, OSError) as exc:
        add("onnxruntime", False, str(exc))
    for name, path, checksum in (
        ("yunet", bundled_yunet(), YUNET_SHA256),
        ("sface", bundled_sface(), SFACE_SHA256),
    ):
        add(name, path.is_file() and digest_file(path)[0] == checksum, str(path))
    if models:
        for name, path in prepared_swap_paths(storage).items():
            record = SWAP_MODELS[name]
            ready = path.is_file() and digest_file(path) == (
                record["sha256"],
                record["size_bytes"],
            )
            add(
                name, ready, str(path) if ready else "content-agent models prepare-swap"
            )
    if generation:
        try:
            from .sdxl_generator import SDXLPortraitGenerator

            SDXLPortraitGenerator._runtime()
            import torch

            add(
                "generation",
                device == "cpu" or torch.cuda.is_available(),
                {
                    "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                },
            )
        except (ExecutionError, ImportError, OSError) as exc:
            add("generation", False, str(exc))
    return {
        "ready": all(item["ready"] for item in checks),
        "storage_root": str(storage.root),
        "device": device,
        "checks": checks,
    }
