"""Explicit preparation of the exact, checksum-pinned native swap weights."""

import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .contracts import ExecutionError
from .storage import UserStorage, digest_file

RELEASE = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0"
)
MODEL_DIRECTORY = "artifacts/models/inswapper-128-models-3.0.0"
SWAP_MODELS = {
    "swap_model": {
        "filename": "inswapper_128.onnx",
        "size_bytes": 555303150,
        "sha256": "a290273ed497312095dac48cdef20feec9d5208298223dd01288ab202b54bea7",
    },
    "arcface_model": {
        "filename": "arcface_w600k_r50.onnx",
        "size_bytes": 174388474,
        "sha256": "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1",
    },
}


class ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    """Allow GitHub's HTTPS asset redirect without accepting arbitrary hosts."""

    def redirect_request(self, request, file, code, message, headers, new_url):
        parsed = urllib.parse.urlparse(new_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname
            not in {
                "github.com",
                "release-assets.githubusercontent.com",
                "objects.githubusercontent.com",
            }
            or parsed.username
            or parsed.password
        ):
            raise ExecutionError("Unsupported swap model download redirect")
        return super().redirect_request(request, file, code, message, headers, new_url)


def prepared_swap_paths(storage: UserStorage) -> dict:
    """Return deterministic paths; preparation/download is always explicit."""
    return {
        name: storage.path(f"{MODEL_DIRECTORY}/{item['filename']}")
        for name, item in SWAP_MODELS.items()
    }


def _matches(path, record) -> bool:
    return (
        not path.is_symlink()
        and path.is_file()
        and path.stat().st_size == record["size_bytes"]
        and digest_file(path)[0] == record["sha256"]
    )


def _download(path, record) -> None:
    """Keep a verified completed download; restart partial bytes on retry."""
    if _matches(path, record):
        return
    partial = path.with_suffix(".partial")
    if path.is_symlink() or partial.is_symlink():
        raise ExecutionError("Swap model downloads must not be symlinks")
    partial.unlink(missing_ok=True)
    print(
        f"Downloading {record['filename']} ({record['size_bytes']} bytes); verifying SHA-256 before publication",
        file=sys.stderr,
        flush=True,
    )
    try:
        with (
            urllib.request.build_opener(ReleaseRedirect()).open(
                f"{RELEASE}/{record['filename']}", timeout=60
            ) as response,
            partial.open("xb") as output,
        ):
            if os.name != "nt":
                os.fchmod(output.fileno(), 0o600)
            size = 0
            progress = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > record["size_bytes"]:
                    raise ExecutionError("Swap model exceeds its pinned size")
                output.write(block)
                if size - progress >= 64 * 1024**2:
                    print(
                        f"{record['filename']}: {size}/{record['size_bytes']} bytes",
                        file=sys.stderr,
                        flush=True,
                    )
                    progress = size
            output.flush()
            os.fsync(output.fileno())
        if not _matches(partial, record):
            raise ExecutionError("Swap model checksum or size mismatch")
        partial.replace(path)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ExecutionError(
            "Swap model download failed; retry models prepare-swap to reuse completed files"
        ) from exc


def prepare_swap_models(storage: UserStorage) -> dict:
    """Publish both pinned weights only after verifying the complete set."""
    storage.initialize()
    paths = prepared_swap_paths(storage)
    destination = storage.path(MODEL_DIRECTORY)
    if destination.exists() and not destination.is_dir():
        raise ExecutionError(
            "Swap model destination is not a directory; existing file preserved"
        )
    with storage.run_lock("prepare-inswapper-models-3-0-0"):
        if all(_matches(paths[name], item) for name, item in SWAP_MODELS.items()):
            return paths
        work = storage.path("temps/swap_model_preparation/models-3.0.0")
        work.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name, record in SWAP_MODELS.items():
            target = storage.path(work.relative_to(storage.root) / record["filename"])
            if _matches(paths[name], record) and not _matches(target, record):
                shutil.copyfile(paths[name], target, follow_symlinks=False)
                target.chmod(0o600)
            _download(target, record)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        backup = storage.path(f"{MODEL_DIRECTORY}.backup-{uuid.uuid4().hex}")
        if destination.exists():
            destination.rename(backup)
        try:
            work.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    return paths
