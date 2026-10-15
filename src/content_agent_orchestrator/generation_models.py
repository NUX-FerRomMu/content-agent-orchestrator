"""Explicit, verified preparation of SDXL weights; inference stays offline."""

import hashlib
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import ClassVar

from .contracts import ExecutionError, PipelineError
from .face_data import read_json
from .pipeline import canonical_json
from .storage import UserStorage, digest_file

MODEL_REPOSITORIES = {
    "realvis-xl": "SG161222/RealVisXL_V5.0",
    "sdxl-base": "stabilityai/stable-diffusion-xl-base-1.0",
}
COMPONENTS = (
    "scheduler",
    "tokenizer",
    "tokenizer_2",
    "text_encoder",
    "text_encoder_2",
    "unet",
    "vae",
)
WEIGHT_COMPONENTS = ("text_encoder", "text_encoder_2", "unet", "vae")
LICENSE = "openrail++"


class ModelRedirect(urllib.request.HTTPRedirectHandler):
    allowed: ClassVar = {
        "huggingface.co",
        "cdn-lfs.huggingface.co",
        "cdn-lfs-us-1.huggingface.co",
        "cas-bridge.xethub.hf.co",
        "us.aws.cdn.hf.co",
    }

    def redirect_request(self, request, file, code, message, headers, new_url):
        parsed = urllib.parse.urlparse(new_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in self.allowed
            or parsed.username
            or parsed.password
        ):
            raise ExecutionError(
                "Model download redirected to an unsupported destination"
            )
        return super().redirect_request(request, file, code, message, headers, new_url)


def model_metadata(repository: str, revision: str) -> dict:
    if (
        repository not in MODEL_REPOSITORIES.values()
        or re.fullmatch(r"[A-Za-z0-9._-]{1,80}", revision) is None
    ):
        raise PipelineError("Unsupported SDXL model repository or revision")
    url = (
        f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true"
    )
    try:
        with urllib.request.build_opener(ModelRedirect()).open(
            url, timeout=30
        ) as response:
            content = response.read(4 * 1024 * 1024 + 1)
        if len(content) > 4 * 1024 * 1024:
            raise ValueError
        metadata = json.loads(content)
        if (
            metadata.get("id") != repository
            or re.fullmatch(r"[0-9a-f]{40}", metadata.get("sha", "")) is None
        ):
            raise ValueError
        if metadata.get("gated"):
            raise ExecutionError(
                "Selected model requires gated access; no credentials are requested automatically"
            )
        if metadata.get("cardData", {}).get("license") != LICENSE:
            raise ExecutionError(
                "Model must declare the verified CreativeML Open RAIL++-M license"
            )
        if not isinstance(metadata.get("siblings"), list):
            raise TypeError
        return metadata
    except urllib.error.HTTPError as exc:
        raise ExecutionError(
            f"Model metadata HTTP error {exc.code}; check Hugging Face network access"
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ExecutionError("Cannot reach Hugging Face model metadata") from None
    except (ValueError, TypeError, AttributeError):
        raise ExecutionError("Invalid model repository metadata") from None


def selected_files(metadata: dict) -> list[dict]:
    entries = {}
    for item in metadata["siblings"]:
        if not isinstance(item, dict) or not isinstance(item.get("rfilename"), str):
            raise ExecutionError("Invalid model file metadata")
        name = item["rfilename"]
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or name in entries:
            raise ExecutionError("Unsafe or duplicate model filename")
        entries[name] = item
    chosen = {"model_index.json", "README.md"}
    chosen.update(
        name
        for name in entries
        if "/" not in name and name.lower().startswith("license")
    )
    for component in COMPONENTS:
        chosen.update(
            name
            for name in entries
            if name.startswith(component + "/")
            and name.endswith((".json", ".txt"))
            and not name.endswith(".index.json")
        )
    for component in WEIGHT_COMPONENTS:
        stem = (
            "model"
            if component.startswith("text_encoder")
            else "diffusion_pytorch_model"
        )
        name = f"{component}/{stem}.fp16.safetensors"
        if name not in entries:
            name = f"{component}/{stem}.safetensors"
        if name not in entries:
            raise ExecutionError(
                "Model needs single-file safetensors weights for every SDXL component"
            )
        chosen.add(name)
    files = []
    for name in sorted(chosen):
        if name not in entries:
            raise ExecutionError(f"Missing SDXL model file: {name}")
        item = entries[name]
        lfs = item.get("lfs")
        if lfs is not None:
            if not isinstance(lfs, dict):
                raise ExecutionError("Invalid model LFS metadata")
            checksum, algorithm, size = lfs.get("sha256"), "sha256", lfs.get("size")
        else:
            checksum, algorithm, size = item.get("blobId"), "git-sha1", item.get("size")
        length = 64 if algorithm == "sha256" else 40
        if (
            not isinstance(checksum, str)
            or re.fullmatch(f"[0-9a-f]{{{length}}}", checksum) is None
            or type(size) is not int
            or not 0 <= size <= 12 * 1024**3
        ):
            raise ExecutionError("Model files need verified checksum and size metadata")
        # Normalize component filenames so Diffusers can load mixed upstream
        # fp16/non-fp16 snapshots without a global variant assumption.
        files.append(
            {
                "path": name.replace(".fp16.safetensors", ".safetensors"),
                "upstream_path": name,
                "upstream_checksum": checksum,
                "upstream_algorithm": algorithm,
                "size_bytes": size,
            }
        )
    if sum(item["size_bytes"] for item in files) > 24 * 1024**3:
        raise ExecutionError("Model snapshot exceeds the 24 GiB download limit")
    return files


def _upstream_digest(path: Path, record: dict) -> str:
    if record["upstream_algorithm"] == "sha256":
        return digest_file(path)[0]
    digest = hashlib.sha1(f"blob {record['size_bytes']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path, record: dict) -> None:
    if path.is_symlink():
        raise ExecutionError("Model files must not be symlinks")
    if (
        path.exists()
        and path.stat().st_size == record["size_bytes"]
        and _upstream_digest(path, record) == record["upstream_checksum"]
    ):
        return
    partial = path.with_name(path.name + ".partial")
    if partial.is_symlink():
        raise ExecutionError("Model download partial must not be a symlink")
    # Completed files are reused; incomplete downloads restart, never get blessed.
    partial.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    print(
        f"Downloading {record['path']} ({record['size_bytes']} bytes); verifying upstream checksum",
        file=sys.stderr,
        flush=True,
    )
    try:
        with (
            urllib.request.build_opener(ModelRedirect()).open(
                url, timeout=60
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
                    raise ExecutionError("Model download exceeds declared size")
                output.write(block)
                if size - progress >= 64 * 1024**2:
                    print(
                        f"{record['path']}: {size}/{record['size_bytes']} bytes",
                        file=sys.stderr,
                        flush=True,
                    )
                    progress = size
            output.flush()
            os.fsync(output.fileno())
        if (
            size != record["size_bytes"]
            or _upstream_digest(partial, record) != record["upstream_checksum"]
        ):
            raise ExecutionError("Model download checksum or size mismatch")
        partial.replace(path)
    except urllib.error.HTTPError as exc:
        raise ExecutionError(
            f"Model download HTTP error {exc.code}; temporaries retained"
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ExecutionError(
            "Model download failed; completed files and temporaries retained"
        ) from None


def prepare_model(
    storage: UserStorage, model: str = "realvis-xl", revision: str = "main"
) -> Path:
    if model not in MODEL_REPOSITORIES:
        raise PipelineError("Unknown portrait checkpoint")
    storage.initialize()
    repository = MODEL_REPOSITORIES[model]
    metadata = model_metadata(repository, revision)
    files = selected_files(metadata)
    commit = metadata["sha"]
    relative = f"artifacts/models/{model}/{commit}"
    destination = storage.path(relative)
    work = storage.path(f"temps/model_preparation/{model}-{commit}")
    with storage.run_lock(f"model-{model}-{commit}"):
        if destination.exists():
            try:
                verify_model(destination)
            except (ExecutionError, OSError):
                # Keep the damaged snapshot until its replacement is verified.
                pass
            else:
                return destination
        work.mkdir(parents=True, exist_ok=True, mode=0o700)
        for record in files:
            path = storage.path(work.relative_to(storage.root) / record["path"])
            url = f"https://huggingface.co/{repository}/resolve/{commit}/{record['upstream_path']}"
            _download(url, path, record)
            record["sha256"] = digest_file(path)[0]
        manifest = {
            "schema_version": 1,
            "repository": repository,
            "revision": commit,
            "license": LICENSE,
            "files": files,
        }
        manifest_path = storage.path(
            work.relative_to(storage.root) / "generation-model.json"
        )
        manifest_path.unlink(missing_ok=True)
        with manifest_path.open("x", encoding="utf-8") as stream:
            if os.name != "nt":
                os.fchmod(stream.fileno(), 0o600)
            stream.write(canonical_json(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        verify_model(work)
        if destination.exists():
            backup = storage.path(
                f"temps/model_preparation/{model}-{commit}-corrupt-{uuid.uuid4().hex}"
            )
            destination.replace(backup)
            try:
                work.replace(destination)
            except BaseException:
                backup.replace(destination)
                raise
            shutil.rmtree(backup)
        else:
            work.replace(destination)
    return destination


def model_manifest(directory: Path) -> tuple[dict, str]:
    path = directory / "generation-model.json"
    if directory.is_symlink() or path.is_symlink() or not path.is_file():
        raise ExecutionError("Prepare a local model first with virtual prepare-model")
    manifest = read_json(path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("repository") not in MODEL_REPOSITORIES.values()
        or manifest.get("license") != LICENSE
        or not isinstance(manifest.get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]) is None
        or not isinstance(manifest.get("files"), list)
    ):
        raise ExecutionError("Invalid local generation model manifest")
    return manifest, digest_file(path)[0]


def verify_model(directory: Path) -> dict:
    manifest, _ = model_manifest(directory)
    seen = set()
    for record in manifest["files"]:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ExecutionError("Invalid generation model file record")
        relative = Path(record["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in record["path"]
            or record["path"] in seen
        ):
            raise ExecutionError("Unsafe model manifest path")
        seen.add(record["path"])
        candidate = directory
        for part in relative.parts:
            candidate /= part
            if candidate.is_symlink():
                raise ExecutionError("Model paths must not contain symlinks")
        if not candidate.is_file() or digest_file(candidate) != (
            record.get("sha256"),
            record.get("size_bytes"),
        ):
            raise ExecutionError("Local model checksum mismatch; inference refused")
    expected = {"model_index.json"}
    for component in WEIGHT_COMPONENTS:
        if not any(
            name.startswith(component + "/") and name.endswith(".safetensors")
            for name in seen
        ):
            raise ExecutionError("Missing verified SDXL component weights")
    paths = list(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ExecutionError("Model snapshot must not contain symlinks")
    if not expected <= seen or any(
        path.suffix in (".bin", ".pt", ".pth", ".pkl", ".py")
        for path in paths
        if path.is_file()
    ):
        raise ExecutionError(
            "Model snapshot must contain only safe local data and safetensors weights"
        )
    actual = {str(path.relative_to(directory)) for path in paths if path.is_file()}
    if actual != seen | {"generation-model.json"}:
        raise ExecutionError("Model snapshot contains unverified files")
    return manifest
