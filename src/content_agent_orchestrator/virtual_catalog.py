"""Bounded virtual portrait bundles, with bytes covered by pipeline hashes."""

import hashlib
import io
import json
import uuid
import zipfile
from pathlib import Path

from .contracts import ExecutionError
from .virtual_face import MAX_PNG_BYTES, appearance_profile, png_image

MAX_CATALOG_BYTES = 128 * 1024 * 1024


def read_catalog(path: Path) -> tuple[dict, dict[str, bytes]]:
    if path.stat().st_size > MAX_CATALOG_BYTES:
        raise ExecutionError("Virtual catalog exceeds the size limit")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise ValueError
            if any(
                info.file_size > MAX_PNG_BYTES
                or info.compress_type != zipfile.ZIP_STORED
                or info.flag_bits & 1
                for info in archive.infolist()
            ):
                raise ValueError
            manifest = json.loads(archive.read("manifest.json"))
            if not isinstance(manifest, dict) or set(manifest) != {
                "schema_version",
                "registry_id",
                "source_revision",
                "revision",
                "faces",
            }:
                raise ValueError
            if manifest["schema_version"] != 1 or not isinstance(
                manifest["faces"], list
            ):
                raise ValueError
            if str(uuid.UUID(manifest["registry_id"])) != manifest["registry_id"]:
                raise ValueError
            if any(
                type(manifest[key]) is not int or manifest[key] < 0
                for key in ("revision", "source_revision")
            ):
                raise ValueError
            expected = {"manifest.json"}
            identities = set()
            portraits = {}
            for item in manifest["faces"]:
                if not isinstance(item, dict) or set(item) != {
                    "identity_id",
                    "virtual_id",
                    "profile",
                    "portrait_sha256",
                    "portrait_file",
                    "reviewed",
                }:
                    raise ValueError
                for key in ("identity_id", "virtual_id"):
                    if str(uuid.UUID(item[key])) != item[key]:
                        raise ValueError
                if (
                    item["identity_id"] in identities
                    or item["reviewed"] is not True
                    or appearance_profile(item["profile"]) != item["profile"]
                ):
                    raise ValueError
                identities.add(item["identity_id"])
                name = f"portraits/{item['virtual_id']}.png"
                if item["portrait_file"] != name or name in expected:
                    raise ValueError
                expected.add(name)
                content = archive.read(name)
                if hashlib.sha256(content).hexdigest() != item["portrait_sha256"]:
                    raise ValueError
                png_image(content)
                portraits[name] = content
            if set(names) != expected:
                raise ValueError
        return manifest, portraits
    except (
        zipfile.BadZipFile,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        RuntimeError,
    ) as exc:
        raise ExecutionError("Invalid virtual portrait catalog") from exc


def selected_targets(
    manifest: dict, portraits: dict[str, bytes], identity_ids: set[str]
) -> bytes:
    selected = {
        key: manifest[key]
        for key in ("schema_version", "registry_id", "source_revision", "revision")
    }
    selected["faces"] = [
        item for item in manifest["faces"] if item["identity_id"] in identity_ids
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            zipfile.ZipInfo("manifest.json"),
            json.dumps(selected, sort_keys=True, separators=(",", ":")),
        )
        for item in selected["faces"]:
            name = item["portrait_file"]
            archive.writestr(zipfile.ZipInfo(name), portraits[name])
    return buffer.getvalue()
