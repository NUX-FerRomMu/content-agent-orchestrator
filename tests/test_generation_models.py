import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

from content_agent_orchestrator.contracts import ExecutionError, PipelineError
from content_agent_orchestrator.generation_models import (
    COMPONENTS,
    MODEL_REPOSITORIES,
    WEIGHT_COMPONENTS,
    ModelRedirect,
    model_metadata,
    prepare_model,
    selected_files,
    verify_model,
)
from content_agent_orchestrator.storage import UserStorage, digest_file


def fake_repository():
    contents = {
        "model_index.json": b'{"_class_name":"StableDiffusionXLPipeline"}',
        "README.md": b"Test metadata, CreativeML Open RAIL++-M",
    }
    for component in COMPONENTS:
        contents[f"{component}/config.json"] = b"{}"
    for component in WEIGHT_COMPONENTS:
        stem = (
            "model"
            if component.startswith("text_encoder")
            else "diffusion_pytorch_model"
        )
        contents[f"{component}/{stem}.fp16.safetensors"] = b"verified test weight bytes"
    siblings = []
    for name, content in contents.items():
        item = {"rfilename": name, "size": len(content)}
        if name.endswith(".safetensors"):
            item["lfs"] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        else:
            item["blobId"] = hashlib.sha1(
                f"blob {len(content)}\0".encode() + content
            ).hexdigest()
        siblings.append(item)
    return {
        "id": MODEL_REPOSITORIES["realvis-xl"],
        "sha": "a" * 40,
        "gated": False,
        "cardData": {"license": "openrail++"},
        "siblings": siblings,
    }, contents


def prepared_fixture(directory: Path):
    metadata, contents = fake_repository()
    files = selected_files(metadata)
    for record in files:
        path = directory / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents[record["upstream_path"]])
        record["sha256"] = digest_file(path)[0]
    manifest = {
        "schema_version": 1,
        "repository": metadata["id"],
        "revision": metadata["sha"],
        "license": "openrail++",
        "files": files,
    }
    (directory / "generation-model.json").write_text(json.dumps(manifest))
    return manifest


class ModelPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.storage = UserStorage(self.root / "user")
        self.metadata, self.contents = fake_repository()

    def opener(self, *, corrupt=False):
        def open_response(url, **kwargs):
            name = str(url).partition("/resolve/" + self.metadata["sha"] + "/")[2]
            content = self.contents[name]
            if corrupt and name.endswith(".safetensors"):
                content = b"x" * len(content)
            return contextlib.closing(io.BytesIO(content))

        return Mock(open=Mock(side_effect=open_response))

    def prepare(self, opener):
        with (
            patch(
                "content_agent_orchestrator.generation_models.model_metadata",
                return_value=self.metadata,
            ),
            patch("urllib.request.build_opener", return_value=opener),
        ):
            return prepare_model(self.storage)

    def test_preparation_pins_commit_verifies_hashes_and_normalizes_fp16_names(self):
        opener = self.opener()
        path = self.prepare(opener)
        self.assertIn(self.metadata["sha"], str(path))
        manifest = verify_model(path)
        self.assertEqual(manifest["revision"], self.metadata["sha"])
        for record in manifest["files"]:
            self.assertNotIn(".fp16.", record["path"])
            self.assertEqual(
                digest_file(path / record["path"]),
                (record["sha256"], record["size_bytes"]),
            )
        self.assertEqual(
            list((self.storage.root / "temps/model_preparation").iterdir()), []
        )
        count = opener.open.call_count
        self.assertEqual(self.prepare(opener), path)
        self.assertEqual(opener.open.call_count, count)

    def test_corrupt_download_never_publishes_model_and_retry_keeps_valid_files(self):
        opener = self.opener(corrupt=True)
        with self.assertRaisesRegex(ExecutionError, "checksum"):
            self.prepare(opener)
        self.assertFalse(
            (
                self.storage.root / "artifacts/models/realvis-xl" / self.metadata["sha"]
            ).exists()
        )
        self.assertTrue(
            list((self.storage.root / "temps/model_preparation").rglob("*.partial"))
        )
        clean = self.opener()
        path = self.prepare(clean)
        verify_model(path)
        self.assertLess(clean.open.call_count, len(self.contents))

    def test_single_fp16_snapshot_omits_indices_for_unselected_sharded_weights(self):
        metadata, _ = fake_repository()
        # RealVisXL publishes a float32 index alongside its single fp16 UNet.
        metadata["siblings"].append(
            {
                "rfilename": "unet/diffusion_pytorch_model.safetensors.index.json",
                "size": 100,
                "blobId": "b" * 40,
            }
        )
        files = selected_files(metadata)
        self.assertFalse(any(item["path"].endswith(".index.json") for item in files))
        self.assertTrue(
            any(
                item["path"] == "unet/diffusion_pytorch_model.safetensors"
                for item in files
            )
        )

    def test_changed_weights_unverified_files_and_symlinks_are_rejected(self):
        model = self.root / "model"
        manifest = prepared_fixture(model)
        weight = model / next(
            item["path"]
            for item in manifest["files"]
            if item["path"].endswith(".safetensors")
        )
        original = weight.read_bytes()
        weight.write_bytes(b"changed")
        with self.assertRaisesRegex(ExecutionError, "checksum"):
            verify_model(model)
        weight.write_bytes(original)
        extra = model / "remote-code.py"
        extra.write_text("raise RuntimeError('never execute')")
        with self.assertRaises(ExecutionError):
            verify_model(model)
        extra.unlink()
        weight.unlink()
        outside = self.root / "outside.safetensors"
        outside.write_bytes(original)
        weight.symlink_to(outside)
        with self.assertRaisesRegex(ExecutionError, "symlink"):
            verify_model(model)

    def test_selection_requires_all_safe_component_weights_and_upstream_hashes(self):
        for edit in (
            lambda data: data["siblings"].append({"rfilename": "../escape"}),
            lambda data: data["siblings"].__delitem__(-1),
            lambda data: data["siblings"][0].update(blobId="bad"),
        ):
            data, _ = fake_repository()
            edit(data)
            with self.subTest(edit=edit), self.assertRaises(ExecutionError):
                selected_files(data)

    def test_metadata_requires_compatible_license_and_ungated_pinned_repository(self):
        for change in (
            {"cardData": {"license": "non-commercial"}},
            {"gated": "auto"},
            {"sha": "main"},
            {"id": "untrusted/model"},
        ):
            response = contextlib.closing(
                io.BytesIO(json.dumps(self.metadata | change).encode())
            )
            with (
                patch(
                    "urllib.request.build_opener",
                    return_value=Mock(open=Mock(return_value=response)),
                ),
                self.assertRaises(ExecutionError),
            ):
                model_metadata(self.metadata["id"], "main")
        with self.assertRaises(PipelineError):
            model_metadata("untrusted/model", "main")
        opener = Mock(
            open=Mock(
                side_effect=urllib.error.HTTPError("", 403, "secret-response", {}, None)
            )
        )
        with (
            patch("urllib.request.build_opener", return_value=opener),
            self.assertRaisesRegex(ExecutionError, "HTTP error 403") as caught,
        ):
            model_metadata(self.metadata["id"], "main")
        self.assertNotIn("secret-response", str(caught.exception))

    def test_redirects_must_use_https_and_supported_model_hosts(self):
        redirect = ModelRedirect()
        for url in (
            "http://huggingface.co/weights",
            "https://example.com/weights",
            "https://user:password@huggingface.co/weights",
        ):
            with self.subTest(url=url), self.assertRaises(ExecutionError):
                redirect.redirect_request(None, None, 302, "", None, url)
        request = urllib.request.Request(
            "https://huggingface.co/model/resolve/commit/weights"
        )
        redirected = redirect.redirect_request(
            request, None, 302, "", {}, "https://us.aws.cdn.hf.co/xet-bridge/weights"
        )
        self.assertEqual(
            redirected.full_url, "https://us.aws.cdn.hf.co/xet-bridge/weights"
        )
