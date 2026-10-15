import hashlib
import io
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from content_agent_orchestrator.contracts import ExecutionError
from content_agent_orchestrator.storage import UserStorage
from content_agent_orchestrator.swap_models import ReleaseRedirect, prepare_swap_models


class SwapPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.storage = UserStorage(Path(self.temporary.name) / "user")
        self.contents = {"first.onnx": b"first model", "second.onnx": b"second model"}
        self.records = {
            name: {
                "filename": filename,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, (filename, content) in zip(
                ("swap_model", "arcface_model"), self.contents.items(), strict=True
            )
        }
        self.addCleanup(patch.stopall)
        patch(
            "content_agent_orchestrator.swap_models.SWAP_MODELS", self.records
        ).start()
        self.opener = (
            patch("content_agent_orchestrator.swap_models.urllib.request.build_opener")
            .start()
            .return_value
        )
        self.opener.open.side_effect = lambda url, **kw: io.BytesIO(
            self.contents[url.rsplit("/", 1)[1]]
        )

    def test_verifies_both_files_private_publication_and_offline_reuse(self):
        paths = prepare_swap_models(self.storage)
        self.assertEqual(self.opener.open.call_count, 2)
        for path in paths.values():
            self.assertEqual(path.read_bytes(), self.contents[path.name])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(prepare_swap_models(self.storage), paths)
        self.assertEqual(self.opener.open.call_count, 2)
        self.assertFalse(
            self.storage.path("temps/swap_model_preparation/models-3.0.0").exists()
        )

    def test_failed_second_download_retains_first_verified_file_and_reuses_it(self):
        self.opener.open.side_effect = [
            io.BytesIO(self.contents["first.onnx"]),
            urllib.error.URLError("offline"),
        ]
        with self.assertRaises(ExecutionError):
            prepare_swap_models(self.storage)
        first = self.storage.path(
            "temps/swap_model_preparation/models-3.0.0/first.onnx"
        )
        self.assertTrue(first.is_file())
        self.opener.open.side_effect = lambda url, **kw: io.BytesIO(
            self.contents[url.rsplit("/", 1)[1]]
        )
        result = prepare_swap_models(self.storage)
        self.assertEqual(self.opener.open.call_count, 3)
        self.assertEqual(result["swap_model"].read_bytes(), self.contents["first.onnx"])

    def test_corrupt_replacement_is_never_published_and_old_snapshot_survives(self):
        paths = prepare_swap_models(self.storage)
        paths["arcface_model"].write_bytes(b"damaged")
        self.opener.open.side_effect = lambda *a, **k: io.BytesIO(b"wrong bytes!")
        with self.assertRaises(ExecutionError):
            prepare_swap_models(self.storage)
        self.assertEqual(paths["swap_model"].read_bytes(), self.contents["first.onnx"])
        self.assertEqual(paths["arcface_model"].read_bytes(), b"damaged")

    def test_partial_and_final_symlink_escape_are_rejected(self):
        work = self.storage.path("temps/swap_model_preparation/models-3.0.0")
        work.mkdir(parents=True)
        outside = Path(self.temporary.name) / "outside"
        outside.write_bytes(b"keep me")
        (work / "first.partial").symlink_to(outside)
        with self.assertRaises(ExecutionError):
            prepare_swap_models(self.storage)
        self.assertEqual(outside.read_bytes(), b"keep me")

    def test_redirects_require_https_and_known_asset_hosts(self):
        request = urllib.request.Request("https://github.com/model.onnx")
        handler = ReleaseRedirect()
        for url in (
            "http://github.com/file",
            "https://evil.invalid/file",
            "https://user:pass@github.com/file",
        ):
            with self.subTest(url=url), self.assertRaises(ExecutionError):
                handler.redirect_request(request, None, 302, "Found", {}, url)
        valid = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://release-assets.githubusercontent.com/file?signature=value",
        )
        self.assertEqual(valid.host, "release-assets.githubusercontent.com")
