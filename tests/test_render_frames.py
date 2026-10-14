import importlib.util
import json
import tempfile
import unittest
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np

from content_agent_orchestrator.contracts import ExecutionError, ModuleContext
from content_agent_orchestrator.render_frames import FrameChunks, archived_frames


@unittest.skipUnless(importlib.util.find_spec("av"), "Optional swap runtime required")
class FrameArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.result = self.root / "result"
        self.result.mkdir()
        self.context = ModuleContext(self.root, self.result)
        self.video = {
            "width": 64,
            "height": 64,
            "frame_count": 3,
            "time_base": "1/90000",
        }
        self.ticks = [45000, 48003, 60000]

    def append(self, cache, index):
        tick = self.ticks[index]
        cache.append(
            {
                "frame_index": index,
                "presentation_timestamp": tick,
                "presentation_time": str(tick / 90000),
            },
            np.full((64, 64, 3), index * 40, np.uint8),
            {"status": "test"},
        )

    def test_real_lossless_chunks_preserve_variable_timestamps_and_pixels(self):
        cache = FrameChunks(self.context, self.video, "signature", 2)
        for index in range(3):
            self.append(cache, index)
        path, _ = cache.finish()
        images = list(archived_frames(path, self.root, self.video))
        self.assertEqual(len(images), 3)
        for index, time, image in images:
            self.assertEqual(time, Fraction(self.ticks[index], 90000))
            np.testing.assert_array_equal(
                image, np.full((64, 64, 3), index * 40, np.uint8)
            )

    def test_resume_uses_only_verified_complete_chunks_and_discards_corrupt_cache(self):
        cache = FrameChunks(self.context, self.video, "signature", 2)
        for index in range(3):
            self.append(cache, index)
        cache.close()
        resumed = FrameChunks(self.context, self.video, "signature", 2)
        self.assertEqual(resumed.processed, 2)
        self.append(resumed, 2)
        path, _ = resumed.finish()
        self.assertEqual(len(list(archived_frames(path, self.root, self.video))), 3)
        movie = self.root / "frame-cache/chunk-00000000.nut"
        movie.write_bytes(b"corrupted")
        self.assertEqual(
            FrameChunks(self.context, self.video, "signature", 2).processed, 0
        )
        self.assertEqual(
            FrameChunks(self.context, self.video, "different-inputs", 2).processed, 0
        )

    def test_frame_archive_rejects_extra_entries_and_changed_chunk_hashes(self):
        for tamper in ("extra", "chunk", "manifest"):
            with self.subTest(tamper=tamper):
                cache = FrameChunks(self.context, self.video, "signature", 2)
                for index in range(cache.processed, 3):
                    self.append(cache, index)
                path, _ = cache.finish()
                with zipfile.ZipFile(path) as archive:
                    contents = {name: archive.read(name) for name in archive.namelist()}
                if tamper == "extra":
                    contents["../../outside"] = b"never extracted"
                elif tamper == "chunk":
                    contents["chunks/00000000.nut"] = b"x" * len(
                        contents["chunks/00000000.nut"]
                    )
                else:
                    manifest = json.loads(contents["manifest.json"])
                    manifest["video"]["width"] = 100000
                    contents["manifest.json"] = json.dumps(manifest).encode()
                with zipfile.ZipFile(
                    path, "w", compression=zipfile.ZIP_STORED
                ) as archive:
                    for name, content in contents.items():
                        archive.writestr(name, content)
                with self.assertRaises(ExecutionError):
                    list(archived_frames(path, self.root, self.video))
                self.assertFalse((self.root.parent / "outside").exists())
