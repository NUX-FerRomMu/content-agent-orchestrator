import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from content_agent_orchestrator.contracts import ExecutionError
from content_agent_orchestrator.storage import digest_file
from content_agent_orchestrator.video_frames import decoded_frames


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
class FrameDecodeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.video = self.root / "video.mkv"
        self.timestamps = self.root / "timestamps.jsonl"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=512x512:rate=3:duration=1",
                "-c:v",
                "ffv1",
                str(self.video),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )

    def timestamps_file(self, ticks):
        self.timestamps.write_text(
            "".join(
                json.dumps(
                    {
                        "frame_index": index,
                        "presentation_timestamp": tick,
                        "presentation_time": f"{tick / 3:.6f}",
                    }
                )
                + "\n"
                for index, tick in enumerate(ticks)
            ),
            encoding="utf-8",
        )

    def frames(self, count, timeout=30):
        return decoded_frames(
            self.video,
            self.timestamps,
            stream_index=0,
            width=512,
            height=512,
            expected_count=count,
            work_dir=self.root,
            timeout=timeout,
        )

    def test_missing_decoder_is_actionable(self):
        with (
            patch(
                "content_agent_orchestrator.video_frames.shutil.which",
                return_value=None,
            ),
            self.assertRaisesRegex(ExecutionError, "FFmpeg is required"),
            self.frames(3),
        ):
            self.fail("Decoder should not start")

    def test_diagnostic_symlink_cannot_overwrite_source(self):
        original = digest_file(self.video)
        try:
            (self.root / "decode.stderr").symlink_to(self.video)
        except OSError:
            self.skipTest("Symlink creation not available")
        with self.assertRaisesRegex(ExecutionError, "symlinks"), self.frames(3):
            self.fail("Decoder must reject the log symlink")
        self.assertEqual(digest_file(self.video), original)

    def test_more_frames_than_timestamps_is_rejected(self):
        self.timestamps_file([0, 1])
        with (
            self.assertRaisesRegex(ExecutionError, "more frames"),
            self.frames(3) as frames,
        ):
            list(frames)

    def test_fewer_frames_than_timestamps_is_rejected(self):
        self.timestamps_file([0, 1, 2, 3])
        with (
            self.assertRaisesRegex(ExecutionError, "fewer frames"),
            self.frames(4) as frames,
        ):
            list(frames)

    def test_bad_timestamps_close_decoder(self):
        self.timestamps_file([0, 0, 1])
        with (
            self.assertRaisesRegex(ExecutionError, "strictly increasing"),
            self.frames(3) as frames,
        ):
            list(frames)

    def test_timeout_kills_decoder_and_reports_failure(self):
        self.timestamps_file([0, 1, 2])
        with (
            self.assertRaisesRegex(ExecutionError, "timeout"),
            self.frames(3, timeout=1) as frames,
        ):
            next(frames)
            time.sleep(1.1)
            list(frames)
