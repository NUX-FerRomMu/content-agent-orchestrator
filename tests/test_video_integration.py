import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from content_agent_orchestrator.cli import main
from content_agent_orchestrator.storage import digest_file


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required"
)
class VideoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "user"
        self.input_dir = self.root / "artifacts" / "raw_footages"
        self.input_dir.mkdir(parents=True)

    def generate(self, name, *, variable=False, codec="mpeg4"):
        target = self.input_dir / name
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x96:rate=12:duration=1",
        ]
        if variable:
            command.extend(
                [
                    "-vf",
                    "setpts='if(lt(N,6),N/(12*TB),(0.5+(N-6)/6)/TB)'",
                    "-fps_mode",
                    "vfr",
                    "-c:v",
                    codec,
                ]
            )
        else:
            command.extend(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=1",
                    "-c:v",
                    codec,
                    "-c:a",
                    "aac",
                    "-shortest",
                ]
            )
        command.extend(["-metadata", "title=private source title", str(target)])
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        return target

    def invoke(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["--root", str(self.root), *arguments])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_inspect_cfr_audio_and_cache_with_original_unchanged(self):
        source = self.generate("constant.mp4")
        original = digest_file(source)
        with contextlib.chdir(self.root):
            code, stdout, stderr = self.invoke(
                ["inspect", "artifacts/raw_footages/constant.mp4"]
            )
        self.assertEqual(code, 0, stderr)
        result = json.loads(stdout)
        report_path = Path(result["outputs"]["report"]["path"])
        self.assertEqual(report_path.parent, self.root / "outputs" / result["run_id"])
        report = json.loads(report_path.read_text())
        self.assertEqual(report["timing"]["mode"], "constant")
        self.assertEqual(report["timing"]["decoded_frame_count"], 12)
        self.assertEqual(report["timing"]["missing_timestamps"], 0)
        video = next(
            stream for stream in report["streams"] if stream["codec_type"] == "video"
        )
        self.assertEqual((video["width"], video["height"]), (128, 96))
        self.assertTrue(
            any(stream["codec_type"] == "audio" for stream in report["streams"])
        )
        self.assertNotIn("private source title", report_path.read_text())
        self.assertNotIn(str(source), report_path.read_text())
        self.assertEqual(digest_file(source), original)
        timestamps = next((self.root / "artifacts" / "runs").rglob("timestamps.jsonl"))
        entries = [json.loads(line) for line in timestamps.read_text().splitlines()]
        self.assertEqual(len(entries), 12)
        self.assertEqual(entries[0]["presentation_time"], "0.000000")
        self.assertEqual(entries[0]["presentation_timestamp"], 0)
        code, stdout, stderr = self.invoke(["inspect", str(source)])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["cache_hits"], ["inspect", "export"])

    def test_variable_fps_is_measured_without_changing_source(self):
        source = self.generate("variable.mp4", variable=True)
        original = digest_file(source)
        code, stdout, stderr = self.invoke(["inspect", str(source)])
        self.assertEqual(code, 0, stderr)
        report = json.loads(
            Path(json.loads(stdout)["outputs"]["report"]["path"]).read_text()
        )
        self.assertEqual(report["timing"]["mode"], "variable")
        self.assertEqual(report["timing"]["decoded_frame_count"], 12)
        self.assertEqual(digest_file(source), original)

    def test_h264_frame_side_data_does_not_break_timestamp_parsing(self):
        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        if "libx264" not in encoders.stdout:
            self.skipTest("FFmpeg built without libx264")
        source = self.generate("h264.mp4", codec="libx264")
        code, stdout, stderr = self.invoke(["inspect", str(source)])
        self.assertEqual(code, 0, stderr)
        report = json.loads(
            Path(json.loads(stdout)["outputs"]["report"]["path"]).read_text()
        )
        self.assertEqual(report["timing"]["decoded_frame_count"], 12)
        self.assertEqual(report["timing"]["mode"], "constant")

    def test_runs_documented_json_pipeline(self):
        source = self.generate("example.mp4")
        example = (
            Path(__file__).resolve().parents[1] / "examples" / "inspect-video.json"
        )
        code, stdout, stderr = self.invoke(
            ["run", str(example), "--input", f"video={source}"]
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["status"], "succeeded")

    def test_relative_inputs_use_cwd_for_inspect_and_run(self):
        source = self.generate("relative.mp4")
        original = digest_file(source)
        # A conflicting storage-relative file must not shadow the CWD input.
        (self.root / source.name).write_bytes(b"not the requested video")
        example = (
            Path(__file__).resolve().parents[1] / "examples" / "inspect-video.json"
        )
        commands = [
            ["inspect", source.name],
            ["run", str(example), "--input", f"video={source.name}"],
        ]
        for command in commands:
            with self.subTest(command=command), contextlib.chdir(self.input_dir):
                code, stdout, stderr = self.invoke(command)
                self.assertEqual(code, 0, stderr)
                report = json.loads(
                    Path(json.loads(stdout)["outputs"]["report"]["path"]).read_text()
                )
                self.assertEqual(report["source"]["sha256"], original[0])
        self.assertEqual(digest_file(source), original)

    def test_invalid_video_retains_diagnostics_and_no_final_output(self):
        source = self.input_dir / "invalid.mp4"
        source.write_bytes(b"not a video")
        code, stdout, stderr = self.invoke(["inspect", str(source)])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("FFprobe failed", stderr)
        self.assertEqual(list((self.root / "outputs").iterdir()), [])
        self.assertTrue(list((self.root / "temps").rglob("probe.stderr")))

    def test_duplicate_input_binding_is_rejected(self):
        source = self.generate("duplicate.mp4")
        example = (
            Path(__file__).resolve().parents[1] / "examples" / "inspect-video.json"
        )
        code, stdout, stderr = self.invoke(
            [
                "run",
                str(example),
                "--input",
                f"video={source}",
                "--input",
                f"video={source}",
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("unique NAME=PATH", stderr)
        self.assertEqual(stdout, "")
