import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from content_agent_orchestrator.cli import face_analysis_definition, main
from content_agent_orchestrator.contracts import Artifact, ExecutionError, ModuleContext
from content_agent_orchestrator.face_data import YUNET_SHA256, bundled_yunet, iter_rows
from content_agent_orchestrator.modules import (
    FaceDetection,
    FaceReport,
    builtin_registry,
)
from content_agent_orchestrator.pipeline import Pipeline
from content_agent_orchestrator.runner import RunFailed, Runner
from content_agent_orchestrator.storage import UserStorage, digest_file


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required"
)
class FaceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.storage = UserStorage(self.root / "user")
        self.registry = builtin_registry()
        self.runner = Runner(self.storage, self.registry)
        self.fixture = Path(__file__).parent / "fixtures" / "astronaut.png"
        self.assertEqual(
            digest_file(self.fixture)[0],
            "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
        )
        self.assertEqual(digest_file(bundled_yunet())[0], YUNET_SHA256)
        self.image = cv2.imread(str(self.fixture))
        self.assertIsNotNone(self.image)
        self.source = self.root / "video.mkv"
        self.parameters = {"max_dimension": 640, "chunk_frames": 2}

    def generate(self, frames, *, variable=False):
        height, width = frames[0].shape[:2]
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "12",
            "-i",
            "pipe:0",
        ]
        if variable:
            command += [
                "-vf",
                "setpts='if(lt(N,3),N/(12*TB),(0.25+(N-3)/6)/TB)'",
                "-fps_mode",
                "vfr",
            ]
        command += ["-c:v", "ffv1", "-y", str(self.source)]
        subprocess.run(
            command,
            input=b"".join(frame.tobytes() for frame in frames),
            check=True,
            capture_output=True,
            timeout=30,
        )

    def run_pipeline(self, **kwargs):
        pipeline = Pipeline.parse(
            face_analysis_definition(self.parameters), self.registry
        )
        return self.runner.run(
            pipeline, {"video": self.source, "detector": bundled_yunet()}, **kwargs
        )

    @staticmethod
    def report(result):
        return json.loads(result.outputs["report"].path.read_text())

    def test_real_detector_tracks_moving_face_and_reuses_cache(self):
        frames = [
            cv2.warpAffine(
                self.image, np.float32([[1, 0, offset], [0, 1, 0]]), (512, 512)
            )
            for offset in range(6)
        ]
        self.generate(frames)
        original = digest_file(self.source)
        result = self.run_pipeline()
        report = self.report(result)
        self.assertEqual(report["counts"]["frame_count"], 6)
        self.assertEqual(report["counts"]["face_detection_count"], 6)
        self.assertEqual(report["counts"]["track_count"], 1)
        self.assertFalse(report["anonymization_performed"])
        self.assertFalse(report["identity_recognition_performed"])
        rows = list(iter_rows(result.outputs["frames"].path))
        self.assertEqual(
            {item["faces"][0]["track_id"] for item in rows}, {"track-000001"}
        )
        self.assertEqual(len(rows[0]["faces"][0]["landmarks"]), 5)
        timestamps = next(
            (self.storage.root / "artifacts" / "runs").rglob("timestamps.jsonl")
        )
        self.assertEqual(
            [row["presentation_timestamp"] for row in rows],
            [row["presentation_timestamp"] for row in iter_rows(timestamps)],
        )
        self.assertEqual(digest_file(self.source), original)
        cached = self.run_pipeline()
        self.assertEqual(cached.cache_hits, ("inspect", "detect", "track", "report"))
        self.assertEqual(
            cached.outputs["frames"].sha256, result.outputs["frames"].sha256
        )
        self.assertEqual(
            list((self.storage.root / "temps" / result.run_id).iterdir()), []
        )

    def test_real_detector_finds_two_faces(self):
        two_faces = np.concatenate((self.image, self.image), axis=1)
        self.generate([two_faces] * 4)
        report = self.report(self.run_pipeline())
        self.assertEqual(report["counts"]["face_detection_count"], 8)
        self.assertEqual(report["counts"]["max_simultaneous_faces"], 2)
        self.assertEqual(report["counts"]["track_count"], 2)

    def test_real_disappearance_is_recorded_and_reacquired(self):
        empty = np.zeros_like(self.image)
        self.generate([self.image, self.image, empty, empty, self.image, self.image])
        result = self.run_pipeline()
        report = self.report(result)
        self.assertEqual(report["counts"]["frames_without_faces"], 2)
        self.assertEqual(report["counts"]["track_count"], 1)
        self.assertEqual(report["counts"]["gap_events"], 1)
        intervals = list(iter_rows(result.outputs["intervals"].path))
        self.assertEqual(
            (intervals[0]["start_frame"], intervals[0]["end_frame"]), (2, 3)
        )

    def test_no_face_video_has_no_tracks(self):
        self.generate([np.zeros_like(self.image)] * 3)
        result = self.run_pipeline()
        self.assertEqual(self.report(result)["counts"]["track_count"], 0)
        self.assertEqual(list(iter_rows(result.outputs["tracks"].path)), [])

    def test_variable_fps_keeps_original_timestamps(self):
        self.generate([self.image] * 6, variable=True)
        original = digest_file(self.source)
        result = self.run_pipeline()
        rows = list(iter_rows(result.outputs["frames"].path))
        differences = [
            right["presentation_timestamp"] - left["presentation_timestamp"]
            for left, right in pairwise(rows)
        ]
        self.assertGreater(max(differences) - min(differences), 2)
        self.assertEqual(len(rows), 6)
        self.assertEqual(digest_file(self.source), original)

    def test_smaller_detection_image_returns_source_coordinates(self):
        self.parameters["max_dimension"] = 256
        self.generate([self.image] * 3)
        result = self.run_pipeline()
        box = next(iter_rows(result.outputs["frames"].path))["faces"][0]["bbox"]
        self.assertGreater(box[0], 150)
        self.assertLess(box[0], 220)
        self.assertGreater(box[2], 60)
        self.assertEqual(self.report(result)["video"]["width"], 512)

    def test_quarter_turn_display_rotation_restores_coded_coordinates(self):
        upright = np.concatenate(
            (self.image, np.zeros((512, 256, 3), np.uint8)), axis=1
        )
        rotations = [
            (90, cv2.ROTATE_90_CLOCKWISE),
            (180, cv2.ROTATE_180),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
        options = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "full"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        if "-display_rotation" not in options.stdout:
            self.skipTest("FFmpeg does not support generating rotation metadata")
        for rotation, operation in rotations:
            with self.subTest(rotation=rotation):
                coded = cv2.rotate(upright, operation)
                self.generate([coded] * 3)
                base = self.root / f"base-{rotation}.mp4"
                source = self.root / f"rotated-{rotation}.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(self.source),
                        "-c:v",
                        "mpeg4",
                        str(base),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-display_rotation",
                        str(rotation),
                        "-i",
                        str(base),
                        "-c",
                        "copy",
                        str(source),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                self.source = source
                original = digest_file(source)
                result = self.run_pipeline(cache="fresh")
                report = self.report(result)
                self.assertEqual(report["counts"]["face_detection_count"], 3)
                self.assertEqual(report["video"]["rotation_degrees"], rotation)
                self.assertEqual(
                    (report["video"]["width"], report["video"]["height"]),
                    (coded.shape[1], coded.shape[0]),
                )
                box = next(iter_rows(result.outputs["frames"].path))["faces"][0]["bbox"]
                if rotation == 90:
                    self.assertGreater(box[0], 300)
                    self.assertLess(box[1], 250)
                elif rotation == 180:
                    self.assertGreater(box[0], 450)
                    self.assertGreater(box[1], 300)
                else:
                    self.assertLess(box[0], 120)
                    self.assertGreater(box[1], 450)
                self.assertEqual(digest_file(source), original)
                self.source = self.root / "video.mkv"

    def assert_detection_resume(self, *, corrupt=False):
        self.generate([self.image] * 8)
        original_detect = FaceDetection._detect_faces
        calls = 0

        def interrupted(*args):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise ExecutionError("simulated detector interruption")
            return original_detect(*args)

        with (
            patch.object(FaceDetection, "_detect_faces", side_effect=interrupted),
            self.assertRaisesRegex(
                RunFailed, "simulated detector interruption"
            ) as caught,
        ):
            self.run_pipeline(cache="fresh")
        run_id = caught.exception.run_id
        cache = self.storage.root / "temps" / run_id / "detect" / "cache"
        progress = json.loads((cache / "checkpoint.json").read_text())
        self.assertEqual([chunk["count"] for chunk in progress["chunks"]], [2, 2])
        self.assertFalse((self.storage.root / "outputs" / run_id).exists())
        if corrupt:
            (cache / "chunk-00000001.jsonl").write_text("corrupt", encoding="utf-8")
        with patch.object(
            FaceDetection, "_detect_faces", wraps=original_detect
        ) as detector:
            resumed = self.run_pipeline(resume=run_id)
            self.assertEqual(detector.call_count, 6 if corrupt else 4)
        self.assertEqual(self.report(resumed)["counts"]["face_detection_count"], 8)
        fresh = self.run_pipeline(cache="fresh")
        self.assertEqual(
            resumed.outputs["frames"].sha256, fresh.outputs["frames"].sha256
        )
        self.assertEqual(
            resumed.outputs["report"].sha256, fresh.outputs["report"].sha256
        )

    def test_detection_resumes_only_complete_verified_chunks(self):
        self.assert_detection_resume()

    def test_corrupt_chunk_is_recomputed_without_losing_valid_prefix(self):
        self.assert_detection_resume(corrupt=True)

    def test_cli_and_json_example_use_bundled_model(self):
        self.generate([self.image] * 3)
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            contextlib.chdir(self.root),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(
                ["--root", str(self.storage.root), "analyze-faces", self.source.name]
            )
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["status"], "succeeded")
        example = Path(__file__).resolve().parents[1] / "examples/analyze-faces.json"
        result = self.runner.run(
            Pipeline.load(example, self.registry),
            {"video": self.source, "detector": bundled_yunet()},
        )
        self.assertEqual(self.report(result)["counts"]["track_count"], 1)

    def test_modified_model_is_rejected_before_loading(self):
        self.generate([self.image])
        model = self.root / "wrong.onnx"
        model.write_bytes(b"not the pinned model")
        pipeline = Pipeline.parse(face_analysis_definition(), self.registry)
        with self.assertRaisesRegex(RunFailed, "checksum") as caught:
            self.runner.run(pipeline, {"video": self.source, "detector": model})
        self.assertFalse(
            (self.storage.root / "outputs" / caught.exception.run_id).exists()
        )

    def test_timestamps_from_another_artifact_are_rejected_before_inference(self):
        self.generate([self.image] * 3)
        inspect = self.registry["video.inspect"]
        original_execute = inspect.execute

        def substituted_timestamps(*args):
            outputs = original_execute(*args)
            path = outputs["timestamps"]
            rows = list(iter_rows(path))
            rows[0]["presentation_time"] = "0.123456"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            return outputs

        with (
            patch.object(inspect, "execute", side_effect=substituted_timestamps),
            patch.object(FaceDetection, "_detect_faces") as detector,
            self.assertRaisesRegex(RunFailed, "Timestamps do not belong") as caught,
        ):
            self.run_pipeline()
        detector.assert_not_called()
        self.assertFalse(
            (self.storage.root / "outputs" / caught.exception.run_id).exists()
        )

    def test_report_rejects_detail_files_from_another_artifact(self):
        self.generate([self.image] * 3)
        result = self.run_pipeline()
        report = result.outputs["report"]
        inputs = {
            name: result.outputs[name] for name in ("frames", "tracks", "intervals")
        }
        inputs["summary"] = Artifact(
            report.path, "face.tracking-summary", report.sha256, report.size
        )
        foreign = self.root / "foreign_frames.jsonl"
        foreign.write_text("not the paired frames\n", encoding="utf-8")
        digest, size = digest_file(foreign)
        inputs["frames"] = Artifact(foreign, "face.track-frames", digest, size)
        output = self.root / "report-result"
        output.mkdir()
        with self.assertRaisesRegex(ExecutionError, "do not belong"):
            FaceReport().execute(inputs, {}, ModuleContext(self.root, output))
        self.assertEqual(list(output.iterdir()), [])
