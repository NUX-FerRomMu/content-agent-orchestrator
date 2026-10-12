import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from content_agent_orchestrator.cli import identity_recognition_definition, main
from content_agent_orchestrator.contracts import ExecutionError, PipelineError
from content_agent_orchestrator.face_data import bundled_yunet, iter_rows, read_json
from content_agent_orchestrator.identity_data import SFACE_SHA256, bundled_sface
from content_agent_orchestrator.identity_registry import IdentityRegistry
from content_agent_orchestrator.modules import FaceEmbedding, builtin_registry
from content_agent_orchestrator.pipeline import Pipeline
from content_agent_orchestrator.runner import RunFailed, Runner
from content_agent_orchestrator.storage import UserStorage, digest_file


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required"
)
class IdentityIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.storage = UserStorage(self.root / "user")
        self.identities = IdentityRegistry(self.storage)
        self.modules = builtin_registry()
        self.runner = Runner(self.storage, self.modules)
        fixtures = Path(__file__).parent / "fixtures"
        self.astronaut = cv2.imread(str(fixtures / "astronaut.png"))
        self.hopper = cv2.imread(str(fixtures / "grace_hopper.jpg"))
        self.assertEqual(
            digest_file(fixtures / "grace_hopper.jpg")[0],
            "a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130",
        )
        self.assertEqual(digest_file(bundled_sface()), (SFACE_SHA256, 38696353))
        self.source = self.root / "clip.mkv"
        self.snapshot = self.identities.snapshot()

    def video(self, frames, *, variable=False):
        height, width = frames[0].shape[:2]
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
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
        subprocess.run(
            command + ["-c:v", "ffv1", "-y", str(self.source)],
            input=b"".join(frame.tobytes() for frame in frames),
            capture_output=True,
            check=True,
            timeout=30,
        )

    def recognize(self, *, embedding_parameters=None, **kwargs):
        definition = identity_recognition_definition(
            embedding_parameters=embedding_parameters
        )
        return self.runner.run(
            Pipeline.parse(definition, self.modules),
            {
                "video": self.source,
                "detector": bundled_yunet(),
                "recognizer": bundled_sface(),
                "registry": self.snapshot,
            },
            **kwargs,
        )

    def private_artifacts(self, result):
        with self.storage.connect() as connection:
            row = connection.execute(
                "SELECT outputs FROM nodes WHERE run_id = ? AND node_id = 'embed'",
                (result.run_id,),
            ).fetchone()
        return self.storage.verified_artifacts(row[0], dict(FaceEmbedding.output_kinds))

    def enroll_result(self, result, **kwargs):
        artifacts = self.private_artifacts(result)
        return self.identities.enroll(
            artifacts["frames"].path,
            artifacts["summary"].path,
            "track-000001",
            **kwargs,
        )

    def test_real_sface_cross_clip_recognition_unknown_person_and_registry_cache(self):
        self.video([self.astronaut] * 4)
        original = digest_file(self.source)
        before = self.recognize()
        self.assertEqual(
            read_json(before.outputs["report"].path)["counts"][
                "unknown_observation_count"
            ],
            4,
        )
        identity = self.enroll_result(before, alias="owner")
        self.snapshot = self.identities.snapshot()
        after = self.recognize()
        self.assertEqual(after.cache_hits, ("inspect", "detect", "track", "embed"))
        self.assertEqual(
            read_json(after.outputs["report"].path)["counts"][
                "known_observation_count"
            ],
            4,
        )
        cached = self.recognize()
        self.assertEqual(
            cached.cache_hits,
            ("inspect", "detect", "track", "embed", "resolve", "report"),
        )
        self.assertEqual(digest_file(self.source), original)
        self.source = self.root / "another-clip.mkv"
        moved = [
            cv2.warpAffine(
                self.astronaut, np.float32([[1, 0, offset], [0, 1, 1]]), (512, 512)
            )
            for offset in range(4)
        ]
        self.video(moved)
        same_person = self.recognize()
        for row in iter_rows(same_person.outputs["frames"].path):
            self.assertEqual(
                row["faces"][0]["identity"]["identity_id"], identity["identity_id"]
            )
        self.source = self.root / "different-person.mkv"
        self.video([self.hopper] * 4)
        unknown = self.recognize()
        self.assertEqual(
            read_json(unknown.outputs["report"].path)["counts"][
                "unknown_observation_count"
            ],
            4,
        )
        for row in iter_rows(unknown.outputs["frames"].path):
            self.assertLess(row["faces"][0]["identity"]["candidates"][0]["score"], 0.6)
        for artifact in same_person.outputs.values():
            self.assertNotIn('"embedding":', artifact.path.read_text())
            self.assertNotIn('"references":', artifact.path.read_text())
        if os.name != "nt":
            for artifact in self.private_artifacts(after).values():
                self.assertEqual(artifact.path.stat().st_mode & 0o777, 0o600)

    def test_variable_timestamps_and_empty_frames_survive_extraction(self):
        self.video(
            [self.astronaut] * 3 + [np.zeros_like(self.astronaut)] * 3, variable=True
        )
        result = self.recognize()
        private = self.private_artifacts(result)
        frames = list(iter_rows(private["frames"].path))
        report = read_json(result.outputs["report"].path)
        self.assertEqual(report["counts"]["frame_count"], 6)
        self.assertEqual(report["counts"]["usable_embedding_count"], 3)
        self.assertEqual(frames[5]["faces"], [])
        ticks = [row["presentation_timestamp"] for row in frames]
        differences = np.diff(ticks)
        self.assertGreater(int(max(differences) - min(differences)), 2)
        self.assertEqual(
            [
                row["presentation_timestamp"]
                for row in iter_rows(result.outputs["frames"].path)
            ],
            ticks,
        )

    def test_low_quality_observations_cannot_be_registered(self):
        self.video([self.astronaut] * 4)
        result = self.recognize(embedding_parameters={"min_face_pixels": 200})
        self.assertEqual(
            read_json(result.outputs["report"].path)["counts"][
                "unusable_observation_count"
            ],
            4,
        )
        with self.assertRaisesRegex(ExecutionError, "three usable"):
            self.enroll_result(result)

    def test_failed_extraction_restarts_safely_and_publishes_no_final(self):
        self.video([self.astronaut] * 4)
        original = FaceEmbedding._embedding
        calls = 0

        def fail(*args):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise ExecutionError("simulated extractor failure")
            return original(*args)

        with (
            patch.object(FaceEmbedding, "_embedding", side_effect=fail),
            self.assertRaisesRegex(RunFailed, "extractor failure") as caught,
        ):
            self.recognize()
        run_id = caught.exception.run_id
        self.assertFalse((self.storage.root / "outputs" / run_id).exists())
        self.assertTrue((self.storage.root / "temps" / run_id / "embed").exists())
        resumed = self.recognize(resume=run_id)
        self.assertEqual(
            read_json(resumed.outputs["report"].path)["counts"][
                "usable_embedding_count"
            ],
            4,
        )
        self.assertEqual(list((self.storage.root / "temps" / run_id).iterdir()), [])

    def test_resume_requires_the_original_snapshot(self):
        self.video([self.astronaut] * 4)
        completed = self.recognize()
        with (
            patch.object(
                self.modules["identity.resolve"],
                "execute",
                side_effect=ExecutionError("resolver interrupted"),
            ),
            self.assertRaises(RunFailed) as caught,
        ):
            self.recognize(cache="fresh")
        original_snapshot = self.snapshot
        self.enroll_result(completed)
        self.snapshot = self.identities.snapshot()
        with self.assertRaisesRegex(PipelineError, "same pipeline, paths and versions"):
            self.recognize(resume=caught.exception.run_id)
        self.snapshot = original_snapshot
        resumed = self.recognize(resume=caught.exception.run_id)
        self.assertEqual(
            read_json(resumed.outputs["report"].path)["registry"]["revision"], 0
        )

    def test_tampered_sface_model_is_rejected_before_loading(self):
        self.video([self.astronaut] * 3)
        model = self.root / "wrong.onnx"
        model.write_bytes(b"wrong model")
        with (
            patch(
                "content_agent_orchestrator.modules.face_embed.cv2.FaceRecognizerSF_create"
            ) as create,
            self.assertRaisesRegex(RunFailed, "checksum"),
        ):
            self.runner.run(
                Pipeline.parse(identity_recognition_definition(), self.modules),
                {
                    "video": self.source,
                    "detector": bundled_yunet(),
                    "recognizer": model,
                    "registry": self.snapshot,
                },
            )
        create.assert_not_called()

    def test_alignment_failure_is_unusable_without_identity_assignment(self):
        self.video([self.astronaut] * 3)
        with patch.object(
            FaceEmbedding, "_embedding", side_effect=cv2.error("alignment failed")
        ):
            result = self.recognize()
        report = read_json(result.outputs["report"].path)
        self.assertEqual(report["counts"]["unusable_observation_count"], 3)
        for row in iter_rows(result.outputs["frames"].path):
            self.assertEqual(
                row["faces"][0]["identity"],
                {"status": "unusable", "reason": "alignment_failed"},
            )

    def test_tracking_timestamp_mismatch_is_rejected_before_embedding(self):
        self.video([self.astronaut] * 3)
        track = self.modules["face.track"]
        original = track.execute

        def inconsistent(*args):
            outputs = original(*args)
            rows = list(iter_rows(outputs["frames"]))
            rows[0]["presentation_timestamp"] += 1
            outputs["frames"].write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = read_json(outputs["summary"])
            summary["file_sha256"]["frames"] = digest_file(outputs["frames"])[0]
            outputs["summary"].write_text(json.dumps(summary), encoding="utf-8")
            return outputs

        with (
            patch.object(track, "execute", side_effect=inconsistent),
            patch.object(FaceEmbedding, "_embedding") as inference,
            self.assertRaisesRegex(RunFailed, "timestamps do not match") as caught,
        ):
            self.recognize()
        inference.assert_not_called()
        self.assertFalse(
            (self.storage.root / "outputs" / caught.exception.run_id).exists()
        )

    def test_real_sface_alignment_recognizes_quarter_turn_video(self):
        self.video([self.astronaut] * 3)
        identity = self.enroll_result(self.recognize())
        self.snapshot = self.identities.snapshot()
        options = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "full"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        if "-display_rotation" not in options.stdout:
            self.skipTest("FFmpeg cannot generate display rotation metadata")
        for rotation, operation in (
            (90, cv2.ROTATE_90_CLOCKWISE),
            (180, cv2.ROTATE_180),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ):
            with self.subTest(rotation=rotation):
                self.source = self.root / f"coded-{rotation}.mkv"
                self.video([cv2.rotate(self.astronaut, operation)] * 3)
                base = self.root / f"base-{rotation}.mp4"
                rotated = self.root / f"display-{rotation}.mp4"
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
                    capture_output=True,
                    check=True,
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
                        str(rotated),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
                self.source = rotated
                result = self.recognize()
                self.assertEqual(
                    read_json(result.outputs["report"].path)["video"][
                        "rotation_degrees"
                    ],
                    rotation,
                )
                for row in iter_rows(result.outputs["frames"].path):
                    self.assertEqual(
                        row["faces"][0]["identity"]["identity_id"],
                        identity["identity_id"],
                    )

    def test_declarative_example_has_one_final_and_no_embeddings_in_outputs(self):
        self.video([self.astronaut] * 3)
        example = Path(__file__).resolve().parents[1] / "examples/recognize-faces.json"
        pipeline = Pipeline.load(example, self.modules)
        self.assertEqual(pipeline.final, "report")
        result = self.runner.run(
            pipeline,
            {
                "video": self.source,
                "detector": bundled_yunet(),
                "recognizer": bundled_sface(),
                "registry": self.snapshot,
            },
        )
        self.assertEqual(set(result.outputs), {"report", "frames", "tracks"})
        for artifact in result.outputs.values():
            self.assertNotIn('"embedding":', artifact.path.read_text())

    def cli(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["--root", str(self.storage.root), *arguments])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertNotIn('"embedding":', stdout.getvalue() + stderr.getvalue())
        return json.loads(stdout.getvalue())

    def test_cli_enroll_recognize_list_snapshot_delete(self):
        self.video([self.astronaut] * 4)
        with contextlib.chdir(self.root):
            first = self.cli("recognize-faces", self.source.name)
        intermediate = first["private_intermediates"]
        enrolled = self.cli(
            "identity",
            "register",
            "--embeddings",
            intermediate["frames"],
            "--summary",
            intermediate["summary"],
            "--track",
            "track-000001",
            "--alias",
            "owner",
        )
        listed = self.cli("identity", "list")
        self.assertEqual(
            listed["identities"][0]["identity_id"], enrolled["identity_id"]
        )
        snapshot = self.cli("identity", "snapshot")
        second = self.cli(
            "recognize-faces", str(self.source), "--snapshot", snapshot["path"]
        )
        report = read_json(Path(second["outputs"]["report"]["path"]))
        self.assertTrue(report["identity_recognition_performed"])
        self.assertFalse(report["anonymization_performed"])
        self.assertEqual(report["counts"]["known_observation_count"], 4)
        self.cli("identity", "delete", enrolled["identity_id"])
        third = self.cli("recognize-faces", str(self.source))
        self.assertEqual(
            read_json(Path(third["outputs"]["report"]["path"]))["counts"][
                "unknown_observation_count"
            ],
            4,
        )
