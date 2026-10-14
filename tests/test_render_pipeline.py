import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from test_swap_engine import tiny_models
from test_virtual_faces import FIXTURE, PROFILE, FixtureGenerator

from content_agent_orchestrator.cli import main, video_anonymization_definition
from content_agent_orchestrator.contracts import ExecutionError
from content_agent_orchestrator.face_data import bundled_yunet, iter_rows, read_json
from content_agent_orchestrator.identity_data import bundled_sface
from content_agent_orchestrator.modules import builtin_registry
from content_agent_orchestrator.modules.video_export import audio_signatures
from content_agent_orchestrator.pipeline import Pipeline
from content_agent_orchestrator.runner import RunFailed, Runner
from content_agent_orchestrator.storage import UserStorage, digest_file
from content_agent_orchestrator.virtual_face import png_image, validate_portrait
from content_agent_orchestrator.virtual_registry import VirtualRegistry

HAS_RUNTIME = bool(
    importlib.util.find_spec("av")
    and importlib.util.find_spec("onnxruntime")
    and importlib.util.find_spec("onnx")
    and shutil.which("ffmpeg")
    and shutil.which("ffprobe")
)


class FixtureSwapper:
    """Synthetic fixture compositing, without claiming trained model quality."""

    def __init__(self, owner):
        self.owner = owner

    def prepare_target(self, name, content):
        self.image = png_image(content)
        self.face = validate_portrait(content)["face"]

    def replace(self, image, face, target):
        self.owner.swap_calls += 1
        if self.owner.interrupt_at == self.owner.swap_calls:
            raise KeyboardInterrupt
        if self.owner.fail_at == self.owner.swap_calls:
            raise ExecutionError("simulated failed face swap")
        if self.owner.mode == "unchanged":
            return image
        transform, _ = cv2.estimateAffinePartial2D(
            np.array(self.face["landmarks"]),
            np.array(face["landmarks"]),
            method=cv2.LMEDS,
        )
        return cv2.warpAffine(
            self.image,
            transform,
            (image.shape[1], image.shape[0]),
            borderValue=(96, 96, 96),
        )


@unittest.skipUnless(HAS_RUNTIME, "Optional swap runtime and FFmpeg required")
class RenderPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.storage = UserStorage(self.root / "user")
        self.modules = builtin_registry()
        self.runner = Runner(self.storage, self.modules)
        self.source = self.root / "source.mkv"
        self.original_image = cv2.imread(str(FIXTURE.parent / "astronaut.png"))
        self.make_video([self.original_image] * 5)
        result = self.cli("recognize-faces", str(self.source))
        paths = result["private_intermediates"]
        self.seed_frames = Path(paths["frames"])
        identity = self.cli(
            "identity",
            "register",
            "--embeddings",
            paths["frames"],
            "--summary",
            paths["summary"],
            "--track",
            "track-000001",
        )
        self.identity_id = identity["identity_id"]
        self.registry = VirtualRegistry(self.storage)
        candidate = self.registry.create(
            identity["identity_id"], PROFILE, FixtureGenerator()
        )
        self.registry.approve(
            identity["identity_id"], candidate["candidate_id"], reviewed=True
        )
        self.snapshot = self.registry.identities.snapshot()
        self.catalog = self.registry.snapshot()
        self.swap_model, self.arcface = tiny_models(self.root)
        self.swap_calls = 0
        self.interrupt_at = self.fail_at = None
        self.mode = "fixture"

    def make_video(self, images, *, variable=False):
        h, w = images[0].shape[:2]
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
            f"{w}x{h}",
            "-framerate",
            "6",
            "-i",
            "pipe:0",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
        ]
        if variable:
            command += [
                "-vf",
                "setpts='if(lt(N,3),N/(6*TB),(0.5+(N-3)/3)/TB)'",
                "-fps_mode",
                "vfr",
            ]
        command += [
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            "-y",
            str(self.source),
        ]
        subprocess.run(
            command,
            input=b"".join(image.tobytes() for image in images),
            capture_output=True,
            check=True,
            timeout=30,
        )

    def cli(self, *arguments, expected=0):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(["--root", str(self.storage.root), *arguments])
        self.assertEqual(code, expected, error.getvalue())
        return json.loads(output.getvalue()) if expected == 0 else error.getvalue()

    def inputs(self):
        return {
            "video": self.source,
            "detector": bundled_yunet(),
            "recognizer": bundled_sface(),
            "registry": self.snapshot,
            "catalog": self.catalog,
            "swap_model": self.swap_model,
            "arcface_model": self.arcface,
        }

    def render(
        self, *, allow_all=False, codec="ffv1", resume=None, cache="reuse", native=False
    ):
        definition = video_anonymization_definition(
            swap_parameters={"chunk_frames": 2},
            export_parameters={"codec": codec, "allow_all_redacted": allow_all},
        )
        pipeline = Pipeline.parse(definition, self.modules)
        factory = (
            contextlib.nullcontext()
            if native
            else patch(
                "content_agent_orchestrator.modules.face_swap.InSwapper",
                side_effect=lambda *a, **k: FixtureSwapper(self),
            )
        )
        with factory:
            return self.runner.run(pipeline, self.inputs(), resume=resume, cache=cache)

    def pictures(self, path):
        import av

        with av.open(str(path)) as container:
            return [
                (
                    Fraction(frame.pts) * frame.time_base,
                    frame.to_ndarray(format="bgr24"),
                )
                for frame in container.decode(video=0)
            ]

    def validator_inputs(self):
        from content_agent_orchestrator.privacy_validation import PrivacyValidator
        from content_agent_orchestrator.virtual_catalog import read_catalog

        embedded = next(iter_rows(self.seed_frames))
        target = read_catalog(self.catalog)[0]["faces"][0]
        face = embedded["faces"][0]
        virtual = {
            **embedded,
            "faces": [
                {key: face[key] for key in ("track_id", "bbox", "landmarks", "score")}
                | {
                    "identity": {"status": "known", "identity_id": self.identity_id},
                    "virtual": {
                        "status": "ready",
                        **{
                            key: target[key]
                            for key in (
                                "virtual_id",
                                "portrait_file",
                                "portrait_sha256",
                            )
                        },
                    },
                }
            ],
        }
        video = {
            "width": 512,
            "height": 512,
            "frame_count": 5,
            "time_base": "1/1000",
            "rotation_degrees": 0,
        }
        validator = PrivacyValidator(
            video, self.catalog, self.snapshot, bundled_yunet(), bundled_sface()
        )
        engine = FixtureSwapper(self)
        engine.prepare_target(target["virtual_id"], FIXTURE.read_bytes())
        image = engine.replace(self.original_image, face, target["virtual_id"])
        return (
            validator,
            image,
            virtual,
            embedded,
            {"swapped": [face["track_id"]], "failed": []},
        )

    def test_independent_detector_missing_extra_and_error_results_require_redaction(
        self,
    ):
        from content_agent_orchestrator.modules.face_detect import FaceDetection

        validator, image, virtual, embedded, record = self.validator_inputs()
        face = virtual["faces"][0]
        for result, reason in (
            ([], "face_coverage_mismatch"),
            ([face, face], "face_coverage_mismatch"),
            (ExecutionError("detector failed"), "output_detection_failed"),
        ):
            with (
                self.subTest(reason=reason),
                patch.object(
                    FaceDetection,
                    "_detect_faces",
                    **(
                        {"side_effect": result}
                        if isinstance(result, Exception)
                        else {"return_value": result}
                    ),
                ),
            ):
                decision = validator.check(image, virtual, embedded, record)
                self.assertEqual(decision["status"], "redact")
                self.assertEqual(decision["reason"], reason)

    def test_output_embedding_failure_requires_redaction(self):
        from content_agent_orchestrator.modules.face_embed import FaceEmbedding

        validator, image, virtual, embedded, record = self.validator_inputs()
        with patch.object(
            FaceEmbedding, "_embedding", side_effect=ExecutionError("invalid embedding")
        ):
            decision = validator.check(image, virtual, embedded, record)
        self.assertEqual(decision["status"], "redact")
        self.assertEqual(decision["reason"], "invalid_output_embedding")

    def test_wrong_target_is_rejected_even_when_the_original_identity_is_absent(self):
        from content_agent_orchestrator.modules.face_detect import FaceDetection

        validator, _, virtual, embedded, record = self.validator_inputs()
        other = cv2.imread(str(FIXTURE.parent / "grace_hopper.jpg"))
        height, width = other.shape[:2]
        detector = cv2.FaceDetectorYN_create(
            str(bundled_yunet()), "", (width, height), 0.7, 0.3, 500
        )
        face = FaceDetection._detect_faces(detector, other, width, height)[0]
        transform, _ = cv2.estimateAffinePartial2D(
            np.array(face["landmarks"]),
            np.array(virtual["faces"][0]["landmarks"]),
            method=cv2.LMEDS,
        )
        image = cv2.warpAffine(other, transform, (512, 512), borderValue=(96, 96, 96))
        decision = validator.check(image, virtual, embedded, record)
        self.assertEqual(decision["status"], "redact")
        self.assertEqual(decision["reason"], "identity_check_failed")
        self.assertLess(decision["faces"][0]["source_similarity"], 0.3)
        self.assertLess(decision["faces"][0]["target_similarity"], 0.45)

    def test_display_rotation_is_normalized_without_scaling_or_losing_frames(self):
        for rotation, operation in (
            (90, cv2.ROTATE_90_CLOCKWISE),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ):
            with self.subTest(rotation=rotation):
                portrait = cv2.copyMakeBorder(
                    self.original_image, 0, 128, 0, 0, cv2.BORDER_CONSTANT
                )
                self.source = self.root / f"coded-{rotation}.mkv"
                self.make_video([cv2.rotate(portrait, operation)] * 5)
                base = self.root / f"base-{rotation}.mp4"
                rotated = self.root / f"rotated-{rotation}.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(self.source),
                        "-c:v",
                        "mpeg4",
                        "-an",
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
                result = self.render()
                report = read_json(result.outputs["report"].path)
                self.assertEqual(
                    (report["delivery"]["width"], report["delivery"]["height"]),
                    (512, 640),
                )
                self.assertEqual(report["counts"], {"validated": 5, "redacted": 0})
                self.assertEqual(len(self.pictures(result.outputs["video"].path)), 5)

    def test_real_cpu_pipeline_replaces_faces_preserves_audio_and_caches_all_stages(
        self,
    ):
        source_hash = digest_file(self.source)
        result = self.render()
        report = read_json(result.outputs["report"].path)
        self.assertEqual(report["counts"], {"validated": 5, "redacted": 0})
        self.assertTrue(report["anonymization_performed"])
        self.assertFalse(report["voice_modified"])
        self.assertEqual(digest_file(self.source), source_hash)
        self.assertEqual(
            audio_signatures(self.source),
            audio_signatures(result.outputs["video"].path),
        )
        self.assertEqual(len(self.pictures(result.outputs["video"].path)), 5)
        self.assertNotIn('"embedding":', result.outputs["audit"].path.read_text())
        again = self.render()
        self.assertEqual(len(again.cache_hits), 10)
        self.assertEqual(self.swap_calls, 5)
        self.assertEqual(
            digest_file(again.outputs["video"].path),
            digest_file(result.outputs["video"].path),
        )
        self.assertEqual(
            list((self.storage.root / "temps" / result.run_id).iterdir()), []
        )

    def test_one_failed_swap_is_opaque_and_frame_count_and_timing_are_unchanged(self):
        self.fail_at = 3
        result = self.render()
        self.assertEqual(
            read_json(result.outputs["report"].path)["counts"],
            {"validated": 4, "redacted": 1},
        )
        original, final = (
            self.pictures(self.source),
            self.pictures(result.outputs["video"].path),
        )
        self.assertEqual(len(final), len(original))
        self.assertEqual([time for time, _ in final], [time for time, _ in original])
        np.testing.assert_array_equal(final[2][1], np.full_like(final[2][1], 0))
        self.assertFalse(np.array_equal(final[0][1], original[0][1]))

    def test_missing_source_detection_is_opaque_and_never_passthrough(self):
        self.make_video(
            [
                self.original_image,
                self.original_image,
                np.zeros_like(self.original_image),
                self.original_image,
                self.original_image,
            ]
        )
        result = self.render()
        report = read_json(result.outputs["report"].path)
        self.assertEqual(report["counts"], {"validated": 4, "redacted": 1})
        np.testing.assert_array_equal(
            self.pictures(result.outputs["video"].path)[2][1],
            np.full_like(self.original_image, 0),
        )

    def test_source_face_passthrough_fails_independent_identity_check_and_publishes_nothing(
        self,
    ):
        self.mode = "unchanged"
        with self.assertRaisesRegex(RunFailed, "No frame passed") as caught:
            self.render()
        self.assertFalse(
            (self.storage.root / "outputs" / caught.exception.run_id).exists()
        )
        with self.storage.connect() as connection:
            row = connection.execute(
                "SELECT outputs FROM nodes WHERE run_id=? AND node_id='validate'",
                (caught.exception.run_id,),
            ).fetchone()
        audit = json.loads(row[0])["audit"]
        reasons = {
            row["reason"] for row in iter_rows(self.storage.root / audit["path"])
        }
        self.assertEqual(reasons, {"identity_check_failed"})

    def test_native_untrained_passthrough_model_is_redacted_not_accepted_as_swap(self):
        result = self.render(native=True, allow_all=True)
        self.assertEqual(
            read_json(result.outputs["report"].path)["counts"],
            {"validated": 0, "redacted": 5},
        )
        for _, image in self.pictures(result.outputs["video"].path):
            np.testing.assert_array_equal(image, np.full_like(image, 0))

    def test_unknown_person_is_fully_redacted(self):
        unknown = cv2.imread(str(FIXTURE.parent / "grace_hopper.jpg"))
        self.make_video([unknown] * 5)
        result = self.render(allow_all=True)
        self.assertEqual(self.swap_calls, 0)
        self.assertEqual(
            read_json(result.outputs["report"].path)["counts"],
            {"validated": 0, "redacted": 5},
        )
        for _, image in self.pictures(result.outputs["video"].path):
            np.testing.assert_array_equal(image, np.full_like(image, 0))

    def test_interrupted_swap_reuses_only_complete_chunks(self):
        self.interrupt_at = 3
        with self.assertRaises(KeyboardInterrupt):
            self.render()
        with self.storage.connect() as connection:
            run_id = connection.execute(
                "SELECT id FROM runs ORDER BY rowid DESC LIMIT 1"
            ).fetchone()[0]
        self.assertFalse((self.storage.root / "outputs" / run_id).exists())
        checkpoint = (
            self.storage.root / "temps" / run_id / "swap/frame-cache/checkpoint.json"
        )
        self.assertEqual(read_json(checkpoint)["chunks"][0]["count"], 2)
        self.interrupt_at = None
        result = self.render(resume=run_id)
        self.assertEqual(self.swap_calls, 6)
        self.assertEqual(result.run_id, run_id)
        self.assertEqual(
            read_json(result.outputs["report"].path)["counts"]["validated"], 5
        )

    def test_fresh_resume_discards_failed_swap_cache(self):
        self.interrupt_at = 3
        with self.assertRaises(KeyboardInterrupt):
            self.render()
        with self.storage.connect() as connection:
            run_id = connection.execute(
                "SELECT id FROM runs ORDER BY rowid DESC LIMIT 1"
            ).fetchone()[0]
        self.interrupt_at = None
        result = self.render(resume=run_id, cache="fresh")
        self.assertEqual(self.swap_calls, 8)
        self.assertEqual(result.cache_hits, ())

    def test_encoder_failure_publishes_nothing_and_resume_reuses_safe_stages(self):
        with (
            patch(
                "content_agent_orchestrator.modules.video_export.VideoWriter",
                side_effect=ExecutionError("simulated encoder failure"),
            ),
            self.assertRaisesRegex(RunFailed, "encoder failure") as caught,
        ):
            self.render()
        self.assertFalse(
            (self.storage.root / "outputs" / caught.exception.run_id).exists()
        )
        result = self.render(resume=caught.exception.run_id)
        self.assertEqual(len(result.cache_hits), 9)
        self.assertEqual(self.swap_calls, 5)

    def test_failed_post_encoding_validation_never_publishes_delivery(self):
        from content_agent_orchestrator.privacy_validation import PrivacyValidator

        def reject(*args):
            validator = PrivacyValidator(*args)
            validator.check = lambda *a: {
                "status": "redact",
                "reason": "simulated encoded-frame failure",
            }
            return validator

        with (
            patch(
                "content_agent_orchestrator.modules.video_export.PrivacyValidator",
                side_effect=reject,
            ),
            self.assertRaisesRegex(RunFailed, "Encoded delivery failed") as caught,
        ):
            self.render()
        self.assertFalse(
            (self.storage.root / "outputs" / caught.exception.run_id).exists()
        )

    def test_publication_checkpoint_failure_rolls_back_complete_delivery(self):
        original = self.runner._node_checkpoint

        def fail(run_id, node_id, fingerprint, status, *args):
            if node_id == "export" and status == "succeeded":
                raise ExecutionError("simulated publication checkpoint failure")
            return original(run_id, node_id, fingerprint, status, *args)

        with (
            patch.object(self.runner, "_node_checkpoint", side_effect=fail),
            self.assertRaisesRegex(RunFailed, "publication checkpoint") as caught,
        ):
            self.render()
        self.assertFalse(
            (self.storage.root / "outputs" / caught.exception.run_id).exists()
        )
        result = self.render(resume=caught.exception.run_id)
        self.assertEqual(
            read_json(result.outputs["report"].path)["counts"]["validated"], 5
        )

    def test_vfr_delivery_keeps_source_presentation_times_and_audio(self):
        self.make_video([self.original_image] * 6, variable=True)
        result = self.render()
        before, after = (
            self.pictures(self.source),
            self.pictures(result.outputs["video"].path),
        )
        self.assertEqual(len(before), len(after))
        for a, b in zip(before, after, strict=True):
            self.assertLessEqual(abs(a[0] - b[0]), Fraction(1, 1000))
        self.assertGreater(
            len({before[i + 1][0] - before[i][0] for i in range(len(before) - 1)}), 1
        )

    def test_h264_delivery_is_rechecked_after_real_lossy_encoding(self):
        import av

        self.fail_at = 3
        result = self.render(codec="h264")
        report = read_json(result.outputs["report"].path)
        self.assertEqual(report["delivery"]["codec"], "h264")
        self.assertEqual(report["counts"], {"validated": 4, "redacted": 1})
        final = self.pictures(result.outputs["video"].path)
        self.assertLessEqual(np.abs(final[2][1].astype(int)).max(), 3)
        with av.open(str(result.outputs["video"].path)) as container:
            self.assertEqual(container.streams.video[0].codec_context.colorspace, 5)
            self.assertEqual(container.streams.video[0].codec_context.color_range, 1)

    def test_cli_runs_the_complete_dag_and_requires_original_resume_inputs(self):
        with patch(
            "content_agent_orchestrator.modules.face_swap.InSwapper",
            side_effect=lambda *a, **k: FixtureSwapper(self),
        ):
            result = self.cli(
                "anonymize-video",
                str(self.source),
                "--swap-model",
                str(self.swap_model),
                "--arcface-model",
                str(self.arcface),
                "--codec",
                "ffv1",
            )
        self.assertEqual(set(result["outputs"]), {"video", "report", "audit"})
        self.assertTrue(Path(result["registry_snapshot"]).exists())
        self.assertTrue(Path(result["virtual_catalog"]).exists())
        error = self.cli(
            "anonymize-video",
            str(self.source),
            "--swap-model",
            str(self.swap_model),
            "--arcface-model",
            str(self.arcface),
            "--resume",
            result["run_id"],
            expected=1,
        )
        self.assertIn("original --snapshot", error)

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_final_and_intermediate_pixels_are_private(self):
        result = self.render()
        for artifact in result.outputs.values():
            self.assertEqual(artifact.path.stat().st_mode & 0o777, 0o600)
        with self.storage.connect() as connection:
            row = connection.execute(
                "SELECT outputs FROM nodes WHERE run_id=? AND node_id='swap'",
                (result.run_id,),
            ).fetchone()
        for item in json.loads(row[0]).values():
            self.assertEqual(
                (self.storage.root / item["path"]).stat().st_mode & 0o777, 0o600
            )
