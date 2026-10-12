import json
import tempfile
import unittest
from pathlib import Path

from content_agent_orchestrator.contracts import (
    Artifact,
    ExecutionError,
    ModuleContext,
    PipelineError,
)
from content_agent_orchestrator.face_data import iter_rows, write_json
from content_agent_orchestrator.modules import FaceDetection, FaceTracking
from content_agent_orchestrator.storage import digest_file


def face(x, y=20, width=20, height=20):
    return {
        "bbox": [x, y, width, height],
        "landmarks": [[x + 5, y + 5]] * 5,
        "score": 0.95,
    }


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.result = self.root / "result"
        self.result.mkdir()
        self.context = ModuleContext(self.root, self.result)

    def run_tracking(self, faces_by_frame, *, parameters=None, ticks=None, edit=None):
        if ticks is None:
            ticks = list(range(len(faces_by_frame)))
        rows = [
            {
                "frame_index": index,
                "presentation_timestamp": tick,
                "presentation_time": f"{tick / 10:.6f}",
                "faces": faces,
            }
            for index, (tick, faces) in enumerate(
                zip(ticks, faces_by_frame, strict=True)
            )
        ]
        if edit:
            edit(rows)
        frames = self.root / "detections.jsonl"
        frames.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        summary = self.root / "summary.json"
        write_json(
            summary,
            {
                "schema_version": 1,
                "source": {"sha256": "a" * 64, "size_bytes": 100},
                "video": {
                    "width": 200,
                    "height": 100,
                    "frame_count": len(faces_by_frame),
                    "time_base": "1/10",
                },
                "counts": {"face_detection_count": sum(map(len, faces_by_frame))},
                "file_sha256": {"frames": digest_file(frames)[0]},
            },
        )
        artifacts = {}
        for name, path, kind in (
            ("frames", frames, "face.detections"),
            ("summary", summary, "face.detection-summary"),
        ):
            digest, size = digest_file(path)
            artifacts[name] = Artifact(path, kind, digest, size)
        outputs = FaceTracking().execute(artifacts, parameters or {}, self.context)
        return {
            name: json.loads(path.read_text())
            if name == "summary"
            else list(iter_rows(path))
            for name, path in outputs.items()
        }

    def test_two_moving_faces_keep_distinct_tracks(self):
        result = self.run_tracking(
            [[face(10), face(100)], [face(12), face(98)], [face(14), face(96)]]
        )
        ids = [[face["track_id"] for face in row["faces"]] for row in result["frames"]]
        self.assertEqual(ids, [["track-000001", "track-000002"]] * 3)
        self.assertEqual(result["summary"]["counts"]["track_count"], 2)
        self.assertEqual([track["observations"] for track in result["tracks"]], [3, 3])

    def test_short_gap_is_explicit_and_reacquired(self):
        result = self.run_tracking([[face(10)], [], [face(10)]])
        last = result["frames"][-1]["faces"][0]
        self.assertEqual(last["track_id"], "track-000001")
        self.assertEqual(last["association"], "reacquired")
        self.assertEqual(last["gap_frames"], 1)
        self.assertEqual(result["tracks"][0]["missing_frames"], 1)
        self.assertEqual(result["intervals"][0]["kind"], "no_faces")
        self.assertEqual(result["intervals"][0]["start_frame"], 1)
        self.assertEqual(result["intervals"][0]["end_frame"], 1)

    def test_long_gap_creates_new_track(self):
        result = self.run_tracking(
            [[face(10)], [], [], [face(10)]], parameters={"max_gap_frames": 1}
        )
        self.assertEqual(result["frames"][-1]["faces"][0]["track_id"], "track-000002")
        self.assertEqual(result["tracks"][0]["end_reason"], "gap_expired")

    def test_variable_timing_uses_elapsed_time_for_gap_limit(self):
        result = self.run_tracking(
            [[face(10)], [], [face(10)]],
            ticks=[0, 1, 20],
            parameters={"max_gap_frames": 10, "max_gap_seconds": 0.5},
        )
        self.assertEqual(result["frames"][-1]["presentation_timestamp"], 20)
        self.assertEqual(result["frames"][-1]["faces"][0]["track_id"], "track-000002")

    def test_ambiguous_merge_retires_old_tracks(self):
        result = self.run_tracking([[face(10), face(22)], [face(16)]])
        detected = result["frames"][-1]["faces"][0]
        self.assertEqual(detected["association"], "ambiguous")
        self.assertEqual(detected["track_id"], "track-000003")
        self.assertEqual(
            [track["end_reason"] for track in result["tracks"][:2]],
            ["ambiguous", "ambiguous"],
        )
        self.assertEqual(result["summary"]["counts"]["ambiguous_frames"], 1)
        self.assertEqual(result["intervals"][-1]["kind"], "ambiguous_association")

    def test_ambiguous_split_does_not_duplicate_old_identity(self):
        result = self.run_tracking([[face(16)], [face(10), face(22)]])
        self.assertEqual(
            [item["association"] for item in result["frames"][-1]["faces"]],
            ["ambiguous", "ambiguous"],
        )
        self.assertEqual(
            len({item["track_id"] for item in result["frames"][-1]["faces"]}), 2
        )

    def test_no_faces_produces_empty_tracks_and_one_interval(self):
        result = self.run_tracking([[], [], []])
        self.assertEqual(result["tracks"], [])
        self.assertEqual(result["summary"]["counts"]["frames_without_faces"], 3)
        self.assertEqual(len(result["intervals"]), 1)
        self.assertEqual(result["intervals"][0]["end_frame"], 2)

    def test_invalid_records_are_rejected(self):
        edits = [
            lambda rows: rows[0].update(frame_index=4),
            lambda rows: rows[0].update(presentation_timestamp=None),
            lambda rows: rows[0].update(presentation_time="NaN"),
            lambda rows: rows[0]["faces"][0].update(bbox=[-1, 20, 20, 20]),
            lambda rows: rows[0]["faces"][0].update(score=float("nan")),
            lambda rows: rows.pop(),
        ]
        for edit in edits:
            with self.subTest(edit=edit), self.assertRaises(ExecutionError):
                self.run_tracking([[face(10)], [face(11)]], edit=edit)

    def test_missing_or_duplicate_timestamps_are_rejected(self):
        with self.assertRaisesRegex(ExecutionError, "strictly increasing"):
            self.run_tracking([[face(10)], [face(11)]], ticks=[0, 0])

    def test_summary_cannot_be_paired_with_other_detection_records(self):
        self.run_tracking([[face(10)]])
        frames = self.root / "detections.jsonl"
        summary = self.root / "summary.json"
        foreign = json.loads(summary.read_text())
        foreign["file_sha256"]["frames"] = "0" * 64
        write_json(summary, foreign)
        inputs = {}
        for name, path, kind in (
            ("frames", frames, "face.detections"),
            ("summary", summary, "face.detection-summary"),
        ):
            digest, size = digest_file(path)
            inputs[name] = Artifact(path, kind, digest, size)
        with self.assertRaisesRegex(ExecutionError, "do not belong"):
            FaceTracking().execute(inputs, {}, self.context)

    def test_detector_and_tracker_reject_invalid_parameters(self):
        cases = [
            (FaceDetection(), {"chunk_frames": True}),
            (FaceDetection(), {"max_dimension": 641}),
            (FaceDetection(), {"score_threshold": float("nan")}),
            (FaceTracking(), {"max_gap_frames": -1}),
            (FaceTracking(), {"max_gap_seconds": float("inf")}),
            (FaceTracking(), {"ambiguity_margin": True}),
        ]
        for module, parameters in cases:
            with (
                self.subTest(module=module.name, parameters=parameters),
                self.assertRaises(PipelineError),
            ):
                module.validate_parameters(parameters)
