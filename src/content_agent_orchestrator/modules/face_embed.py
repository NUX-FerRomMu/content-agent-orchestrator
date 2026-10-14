"""Align original coded frames and extract one SFace vector per observation."""

import json
import math
from collections.abc import Mapping
from typing import ClassVar

import cv2
import numpy as np

from ..contracts import ExecutionError, PipelineError
from ..face_data import (
    analysis_summary,
    iter_rows,
    validate_faces,
    validate_timestamp,
    write_json,
)
from ..identity_data import RECOGNIZER, SFACE_SHA256, TRACK_PATTERN
from ..storage import digest_file
from ..video_frames import decoded_frames
from .face_detect import FaceDetection


class FaceEmbedding:
    name = "face.embed"
    version = f"1-opencv-{cv2.__version__}"
    input_kinds: ClassVar = {
        "video": "video.source",
        "inspection": "video.manifest",
        "timestamps": "video.timestamps",
        "frames": "face.track-frames",
        "summary": "face.tracking-summary",
        "model": "face.recognizer.sface",
    }
    output_kinds: ClassVar = {
        "frames": "face.embeddings",
        "summary": "face.embedding-summary",
    }
    defaults: ClassVar = {
        "min_face_pixels": 64,
        "min_confidence": 0.9,
        "timeout_seconds": 600,
    }

    def validate_parameters(self, parameters: Mapping) -> None:
        if parameters.keys() - self.defaults.keys():
            raise PipelineError("Unknown face.embed parameter")
        values = self.defaults | dict(parameters)
        for name, lower, upper in (
            ("min_face_pixels", 16, 2048),
            ("timeout_seconds", 1, 86400),
        ):
            if type(values[name]) is not int or not lower <= values[name] <= upper:
                raise PipelineError(f"Invalid {name}")
        value = values["min_confidence"]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 < value <= 1
        ):
            raise PipelineError("Invalid min_confidence")

    @staticmethod
    def _embedding(recognizer, image, face):
        detection = np.asarray(
            [
                *face["bbox"],
                *(v for point in face["landmarks"] for v in point),
                face["score"],
            ],
            dtype=np.float32,
        )
        aligned = recognizer.alignCrop(image, detection)
        features = np.asarray(recognizer.feature(aligned), dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(features))
        if (
            features.size != 128
            or not np.isfinite(features).all()
            or not math.isfinite(norm)
            or norm <= 1e-12
        ):
            raise ExecutionError("SFace produced an invalid embedding")
        return (features / norm).tolist()

    def execute(self, inputs, parameters, context):
        self.validate_parameters(parameters)
        values = self.defaults | dict(parameters)
        _, video = FaceDetection._inspection(inputs)
        tracking = analysis_summary(inputs["summary"].path)
        if (
            tracking.get("identity_scope") != "clip_trajectory_only"
            or tracking["source"]
            != {"sha256": inputs["video"].sha256, "size_bytes": inputs["video"].size}
            or tracking["video"] != video
        ):
            raise ExecutionError("Tracking does not belong to this inspected video")
        if tracking["file_sha256"].get("frames") != inputs["frames"].sha256:
            raise ExecutionError("Tracking frames do not belong to this summary")
        if (
            inputs["model"].sha256 != SFACE_SHA256
            or digest_file(inputs["model"].path)[0] != SFACE_SHA256
        ):
            raise ExecutionError("SFace model checksum does not match the pinned model")
        cv2.setNumThreads(4)
        try:
            recognizer = cv2.FaceRecognizerSF_create(
                str(inputs["model"].path),
                "",
                cv2.dnn.DNN_BACKEND_OPENCV,
                cv2.dnn.DNN_TARGET_CPU,
            )
        except cv2.error as exc:
            raise ExecutionError("Cannot load pinned SFace model") from exc
        frames_path = context.result_dir / "embeddings.jsonl"
        # Private vectors, including incomplete results, must not use the default umask.
        frames_path.touch(mode=0o600)
        rows = iter(iter_rows(inputs["frames"].path))
        frame_count = face_count = usable_count = 0
        previous = None
        with (
            frames_path.open("w", encoding="utf-8") as output,
            decoded_frames(
                inputs["video"].path,
                inputs["timestamps"].path,
                stream_index=video["stream_index"],
                width=video["width"],
                height=video["height"],
                expected_count=video["frame_count"],
                work_dir=context.work_dir,
                timeout=values["timeout_seconds"],
            ) as frames,
        ):
            for clock, image in frames:
                row = next(rows, None)
                if row is None:
                    raise ExecutionError("Missing tracking frame")
                previous = validate_timestamp(row, frame_count, previous)
                if any(row.get(key) != value for key, value in clock.items()):
                    raise ExecutionError("Tracking timestamps do not match the video")
                faces = validate_faces(row, video["width"], video["height"])
                seen = set()
                observations = []
                for face in faces:
                    track_id = face.get("track_id")
                    if (
                        not isinstance(track_id, str)
                        or TRACK_PATTERN.fullmatch(track_id) is None
                        or track_id in seen
                        or face.get("association")
                        not in ("new", "matched", "reacquired", "ambiguous")
                    ):
                        raise ExecutionError("Invalid tracking observation")
                    seen.add(track_id)
                    observation = dict(face)
                    if min(face["bbox"][2:]) < values["min_face_pixels"]:
                        status = "small_face"
                    elif face["score"] < values["min_confidence"]:
                        status = "low_confidence"
                    else:
                        try:
                            observation["embedding"] = self._embedding(
                                recognizer, image, face
                            )
                            status = "usable"
                            usable_count += 1
                        except cv2.error:
                            status = "alignment_failed"
                    observation["embedding_status"] = status
                    observations.append(observation)
                output.write(
                    json.dumps(clock | {"faces": observations}, allow_nan=False) + "\n"
                )
                face_count += len(faces)
                frame_count += 1
        if (
            next(rows, None) is not None
            or face_count != tracking["counts"]["face_detection_count"]
        ):
            raise ExecutionError("Tracking records do not match their summary")
        summary_path = context.result_dir / "embedding_summary.json"
        write_json(
            summary_path,
            {
                "schema_version": 1,
                "source": tracking["source"],
                "video": video,
                "recognizer": RECOGNIZER,
                "extraction": values,
                "counts": {
                    "frame_count": frame_count,
                    "face_detection_count": face_count,
                    "usable_embedding_count": usable_count,
                },
                "file_sha256": {"frames": digest_file(frames_path)[0]},
            },
        )
        return {"frames": frames_path, "summary": summary_path}
