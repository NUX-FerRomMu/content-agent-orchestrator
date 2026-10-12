"""YuNet detection with verified model bytes and resumable complete chunks."""

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from ..contracts import Artifact, ExecutionError, ModuleContext, PipelineError
from ..face_data import (
    YUNET_SHA256,
    atomic_json,
    iter_rows,
    read_json,
    safe_child,
    validate_faces,
    validate_timestamp,
    write_json,
)
from ..pipeline import canonical_json
from ..storage import digest_file
from ..video_frames import decoded_frames


class FaceDetection:
    name = "face.detect"
    version = f"1-opencv-{cv2.__version__}"
    input_kinds: ClassVar = {
        "video": "video.source",
        "inspection": "video.manifest",
        "timestamps": "video.timestamps",
        "model": "face.detector.yunet",
    }
    output_kinds: ClassVar = {
        "frames": "face.detections",
        "summary": "face.detection-summary",
    }
    defaults: ClassVar = {
        "max_dimension": 640,
        "score_threshold": 0.85,
        "nms_threshold": 0.3,
        "top_k": 100,
        "chunk_frames": 64,
        "timeout_seconds": 600,
    }

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        if parameters.keys() - self.defaults.keys():
            raise PipelineError("Unknown face.detect parameter")
        values = self.defaults | dict(parameters)
        for name, lower, upper in (
            ("max_dimension", 64, 2048),
            ("top_k", 1, 500),
            ("chunk_frames", 1, 1024),
            ("timeout_seconds", 1, 86400),
        ):
            if type(values[name]) is not int or not lower <= values[name] <= upper:
                raise PipelineError(
                    f"{name} must be an integer between {lower} and {upper}"
                )
        if values["max_dimension"] % 32:
            raise PipelineError("max_dimension must be a multiple of 32")
        for name in ("score_threshold", "nms_threshold"):
            value = values[name]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise PipelineError(f"{name} must be a finite number in (0, 1]")

    @staticmethod
    def _inspection(inputs: Mapping[str, Artifact]) -> tuple[dict, dict]:
        inspection = read_json(inputs["inspection"].path)
        try:
            if inspection["schema_version"] != 1 or inspection["source"] != {
                "sha256": inputs["video"].sha256,
                "size_bytes": inputs["video"].size,
            }:
                raise ValueError
            if inspection["timestamps_sha256"] != inputs["timestamps"].sha256:
                raise ExecutionError("Timestamps do not belong to this inspected video")
            stream = next(
                item
                for item in inspection["streams"]
                if item["index"] == inspection["primary_video_stream"]
            )
            count = inspection["timing"]["decoded_frame_count"]
            if (
                stream["codec_type"] != "video"
                or type(stream["index"]) is not int
                or stream["index"] < 0
            ):
                raise ValueError
            if any(
                type(value) is not int or value < 1
                for value in (stream["width"], stream["height"], count)
            ):
                raise ValueError
            video = {key: stream[key] for key in ("width", "height", "time_base")}
            video.update(
                {
                    "frame_count": count,
                    "stream_index": stream["index"],
                    "coordinate_system": "coded_pixels",
                    "rotation_degrees": stream.get("rotation_degrees", 0),
                }
            )
            rotation = float(video["rotation_degrees"])
            if not math.isfinite(rotation):
                raise ValueError
            normalized = round(rotation / 90) * 90
            if abs(rotation - normalized) > 0.001:
                raise ValueError
            video["rotation_degrees"] = normalized % 360
        except (KeyError, TypeError, ValueError, StopIteration) as exc:
            raise ExecutionError(
                "Inspection must describe this video and a readable primary stream"
            ) from exc
        return inspection, video

    @staticmethod
    def _detect_faces(
        detector, image: np.ndarray, width: int, height: int, rotation: int = 0
    ) -> list[dict]:
        coded_height, coded_width = image.shape[:2]
        if rotation:
            operation = {
                90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_CLOCKWISE,
            }[rotation]
            image = cv2.rotate(image, operation)
        try:
            _, detections = detector.detect(image)
        except cv2.error as exc:
            raise ExecutionError(f"YuNet detection failed: {exc}") from exc
        if detections is None:
            return []
        scale_x, scale_y = width / coded_width, height / coded_height

        def restore(points, *, centers=False):
            points = np.asarray(points, dtype=float)
            x, y = points[:, 0], points[:, 1]
            edge_x, edge_y = coded_width - int(centers), coded_height - int(centers)
            if rotation == 90:
                x, y = edge_x - y, x
            elif rotation == 180:
                x, y = edge_x - x, edge_y - y
            elif rotation == 270:
                x, y = y, edge_y - x
            return np.column_stack((x * scale_x, y * scale_y))

        faces = []
        for detection in detections:
            if len(detection) != 15 or not np.isfinite(detection).all():
                raise ExecutionError("YuNet returned invalid coordinates")
            x, y, w, h = (float(value) for value in detection[:4])
            corners = restore([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            left, top = (
                max(0, min(width, float(corners[:, 0].min()))),
                max(0, min(height, float(corners[:, 1].min()))),
            )
            right, bottom = (
                max(0, min(width, float(corners[:, 0].max()))),
                max(0, min(height, float(corners[:, 1].max()))),
            )
            if right <= left or bottom <= top:
                continue
            points = restore(detection[4:14].reshape(5, 2), centers=True)
            faces.append(
                {
                    "bbox": [
                        round(left, 3),
                        round(top, 3),
                        round(right - left, 3),
                        round(bottom - top, 3),
                    ],
                    "landmarks": [
                        [
                            round(max(0, min(width, float(px))), 3),
                            round(max(0, min(height, float(py))), 3),
                        ]
                        for px, py in points
                    ],
                    "score": round(float(detection[14]), 6),
                }
            )
        faces.sort(key=lambda face: (face["bbox"][0], face["bbox"][1], -face["score"]))
        validate_faces({"faces": faces}, width, height)
        return faces

    @staticmethod
    def _load_chunks(
        cache: Path, signature: str, video: dict, chunk_frames: int
    ) -> list[dict]:
        checkpoint = safe_child(cache, "checkpoint.json")
        try:
            saved = read_json(checkpoint)
            entries = saved["chunks"]
            if saved["signature"] != signature or not isinstance(entries, list):
                return []
            if len(entries) > math.ceil(video["frame_count"] / chunk_frames):
                return []
        except (ExecutionError, KeyError):
            return []
        chunks = []
        count, previous = 0, None
        for index, entry in enumerate(entries):
            path = safe_child(cache, f"chunk-{index:08d}.jsonl")
            try:
                if (
                    not isinstance(entry, dict)
                    or type(entry["count"]) is not int
                    or not 1 <= entry["count"] <= chunk_frames
                ):
                    break
                if digest_file(path) != (entry["sha256"], entry["size"]):
                    break
                rows = 0
                for row in iter_rows(path):
                    previous = validate_timestamp(row, count + rows, previous)
                    validate_faces(row, video["width"], video["height"])
                    rows += 1
                if rows != entry["count"] or count + rows > video["frame_count"]:
                    break
                if rows < chunk_frames and count + rows != video["frame_count"]:
                    break
                count += rows
                chunks.append(entry)
            except (ExecutionError, OSError, KeyError, TypeError):
                break
        return chunks

    @staticmethod
    def _commit_chunk(
        stream,
        partial: Path,
        cache: Path,
        chunks: list[dict],
        count: int,
        signature: str,
    ) -> None:
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        destination = safe_child(cache, f"chunk-{len(chunks):08d}.jsonl")
        partial.replace(destination)
        digest, size = digest_file(destination)
        chunks.append({"count": count, "sha256": digest, "size": size})
        atomic_json(
            safe_child(cache, "checkpoint.json"),
            {"signature": signature, "chunks": chunks},
        )

    def execute(
        self,
        inputs: Mapping[str, Artifact],
        parameters: Mapping[str, Any],
        context: ModuleContext,
    ) -> Mapping[str, Path]:
        self.validate_parameters(parameters)
        values = self.defaults | dict(parameters)
        model = inputs["model"]
        if model.sha256 != YUNET_SHA256 or digest_file(model.path)[0] != YUNET_SHA256:
            raise ExecutionError(
                "YuNet model checksum does not match the supported pinned model"
            )
        inspection, video = self._inspection(inputs)
        width, height = video["width"], video["height"]
        factor = min(1, values["max_dimension"] / max(width, height))
        decode_width = max(32, round(width * factor / 32) * 32)
        decode_height = max(32, round(height * factor / 32) * 32)
        detector_size = (
            (decode_height, decode_width)
            if video["rotation_degrees"] in (90, 270)
            else (decode_width, decode_height)
        )
        try:
            detector = cv2.FaceDetectorYN_create(
                str(model.path),
                "",
                detector_size,
                values["score_threshold"],
                values["nms_threshold"],
                values["top_k"],
                cv2.dnn.DNN_BACKEND_OPENCV,
                cv2.dnn.DNN_TARGET_CPU,
            )
        except cv2.error as exc:
            raise ExecutionError(f"Cannot initialize YuNet: {exc}") from exc
        signature = hashlib.sha256(
            canonical_json(
                {
                    "version": self.version,
                    "parameters": values,
                    "inputs": {name: item.sha256 for name, item in inputs.items()},
                }
            ).encode()
        ).hexdigest()
        cache = safe_child(context.work_dir, "cache")
        cache.mkdir(exist_ok=True, mode=0o700)
        chunks = self._load_chunks(cache, signature, video, values["chunk_frames"])
        processed = sum(chunk["count"] for chunk in chunks)
        atomic_json(
            safe_child(cache, "checkpoint.json"),
            {"signature": signature, "chunks": chunks},
        )
        pending_stream, pending_count = None, 0
        try:
            with decoded_frames(
                inputs["video"].path,
                inputs["timestamps"].path,
                stream_index=video["stream_index"],
                width=decode_width,
                height=decode_height,
                expected_count=video["frame_count"],
                work_dir=context.work_dir,
                timeout=values["timeout_seconds"],
            ) as frames:
                for clock, image in frames:
                    if clock["frame_index"] < processed:
                        continue
                    if pending_stream is None:
                        partial = safe_child(cache, f"chunk-{len(chunks):08d}.part")
                        pending_stream = partial.open("w", encoding="utf-8")
                    faces = self._detect_faces(
                        detector, image, width, height, video["rotation_degrees"]
                    )
                    pending_stream.write(
                        json.dumps(clock | {"faces": faces}, allow_nan=False) + "\n"
                    )
                    pending_count += 1
                    if pending_count == values["chunk_frames"]:
                        self._commit_chunk(
                            pending_stream,
                            partial,
                            cache,
                            chunks,
                            pending_count,
                            signature,
                        )
                        pending_stream, pending_count = None, 0
            if pending_stream is not None:
                self._commit_chunk(
                    pending_stream, partial, cache, chunks, pending_count, signature
                )
        finally:
            if pending_stream is not None:
                pending_stream.close()
        output = context.result_dir / "detections.jsonl"
        count = faces_count = empty_count = maximum = 0
        with output.open("w", encoding="utf-8") as stream:
            for index in range(len(chunks)):
                for row in iter_rows(safe_child(cache, f"chunk-{index:08d}.jsonl")):
                    faces = row["faces"]
                    count += 1
                    faces_count += len(faces)
                    empty_count += not faces
                    maximum = max(maximum, len(faces))
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        if count != video["frame_count"]:
            raise ExecutionError("Detection cache does not cover every inspected frame")
        summary_path = context.result_dir / "summary.json"
        write_json(
            summary_path,
            {
                "schema_version": 1,
                "source": inspection["source"],
                "video": video,
                "detector": {
                    "name": "YuNet",
                    "model_sha256": model.sha256,
                    "opencv_version": cv2.__version__,
                    "backend": "cpu",
                    "parameters": values,
                },
                "counts": {
                    "frame_count": count,
                    "face_detection_count": faces_count,
                    "frames_without_faces": empty_count,
                    "max_simultaneous_faces": maximum,
                },
                "file_sha256": {"frames": digest_file(output)[0]},
            },
        )
        return {"frames": output, "summary": summary_path}
