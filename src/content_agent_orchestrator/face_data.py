"""Shared, bounded readers and validation for face-analysis artifacts."""

import json
import math
import os
import re
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from .contracts import ExecutionError

YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def bundled_yunet() -> Path:
    return Path(__file__).parent / "assets/yunet/face_detection_yunet_2023mar.onnx"


def safe_child(base: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ExecutionError("Analysis workspace paths must be relative")
    candidate = base
    for part in path.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ExecutionError("Analysis workspace must not contain symlinks")
    return candidate


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExecutionError(f"Cannot read analysis JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ExecutionError("Analysis JSON must be an object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def atomic_json(path: Path, value: Any) -> None:
    temporary = safe_child(path.parent, path.name + ".tmp")
    safe_child(path.parent, path.name)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ExecutionError("Invalid analysis JSONL record") from exc
            if not isinstance(row, dict):
                raise ExecutionError("Analysis JSONL records must be objects")
            yield row


def validate_timestamp(row: dict, frame_index: int, previous: int | None) -> int:
    tick = row.get("presentation_timestamp")
    if type(row.get("frame_index")) is not int or row["frame_index"] != frame_index:
        raise ExecutionError("Analysis frame indices must be contiguous")
    if type(tick) is not int or (previous is not None and tick <= previous):
        raise ExecutionError("Analysis needs present, strictly increasing timestamps")
    token = row.get("presentation_time")
    try:
        if not isinstance(token, str) or not Decimal(token).is_finite():
            raise InvalidOperation
    except InvalidOperation as exc:
        raise ExecutionError("Invalid presentation time") from exc
    return tick


def validate_faces(row: dict, width: int, height: int) -> list[dict]:
    faces = row.get("faces")
    if not isinstance(faces, list):
        raise ExecutionError("Each detection frame needs a faces list")
    for face in faces:
        if not isinstance(face, dict):
            raise ExecutionError("Invalid face detection")
        bbox, landmarks, score = (
            face.get(key) for key in ("bbox", "landmarks", "score")
        )
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not isinstance(landmarks, list)
            or len(landmarks) != 5
            or any(
                not isinstance(point, list) or len(point) != 2 for point in landmarks
            )
        ):
            raise ExecutionError("Faces need a bounding box and five landmarks")
        numbers = [*bbox, score, *(value for point in landmarks for value in point)]
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in numbers
        ):
            raise ExecutionError("Face coordinates and scores must be finite numbers")
        x, y, w, h = bbox
        if not (
            0 <= x < width
            and 0 <= y < height
            and w > 0
            and h > 0
            and x + w <= width + 0.001
            and y + h <= height + 0.001
            and 0 <= score <= 1
        ):
            raise ExecutionError("Face bounding box or confidence is out of range")
        if any(not (0 <= x <= width and 0 <= y <= height) for x, y in landmarks):
            raise ExecutionError("Face landmarks are outside the video")
    return faces


def analysis_summary(path: Path) -> dict:
    summary = read_json(path)
    try:
        video, source = summary["video"], summary["source"]
        if summary["schema_version"] != 1:
            raise ValueError
        if any(
            type(video[key]) is not int or video[key] < 1
            for key in ("width", "height", "frame_count")
        ):
            raise ValueError
        if not isinstance(video["time_base"], str) or Fraction(video["time_base"]) <= 0:
            raise ValueError
        if (
            not isinstance(source["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None
        ):
            raise ValueError
        if type(source["size_bytes"]) is not int or source["size_bytes"] < 0:
            raise ValueError
        counts = summary["counts"]
        if not isinstance(counts, dict) or any(
            type(value) is not int or value < 0 for value in counts.values()
        ):
            raise ValueError
        if type(counts.get("face_detection_count")) is not int:
            raise ValueError
        if "frame_count" in counts and counts["frame_count"] != video["frame_count"]:
            raise ValueError
        if not isinstance(summary.get("file_sha256"), dict):
            raise TypeError
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ExecutionError("Invalid face-analysis summary") from exc
    return summary
