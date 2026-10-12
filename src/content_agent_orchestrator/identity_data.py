"""Versioned SFace contracts; vectors never belong in public reports."""

import math
import re
import uuid
from pathlib import Path

import numpy as np

from .contracts import ExecutionError
from .face_data import analysis_summary, iter_rows, validate_faces, validate_timestamp
from .storage import digest_file

SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
RECOGNIZER = {
    "model_sha256": SFACE_SHA256,
    "preprocessing": "sface-aligncrop-v1",
    "dimensions": 128,
}
TRACK_PATTERN = re.compile(r"track-[0-9]{6,}")


def bundled_sface() -> Path:
    return Path(__file__).parent / "assets/sface/face_recognition_sface_2021dec.onnx"


def vector(value) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != 128
        or any(
            type(item) not in (int, float) or not math.isfinite(item) for item in value
        )
    ):
        raise ExecutionError("Embedding must contain 128 finite numbers")
    result = np.asarray(value, dtype=np.float64)
    if np.any(np.abs(result) > 1.001):
        raise ExecutionError("Embedding must have unit norm")
    if abs(float(np.linalg.norm(result)) - 1) > 0.001:
        raise ExecutionError("Embedding must have unit norm")
    return result


def embedding_summary(summary_path: Path, frames_path: Path) -> dict:
    summary = analysis_summary(summary_path)
    if summary.get("recognizer") != RECOGNIZER:
        raise ExecutionError("Incompatible recognition model or preprocessing")
    if summary["file_sha256"].get("frames") != digest_file(frames_path)[0]:
        raise ExecutionError("Embeddings do not belong to this summary")
    return summary


def embedding_rows(path: Path, summary: dict):
    video = summary["video"]
    previous = None
    frame_count = face_count = usable_count = 0
    for row in iter_rows(path):
        previous = validate_timestamp(row, frame_count, previous)
        if frame_count >= video["frame_count"]:
            raise ExecutionError("Too many embedding frames")
        faces = validate_faces(row, video["width"], video["height"])
        seen = set()
        for face in faces:
            track_id = face.get("track_id")
            if (
                not isinstance(track_id, str)
                or TRACK_PATTERN.fullmatch(track_id) is None
                or track_id in seen
            ):
                raise ExecutionError("Invalid or duplicate track ID")
            seen.add(track_id)
            if face.get("association") not in (
                "new",
                "matched",
                "reacquired",
                "ambiguous",
            ):
                raise ExecutionError("Invalid track association")
            if face.get("embedding_status") == "usable":
                vector(face.get("embedding"))
                usable_count += 1
            elif face.get("embedding_status") in (
                "small_face",
                "low_confidence",
                "alignment_failed",
            ):
                if "embedding" in face:
                    raise ExecutionError(
                        "Unusable observations must not include embeddings"
                    )
            else:
                raise ExecutionError("Invalid embedding status")
        frame_count += 1
        face_count += len(faces)
        yield row
    if (frame_count, face_count, usable_count) != (
        video["frame_count"],
        summary["counts"]["face_detection_count"],
        summary["counts"].get("usable_embedding_count"),
    ):
        raise ExecutionError("Embedding records do not match their summary")


def validate_snapshot(snapshot: dict) -> dict:
    try:
        if snapshot["schema_version"] != 1 or snapshot["recognizer"] != RECOGNIZER:
            raise ValueError
        uuid.UUID(snapshot["registry_id"])
        if type(snapshot["revision"]) is not int or snapshot["revision"] < 0:
            raise ValueError
        identities = snapshot["identities"]
        if not isinstance(identities, list):
            raise TypeError
        seen = set()
        for item in identities:
            identifier = item["identity_id"]
            uuid.UUID(identifier)
            if identifier in seen:
                raise ValueError
            seen.add(identifier)
            alias = item["alias"]
            if alias is not None and (
                not isinstance(alias, str) or not alias.strip() or len(alias) > 128
            ):
                raise ValueError
            references = item["references"]
            if not isinstance(references, list) or not 3 <= len(references) <= 32:
                raise ValueError
            for reference in references:
                vector(reference)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ExecutionError("Invalid identity registry snapshot") from exc
    return snapshot
