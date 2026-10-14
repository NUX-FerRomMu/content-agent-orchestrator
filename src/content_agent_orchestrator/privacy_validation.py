"""Independent pixel checks: source separation AND approved-target similarity."""

from fractions import Fraction

import cv2
import numpy as np

from .contracts import ExecutionError, PipelineError
from .face_data import YUNET_SHA256, read_json, validate_faces, validate_timestamp
from .identity_data import SFACE_SHA256, validate_snapshot, vector
from .modules.face_detect import FaceDetection
from .modules.face_embed import FaceEmbedding
from .storage import digest_file
from .virtual_catalog import read_catalog
from .virtual_face import validate_portrait

CLOCK_FIELDS = ("frame_index", "presentation_timestamp", "presentation_time")
REDACTION_COLOR = 0


def render_parameters(parameters, *, swap=False):
    defaults = {"chunk_frames": 32, "timeout_seconds": 600}
    if swap:
        defaults["device"] = "cpu"
    if parameters.keys() - defaults.keys():
        raise PipelineError("Unknown render parameter")
    values = defaults | dict(parameters)
    for key, low, high in (("chunk_frames", 1, 1024), ("timeout_seconds", 1, 86400)):
        if type(values[key]) is not int or not low <= values[key] <= high:
            raise PipelineError(f"Invalid {key}")
    if swap and values["device"] not in ("cpu", "cuda"):
        raise PipelineError("Invalid swap device")
    return values


def bound_render_summary(inputs, *, archive="frames", records="records"):
    summary = read_json(inputs["summary"].path)
    if summary.get("schema_version") != 1 or any(
        summary.get("file_sha256", {}).get(key) != inputs[key].sha256
        for key in (archive, records)
    ):
        raise ExecutionError("Render artifacts do not match their summary")
    return summary


def clock_row(iterator, clock):
    row = next(iterator, None)
    if row is None or any(row.get(key) != clock[key] for key in CLOCK_FIELDS):
        raise ExecutionError("Render records do not match source timestamps")
    return row


def checked_clock(iterator, index, timestamp, video, previous):
    row = next(iterator, None)
    if row is None:
        raise ExecutionError("Missing original frame timestamp")
    previous = validate_timestamp(row, index, previous)
    if abs(
        timestamp - row["presentation_timestamp"] * Fraction(video["time_base"])
    ) > Fraction(1, 1000000):
        raise ExecutionError("Lossless render changed a frame timestamp")
    return row, previous


def no_extra(*iterators):
    if any(next(iterator, None) is not None for iterator in iterators):
        raise ExecutionError("Unexpected extra render records")


def overlap(left, right):
    x, y = max(left[0], right[0]), max(left[1], right[1])
    w = max(0, min(left[0] + left[2], right[0] + right[2]) - x)
    h = max(0, min(left[1] + left[3], right[1] + right[3]) - y)
    intersection = w * h
    return intersection / max(1, left[2] * left[3] + right[2] * right[3] - intersection)


class PrivacyValidator:
    def __init__(self, video, targets, registry, detector_model, recognizer_model):
        cv2.setNumThreads(4)
        for path, digest in (
            (detector_model, YUNET_SHA256),
            (recognizer_model, SFACE_SHA256),
        ):
            if digest_file(path)[0] != digest:
                raise ExecutionError("Privacy validator model checksum mismatch")
        self.video = video
        factor = min(1, 1280 / max(video["width"], video["height"]))
        self.width = max(32, round(video["width"] * factor / 32) * 32)
        self.height = max(32, round(video["height"] * factor / 32) * 32)
        size = (
            (self.height, self.width)
            if video["rotation_degrees"] in (90, 270)
            else (self.width, self.height)
        )
        self.detector = cv2.FaceDetectorYN_create(
            str(detector_model), "", size, 0.7, 0.3, 500
        )
        self.recognizer = cv2.FaceRecognizerSF_create(str(recognizer_model), "")
        frozen = validate_snapshot(read_json(registry))
        manifest, portraits = read_catalog(targets)
        if (
            manifest["registry_id"] != frozen["registry_id"]
            or manifest["source_revision"] != frozen["revision"]
        ):
            raise ExecutionError("Privacy references must use the same frozen registry")
        self.references = [
            vector(ref)
            for person in frozen["identities"]
            for ref in person["references"]
        ]
        self.targets = {}
        for item in manifest["faces"]:
            embedding = vector(
                validate_portrait(portraits[item["portrait_file"]])["embedding"]
            )
            if (
                self.references
                and max(float(embedding @ ref) for ref in self.references) >= 0.3
            ):
                raise ExecutionError(
                    "Approved target is too similar to a frozen source reference"
                )
            self.targets[item["virtual_id"]] = (item, embedding)

    def check(self, image, virtual, embedded, swap_record):
        expected = validate_faces(virtual, self.video["width"], self.video["height"])
        originals = validate_faces(embedded, self.video["width"], self.video["height"])
        if len(expected) != len(originals) or any(
            a.get("track_id") != b.get("track_id") or a["bbox"] != b["bbox"]
            for a, b in zip(expected, originals, strict=True)
        ):
            raise ExecutionError(
                "Source face observations do not match virtual assignments"
            )
        if not expected:
            return {"status": "redact", "reason": "no_source_face", "faces": []}
        if any(
            face.get("virtual", {}).get("status") != "ready"
            or original.get("embedding_status") != "usable"
            for face, original in zip(expected, originals, strict=True)
        ):
            return {"status": "redact", "reason": "unresolved_source", "faces": []}
        identifiers = [face["track_id"] for face in expected]
        if swap_record.get("swapped") != identifiers or swap_record.get("failed"):
            return {"status": "redact", "reason": "swap_failed", "faces": []}
        try:
            working = cv2.resize(image, (self.width, self.height))
            try:
                detected = FaceDetection._detect_faces(
                    self.detector,
                    working,
                    self.video["width"],
                    self.video["height"],
                    self.video["rotation_degrees"],
                )
            except ExecutionError:
                return {
                    "status": "redact",
                    "reason": "output_detection_failed",
                    "faces": [],
                }
            if len(detected) != len(expected):
                return {
                    "status": "redact",
                    "reason": "face_coverage_mismatch",
                    "faces": [],
                }
            references = self.references + [
                vector(face["embedding"]) for face in originals
            ]
            used, scores = set(), []
            for source in expected:
                candidates = [
                    i
                    for i, face in enumerate(detected)
                    if overlap(source["bbox"], face["bbox"]) >= 0.2
                ]
                if len(candidates) != 1 or candidates[0] in used:
                    return {
                        "status": "redact",
                        "reason": "ambiguous_face_location",
                        "faces": [],
                    }
                index = candidates[0]
                used.add(index)
                face = detected[index]
                if face["score"] < 0.9 or min(face["bbox"][2:]) < 64:
                    return {
                        "status": "redact",
                        "reason": "low_output_quality",
                        "faces": [],
                    }
                item, target = self.targets.get(
                    source["virtual"]["virtual_id"], (None, None)
                )
                if (
                    item is None
                    or any(
                        source["virtual"].get(key) != item[key]
                        for key in ("portrait_file", "portrait_sha256")
                    )
                    or source["identity"].get("identity_id") != item["identity_id"]
                ):
                    raise ExecutionError(
                        "Swap target does not match the approved catalog"
                    )
                try:
                    embedding = vector(
                        FaceEmbedding._embedding(self.recognizer, image, face)
                    )
                except ExecutionError:
                    return {
                        "status": "redact",
                        "reason": "invalid_output_embedding",
                        "faces": [],
                    }
                source_score = max(float(embedding @ ref) for ref in references)
                target_score = float(embedding @ target)
                metric = {
                    "track_id": source["track_id"],
                    "source_similarity": source_score,
                    "target_similarity": target_score,
                }
                if (
                    source_score >= 0.3
                    or target_score < 0.45
                    or target_score - source_score < 0.15
                ):
                    return {
                        "status": "redact",
                        "reason": "identity_check_failed",
                        "faces": [metric],
                    }
                scores.append(metric)
            return {
                "status": "validated",
                "reason": "source_separated_target_matched",
                "faces": scores,
            }
        except (cv2.error, np.linalg.LinAlgError):
            return {"status": "redact", "reason": "validation_failed", "faces": []}
