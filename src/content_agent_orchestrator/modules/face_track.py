"""Conservative geometric tracks scoped to one clip, not person identities."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

from ..contracts import Artifact, ExecutionError, ModuleContext, PipelineError
from ..face_data import (
    analysis_summary,
    iter_rows,
    validate_faces,
    validate_timestamp,
    write_json,
)
from ..storage import digest_file


def intersection_over_union(left: list[float], right: list[float]) -> float:
    ax, ay, aw, ah = left
    bx, by, bw, bh = right
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0


@dataclass
class Track:
    id: str
    bbox: list[float]
    first: dict
    last: dict
    observations: int = 1
    gap_events: int = 0
    missing_frames: int = 0

    def observe(self, bbox: list[float], clock: dict) -> int:
        gap = clock["frame_index"] - self.last["frame_index"] - 1
        if gap:
            self.gap_events += 1
            self.missing_frames += gap
        self.bbox, self.last = bbox, clock
        self.observations += 1
        return gap

    def summary(self, reason: str) -> dict:
        return {
            "track_id": self.id,
            "first_frame": self.first["frame_index"],
            "last_frame": self.last["frame_index"],
            "first_timestamp": self.first["presentation_timestamp"],
            "last_timestamp": self.last["presentation_timestamp"],
            "observations": self.observations,
            "gap_events": self.gap_events,
            "missing_frames": self.missing_frames,
            "end_reason": reason,
        }


def associations(
    active: dict[str, Track], faces: list[dict], min_iou: float, margin: float
) -> tuple[dict[int, str], set[int], set[str]]:
    by_face = {}
    by_track = {track_id: [] for track_id in active}
    for index, face in enumerate(faces):
        candidates = []
        for track_id, track in active.items():
            score = intersection_over_union(track.bbox, face["bbox"])
            if score >= min_iou:
                candidates.append((score, track_id))
                by_track[track_id].append((score, index))
        by_face[index] = sorted(candidates, reverse=True)
    ambiguous = {
        index
        for index, candidates in by_face.items()
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] <= margin
    }
    for candidates in by_track.values():
        candidates.sort(reverse=True)
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] <= margin:
            ambiguous.update(index for _, index in candidates)
    # Retire the whole contested component instead of silently choosing a person.
    retired = set()
    while True:
        affected = {track_id for index in ambiguous for _, track_id in by_face[index]}
        expanded = {
            index
            for index, candidates in by_face.items()
            if any(track_id in affected for _, track_id in candidates)
        }
        if expanded == ambiguous and affected == retired:
            break
        ambiguous, retired = expanded, affected
    matches = {}
    for index, candidates in by_face.items():
        if index in ambiguous or not candidates:
            continue
        _, track_id = candidates[0]
        if track_id not in retired and by_track[track_id][0][1] == index:
            matches[index] = track_id
    return matches, ambiguous, retired


class FaceTracking:
    name = "face.track"
    version = "1"
    input_kinds: ClassVar = {
        "frames": "face.detections",
        "summary": "face.detection-summary",
    }
    output_kinds: ClassVar = {
        "frames": "face.track-frames",
        "tracks": "face.tracks",
        "intervals": "face.intervals",
        "summary": "face.tracking-summary",
    }
    defaults: ClassVar = {
        "min_iou": 0.3,
        "ambiguity_margin": 0.05,
        "max_gap_frames": 3,
        "max_gap_seconds": 0.5,
    }

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        if parameters.keys() - self.defaults.keys():
            raise PipelineError("Unknown face.track parameter")
        values = self.defaults | dict(parameters)
        if (
            type(values["max_gap_frames"]) is not int
            or not 0 <= values["max_gap_frames"] <= 300
        ):
            raise PipelineError("max_gap_frames must be an integer between 0 and 300")
        for name, lower, upper in (
            ("min_iou", 0, 1),
            ("ambiguity_margin", 0, 1),
            ("max_gap_seconds", 0, 30),
        ):
            value = values[name]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not lower <= value <= upper
                or (name == "min_iou" and value == 0)
            ):
                raise PipelineError(f"Invalid {name}")

    def execute(
        self,
        inputs: Mapping[str, Artifact],
        parameters: Mapping[str, Any],
        context: ModuleContext,
    ) -> Mapping[str, Path]:
        self.validate_parameters(parameters)
        values = self.defaults | dict(parameters)
        detection = analysis_summary(inputs["summary"].path)
        if detection.get("file_sha256", {}).get("frames") != inputs["frames"].sha256:
            raise ExecutionError("Detection frames do not belong to this summary")
        video = detection["video"]
        time_base = Fraction(video["time_base"])
        max_gap_seconds = Fraction(str(values["max_gap_seconds"]))
        frames_path = context.result_dir / "track_frames.jsonl"
        tracks_path = context.result_dir / "tracks.jsonl"
        intervals_path = context.result_dir / "intervals.jsonl"
        active: dict[str, Track] = {}
        open_intervals = {}
        frame_count = face_count = empty_count = ambiguous_count = maximum = (
            gap_events
        ) = 0
        next_track = 1
        previous = None
        with (
            frames_path.open("w", encoding="utf-8") as frames_out,
            tracks_path.open("w", encoding="utf-8") as tracks_out,
            intervals_path.open("w", encoding="utf-8") as intervals_out,
        ):

            def retire(track_id, reason):
                track = active.pop(track_id)
                tracks_out.write(
                    json.dumps(track.summary(reason), allow_nan=False) + "\n"
                )

            for row in iter_rows(inputs["frames"].path):
                previous = validate_timestamp(row, frame_count, previous)
                faces = validate_faces(row, video["width"], video["height"])
                if frame_count >= video["frame_count"]:
                    raise ExecutionError("Too many detection frames")
                clock = {
                    key: row[key]
                    for key in (
                        "frame_index",
                        "presentation_timestamp",
                        "presentation_time",
                    )
                }
                for track_id, track in list(active.items()):
                    gap = frame_count - track.last["frame_index"] - 1
                    elapsed = (
                        previous - track.last["presentation_timestamp"]
                    ) * time_base
                    if gap > values["max_gap_frames"] or (
                        gap > 0 and elapsed > max_gap_seconds
                    ):
                        retire(track_id, "gap_expired")
                matches, ambiguous, retired = associations(
                    active, faces, values["min_iou"], values["ambiguity_margin"]
                )
                for track_id in sorted(retired):
                    retire(track_id, "ambiguous")
                tracked = []
                for index, face in enumerate(faces):
                    gap = 0
                    if index in matches:
                        track = active[matches[index]]
                        gap = track.observe(face["bbox"], clock)
                        association = "reacquired" if gap else "matched"
                        gap_events += bool(gap)
                    else:
                        track = Track(
                            f"track-{next_track:06d}", face["bbox"], clock, clock
                        )
                        next_track += 1
                        active[track.id] = track
                        association = "ambiguous" if index in ambiguous else "new"
                    tracked.append(
                        face
                        | {
                            "track_id": track.id,
                            "association": association,
                            "gap_frames": gap,
                        }
                    )
                frames_out.write(
                    json.dumps(clock | {"faces": tracked}, allow_nan=False) + "\n"
                )
                signals = {
                    "no_faces": not faces,
                    "ambiguous_association": bool(ambiguous),
                }
                for kind, present in signals.items():
                    if present:
                        interval = open_intervals.setdefault(
                            kind,
                            {
                                "kind": kind,
                                "start_frame": frame_count,
                                "start_timestamp": previous,
                                "start_time": clock["presentation_time"],
                            },
                        )
                        interval.update(
                            {
                                "end_frame": frame_count,
                                "end_timestamp": previous,
                                "end_time": clock["presentation_time"],
                            }
                        )
                    elif kind in open_intervals:
                        intervals_out.write(
                            json.dumps(open_intervals.pop(kind), allow_nan=False) + "\n"
                        )
                frame_count += 1
                face_count += len(faces)
                empty_count += not faces
                ambiguous_count += bool(ambiguous)
                maximum = max(maximum, len(faces))
            if frame_count != video["frame_count"] or face_count != detection.get(
                "counts", {}
            ).get("face_detection_count"):
                raise ExecutionError("Detection records do not match their summary")
            for track_id in sorted(active):
                retire(track_id, "end_of_clip")
            for interval in open_intervals.values():
                intervals_out.write(json.dumps(interval, allow_nan=False) + "\n")
        summary_path = context.result_dir / "summary.json"
        write_json(
            summary_path,
            {
                "schema_version": 1,
                "source": detection["source"],
                "video": video,
                "detector": detection.get("detector"),
                "tracking": values,
                "identity_scope": "clip_trajectory_only",
                "file_sha256": {
                    name: digest_file(path)[0]
                    for name, path in (
                        ("frames", frames_path),
                        ("tracks", tracks_path),
                        ("intervals", intervals_path),
                    )
                },
                "counts": {
                    "frame_count": frame_count,
                    "face_detection_count": face_count,
                    "track_count": next_track - 1,
                    "frames_without_faces": empty_count,
                    "ambiguous_frames": ambiguous_count,
                    "max_simultaneous_faces": maximum,
                    "gap_events": gap_events,
                },
            },
        )
        return {
            "frames": frames_path,
            "tracks": tracks_path,
            "intervals": intervals_path,
            "summary": summary_path,
        }
