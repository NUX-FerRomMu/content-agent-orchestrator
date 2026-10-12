"""Resolve individual observations against an immutable biometric snapshot."""

import json
import math
from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from ..contracts import PipelineError
from ..face_data import read_json, write_json
from ..identity_data import embedding_rows, embedding_summary, validate_snapshot, vector
from ..storage import digest_file


class IdentityResolution:
    name = "identity.resolve"
    version = "1"
    input_kinds: ClassVar = {
        "frames": "face.embeddings",
        "summary": "face.embedding-summary",
        "registry": "identity.registry-snapshot",
    }
    output_kinds: ClassVar = {
        "frames": "identity.observations",
        "tracks": "identity.tracks",
        "summary": "identity.resolution-summary",
    }
    defaults: ClassVar = {"min_similarity": 0.6, "ambiguity_margin": 0.12}

    def validate_parameters(self, parameters: Mapping) -> None:
        if parameters.keys() - self.defaults.keys():
            raise PipelineError("Unknown identity.resolve parameter")
        values = self.defaults | dict(parameters)
        for name, lower, upper in (
            ("min_similarity", 0.0, 1.0),
            ("ambiguity_margin", 0.0, 1.0),
        ):
            value = values[name]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not lower < value <= upper
            ):
                raise PipelineError(f"Invalid {name}")

    @staticmethod
    def _resolve(face, identities, values):
        if face["embedding_status"] != "usable":
            return {"status": "unusable", "reason": face["embedding_status"]}
        features = vector(face["embedding"])
        scores = sorted(
            (
                (float(np.clip(np.max(references @ features), -1, 1)), item)
                for item, references in identities
            ),
            key=lambda pair: (-pair[0], pair[1]["identity_id"]),
        )
        candidates = [
            {"identity_id": item["identity_id"], "score": round(score, 6)}
            for score, item in scores[:2]
        ]
        result = {"candidates": candidates}
        if face["association"] == "ambiguous":
            return result | {"status": "ambiguous", "reason": "track_association"}
        if not scores or scores[0][0] < values["min_similarity"]:
            return result | {"status": "unknown", "reason": "no_matching_reference"}
        if len(scores) > 1 and scores[0][0] - scores[1][0] < values["ambiguity_margin"]:
            return result | {"status": "ambiguous", "reason": "similar_identity_scores"}
        identity = scores[0][1]
        return result | {
            "status": "known",
            "identity_id": identity["identity_id"],
            "alias": identity["alias"],
        }

    def execute(self, inputs, parameters, context):
        self.validate_parameters(parameters)
        values = self.defaults | dict(parameters)
        summary = embedding_summary(inputs["summary"].path, inputs["frames"].path)
        snapshot = validate_snapshot(read_json(inputs["registry"].path))
        identities = [
            (item, np.stack([vector(reference) for reference in item["references"]]))
            for item in snapshot["identities"]
        ]
        frames_path = context.result_dir / "identity_frames.jsonl"
        tracks_path = context.result_dir / "identity_tracks.jsonl"
        tracks = {}
        totals = dict.fromkeys(("known", "unknown", "ambiguous", "unusable"), 0)
        with frames_path.open("w", encoding="utf-8") as output:
            for row in embedding_rows(inputs["frames"].path, summary):
                faces = []
                for face in row["faces"]:
                    resolution = self._resolve(face, identities, values)
                    status = resolution["status"]
                    totals[status] += 1
                    # Use an allowlist: neither a vector nor unexpected input fields reach outputs.
                    observation = {
                        key: face[key]
                        for key in (
                            "track_id",
                            "bbox",
                            "landmarks",
                            "score",
                            "association",
                            "embedding_status",
                        )
                    }
                    faces.append(observation | {"identity": resolution})
                    track = tracks.setdefault(
                        face["track_id"],
                        {
                            "track_id": face["track_id"],
                            "first_frame": row["frame_index"],
                            "counts": dict.fromkeys(totals, 0),
                            "identities": set(),
                            "identity_changes": [],
                            "last_identity": None,
                        },
                    )
                    track["last_frame"] = row["frame_index"]
                    track["counts"][status] += 1
                    if status == "known":
                        identity_id = resolution["identity_id"]
                        track["identities"].add(identity_id)
                        if (
                            track["last_identity"] is not None
                            and track["last_identity"] != identity_id
                        ):
                            track["identity_changes"].append(
                                {
                                    "frame_index": row["frame_index"],
                                    "presentation_timestamp": row[
                                        "presentation_timestamp"
                                    ],
                                    "from": track["last_identity"],
                                    "to": identity_id,
                                }
                            )
                        track["last_identity"] = identity_id
                clock = {
                    key: row[key]
                    for key in (
                        "frame_index",
                        "presentation_timestamp",
                        "presentation_time",
                    )
                }
                output.write(
                    json.dumps(clock | {"faces": faces}, allow_nan=False) + "\n"
                )
        mixed_count = 0
        with tracks_path.open("w", encoding="utf-8") as output:
            for track in tracks.values():
                track.pop("last_identity")
                track["identity_ids"] = sorted(track.pop("identities"))
                counts = track["counts"]
                if len(track["identity_ids"]) > 1:
                    track["status"] = "mixed"
                    mixed_count += 1
                elif counts["known"] and not any(
                    counts[key] for key in ("unknown", "ambiguous", "unusable")
                ):
                    track["status"] = "known"
                elif counts["known"] or counts["ambiguous"] or counts["unusable"]:
                    track["status"] = "uncertain"
                else:
                    track["status"] = "unknown"
                output.write(json.dumps(track, allow_nan=False) + "\n")
        summary_path = context.result_dir / "identity_summary.json"
        write_json(
            summary_path,
            {
                "schema_version": 1,
                "source": summary["source"],
                "video": summary["video"],
                "recognizer": summary["recognizer"],
                "resolution": values,
                "registry": {key: snapshot[key] for key in ("registry_id", "revision")}
                | {
                    "sha256": inputs["registry"].sha256,
                    "identity_count": len(identities),
                },
                "counts": summary["counts"]
                | {f"{key}_observation_count": value for key, value in totals.items()}
                | {"track_count": len(tracks), "mixed_track_count": mixed_count},
                "file_sha256": {
                    "frames": digest_file(frames_path)[0],
                    "tracks": digest_file(tracks_path)[0],
                },
            },
        )
        return {"frames": frames_path, "tracks": tracks_path, "summary": summary_path}
