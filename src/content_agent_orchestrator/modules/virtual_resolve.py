"""Bind only known observations to approved, immutable synthetic targets."""

import json
import uuid
from typing import ClassVar

from ..contracts import ExecutionError, PipelineError
from ..face_data import (
    analysis_summary,
    iter_rows,
    validate_faces,
    validate_timestamp,
    write_json,
)
from ..identity_data import RECOGNIZER, TRACK_PATTERN
from ..storage import digest_file
from ..virtual_catalog import read_catalog, selected_targets


class VirtualResolution:
    name = "virtual.resolve"
    version = "1"
    input_kinds: ClassVar = {
        "frames": "identity.observations",
        "summary": "identity.resolution-summary",
        "catalog": "virtual.catalog",
    }
    output_kinds: ClassVar = {
        "frames": "virtual.observations",
        "targets": "virtual.targets",
        "summary": "virtual.resolution-summary",
    }

    def validate_parameters(self, parameters):
        if parameters:
            raise PipelineError("virtual.resolve does not accept parameters")

    def execute(self, inputs, parameters, context):
        self.validate_parameters(parameters)
        summary = analysis_summary(inputs["summary"].path)
        if (
            summary.get("recognizer") != RECOGNIZER
            or summary["file_sha256"].get("frames") != inputs["frames"].sha256
        ):
            raise ExecutionError("Identity observations do not belong to this summary")
        manifest, portraits = read_catalog(inputs["catalog"].path)
        registry = summary.get("registry", {})
        if not isinstance(registry, dict):
            raise ExecutionError("Invalid source registry metadata")
        if manifest["registry_id"] != registry.get("registry_id") or manifest[
            "source_revision"
        ] != registry.get("revision"):
            raise ExecutionError(
                "Virtual catalog and recognition must use the same source registry revision"
            )
        mapping = {item["identity_id"]: item for item in manifest["faces"]}
        frames_path = context.result_dir / "virtual_frames.jsonl"
        targets_path = context.result_dir / "targets.zip"
        selected = set()
        counts = dict.fromkeys(("ready", "missing_target", "unresolved_source"), 0)
        identity_counts = dict.fromkeys(
            ("known", "unknown", "ambiguous", "unusable"), 0
        )
        frame_count = face_count = 0
        previous = None
        video = summary["video"]
        with frames_path.open("w", encoding="utf-8") as output:
            for row in iter_rows(inputs["frames"].path):
                previous = validate_timestamp(row, frame_count, previous)
                if frame_count >= video["frame_count"]:
                    raise ExecutionError("Too many identity frames")
                faces = validate_faces(row, video["width"], video["height"])
                resolved = []
                seen = set()
                for face in faces:
                    track_id = face.get("track_id")
                    if (
                        not isinstance(track_id, str)
                        or TRACK_PATTERN.fullmatch(track_id) is None
                        or track_id in seen
                    ):
                        raise ExecutionError("Invalid virtual source track ID")
                    seen.add(track_id)
                    identity = face.get("identity")
                    if (
                        not isinstance(identity, dict)
                        or identity.get("status") not in identity_counts
                    ):
                        raise ExecutionError("Invalid identity observation status")
                    status = identity["status"]
                    association = face.get("association")
                    quality = face.get("embedding_status")
                    supported_quality = (
                        ("small_face", "low_confidence", "alignment_failed")
                        if status == "unusable"
                        else ("usable",)
                    )
                    if (
                        association
                        not in (
                            "new",
                            "matched",
                            "reacquired",
                            "ambiguous",
                        )
                        or quality not in supported_quality
                    ):
                        raise ExecutionError(
                            "Identity observation has inconsistent quality or association"
                        )
                    identity_counts[status] += 1
                    public_identity = {"status": status}
                    if status == "known":
                        identifier = identity.get("identity_id")
                        try:
                            if str(uuid.UUID(identifier)) != identifier:
                                raise ValueError
                        except (ValueError, TypeError, AttributeError) as exc:
                            raise ExecutionError(
                                "Known observation requires a valid identity ID"
                            ) from exc
                        if face.get("association") == "ambiguous":
                            raise ExecutionError(
                                "Ambiguous track association cannot be a known source"
                            )
                        public_identity["identity_id"] = identifier
                        target = mapping.get(identifier)
                        if target is None:
                            binding = {"status": "missing_target"}
                        else:
                            selected.add(identifier)
                            binding = {
                                "status": "ready",
                                "virtual_id": target["virtual_id"],
                                "portrait_file": target["portrait_file"],
                                "portrait_sha256": target["portrait_sha256"],
                            }
                    else:
                        if "identity_id" in identity:
                            raise ExecutionError(
                                "Unresolved observation must not assign a source identity"
                            )
                        binding = {"status": "unresolved_source", "reason": status}
                    counts[binding["status"]] += 1
                    resolved.append(
                        {
                            key: face[key]
                            for key in ("track_id", "bbox", "landmarks", "score")
                        }
                        | {"identity": public_identity, "virtual": binding}
                    )
                clock = {
                    key: row[key]
                    for key in (
                        "frame_index",
                        "presentation_timestamp",
                        "presentation_time",
                    )
                }
                output.write(
                    json.dumps(clock | {"faces": resolved}, allow_nan=False) + "\n"
                )
                frame_count += 1
                face_count += len(faces)
        if (
            frame_count != video["frame_count"]
            or face_count != summary["counts"]["face_detection_count"]
            or any(
                summary["counts"].get(f"{key}_observation_count") != value
                for key, value in identity_counts.items()
            )
        ):
            raise ExecutionError("Identity observations do not match their summary")
        targets_path.write_bytes(selected_targets(manifest, portraits, selected))
        summary_path = context.result_dir / "virtual_summary.json"
        write_json(
            summary_path,
            summary
            | {
                "virtual_catalog": {
                    "sha256": inputs["catalog"].sha256,
                    "revision": manifest["revision"],
                },
                "counts": summary["counts"]
                | {f"{key}_observation_count": value for key, value in counts.items()}
                | {"selected_virtual_face_count": len(selected)},
                "file_sha256": {
                    "frames": digest_file(frames_path)[0],
                    "targets": digest_file(targets_path)[0],
                },
            },
        )
        return {"frames": frames_path, "targets": targets_path, "summary": summary_path}
