"""Redetect every rendered frame and audit original/target identity similarity."""

from typing import ClassVar

import cv2

from ..contracts import ExecutionError
from ..face_data import iter_rows, write_json
from ..privacy_validation import (
    PrivacyValidator,
    bound_render_summary,
    checked_clock,
    clock_row,
    no_extra,
)
from ..render_frames import archived_frames, runtime_version
from ..storage import digest_file


class FaceValidation:
    name = "face.validate"
    version = f"1-opencv-{cv2.__version__}-av-{runtime_version('av')}"
    input_kinds: ClassVar = {
        "frames": "video.unvalidated-frames",
        "records": "face.swap-records",
        "summary": "face.swap-summary",
        "timestamps": "video.timestamps",
        "assignments": "virtual.observations",
        "embeddings": "face.embeddings",
        "targets": "virtual.targets",
        "registry": "identity.registry-snapshot",
        "detector": "face.detector.yunet",
        "recognizer": "face.recognizer.sface",
    }
    output_kinds: ClassVar = {
        "audit": "face.privacy-audit",
        "summary": "face.validation-summary",
    }

    def validate_parameters(self, parameters):
        if parameters:
            from ..contracts import PipelineError

            raise PipelineError("face.validate does not accept parameters")

    def execute(self, inputs, parameters, context):
        self.validate_parameters(parameters)
        summary = bound_render_summary(inputs)
        if (
            summary.get("assignments_sha256") != inputs["assignments"].sha256
            or summary.get("targets_sha256") != inputs["targets"].sha256
        ):
            raise ExecutionError("Validation inputs do not match the swap")
        video = summary["video"]
        validator = PrivacyValidator(
            video,
            inputs["targets"].path,
            inputs["registry"].path,
            inputs["detector"].path,
            inputs["recognizer"].path,
        )
        clocks, virtual, embedded, swapped = (
            iter(iter_rows(inputs[key].path))
            for key in ("timestamps", "assignments", "embeddings", "records")
        )
        previous = None
        counts = {"validated": 0, "redact": 0}
        audit = context.result_dir / "audit.jsonl"
        audit.touch(mode=0o600)
        with audit.open("w", encoding="utf-8") as output:
            import json

            for index, time, image in archived_frames(
                inputs["frames"].path, context.work_dir, video
            ):
                clock, previous = checked_clock(clocks, index, time, video, previous)
                decision = validator.check(
                    image,
                    clock_row(virtual, clock),
                    clock_row(embedded, clock),
                    clock_row(swapped, clock),
                )
                counts[decision["status"]] += 1
                output.write(json.dumps(clock | decision, allow_nan=False) + "\n")
        no_extra(clocks, virtual, embedded, swapped)
        result = context.result_dir / "summary.json"
        write_json(
            result,
            {
                "schema_version": 1,
                "source": summary["source"],
                "video": video,
                "counts": counts,
                "frames_sha256": inputs["frames"].sha256,
                "records_sha256": inputs["records"].sha256,
                "file_sha256": {"audit": digest_file(audit)[0]},
            },
        )
        return {"audit": audit, "summary": result}
