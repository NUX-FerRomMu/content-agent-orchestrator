"""Replace every unvalidated frame with opaque pixels, retaining timestamps."""

from typing import ClassVar

import numpy as np

from ..contracts import ExecutionError
from ..face_data import iter_rows, read_json, write_json
from ..privacy_validation import (
    REDACTION_COLOR,
    bound_render_summary,
    checked_clock,
    clock_row,
    no_extra,
    render_parameters,
)
from ..render_frames import FrameChunks, archived_frames, runtime_version, signature
from ..storage import digest_file


class FaceRedaction:
    name = "face.redact"
    version = f"1-av-{runtime_version('av')}"
    input_kinds: ClassVar = {
        "frames": "video.unvalidated-frames",
        "records": "face.swap-records",
        "summary": "face.swap-summary",
        "audit": "face.privacy-audit",
        "validation": "face.validation-summary",
        "timestamps": "video.timestamps",
    }
    output_kinds: ClassVar = {
        "frames": "video.redacted-frames",
        "records": "face.redaction-records",
        "summary": "face.redaction-summary",
    }

    def validate_parameters(self, parameters):
        render_parameters(parameters)

    def execute(self, inputs, parameters, context):
        values = render_parameters(parameters)
        original = bound_render_summary(inputs)
        validation = read_json(inputs["validation"].path)
        if (
            validation.get("frames_sha256") != inputs["frames"].sha256
            or validation.get("records_sha256") != inputs["records"].sha256
            or validation.get("file_sha256", {}).get("audit") != inputs["audit"].sha256
            or validation.get("source") != original["source"]
            or validation.get("video") != original["video"]
        ):
            raise ExecutionError("Redaction audit does not belong to these pixels")
        video = original["video"]
        clocks, audits = (
            iter(iter_rows(inputs["timestamps"].path)),
            iter(iter_rows(inputs["audit"].path)),
        )
        previous = None
        counts = {"validated": 0, "redacted": 0}
        cache = FrameChunks(
            context, video, signature(self, inputs, values), values["chunk_frames"]
        )
        try:
            for index, time, image in archived_frames(
                inputs["frames"].path, context.work_dir, video
            ):
                clock, previous = checked_clock(clocks, index, time, video, previous)
                audit = clock_row(audits, clock)
                if audit.get("status") not in ("validated", "redact"):
                    raise ExecutionError("Unknown privacy audit decision")
                status = "redacted" if audit["status"] == "redact" else "validated"
                counts[status] += 1
                if index < cache.processed:
                    continue
                result = (
                    np.full_like(image, REDACTION_COLOR)
                    if status == "redacted"
                    else image
                )
                cache.append(
                    clock, result, {"status": status, "reason": audit["reason"]}
                )
            no_extra(clocks, audits)
            archive, records = cache.finish()
        finally:
            cache.close()
        if counts != {
            "validated": validation["counts"]["validated"],
            "redacted": validation["counts"]["redact"],
        }:
            raise ExecutionError("Audit coverage does not match the validation summary")
        summary = context.result_dir / "summary.json"
        write_json(
            summary,
            {
                "schema_version": 1,
                "source": original["source"],
                "video": video,
                "counts": counts,
                "engine": original["engine"],
                "policy": "opaque-entire-failed-frame-v1",
                "file_sha256": {
                    "frames": digest_file(archive)[0],
                    "records": digest_file(records)[0],
                },
            },
        )
        return {"frames": archive, "records": records, "summary": summary}
