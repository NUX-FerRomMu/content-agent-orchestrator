"""Final recognition report without pixels or biometric vectors."""

import shutil
from typing import ClassVar

from ..contracts import ExecutionError, PipelineError
from ..face_data import analysis_summary, write_json
from ..identity_data import RECOGNIZER


class IdentityReport:
    name = "identity.report"
    version = "1"
    input_kinds: ClassVar = {
        "summary": "identity.resolution-summary",
        "frames": "identity.observations",
        "tracks": "identity.tracks",
    }
    output_kinds: ClassVar = {
        "report": "identity.report",
        "frames": "identity.observations",
        "tracks": "identity.tracks",
    }

    def validate_parameters(self, parameters):
        if parameters:
            raise PipelineError("identity.report does not accept parameters")

    def execute(self, inputs, parameters, context):
        self.validate_parameters(parameters)
        summary = analysis_summary(inputs["summary"].path)
        if (
            summary.get("recognizer") != RECOGNIZER
            or "registry" not in summary
            or any(
                summary["file_sha256"].get(name) != inputs[name].sha256
                for name in ("frames", "tracks")
            )
        ):
            raise ExecutionError(
                "Identity report artifacts do not belong to this summary"
            )
        outputs = {}
        for name in ("frames", "tracks"):
            outputs[name] = context.result_dir / f"{name}.jsonl"
            shutil.copyfile(inputs[name].path, outputs[name])
        outputs["report"] = context.result_dir / "report.json"
        write_json(
            outputs["report"],
            summary
            | {
                "purpose": "face_identity_recognition",
                "identity_recognition_performed": True,
                "anonymization_performed": False,
                "files": {name: f"{name}.jsonl" for name in ("frames", "tracks")},
                "limitations": [
                    "Similarity is not a probability or proof of identity.",
                    "Unknown, ambiguous and unusable observations must not be assigned a person automatically.",
                    "A track may contain different identities; inspect per-frame results.",
                    "The snapshot is frozen; this report does not reflect subsequent registry changes.",
                ],
            },
        )
        return outputs
