"""Publish assignments and selected synthetic portraits with a single final node."""

import shutil
from typing import ClassVar

from ..contracts import ExecutionError, PipelineError
from ..face_data import analysis_summary, write_json
from ..identity_data import RECOGNIZER


class VirtualReport:
    name = "virtual.report"
    version = "1"
    input_kinds: ClassVar = {
        "frames": "virtual.observations",
        "targets": "virtual.targets",
        "summary": "virtual.resolution-summary",
    }
    output_kinds: ClassVar = {
        "frames": "virtual.observations",
        "targets": "virtual.targets",
        "report": "virtual.report",
    }

    def validate_parameters(self, parameters):
        if parameters:
            raise PipelineError("virtual.report does not accept parameters")

    def execute(self, inputs, parameters, context):
        self.validate_parameters(parameters)
        summary = analysis_summary(inputs["summary"].path)
        if (
            summary.get("recognizer") != RECOGNIZER
            or "virtual_catalog" not in summary
            or any(
                summary["file_sha256"].get(name) != inputs[name].sha256
                for name in ("frames", "targets")
            )
        ):
            raise ExecutionError("Virtual report inputs do not belong to this summary")
        outputs = {}
        for name, extension in (("frames", ".jsonl"), ("targets", ".zip")):
            outputs[name] = context.result_dir / f"{name}{extension}"
            shutil.copyfile(inputs[name].path, outputs[name])
        outputs["report"] = context.result_dir / "report.json"
        write_json(
            outputs["report"],
            summary
            | {
                "purpose": "virtual_face_assignment",
                "anonymization_performed": False,
                "face_swap_performed": False,
                "files": {"frames": "frames.jsonl", "targets": "targets.zip"},
                "limitations": [
                    "Only ready observations have an approved target; unresolved observations must be handled conservatively by a later swap module.",
                    "Generated provenance and visual review cannot prove that a fictional face resembles no real person.",
                    "No detection does not prove that a face is absent.",
                    "Portrait geometry and source separation do not certify age, gender, skin tone or absence of eyewear.",
                ],
            },
        )
        return outputs
