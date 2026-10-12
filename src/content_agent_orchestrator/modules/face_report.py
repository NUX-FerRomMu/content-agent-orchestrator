"""Publish metadata analysis, explicitly without an anonymization claim."""

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from ..contracts import Artifact, ExecutionError, ModuleContext, PipelineError
from ..face_data import analysis_summary, write_json


class FaceReport:
    name = "face.report"
    version = "1"
    input_kinds: ClassVar = {
        "summary": "face.tracking-summary",
        "frames": "face.track-frames",
        "tracks": "face.tracks",
        "intervals": "face.intervals",
    }
    output_kinds: ClassVar = {
        "report": "face.analysis-report",
        "frames": "face.track-frames",
        "tracks": "face.tracks",
        "intervals": "face.intervals",
    }

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise PipelineError("face.report does not accept parameters")

    def execute(
        self,
        inputs: Mapping[str, Artifact],
        parameters: Mapping[str, Any],
        context: ModuleContext,
    ) -> Mapping[str, Path]:
        self.validate_parameters(parameters)
        summary = analysis_summary(inputs["summary"].path)
        if summary.get("identity_scope") != "clip_trajectory_only":
            raise ExecutionError("Report needs a clip-scoped tracking summary")
        if any(
            summary.get("file_sha256", {}).get(name) != inputs[name].sha256
            for name in ("frames", "tracks", "intervals")
        ):
            raise ExecutionError(
                "Report artifacts do not belong to this tracking summary"
            )
        outputs = {}
        for name in ("frames", "tracks", "intervals"):
            destination = context.result_dir / f"{name}.jsonl"
            shutil.copyfile(inputs[name].path, destination)
            outputs[name] = destination
        report = context.result_dir / "report.json"
        write_json(
            report,
            summary
            | {
                "purpose": "face_detection_and_tracking",
                "anonymization_performed": False,
                "identity_recognition_performed": False,
                "files": {name: f"{name}.jsonl" for name in outputs},
                "limitations": [
                    "No detection does not prove that a face is absent.",
                    "Track IDs describe geometric continuity within this clip, not person identities.",
                ],
            },
        )
        outputs["report"] = report
        return outputs
