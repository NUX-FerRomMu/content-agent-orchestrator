"""Final module for publishing an inspection report."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from ..contracts import Artifact, ExecutionError, ModuleContext, PipelineError


class ReportExport:
    name = "report.export"
    version = "1"
    input_kinds: ClassVar = {"inspection": "video.manifest"}
    output_kinds: ClassVar = {"report": "video.report"}

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        if parameters:
            raise PipelineError("report.export does not accept parameters")

    def execute(
        self,
        inputs: Mapping[str, Artifact],
        parameters: Mapping[str, Any],
        context: ModuleContext,
    ) -> Mapping[str, Path]:
        manifest = json.loads(inputs["inspection"].path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("timing"), dict)
            or type(manifest["timing"].get("decoded_frame_count")) is not int
            or manifest["timing"]["decoded_frame_count"] < 1
        ):
            raise ExecutionError("Cannot export an invalid inspection manifest")
        target = context.result_dir / "report.json"
        target.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return {"report": target}
