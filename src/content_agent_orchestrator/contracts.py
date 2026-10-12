"""Public contracts for composable, file-producing modules."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class PipelineError(ValueError):
    """Invalid configuration or an unsafe storage operation."""


class ExecutionError(RuntimeError):
    """An execution failed without publishing its final result."""


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModuleContext:
    """Only files inside result_dir become artifacts; work_dir is disposable.

    Modules must leave inputs unchanged and derive results from input contents,
    parameters and their version. Bump the version when output behavior changes.
    """

    work_dir: Path
    result_dir: Path


class Module(Protocol):
    name: str
    version: str
    input_kinds: Mapping[str, str]
    output_kinds: Mapping[str, str]

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None: ...

    def execute(
        self,
        inputs: Mapping[str, Artifact],
        parameters: Mapping[str, Any],
        context: ModuleContext,
    ) -> Mapping[str, Path]: ...
