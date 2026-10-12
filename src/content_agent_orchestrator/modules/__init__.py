"""Built-in module registry; execution accepts custom registries as well."""

from ..contracts import Module
from .report import ReportExport
from .video import VideoInspection

__all__ = ["ReportExport", "VideoInspection", "builtin_registry"]


def builtin_registry() -> dict[str, Module]:
    modules = (VideoInspection(), ReportExport())
    return {module.name: module for module in modules}
