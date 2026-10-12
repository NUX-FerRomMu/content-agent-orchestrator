"""Built-in module registry; execution accepts custom registries as well."""

from ..contracts import Module
from .face_detect import FaceDetection
from .face_report import FaceReport
from .face_track import FaceTracking
from .report import ReportExport
from .video import VideoInspection

__all__ = [
    "FaceDetection",
    "FaceReport",
    "FaceTracking",
    "ReportExport",
    "VideoInspection",
    "builtin_registry",
]


def builtin_registry() -> dict[str, Module]:
    modules = (
        VideoInspection(),
        ReportExport(),
        FaceDetection(),
        FaceTracking(),
        FaceReport(),
    )
    return {module.name: module for module in modules}
