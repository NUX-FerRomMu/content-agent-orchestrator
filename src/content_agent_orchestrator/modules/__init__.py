"""Built-in module registry; execution accepts custom registries as well."""

from ..contracts import Module
from .face_detect import FaceDetection
from .face_embed import FaceEmbedding
from .face_report import FaceReport
from .face_track import FaceTracking
from .identity_report import IdentityReport
from .identity_resolve import IdentityResolution
from .report import ReportExport
from .video import VideoInspection

__all__ = [
    "FaceDetection",
    "FaceEmbedding",
    "FaceReport",
    "FaceTracking",
    "IdentityReport",
    "IdentityResolution",
    "ReportExport",
    "VideoInspection",
    "builtin_registry",
]


def builtin_registry() -> dict[str, Module]:
    modules = (
        VideoInspection(),
        ReportExport(),
        FaceDetection(),
        FaceEmbedding(),
        FaceTracking(),
        FaceReport(),
        IdentityResolution(),
        IdentityReport(),
    )
    return {module.name: module for module in modules}
