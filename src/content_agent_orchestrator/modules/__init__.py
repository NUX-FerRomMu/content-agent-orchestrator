"""Built-in module registry; execution accepts custom registries as well."""

from ..contracts import Module
from .face_detect import FaceDetection
from .face_embed import FaceEmbedding
from .face_redact import FaceRedaction
from .face_report import FaceReport
from .face_swap import FaceSwap
from .face_track import FaceTracking
from .face_validate import FaceValidation
from .identity_report import IdentityReport
from .identity_resolve import IdentityResolution
from .report import ReportExport
from .video import VideoInspection
from .video_export import VideoExport
from .virtual_report import VirtualReport
from .virtual_resolve import VirtualResolution

__all__ = [
    "FaceDetection",
    "FaceEmbedding",
    "FaceRedaction",
    "FaceReport",
    "FaceSwap",
    "FaceTracking",
    "FaceValidation",
    "IdentityReport",
    "IdentityResolution",
    "ReportExport",
    "VideoExport",
    "VideoInspection",
    "VirtualReport",
    "VirtualResolution",
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
        VirtualResolution(),
        VirtualReport(),
        FaceSwap(),
        FaceValidation(),
        FaceRedaction(),
        VideoExport(),
    )
    return {module.name: module for module in modules}
