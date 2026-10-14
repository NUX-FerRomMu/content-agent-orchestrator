"""Apply fixed target portraits; intermediate pixels are NOT safe to publish."""

from typing import ClassVar

import cv2

from ..contracts import ExecutionError
from ..face_data import analysis_summary, iter_rows, validate_faces, write_json
from ..privacy_validation import clock_row, no_extra, render_parameters
from ..render_frames import FrameChunks, check_image, runtime_version, signature
from ..storage import digest_file
from ..swap_engine import InSwapper
from ..video_frames import decoded_frames
from ..virtual_catalog import read_catalog
from .face_detect import FaceDetection


class FaceSwap:
    name = "face.swap"
    can_be_final = False
    version = f"1-opencv-{cv2.__version__}-av-{runtime_version('av')}-ort-{runtime_version('onnxruntime')}-{runtime_version('onnxruntime-gpu')}"
    input_kinds: ClassVar = {
        "video": "video.source",
        "inspection": "video.manifest",
        "timestamps": "video.timestamps",
        "assignments": "virtual.observations",
        "summary": "virtual.resolution-summary",
        "targets": "virtual.targets",
        "swap_model": "face.swapper.inswapper",
        "arcface_model": "face.recognizer.arcface",
    }
    output_kinds: ClassVar = {
        "frames": "video.unvalidated-frames",
        "records": "face.swap-records",
        "summary": "face.swap-summary",
    }

    def validate_parameters(self, parameters):
        render_parameters(parameters, swap=True)

    def execute(self, inputs, parameters, context):
        values = render_parameters(parameters, swap=True)
        _, video = FaceDetection._inspection(inputs)
        source = {"sha256": inputs["video"].sha256, "size_bytes": inputs["video"].size}
        assignments = analysis_summary(inputs["summary"].path)
        if (
            assignments["video"] != video
            or assignments["source"] != source
            or assignments["file_sha256"].get("frames") != inputs["assignments"].sha256
            or assignments["file_sha256"].get("targets") != inputs["targets"].sha256
        ):
            raise ExecutionError("Virtual assignments do not belong to this video")
        manifest, portraits = read_catalog(inputs["targets"].path)
        targets = {item["virtual_id"]: item for item in manifest["faces"]}
        engine = InSwapper(
            inputs["swap_model"].path,
            inputs["arcface_model"].path,
            device=values["device"],
        )
        for item in manifest["faces"]:
            engine.prepare_target(item["virtual_id"], portraits[item["portrait_file"]])
        cache = FrameChunks(
            context, video, signature(self, inputs, values), values["chunk_frames"]
        )
        rows = iter(iter_rows(inputs["assignments"].path))
        try:
            with decoded_frames(
                inputs["video"].path,
                inputs["timestamps"].path,
                stream_index=video["stream_index"],
                width=video["width"],
                height=video["height"],
                expected_count=video["frame_count"],
                work_dir=context.work_dir,
                timeout=values["timeout_seconds"],
            ) as frames:
                for clock, image in frames:
                    row = clock_row(rows, clock)
                    faces = validate_faces(row, video["width"], video["height"])
                    if clock["frame_index"] < cache.processed:
                        continue
                    result, swapped, failed = image.copy(), [], []
                    for face in faces:
                        binding = face.get("virtual", {})
                        if binding.get("status") != "ready":
                            continue
                        target = targets.get(binding.get("virtual_id"))
                        if (
                            target is None
                            or any(
                                binding.get(key) != target[key]
                                for key in ("portrait_file", "portrait_sha256")
                            )
                            or face.get("identity", {}).get("identity_id")
                            != target["identity_id"]
                        ):
                            raise ExecutionError("Invalid target assignment")
                        try:
                            candidate = engine.replace(
                                result.copy(), face, target["virtual_id"]
                            )
                            check_image(candidate, video)
                            result = candidate
                            swapped.append(face["track_id"])
                        except (ExecutionError, cv2.error, RuntimeError, ValueError):
                            failed.append(face["track_id"])
                    cache.append(clock, result, {"swapped": swapped, "failed": failed})
            no_extra(rows)
            archive, records = cache.finish()
        finally:
            cache.close()
        summary = context.result_dir / "summary.json"
        write_json(
            summary,
            {
                "schema_version": 1,
                "source": source,
                "video": video,
                "assignments_sha256": inputs["assignments"].sha256,
                "targets_sha256": inputs["targets"].sha256,
                "engine": {
                    "name": "inswapper-128",
                    "device": values["device"],
                    "swap_model_sha256": inputs["swap_model"].sha256,
                    "arcface_model_sha256": inputs["arcface_model"].sha256,
                },
                "file_sha256": {
                    "frames": digest_file(archive)[0],
                    "records": digest_file(records)[0],
                },
            },
        )
        return {"frames": archive, "records": records, "summary": summary}
