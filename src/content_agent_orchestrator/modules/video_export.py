"""Encode, copy only audio, then validate the actual decoded delivery pixels."""

import hashlib
import json
import shutil
import subprocess
from fractions import Fraction
from typing import ClassVar

import cv2
import numpy as np

from ..contracts import ExecutionError, PipelineError
from ..face_data import iter_rows, validate_timestamp, write_json
from ..privacy_validation import (
    REDACTION_COLOR,
    PrivacyValidator,
    bound_render_summary,
    clock_row,
    no_extra,
)
from ..render_frames import VideoWriter, archived_frames, av_runtime, runtime_version
from ..storage import digest_file


def audio_signatures(path):
    av = av_runtime()
    result = []
    with av.open(str(path)) as container:
        streams = list(container.streams.audio)
        hashes = {stream.index: hashlib.sha256() for stream in streams}
        counts = dict.fromkeys(hashes, 0)
        # Hash codec payloads without decoding or modifying voice content.
        for packet in container.demux(streams) if streams else ():
            if not packet.size:
                continue
            hashes[packet.stream.index].update(bytes(packet))
            counts[packet.stream.index] += 1
            if packet.pts is None:
                raise ExecutionError(
                    "Audio needs explicit timestamps for verified synchronization"
                )
        for stream in streams:
            result.append(
                {
                    "codec": stream.codec_context.name,
                    "sample_rate": stream.codec_context.sample_rate,
                    "channels": stream.codec_context.layout.nb_channels,
                    "packet_count": counts[stream.index],
                    "sha256": hashes[stream.index].hexdigest(),
                    "stream_index": stream.index,
                }
            )
    return result


def audio_timestamps(path, index):
    with av_runtime().open(str(path)) as container:
        for packet in container.demux(container.streams[index]):
            if packet.size:
                if packet.pts is None:
                    raise ExecutionError("Audio packet has no timestamp")
                yield Fraction(packet.pts) * packet.time_base


def source_tail_duration(path, stream_index, time_base, last_tick, previous_interval):
    av = av_runtime()
    last_duration = None
    with av.open(str(path)) as container:
        for packet in container.demux(container.streams[stream_index]):
            if (
                packet.pts is not None
                and packet.pts * packet.time_base == last_tick * time_base
                and packet.duration > 0
            ):
                last_duration = Fraction(packet.duration) * packet.time_base / time_base
    if last_duration is not None and last_duration.denominator == 1:
        return int(last_duration), "source_packet"
    if previous_interval is not None and previous_interval > 0:
        return previous_interval, "inferred_previous_interval"
    raise ExecutionError("Single-frame input needs an explicit frame duration")


def rotate(image, degrees):
    return (
        cv2.rotate(
            image,
            {
                90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_CLOCKWISE,
            }[degrees],
        )
        if degrees
        else image
    )


class VideoExport:
    name = "video.export"
    version = f"1-opencv-{cv2.__version__}-av-{runtime_version('av')}"
    input_kinds: ClassVar = {
        "video": "video.source",
        "inspection": "video.manifest",
        "timestamps": "video.timestamps",
        "frames": "video.redacted-frames",
        "records": "face.redaction-records",
        "summary": "face.redaction-summary",
        "assignments": "virtual.observations",
        "embeddings": "face.embeddings",
        "swap_records": "face.swap-records",
        "targets": "virtual.targets",
        "registry": "identity.registry-snapshot",
        "detector": "face.detector.yunet",
        "recognizer": "face.recognizer.sface",
    }
    output_kinds: ClassVar = {
        "video": "video.face-anonymized",
        "report": "video.privacy-report",
        "audit": "video.delivery-audit",
    }

    @staticmethod
    def parameters(parameters):
        defaults = {
            "codec": "h264",
            "crf": 18,
            "timeout_seconds": 600,
            "allow_all_redacted": False,
        }
        if parameters.keys() - defaults.keys():
            raise PipelineError("Unknown video.export parameter")
        values = defaults | dict(parameters)
        if (
            values["codec"] not in ("h264", "ffv1")
            or type(values["crf"]) is not int
            or not 0 <= values["crf"] <= 23
            or type(values["timeout_seconds"]) is not int
            or not 1 <= values["timeout_seconds"] <= 86400
            or type(values["allow_all_redacted"]) is not bool
        ):
            raise PipelineError("Invalid export codec, quality or timeout")
        return values

    def validate_parameters(self, parameters):
        self.parameters(parameters)

    def execute(self, inputs, parameters, context):
        values = self.parameters(parameters)
        summary = bound_render_summary(inputs)
        video = summary["video"]
        from .face_detect import FaceDetection

        inspection, inspected_video = FaceDetection._inspection(inputs)
        if inspected_video != video or summary["source"] != inspection["source"]:
            raise ExecutionError("Export does not belong to the inspected source")
        if summary["counts"]["validated"] == 0 and not values["allow_all_redacted"]:
            raise ExecutionError(
                "No frame passed face replacement validation; nothing published"
            )
        primary = next(
            item
            for item in inspection["streams"]
            if item["index"] == video["stream_index"]
        )
        av = av_runtime()
        if primary.get("color_transfer") in ("smpte2084", "arib-std-b67") or any(
            component.bits > 8
            for component in av.VideoFormat(primary["pix_fmt"]).components
        ):
            raise ExecutionError(
                "This renderer supports 8-bit SDR; HDR/high-bit-depth input requires another renderer"
            )
        rotation = video["rotation_degrees"]
        delivery = dict(video)
        delivery["rotation_degrees"] = 0
        if rotation in (90, 270):
            delivery["width"], delivery["height"] = video["height"], video["width"]
        if values["codec"] == "h264" and (
            delivery["width"] % 2 or delivery["height"] % 2
        ):
            raise ExecutionError(
                "H.264 needs even dimensions; choose the lossless ffv1 codec"
            )
        last_clock, interval = None, None
        for row in iter_rows(inputs["timestamps"].path):
            if last_clock is not None:
                interval = (
                    row["presentation_timestamp"] - last_clock["presentation_timestamp"]
                )
            last_clock = row
        tail, tail_basis = source_tail_duration(
            inputs["video"].path,
            video["stream_index"],
            Fraction(video["time_base"]),
            last_clock["presentation_timestamp"],
            interval,
        )
        delivery["frame_rate"] = str(1 / (tail * Fraction(video["time_base"])))
        encoded = context.work_dir / "encoded.nut"
        writer = VideoWriter(
            encoded,
            delivery,
            codec="libx264" if values["codec"] == "h264" else "ffv1",
            crf=values["crf"],
        )
        clocks = iter(iter_rows(inputs["timestamps"].path))
        decisions = iter(iter_rows(inputs["records"].path))
        clock = next(clocks, None)
        previous = None
        try:
            for index, time, image in archived_frames(
                inputs["frames"].path, context.work_dir, video
            ):
                previous = validate_timestamp(clock, index, previous)
                if abs(
                    time
                    - clock["presentation_timestamp"] * Fraction(video["time_base"])
                ) > Fraction(1, 1000000):
                    raise ExecutionError("Redacted frame timestamps changed")
                next_clock = next(clocks, None)
                duration = (
                    next_clock["presentation_timestamp"]
                    - clock["presentation_timestamp"]
                    if next_clock
                    else tail
                )
                decision = clock_row(decisions, clock)
                if decision.get("status") not in ("validated", "redacted"):
                    raise ExecutionError("Invalid redaction decision before encoding")
                writer.append(
                    clock
                    | {
                        "duration_ticks": duration,
                        "redacted": decision["status"] == "redacted",
                    },
                    rotate(image, rotation),
                )
                clock = next_clock
            if clock is not None:
                raise ExecutionError("Incomplete redacted frame coverage")
            no_extra(decisions)
            writer.finish()
        except BaseException:
            writer.abort()
            raise
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise ExecutionError("FFmpeg is required for verified audio muxing")
        output = context.result_dir / "video.mkv"
        output.touch(mode=0o600)
        command = [
            executable,
            "-v",
            "error",
            "-nostdin",
            "-copyts",
            "-i",
            str(encoded),
            "-i",
            str(inputs["video"].path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c",
            "copy",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-avoid_negative_ts",
            "disabled",
            "-y",
            str(output),
        ]
        try:
            with (context.work_dir / "mux.stderr").open("wb") as error:
                result = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=error,
                    timeout=values["timeout_seconds"],
                    check=False,
                )
        except subprocess.TimeoutExpired:
            raise ExecutionError("Audio mux timeout; no output published") from None
        if result.returncode:
            raise ExecutionError("Audio mux failed; no output published")
        before, after = audio_signatures(inputs["video"].path), audio_signatures(output)
        if len(before) != len(after):
            raise ExecutionError("Audio stream count changed")
        for original, final in zip(before, after, strict=True):
            if any(
                original[key] != final[key] for key in original if key != "stream_index"
            ) or any(
                abs(a - b) > Fraction(1, 1000)
                for a, b in zip(
                    audio_timestamps(inputs["video"].path, original["stream_index"]),
                    audio_timestamps(output, final["stream_index"]),
                    strict=True,
                )
            ):
                raise ExecutionError("Audio content or synchronization changed")
        validator = PrivacyValidator(
            video,
            inputs["targets"].path,
            inputs["registry"].path,
            inputs["detector"].path,
            inputs["recognizer"].path,
        )
        clocks, records, virtual, embedded, swapped = (
            iter(iter_rows(inputs[key].path))
            for key in (
                "timestamps",
                "records",
                "assignments",
                "embeddings",
                "swap_records",
            )
        )
        audit = context.result_dir / "delivery_audit.jsonl"
        audit.touch(mode=0o600)
        counts = {"validated": 0, "redacted": 0}
        previous = None
        previous_delivery_time = None
        with (
            av.open(str(output)) as container,
            audit.open("w", encoding="utf-8") as audit_output,
        ):
            if len(container.streams.video) != 1:
                raise ExecutionError("Delivery must contain exactly one video stream")
            index = 0
            for frame in container.decode(video=0):
                clock = next(clocks, None)
                if clock is None:
                    raise ExecutionError("Delivery contains extra video frames")
                previous = validate_timestamp(clock, index, previous)
                delivery_time = (
                    frame.pts * frame.time_base if frame.pts is not None else None
                )
                if (
                    frame.pts is None
                    or (
                        previous_delivery_time is not None
                        and delivery_time <= previous_delivery_time
                    )
                    or abs(
                        frame.pts * frame.time_base
                        - clock["presentation_timestamp"] * Fraction(video["time_base"])
                    )
                    > Fraction(1, 1000)
                    or (frame.width, frame.height)
                    != (delivery["width"], delivery["height"])
                ):
                    raise ExecutionError("Delivery changed frame timing or resolution")
                previous_delivery_time = delivery_time
                record = clock_row(records, clock)
                assigned, original, swap = (
                    clock_row(virtual, clock),
                    clock_row(embedded, clock),
                    clock_row(swapped, clock),
                )
                image = rotate(frame.to_ndarray(format="bgr24"), (-rotation) % 360)
                if record.get("status") == "redacted":
                    if np.abs(image.astype(np.int16) - REDACTION_COLOR).max() > 3:
                        raise ExecutionError(
                            "Delivery did not preserve opaque redaction"
                        )
                    decision = {"status": "redacted", "reason": record["reason"]}
                elif record.get("status") == "validated":
                    decision = validator.check(image, assigned, original, swap)
                    if decision["status"] != "validated":
                        raise ExecutionError(
                            "Encoded delivery failed face privacy validation; nothing published"
                        )
                else:
                    raise ExecutionError("Unknown delivery redaction decision")
                counts[decision["status"]] += 1
                audit_output.write(json.dumps(clock | decision, allow_nan=False) + "\n")
                index += 1
            if index and abs(
                frame.duration * frame.time_base - tail * Fraction(video["time_base"])
            ) > Fraction(1, 1000):
                raise ExecutionError("Delivery changed the last-frame duration")
        no_extra(clocks, records, virtual, embedded, swapped)
        if counts != summary["counts"] or index != video["frame_count"]:
            raise ExecutionError("Delivery audit coverage is incomplete")
        report = context.result_dir / "privacy_report.json"
        write_json(
            report,
            {
                "schema_version": 1,
                "anonymization_performed": True,
                "voice_modified": False,
                "source": summary["source"],
                "engine": summary["engine"],
                "counts": counts,
                "delivery": {
                    "width": delivery["width"],
                    "height": delivery["height"],
                    "codec": values["codec"],
                    "timing_tolerance_seconds": 0.001,
                    "audio_streams_copied": len(after),
                    "last_frame_duration_seconds": float(
                        tail * Fraction(video["time_base"])
                    ),
                    "last_frame_duration_basis": tail_basis,
                },
                "policy": summary["policy"],
                "validation": "independent-YuNet-SFace-before-and-after-encoding-v1",
                "file_sha256": {
                    "video": digest_file(output)[0],
                    "audit": digest_file(audit)[0],
                },
            },
        )
        return {"video": output, "report": report, "audit": audit}
