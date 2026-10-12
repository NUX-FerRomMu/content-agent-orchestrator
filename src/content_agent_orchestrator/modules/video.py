"""Built-in inspection modules; no face processing or GPU dependencies yet."""

import csv
import json
import shutil
import subprocess
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

from ..contracts import Artifact, ExecutionError, ModuleContext, PipelineError
from ..storage import digest_file


class VideoInspection:
    name = "video.inspect"
    version = "2"
    input_kinds: ClassVar = {"video": "video.source"}
    output_kinds: ClassVar = {
        "manifest": "video.manifest",
        "timestamps": "video.timestamps",
    }

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        if parameters.keys() - {"timeout_seconds"}:
            raise PipelineError("Unknown inspection parameter")
        timeout = parameters.get("timeout_seconds", 600)
        if type(timeout) is not int or not 1 <= timeout <= 86400:
            raise PipelineError(
                "timeout_seconds must be an integer between 1 and 86400"
            )

    @staticmethod
    def _probe(command: list[str], destination: Path, timeout: int) -> None:
        # Large timestamp listings go to disk, not to a Python string in memory.
        with (
            destination.open("wb") as stdout,
            destination.with_suffix(".stderr").open("w+b") as stderr,
        ):
            try:
                result = subprocess.run(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                    check=False,
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired as exc:
                raise ExecutionError(
                    f"FFprobe exceeded its {timeout}s timeout"
                ) from exc
            stderr.seek(0, 2)
            if result.returncode or stderr.tell():
                stderr.seek(max(0, stderr.tell() - 2000))
                detail = stderr.read().decode("utf-8", errors="replace").strip()
                raise ExecutionError(f"FFprobe failed ({result.returncode}): {detail}")

    def execute(
        self,
        inputs: Mapping[str, Artifact],
        parameters: Mapping[str, Any],
        context: ModuleContext,
    ) -> Mapping[str, Path]:
        executable = shutil.which("ffprobe")
        if executable is None:
            raise ExecutionError(
                "FFprobe is required; install FFmpeg and add it to PATH"
            )
        source = inputs["video"]
        timeout = parameters.get("timeout_seconds", 600)
        metadata_path = context.work_dir / "probe.json"
        self._probe(
            [
                executable,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                "-i",
                str(source.path),
            ],
            metadata_path,
            timeout,
        )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            streams = metadata["streams"]
            primary = next(
                stream
                for stream in streams
                if stream.get("codec_type") == "video"
                and not stream.get("disposition", {}).get("attached_pic")
            )
        except (ValueError, KeyError, TypeError, StopIteration) as exc:
            raise ExecutionError("Input has no readable video stream") from exc
        raw_timestamps = context.work_dir / "frame_timestamps.csv"
        self._probe(
            [
                executable,
                "-v",
                "error",
                "-select_streams",
                str(primary["index"]),
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp,best_effort_timestamp_time:frame_side_data=",
                "-of",
                "csv=p=0",
                "-i",
                str(source.path),
            ],
            raw_timestamps,
            timeout,
        )
        timestamps_path = context.result_dir / "timestamps.jsonl"
        count = missing = non_monotonic = intervals = 0
        previous = minimum = maximum = first = last = None
        with (
            raw_timestamps.open(encoding="utf-8", newline="") as raw,
            timestamps_path.open("w", encoding="utf-8") as output,
        ):
            for row in csv.reader(raw):
                if not row:
                    continue
                # FFprobe's CSV writer adds empty side-data columns for some
                # codecs (e.g. H.264 SEI), even when their fields are excluded.
                if len(row) < 2 or any(field.strip() for field in row[2:]):
                    raise ExecutionError(
                        "FFprobe returned an incomplete frame timestamp"
                    )
                raw_tick, token = (field.strip() for field in row[:2])
                try:
                    presentation_timestamp = (
                        int(raw_tick) if raw_tick not in {"", "N/A"} else None
                    )
                except ValueError as exc:
                    raise ExecutionError(
                        "FFprobe returned an invalid timestamp tick"
                    ) from exc
                if token in {"", "N/A"}:
                    value = None
                else:
                    try:
                        value = Decimal(token)
                        if not value.is_finite():
                            raise InvalidOperation
                    except InvalidOperation as exc:
                        raise ExecutionError(
                            "FFprobe returned an invalid frame timestamp"
                        ) from exc
                output.write(
                    json.dumps(
                        {
                            "frame_index": count,
                            "presentation_timestamp": presentation_timestamp,
                            "presentation_time": str(value)
                            if value is not None
                            else None,
                        }
                    )
                    + "\n"
                )
                count += 1
                if value is None:
                    missing += 1
                    previous = None
                    continue
                if first is None:
                    first = value
                last = value
                if previous is not None:
                    difference = value - previous
                    if difference <= 0:
                        non_monotonic += 1
                    else:
                        intervals += 1
                        minimum = (
                            difference if minimum is None else min(minimum, difference)
                        )
                        maximum = (
                            difference if maximum is None else max(maximum, difference)
                        )
                previous = value
        if count == 0:
            raise ExecutionError("FFprobe could not decode any video frames")
        try:
            tick = Fraction(primary.get("time_base", "0/1"))
            tolerance = max(
                Decimal(2 * tick.numerator) / Decimal(tick.denominator),
                Decimal("0.000002"),
            )
        except (ValueError, ZeroDivisionError):
            tolerance = Decimal("0.000002")
        mode = "unknown"
        if not missing and not non_monotonic and intervals:
            mode = "variable" if maximum - minimum > tolerance else "constant"
        fields = (
            "index",
            "codec_type",
            "codec_name",
            "width",
            "height",
            "pix_fmt",
            "color_range",
            "color_space",
            "color_transfer",
            "color_primaries",
            "sample_aspect_ratio",
            "display_aspect_ratio",
            "time_base",
            "start_time",
            "duration",
            "duration_ts",
            "avg_frame_rate",
            "r_frame_rate",
            "nb_frames",
            "sample_rate",
            "channels",
            "channel_layout",
        )
        selected_streams = []
        for stream in streams:
            selected = {key: stream[key] for key in fields if key in stream}
            rotations = [
                item["rotation"]
                for item in stream.get("side_data_list", [])
                if "rotation" in item
            ]
            if rotations:
                selected["rotation_degrees"] = rotations[0]
            elif "rotate" in stream.get("tags", {}):
                selected["rotation_degrees"] = stream["tags"]["rotate"]
            selected_streams.append(selected)
        manifest = {
            "schema_version": 1,
            "source": {"sha256": source.sha256, "size_bytes": source.size},
            "timestamps_sha256": digest_file(timestamps_path)[0],
            "primary_video_stream": primary["index"],
            "container": {
                key: metadata.get("format", {})[key]
                for key in ("format_name", "start_time", "duration", "bit_rate")
                if key in metadata.get("format", {})
            },
            "streams": selected_streams,
            "timing": {
                "mode": mode,
                "interval_tolerance_seconds": str(tolerance),
                "decoded_frame_count": count,
                "missing_timestamps": missing,
                "non_monotonic_timestamps": non_monotonic,
                "first_timestamp": str(first) if first is not None else None,
                "last_timestamp": str(last) if last is not None else None,
                "min_interval_seconds": str(minimum) if minimum is not None else None,
                "max_interval_seconds": str(maximum) if maximum is not None else None,
            },
        }
        manifest_path = context.result_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return {"manifest": manifest_path, "timestamps": timestamps_path}
