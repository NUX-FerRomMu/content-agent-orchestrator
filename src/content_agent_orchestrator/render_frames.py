"""Bounded lossless frame chunks with checksummed, resumable journals."""

import hashlib
import json
import os
import shutil
import zipfile
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from .contracts import ExecutionError
from .face_data import atomic_json, iter_rows, read_json, safe_child, validate_timestamp
from .pipeline import canonical_json
from .storage import digest_file


def runtime_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def av_runtime():
    try:
        import av
    except ImportError:
        raise ExecutionError(
            "Install the optional swap or swap-cuda uv group"
        ) from None
    return av


def signature(module, inputs, parameters):
    return hashlib.sha256(
        canonical_json(
            {
                "module": module.name,
                "version": module.version,
                "inputs": {key: value.sha256 for key, value in inputs.items()},
                "parameters": parameters,
            }
        ).encode()
    ).hexdigest()


def check_image(image, video):
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.shape != (video["height"], video["width"], 3)
    ):
        raise ExecutionError("Render must preserve original 8-bit frame dimensions")


class VideoWriter:
    def __init__(self, path, video, *, codec="ffv1", crf=18):
        self.av = av_runtime()
        path.touch(mode=0o600)
        self.container = self.av.open(str(path), "w", format="nut")
        self.video = video
        self.durations = {}
        self.stream = self.container.add_stream(
            codec, rate=Fraction(video.get("frame_rate", "30"))
        )
        self.stream.width, self.stream.height = video["width"], video["height"]
        self.stream.pix_fmt = "bgr0" if codec == "ffv1" else "yuv420p"
        self.stream.time_base = Fraction(video["time_base"])
        self.stream.codec_context.time_base = self.stream.time_base
        self.stream.codec_context.thread_count = 4
        if codec == "libx264":
            # RGB -> yuv420p uses libswscale's BT.601 matrix. Tag it explicitly
            # so HD players do not infer BT.709 and shift the rendered skin tone.
            self.stream.codec_context.colorspace = 5
            self.stream.codec_context.color_range = 1
            self.stream.options = {"crf": str(crf), "preset": "slow", "bf": "0"}

    def append(self, clock, image):
        check_image(image, self.video)
        frame = self.av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = clock["presentation_timestamp"]
        frame.time_base = Fraction(self.video["time_base"])
        if clock.get("redacted"):
            frame.pict_type = self.av.video.frame.PictureType.I
        self.durations[frame.pts] = clock.get("duration_ticks")
        for packet in self.stream.encode(frame):
            duration = self.durations.pop(packet.pts, None)
            if duration is not None:
                packet.duration = duration
            self.container.mux(packet)

    def finish(self):
        try:
            for packet in self.stream.encode():
                duration = self.durations.pop(packet.pts, None)
                if duration is not None:
                    packet.duration = duration
                self.container.mux(packet)
        finally:
            self.container.close()

    def abort(self):
        self.container.close()


class FrameChunks:
    def __init__(self, context, video, digest, chunk_frames=32):
        self.context, self.video, self.digest = context, video, digest
        self.chunk_frames = chunk_frames
        self.cache = safe_child(context.work_dir, "frame-cache")
        self.cache.mkdir(exist_ok=True, mode=0o700)
        self.chunks = self._load()
        self.count = self.processed = sum(item["count"] for item in self.chunks)
        self.writer = self.log = None
        self.pending = 0

    def _load(self):
        try:
            checkpoint = read_json(safe_child(self.cache, "checkpoint.json"))
            if checkpoint.get("signature") != self.digest or not isinstance(
                checkpoint.get("chunks"), list
            ):
                return []
            chunks, count, previous = [], 0, None
            for index, item in enumerate(checkpoint["chunks"]):
                if not isinstance(item, dict):
                    break
                if (
                    type(item.get("count")) is not int
                    or not 1 <= item["count"] <= self.chunk_frames
                ):
                    break
                if count + item["count"] > self.video["frame_count"]:
                    break
                if (
                    item["count"] < self.chunk_frames
                    and count + item["count"] != self.video["frame_count"]
                ):
                    break
                movie = safe_child(self.cache, f"chunk-{index:08d}.nut")
                log = safe_child(self.cache, f"chunk-{index:08d}.jsonl")
                if digest_file(movie) != (
                    item["sha256"],
                    item["size_bytes"],
                ) or digest_file(log) != (
                    item["records_sha256"],
                    item["records_size_bytes"],
                ):
                    break
                rows = 0
                for row in iter_rows(log):
                    previous = validate_timestamp(row, count + rows, previous)
                    rows += 1
                if rows != item["count"]:
                    break
                chunks.append(item)
                count += rows
            return chunks
        except (ExecutionError, OSError, KeyError, TypeError, ValueError):
            return []

    def append(self, clock, image, record):
        if clock["frame_index"] != self.count:
            raise ExecutionError("Incomplete render cache coverage")
        if self.writer is None:
            index = len(self.chunks)
            self.partial = safe_child(self.cache, f"chunk-{index:08d}.part")
            self.writer = VideoWriter(self.partial, self.video)
            self.log_partial = safe_child(self.cache, f"chunk-{index:08d}.jsonl.part")
            self.log_partial.touch(mode=0o600)
            self.log = self.log_partial.open("w", encoding="utf-8")
        self.writer.append(clock, image)
        self.log.write(json.dumps(clock | record, allow_nan=False) + "\n")
        self.pending += 1
        self.count += 1
        if self.pending == self.chunk_frames:
            self._commit()

    def _commit(self):
        self.writer.finish()
        self.writer = None
        self.log.flush()
        os.fsync(self.log.fileno())
        self.log.close()
        self.log = None
        index = len(self.chunks)
        movie = safe_child(self.cache, f"chunk-{index:08d}.nut")
        records = safe_child(self.cache, f"chunk-{index:08d}.jsonl")
        self.partial.replace(movie)
        self.log_partial.replace(records)
        sha, size = digest_file(movie)
        if size > 4 * 1024**3:
            raise ExecutionError("Frame chunk exceeds 4 GiB; reduce chunk_frames")
        log_sha, log_size = digest_file(records)
        self.chunks.append(
            {
                "count": self.pending,
                "sha256": sha,
                "size_bytes": size,
                "records_sha256": log_sha,
                "records_size_bytes": log_size,
            }
        )
        self.pending = 0
        atomic_json(
            safe_child(self.cache, "checkpoint.json"),
            {"signature": self.digest, "chunks": self.chunks},
        )

    def finish(self):
        if self.count != self.video["frame_count"]:
            raise ExecutionError("Render did not cover every source frame")
        if self.writer is not None:
            self._commit()
        path = self.context.result_dir / "frames.zip"
        records = self.context.result_dir / "records.jsonl"
        path.touch(mode=0o600)
        records.touch(mode=0o600)
        manifest = {"schema_version": 1, "video": self.video, "chunks": self.chunks}
        with (
            zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive,
            records.open("wb") as output,
        ):
            archive.writestr("manifest.json", canonical_json(manifest))
            for index in range(len(self.chunks)):
                archive.write(
                    safe_child(self.cache, f"chunk-{index:08d}.nut"),
                    f"chunks/{index:08d}.nut",
                )
                with safe_child(self.cache, f"chunk-{index:08d}.jsonl").open(
                    "rb"
                ) as source:
                    shutil.copyfileobj(source, output, 1024 * 1024)
        return path, records

    def close(self):
        if self.writer is not None:
            self.writer.abort()
        if self.log is not None:
            self.log.close()


def archived_frames(path, work_dir, video):
    """Decode each verified chunk in order; never extract caller-chosen names."""
    av = av_runtime()
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise ExecutionError("Invalid frame archive entries")
            if archive.getinfo("manifest.json").file_size > 16 * 1024**2:
                raise ExecutionError("Frame archive manifest is too large")
            manifest = json.loads(archive.read("manifest.json"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema_version") != 1
                or manifest.get("video") != video
                or not isinstance(manifest.get("chunks"), list)
            ):
                raise ExecutionError("Frame archive does not describe this video")
            expected = [
                "manifest.json",
                *(
                    f"chunks/{index:08d}.nut"
                    for index in range(len(manifest["chunks"]))
                ),
            ]
            if set(names) != set(expected) or any(
                info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1
                for info in archive.infolist()
            ):
                raise ExecutionError("Unexpected or compressed frame archive entries")
            count = 0
            for index, item in enumerate(manifest["chunks"]):
                name = f"chunks/{index:08d}.nut"
                if (
                    type(item.get("count")) is not int
                    or not 1 <= item["count"] <= 1024
                    or not 0 < item.get("size_bytes", 0) <= 4 * 1024**3
                    or archive.getinfo(name).file_size != item["size_bytes"]
                ):
                    raise ExecutionError("Invalid frame chunk bounds")
                temporary = safe_child(work_dir, "reading.nut")
                temporary.touch(mode=0o600)
                with archive.open(name) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if digest_file(temporary) != (item["sha256"], item["size_bytes"]):
                    raise ExecutionError("Frame chunk checksum mismatch")
                frames = 0
                with av.open(str(temporary), "r") as container:
                    streams = list(container.streams.video)
                    if (
                        len(streams) != 1
                        or streams[0].codec_context.name != "ffv1"
                        or (streams[0].width, streams[0].height)
                        != (video["width"], video["height"])
                    ):
                        raise ExecutionError(
                            "Frame chunk has unexpected codec or dimensions"
                        )
                    for frame in container.decode(video=0):
                        if (
                            frames >= item["count"]
                            or frame.width != video["width"]
                            or frame.height != video["height"]
                            or frame.pts is None
                        ):
                            raise ExecutionError("Invalid decoded frame chunk")
                        yield (
                            count,
                            Fraction(frame.pts) * frame.time_base,
                            frame.to_ndarray(format="bgr24"),
                        )
                        count += 1
                        frames += 1
                temporary.unlink()
                if frames != item["count"]:
                    raise ExecutionError("Frame chunk is incomplete")
            if count != video["frame_count"]:
                raise ExecutionError("Frame archive coverage is incomplete")
    except (zipfile.BadZipFile, KeyError, TypeError, ValueError) as exc:
        raise ExecutionError("Invalid frame archive") from exc
