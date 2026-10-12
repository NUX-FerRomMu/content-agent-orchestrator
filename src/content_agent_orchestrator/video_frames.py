"""Software decode with original ordering, bounded frames and no autorotation."""

import shutil
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .contracts import ExecutionError
from .face_data import iter_rows, safe_child, validate_timestamp


def _read_exact(stream, size: int) -> bytes:
    buffer = bytearray()
    while len(buffer) < size:
        block = stream.read(size - len(buffer))
        if not block:
            break
        buffer.extend(block)
    return bytes(buffer)


@contextmanager
def decoded_frames(
    video: Path,
    timestamps: Path,
    *,
    stream_index: int,
    width: int,
    height: int,
    expected_count: int,
    work_dir: Path,
    timeout: int,
) -> Iterator[Iterator[tuple[dict, np.ndarray]]]:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ExecutionError("FFmpeg is required to decode faces; add it to PATH")
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-noautorotate",
        "-i",
        str(video),
        "-map",
        f"0:{stream_index}",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"scale={width}:{height}",
        "-fps_mode",
        "passthrough",
        "-threads",
        "1",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    with safe_child(work_dir, "decode.stderr").open("w+b") as stderr:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=stderr, stdin=subprocess.DEVNULL
        )
        timed_out = threading.Event()

        def expire():
            timed_out.set()
            try:
                process.kill()
            except ProcessLookupError:
                pass

        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()

        def check_decoder():
            code = process.wait()
            if timed_out.is_set():
                raise ExecutionError(f"Frame decoding exceeded its {timeout}s timeout")
            stderr.seek(0, 2)
            if code or stderr.tell():
                stderr.seek(max(0, stderr.tell() - 2000))
                detail = stderr.read().decode("utf-8", errors="replace").strip()
                raise ExecutionError(f"FFmpeg frame decode failed ({code}): {detail}")

        def frames():
            previous = None
            count = 0
            for row in iter_rows(timestamps):
                previous = validate_timestamp(row, count, previous)
                if count >= expected_count:
                    raise ExecutionError(
                        "Timestamp count exceeds the inspected frame count"
                    )
                data = _read_exact(process.stdout, width * height * 3)
                if len(data) != width * height * 3:
                    check_decoder()
                    raise ExecutionError(
                        "Decoded video has fewer frames than its timestamps"
                    )
                yield row, np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
                count += 1
            if process.stdout.read(1):
                raise ExecutionError(
                    "Decoded video has more frames than its timestamps"
                )
            check_decoder()
            if count != expected_count:
                raise ExecutionError(
                    "Timestamp count does not match the inspected video"
                )

        try:
            yield frames()
        finally:
            timer.cancel()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
