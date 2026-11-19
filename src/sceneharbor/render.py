import subprocess
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, suppress
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import BinaryIO

from PIL import Image

from .media import (
    INPUT_OPTIONS,
    MODEL_SCALE,
    Engine,
    MediaError,
    Shot,
    VideoInfo,
    command_error,
    probe,
    raw_decoder,
    run_command,
    terminate,
)

PROFILES = ("original", "gentle", "restore")
GENTLE_BLEND = 0.35
MAX_BATCH_FRAMES = 24
AI_DISK_PIXELS = 420_000_000
AI_BATCH_PIXELS = 16_000_000
AI_WORKING_PIXELS = 58_000_000
COPYABLE_AUDIO = frozenset({"aac", "ac3", "alac", "eac3", "mp3"})


def profile_frame(
    original: Image.Image,
    upscaled: Image.Image | None,
    size: tuple[int, int],
    profile: str,
) -> Image.Image:
    """Produces a source or restored frame at the requested display size."""
    if profile not in PROFILES:
        raise MediaError("Choose the original, gentle, or restore profile.")
    if profile == "original":
        return _rgb_at_size(original, size)
    if upscaled is None:
        raise MediaError("This restoration profile requires the Real-ESRGAN model.")
    restored = _rgb_at_size(upscaled, size)
    if profile == "restore":
        return restored
    baseline = _rgb_at_size(original, size)
    return Image.blend(baseline, restored, GENTLE_BLEND)


def render_shot(
    source: Path,
    info: VideoInfo,
    shot: Shot,
    profile: str,
    size: tuple[int, int],
    engine: Engine | None,
    dest: Path,
    cancel: Callable[[], None],
    progress: Callable[[int], None],
    crf: int = 18,
) -> None:
    """Encodes a validated, atomic checkpoint for one half-open shot."""
    if not 0 <= shot.start < shot.end <= info.frames:
        raise MediaError("The checkpoint covers an invalid source frame range.")
    if profile not in PROFILES:
        raise MediaError("Choose the original, gentle, or restore profile.")
    if profile != "original" and engine is None:
        raise MediaError(
            "Install Real-ESRGAN NCNN Vulkan before rendering an AI profile."
        )
    if not 0 <= crf <= 51 or any(side < 2 or side % 2 for side in size):
        raise MediaError(
            "Choose even output dimensions and a valid H.264 quality setting."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".sceneharbor-shot-") as temporary:
        candidate = Path(temporary) / "checkpoint.mp4"
        _transcode(
            source, info, shot, profile, size, engine, candidate, cancel, progress, crf
        )
        cancel()
        output = probe(candidate, cancel)
        if (output.width, output.height, output.frames, output.fps) != (
            *size,
            shot.end - shot.start,
            info.fps,
        ):
            raise MediaError(
                "The checkpoint did not contain the expected video frames."
            )
        cancel()
        candidate.replace(dest)


def mux(
    source: Path,
    parts: list[Path],
    dest: Path,
    info: VideoInfo,
    cancel: Callable[[], None],
) -> None:
    """Joins validated checkpoints and aligns every original audio track to the video."""
    if not parts:
        raise MediaError("At least one rendered checkpoint is required.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=dest.parent, prefix=".sceneharbor-export-"
    ) as temporary:
        folder = Path(temporary)
        concat, output = folder / "checkpoints.txt", folder / "output.mp4"
        first = _write_concat(parts, concat, info, cancel)
        run_command(_mux_args(source, concat, output, info), cancel)
        cancel()
        result = probe(output, cancel)
        if (
            result.width,
            result.height,
            result.fps,
            result.frames,
            result.audio_tracks,
            result.chapters,
        ) != (
            first.width,
            first.height,
            info.fps,
            info.frames,
            info.audio_tracks,
            info.chapters,
        ):
            raise MediaError(
                "The finished export did not contain the expected frames, audio tracks, and chapters."
            )
        cancel()
        output.replace(dest)


def _transcode(
    source: Path,
    info: VideoInfo,
    shot: Shot,
    profile: str,
    size: tuple[int, int],
    engine: Engine | None,
    destination: Path,
    cancel: Callable[[], None],
    progress: Callable[[int], None],
    crf: int,
) -> None:
    count = shot.end - shot.start
    decode = raw_decoder(source, info, shot.start, count)
    encode = _encoder_args(info, size, destination, crf)
    batch_limit = _batch_limit(info, size, profile)
    with ExitStack() as stack:
        decoder = stack.enter_context(_media_pipe(decode, decode=True, cancel=cancel))
        encoder = stack.enter_context(_media_pipe(encode, decode=False, cancel=cancel))
        if decoder.stdout is None or encoder.stdin is None:
            raise MediaError("FFmpeg did not open its media stream.")
        done = 0
        for batch in _batches(decoder.stdout, info, count, batch_limit, cancel):
            restored = (
                engine.upscale(batch, cancel, target_size=size)
                if engine and profile != "original"
                else []
            )
            try:
                for index, image in enumerate(batch):
                    cancel()
                    candidate = restored[index] if restored else None
                    rendered = profile_frame(image, candidate, size, profile)
                    encoder.stdin.write(rendered.tobytes())
            except BrokenPipeError:
                break
            done += len(batch)
            progress(done)
        with suppress(BrokenPipeError):
            encoder.stdin.close()


def _rgb_at_size(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    return rgb if rgb.size == size else rgb.resize(size, Image.Resampling.LANCZOS)


def _batch_limit(info: VideoInfo, size: tuple[int, int], profile: str) -> int:
    if profile == "original":
        return 2
    source_pixels = info.width * info.height
    delivery_pixels = size[0] * size[1]
    return max(
        1,
        min(
            MAX_BATCH_FRAMES,
            AI_DISK_PIXELS // (source_pixels * MODEL_SCALE**2),
            AI_WORKING_PIXELS // (source_pixels + delivery_pixels),
        ),
    )


def _batches(
    stream: BinaryIO,
    info: VideoInfo,
    count: int,
    limit: int,
    cancel: Callable[[], None],
) -> Iterator[list[Image.Image]]:
    frame_bytes = info.width * info.height * 3
    remaining = count
    while remaining:
        batch = []
        for _ in range(min(limit, remaining)):
            cancel()
            pixels = stream.read(frame_bytes)
            if len(pixels) != frame_bytes:
                raise MediaError("FFmpeg stopped before the checkpoint's last frame.")
            batch.append(Image.frombytes("RGB", (info.width, info.height), pixels))
        remaining -= len(batch)
        yield batch


@contextmanager
def _media_pipe(
    arguments: list[str],
    *,
    decode: bool,
    cancel: Callable[[], None],
) -> Iterator[subprocess.Popen[bytes]]:
    with TemporaryFile() as errors:
        try:
            process = subprocess.Popen(
                arguments,
                stderr=errors,
                stdin=subprocess.DEVNULL if decode else subprocess.PIPE,
                stdout=subprocess.PIPE if decode else subprocess.DEVNULL,
            )
        except OSError as error:
            raise MediaError(
                "Could not start FFmpeg. Check its installation."
            ) from error
        try:
            yield process
            while process.poll() is None:
                cancel()
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode:
                end = errors.seek(0, 2)
                errors.seek(max(0, end - 2000))
                raise MediaError(command_error(arguments[0], errors.read()))
        finally:
            terminate(process)
            if process.stdin is not None:
                with suppress(BrokenPipeError):
                    process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()


def _encoder_args(
    info: VideoInfo, size: tuple[int, int], dest: Path, crf: int
) -> list[str]:
    rate = Fraction(info.fps)
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{size[0]}x{size[1]}",
        "-framerate",
        info.fps,
        "-i",
        "-",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        "scale=out_color_matrix=bt709:out_range=limited,format=yuv420p,setsar=1",
        "-c:v",
        "libx264",
        "-threads",
        "4",
        "-preset",
        "fast",
        "-crf",
        str(crf),
        "-bf",
        "0",
        "-flags",
        "+cgop",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-video_track_timescale",
        str(rate.numerator),
        "-movflags",
        "+faststart",
        "-y",
        str(dest),
    ]


def _write_concat(
    parts: list[Path],
    concat: Path,
    source: VideoInfo,
    cancel: Callable[[], None],
) -> VideoInfo:
    lines = ["ffconcat version 1.0"]
    first: VideoInfo | None = None
    total = 0
    rate = Fraction(source.fps)
    for part in parts:
        cancel()
        info = probe(part, cancel)
        first = first or info
        if (info.width, info.height, info.fps, info.audio_tracks) != (
            first.width,
            first.height,
            source.fps,
            0,
        ):
            raise MediaError(
                "Rendered checkpoints must share one video format without audio tracks."
            )
        path = part.resolve().as_posix()
        if "\n" in path or "\r" in path:
            raise MediaError("Checkpoint paths cannot contain line breaks.")
        escaped = path.replace("'", "'\\''")
        lines.extend((f"file '{escaped}'", f"duration {float(info.frames / rate):.9f}"))
        total += info.frames
    if first is None or total != source.frames:
        raise MediaError(
            "Rendered checkpoints do not cover the full source frame count."
        )
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return first


def _mux_args(source: Path, concat: Path, output: Path, info: VideoInfo) -> list[str]:
    rate = Fraction(info.fps)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-itsoffset",
        f"{info.seek_offset:.9f}",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        *INPUT_OPTIONS,
        "-i",
        str(source.resolve()),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
    ]
    for index, codec in enumerate(info.audio_codecs):
        if codec not in COPYABLE_AUDIO:
            command.extend((f"-c:a:{index}", "aac", f"-b:a:{index}", "192k"))
    command.extend(
        (
            "-avoid_negative_ts",
            "disabled",
            "-video_track_timescale",
            str(rate.numerator),
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        )
    )
    return command
