import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from time import monotonic
from typing import BinaryIO, cast

from PIL import Image

MAX_SHOT_SECONDS = 12
SCENE_THRESHOLD = 0.30
MODEL_NAME = "realesrgan-x4plus"
MODEL_SCALE = 4
PROCESS_POLL_SECONDS = 0.20
INPUT_OPTIONS = ("-protocol_whitelist", "file,pipe", "-threads", "2")
CLOCK = re.compile(
    rb"\[showinfo@clock\s.*?\]\s+n:\s*(\d+)\s+pts:\s*(-?\d+).*?\bi:([PTB])"
)
CUT = re.compile(rb"\[showinfo@cut\s.*?\]\s+n:\s*\d+\s+pts:\s*(-?\d+)")


class MediaError(RuntimeError):
    """Carries an operator-facing media processing failure."""


@dataclass(frozen=True)
class VideoInfo:
    """Describes the supported video track and its audio streams."""

    width: int
    height: int
    fps: str
    frames: int
    duration: float
    audio_tracks: int
    video_index: int = 0
    seek_offset: float = 0.0
    sar: str = "1/1"
    audio_codecs: tuple[str, ...] = ()
    subtitle_tracks: int = 0
    chapters: int = 0


@dataclass(frozen=True)
class Shot:
    """Identifies a half-open interval in the source frame timeline."""

    index: int
    start: int
    end: int


@dataclass(frozen=True)
class Engine:
    """Runs Real-ESRGAN's Vulkan executable on a bounded batch of images."""

    binary: Path
    gpu: int | None = None
    tile: int = 0

    def upscale(
        self,
        images: list[Image.Image],
        cancel: Callable[[], None],
        *,
        target_size: tuple[int, int] | None = None,
    ) -> list[Image.Image]:
        """Returns ordered RGB outputs, optionally downsizing each frame as it is read."""
        if not images:
            return []
        binary = self.binary.resolve()
        with TemporaryDirectory(prefix="sceneharbor-ai-") as temporary:
            folder = Path(temporary)
            inputs, outputs = folder / "inputs", folder / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            names = []
            for index, image in enumerate(images):
                cancel()
                name = f"{index:06}.png"
                image.convert("RGB").save(inputs / name, compress_level=1)
                names.append(name)
            arguments = [
                str(binary),
                "-i",
                str(inputs),
                "-o",
                str(outputs),
                "-m",
                str(binary.parent / "models"),
                "-n",
                MODEL_NAME,
                "-s",
                str(MODEL_SCALE),
                "-t",
                str(self.tile),
                "-f",
                "png",
                "-j",
                "1:1:1",
            ]
            if self.gpu is not None:
                arguments.extend(("-g", str(self.gpu)))
            run_command(arguments, cancel)
            result = []
            for name, original in zip(names, images, strict=True):
                cancel()
                try:
                    with Image.open(outputs / name) as image:
                        if image.size != (
                            original.width * MODEL_SCALE,
                            original.height * MODEL_SCALE,
                        ):
                            raise MediaError(
                                "Real-ESRGAN returned a frame at an unexpected size."
                            )
                        frame = image.convert("RGB")
                        if target_size is not None and frame.size != target_size:
                            frame = frame.resize(target_size, Image.Resampling.LANCZOS)
                        result.append(frame)
                    (outputs / name).unlink()
                except OSError as error:
                    raise MediaError(
                        "Real-ESRGAN did not create every expected output frame."
                    ) from error
            return result


def probe(path: Path, cancel: Callable[[], None] | None = None) -> VideoInfo:
    """Reads a local progressive SDR video and obtains its nominal frame count."""
    source = _local_path(path)
    metadata = _probe_json(
        [
            "-show_format",
            "-show_streams",
            "-show_chapters",
            "-show_entries",
            (
                "format=start_time:stream=index,codec_name,codec_type,width,height,avg_frame_rate,r_frame_rate,"
                "nb_frames,start_time,field_order,color_transfer,color_primaries,sample_aspect_ratio:"
                "stream_disposition=attached_pic:stream_side_data=rotation:chapter=id"
            ),
            source,
        ],
        cancel=cancel,
    )
    streams = cast(list[dict[str, object]], metadata.get("streams", []))
    video = next((stream for stream in streams if _is_video(stream)), None)
    if video is None:
        raise MediaError("The source contains no video track.")
    if video.get("field_order") not in {None, "unknown", "progressive"}:
        raise MediaError(
            "The source is interlaced. Convert it to progressive video before importing."
        )
    if video.get("color_transfer") in {"smpte2084", "arib-std-b67"} or (
        video.get("color_primaries") == "bt2020"
    ):
        raise MediaError(
            "The source uses HDR or a wide color gamut. Convert it to SDR before importing."
        )

    rate = _frame_rate(video)
    index = _nonnegative(video.get("index"))
    frames = _nonnegative(video.get("nb_frames"))
    if not frames:
        result = _probe_json(
            [
                "-select_streams",
                str(index),
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                source,
            ],
            timeout=600,
            cancel=cancel,
        )
        counted = cast(list[dict[str, object]], result.get("streams", []))
        frames = _nonnegative(counted[0].get("nb_read_frames")) if counted else 0
    if not frames:
        raise MediaError("FFprobe could not determine the source video frame count.")

    width = _nonnegative(video.get("width"))
    height = _nonnegative(video.get("height"))
    if not width or not height:
        raise MediaError("FFprobe could not determine the source video dimensions.")
    rotation = _video_rotation(video)
    sar = _ratio(str(video.get("sample_aspect_ratio", "1:1")).replace(":", "/"))
    if rotation in (90, 270):
        width, height = height, width
        sar = 1 / sar
    file_format = cast(dict[str, object], metadata.get("format", {}))
    file_start = _decimal(file_format.get("start_time"))
    video_start = _decimal(video.get("start_time"), file_start)
    audio = tuple(
        str(stream.get("codec_name", ""))
        for stream in streams
        if stream.get("codec_type") == "audio"
    )
    return VideoInfo(
        width,
        height,
        str(rate),
        frames,
        float(frames / rate),
        len(audio),
        index,
        video_start - file_start,
        str(sar),
        audio,
        sum(stream.get("codec_type") == "subtitle" for stream in streams),
        len(cast(list[dict[str, object]], metadata.get("chapters", []))),
    )


def detect_shots(path: Path, info: VideoInfo, cancel: Callable[[], None]) -> list[Shot]:
    """Finds scene boundaries and checks the decoded timeline frame by frame."""
    rate = Fraction(info.fps)
    filters = (
        f"setpts=PTS-STARTPTS,settb={rate.denominator}/{rate.numerator},"
        "scale=w='min(iw,320)':h=-2:flags=fast_bilinear,"
        f"showinfo@clock=checksum=0,select='gt(scene,{SCENE_THRESHOLD})',showinfo@cut=checksum=0"
    )
    with TemporaryFile() as diagnostics:
        run_command(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "info",
                *INPUT_OPTIONS,
                "-i",
                _local_path(path),
                "-map",
                f"0:{info.video_index}",
                "-vf",
                filters,
                "-an",
                "-sn",
                "-dn",
                "-fps_mode",
                "passthrough",
                "-f",
                "null",
                "-",
            ],
            cancel,
            stderr=diagnostics,
        )
        diagnostics.seek(0)
        cuts = _scene_boundaries(diagnostics, info, cancel)
    return _partition(cuts, info.frames, max(1, int(rate * MAX_SHOT_SECONDS)))


def output_size(info: VideoInfo, height: int) -> tuple[int, int]:
    """Derives an even output size while preserving the display aspect ratio."""
    if not 2 <= height <= 4320:
        raise MediaError("Output height must be between 2 and 4320 pixels.")
    even_height = height - height % 2
    aspect = Fraction(info.width, info.height) * Fraction(info.sar)
    width = max(2, round(aspect * even_height / 2) * 2)
    if width > 7680:
        raise MediaError("The selected output is wider than 7680 pixels.")
    return width, even_height


def sample_frames(
    path: Path,
    info: VideoInfo,
    indices: list[int],
    cancel: Callable[[], None],
) -> list[Image.Image]:
    """Decodes individual RGB frames at the requested source indices."""
    images = []
    for number in indices:
        cancel()
        if not 0 <= number < info.frames:
            raise MediaError("The requested frame is outside the video.")
        pixels = run_command(raw_decoder(path, info, number, 1), cancel, timeout=120)
        if len(pixels) != info.width * info.height * 3:
            raise MediaError("FFmpeg could not decode the requested preview frame.")
        images.append(Image.frombytes("RGB", (info.width, info.height), pixels))
    return images


def raw_decoder(path: Path, info: VideoInfo, start: int, count: int) -> list[str]:
    """Builds an accurate keyframe-seeking decoder for a source frame interval."""
    rate = Fraction(info.fps)
    seek = max(0.0, info.seek_offset + float((start - Fraction(1, 2)) / rate))
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        *INPUT_OPTIONS,
        "-ss",
        f"{seek:.9f}",
        "-i",
        _local_path(path),
        "-map",
        f"0:{info.video_index}",
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        str(count),
        "-fps_mode",
        "passthrough",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]


def run_command(
    arguments: list[str],
    cancel: Callable[[], None] | None = None,
    *,
    timeout: float | None = None,
    stderr: BinaryIO | None = None,
) -> bytes:
    """Polls a subprocess for cancellation and returns its captured standard output."""
    started = monotonic()
    try:
        process = subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr if stderr is not None else subprocess.PIPE,
        )
    except OSError as error:
        raise MediaError(
            f"Could not start {Path(arguments[0]).name}. Check its installation."
        ) from error

    try:
        while True:
            if cancel is not None:
                cancel()
            if timeout is not None and monotonic() - started >= timeout:
                raise MediaError(
                    f"{Path(arguments[0]).name} exceeded its processing time limit."
                )
            try:
                output, error_output = process.communicate(timeout=PROCESS_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        terminate(process)
        process.communicate()
        raise
    if process.returncode:
        if stderr is not None:
            end = stderr.seek(0, 2)
            stderr.seek(max(0, end - 2000))
            error_output = stderr.read()
        raise MediaError(command_error(arguments[0], error_output or b""))
    return output


def terminate(process: subprocess.Popen[bytes]) -> None:
    """Stops an owned media child and waits for it to exit."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def command_error(tool: str, output: bytes) -> str:
    """Formats a short explanation of a failed media child process."""
    lines = output.decode("utf-8", errors="replace").strip().splitlines()
    details = "\n".join(lines[-4:])[-800:]
    label = Path(tool).name
    return f"{label} could not process the media. {details}".strip()


def _probe_json(
    arguments: list[str], *, timeout: float = 30, cancel: Callable[[], None] | None = None
) -> dict[str, object]:
    raw = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-of",
            "json",
            *arguments,
        ],
        cancel=cancel,
        timeout=timeout,
    )
    try:
        return cast(dict[str, object], json.loads(raw))
    except ValueError as error:
        raise MediaError("FFprobe returned invalid media information.") from error


def _local_path(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise MediaError("The source video was not found.")
    return str(resolved)


def _nonnegative(value: object) -> int:
    try:
        return max(0, int(str(value)))
    except ValueError:
        return 0


def _decimal(value: object, fallback: float = 0.0) -> float:
    try:
        return float(str(value))
    except ValueError:
        return fallback


def _ratio(value: str) -> Fraction:
    try:
        ratio = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return Fraction(1)
    return ratio if ratio > 0 else Fraction(1)


def _is_video(stream: dict[str, object]) -> bool:
    disposition = cast(dict[str, object], stream.get("disposition", {}))
    return stream.get("codec_type") == "video" and not disposition.get("attached_pic")


def _frame_rate(video: dict[str, object]) -> Fraction:
    for key in ("r_frame_rate", "avg_frame_rate"):
        try:
            rate = Fraction(str(video.get(key)))
        except (ValueError, ZeroDivisionError):
            continue
        if 0 < rate <= 120:
            return rate
    raise MediaError(
        "The source has an unsupported frame rate. Use progressive video up to 120 fps."
    )


def _video_rotation(video: dict[str, object]) -> int:
    for item in cast(list[dict[str, object]], video.get("side_data_list", [])):
        if "rotation" in item:
            rotation = int(float(str(item["rotation"]))) % 360
            if rotation % 90:
                raise MediaError("The source video uses an unsupported rotation angle.")
            return rotation
    return 0


def _scene_boundaries(
    diagnostics: BinaryIO,
    info: VideoInfo,
    cancel: Callable[[], None],
) -> list[int]:
    count = 0
    cuts = []
    for line in diagnostics:
        if match := CLOCK.search(line):
            number, timestamp = int(match[1]), int(match[2])
            if match[3] != b"P":
                raise MediaError(
                    "The source is interlaced. Convert it to progressive video first."
                )
            if number != count or timestamp != number:
                raise MediaError(
                    "The source uses variable frame timing. Convert it to constant frame rate first."
                )
            count += 1
            if count % 120 == 0:
                cancel()
        elif cut := CUT.search(line):
            cuts.append(int(cut[1]))
    if count != info.frames:
        raise MediaError(
            f"FFmpeg decoded {count} frames, but the source reports {info.frames}."
        )
    return cuts


def _partition(cuts: list[int], frames: int, maximum: int) -> list[Shot]:
    start = 0
    shots = []
    for boundary in [*sorted({cut for cut in cuts if 0 < cut < frames}), frames]:
        while start < boundary:
            end = min(start + maximum, boundary)
            shots.append(Shot(len(shots), start, end))
            start = end
    return shots
