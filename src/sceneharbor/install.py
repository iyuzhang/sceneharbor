"""Installs the pinned upstream inference engine and probes host dependencies."""

import os
import platform
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image

from .media import MODEL_NAME, Engine, MediaError, run_command
from .store import fingerprint

ASSET = "realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
URL = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/{ASSET}"
ARCHIVE_SHA256 = "e5aa6eb131234b87c0c51f82b89390f5e3e642b7b70f2b9bbe95b6a285a40c96"
ARCHIVE_LIMIT = 64 * 1024**2
BINARY = "realesrgan-ncnn-vulkan"
VULKAN_CHECK_SECONDS = 45


@dataclass(frozen=True)
class EngineCheck:
    """Returns a usable engine and identity, or instructions for recovery."""

    engine: Engine | None
    identity: str
    message: str


def data_root() -> Path:
    """Resolves the application's persistent data directory."""
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "sceneharbor"


def default_binary() -> Path:
    """Returns the executable under the per-user pinned install."""
    return data_root() / "engine" / BINARY


def install(local_archive: Path | None = None) -> Path:
    """Downloads the official Linux archive and installs the required files."""
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("The bundled installer supports Linux x86-64. Use a compatible build with --engine.")
    destination = default_binary().parent
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists():
        _require_files(destination / BINARY)
        return destination / BINARY
    with TemporaryDirectory(dir=destination.parent, prefix=".engine-install-") as temporary:
        folder = Path(temporary)
        archive = local_archive or folder / ASSET
        if local_archive is None:
            _download(archive)
        if archive.stat().st_size > ARCHIVE_LIMIT or fingerprint(archive) != ARCHIVE_SHA256:
            raise RuntimeError("Upstream engine archive failed its SHA-256 verification.")
        staging = folder / "engine"
        _extract(archive, staging)
        _require_files(staging / BINARY)
        try:
            staging.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            _require_files(destination / BINARY)
    return destination / BINARY


def check_engine(binary: Path, gpu: int | None, tile: int) -> EngineCheck:
    """Checks FFmpeg, the H.264 encoder and a real Vulkan inference call."""
    try:
        for program in ("ffmpeg", "ffprobe"):
            if not shutil.which(program):
                raise RuntimeError(f"Install {program} and make it available on PATH.")
        encoders = run_command(["ffmpeg", "-hide_banner", "-encoders"], timeout=8).decode()
        if " libx264 " not in encoders:
            raise RuntimeError("Install an FFmpeg build that includes the libx264 encoder.")
        binary = binary.expanduser().resolve()
        files = _require_files(binary)
        identity = ":".join([*(fingerprint(path) for path in files), f"gpu={gpu}", f"tile={tile}"])
        engine = Engine(binary=binary, gpu=gpu, tile=tile)
        deadline = time.monotonic() + VULKAN_CHECK_SECONDS

        def check_timeout() -> None:
            """Stops a stalled driver during the startup diagnostic."""
            if time.monotonic() > deadline:
                raise MediaError(
                    f"The Vulkan engine did not respond within {VULKAN_CHECK_SECONDS} seconds. "
                    "Let other GPU jobs finish and run sceneharbor doctor again."
                )

        engine.upscale([Image.new("RGB", (32, 32), "#968477")], check_timeout)
        return EngineCheck(engine, identity, "Real-ESRGAN and FFmpeg are ready.")
    except (OSError, MediaError, RuntimeError) as error:
        return EngineCheck(None, "", str(error))


def _require_files(binary: Path) -> tuple[Path, Path, Path]:
    """Checks that the selected executable is paired with its model."""
    model = binary.parent / "models" / MODEL_NAME
    files = (binary, model.with_suffix(".param"), model.with_suffix(".bin"))
    if not all(path.is_file() for path in files) or not os.access(binary, os.X_OK):
        raise RuntimeError("Run sceneharbor install-engine or provide an executable with its x4plus model folder.")
    return files


def _download(archive: Path) -> None:
    """Streams the pinned release into a bounded temporary file."""
    request = urllib.request.Request(URL, headers={"User-Agent": "sceneharbor/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, archive.open("xb") as handle:
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > ARCHIVE_LIMIT:
                raise RuntimeError("Upstream engine archive exceeds 64 MiB.")
            handle.write(chunk)


def _extract(archive: Path, staging: Path) -> None:
    """Extracts named files from the verified archive without trusting member paths."""
    staging.mkdir(mode=0o700)
    (staging / "models").mkdir()
    expected = {
        BINARY: staging / BINARY,
        f"models/{MODEL_NAME}.bin": staging / "models" / f"{MODEL_NAME}.bin",
        f"models/{MODEL_NAME}.param": staging / "models" / f"{MODEL_NAME}.param",
    }
    with ZipFile(archive) as zipped:
        for name, destination in expected.items():
            matches = [entry for entry in zipped.namelist() if entry == name or entry.endswith(f"/{name}")]
            if len(matches) != 1:
                raise RuntimeError(f"Upstream engine archive is missing {name}.")
            with zipped.open(matches[0]) as source, destination.open("xb") as handle:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
    (staging / BINARY).chmod(0o700)
