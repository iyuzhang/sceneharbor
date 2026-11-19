"""Reproduces paired-reference scene selection measurements with local AI."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

from sceneharbor.media import MODEL_NAME, Engine, Shot, probe, sample_frames
from sceneharbor.quality import (
    EVALUATION_VERSION,
    Candidate,
    FrameScores,
    evaluate_shot,
    score_frames,
)
from sceneharbor.render import PROFILES, profile_frame

if TYPE_CHECKING:
    from sceneharbor.models import Profile


SOURCE_URL = "https://media.w3.org/2010/05/sintel/trailer.mp4"
SOURCE_SHA256 = "b670602fa00934ca27c4351bb0efe7ea7a07fae57284e44226025eeed7c51254"
REFERENCE_SIZE = (384, 216)
RATE = 12
FRAMES = 12


class Case(NamedTuple):
    name: str
    start: float
    transform: str | None
    crf: int | None


class CaseResult(TypedDict):
    scene: str
    start_seconds: float
    degradation: dict[str, str | int | None]
    input_size: list[int]
    selected: Profile
    reason: str
    trial: dict[Profile, Candidate]
    metrics: dict[str, FrameScores]
    analysis_seconds: float
    full_ai_seconds: float


CASES = {
    "pilot": (
        Case("pilot_clean", 3.0, None, None),
        Case("pilot_downscaled", 9.0, "scale=192:108:flags=lanczos", None),
    ),
    "holdout": (
        Case("clean", 14.0, None, None),
        Case("downscaled", 20.0, "scale=192:108:flags=lanczos", None),
        Case("compressed", 24.1, None, 40),
        Case(
            "noisy_downscaled",
            30.0,
            "scale=192:108:flags=lanczos,noise=all_seed=5234:alls=15:allf=t+u",
            None,
        ),
    ),
}
METHODS = ("fixed_lanczos", "fixed_gentle", "fixed_ai", "adaptive")


def main() -> None:
    """Runs the public-video benchmark and emits a machine-readable report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", type=Path, required=True, help="Path to realesrgan-ncnn-vulkan"
    )
    parser.add_argument(
        "--source", type=Path, help="Local copy of the W3C-hosted Sintel trailer"
    )
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--phase", choices=tuple(CASES), default="holdout")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg must be on PATH.")
    if args.gpu is not None and args.gpu < 0:
        parser.error("GPU index must be nonnegative.")
    if args.tile < 0:
        parser.error("Tile size must be nonnegative.")
    if not args.engine.is_file():
        parser.error("--engine must point to a Real-ESRGAN NCNN Vulkan executable.")
    if args.source is not None and not args.source.is_file():
        parser.error("--source must point to a video.")
    engine = Engine(args.engine.resolve(), args.gpu, args.tile)

    with tempfile.TemporaryDirectory(prefix="sceneharbor-proof-") as temporary:
        scratch = Path(temporary)
        source = args.source.resolve() if args.source else _fetch_source(scratch)
        if _sha256(source) != SOURCE_SHA256:
            parser.error("The teaser checksum differs from the benchmark fixture.")
        report = _benchmark(source, engine, scratch, CASES[args.phase], args.phase)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report:
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


def _fetch_source(scratch: Path) -> Path:
    source = scratch / "sintel-teaser.mp4"
    with (
        urllib.request.urlopen(SOURCE_URL, timeout=25) as response,
        source.open("xb") as output,
    ):
        max_bytes = 10 * 1024 * 1024
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("The upstream teaser exceeds the expected size limit.")
        output.write(data)
    return source


def _benchmark(
    source: Path,
    engine: Engine,
    scratch: Path,
    cases: Sequence[Case],
    phase: str,
) -> dict[str, object]:
    records = [_run_case(source, engine, scratch, case) for case in cases]
    methods = {name: _mean_and_worst(records, name) for name in METHODS}
    best_fixed = max(METHODS[:-1], key=lambda method: methods[method]["mean_ssim"])
    return {
        "attribution": "Sintel teaser © Blender Foundation, CC BY 3.0; video hosted by W3C.",
        "evaluation_version": EVALUATION_VERSION,
        "source_url": SOURCE_URL,
        "source_sha256": _sha256(source),
        "engine_sha256": _sha256(engine.binary),
        "model_sha256": {
            suffix: _sha256(engine.binary.parent / "models" / f"{MODEL_NAME}.{suffix}")
            for suffix in ("param", "bin")
        },
        "phase": phase,
        "frames_per_shot": FRAMES,
        "output_size": list(REFERENCE_SIZE),
        "profiles_selected": dict(Counter(record["selected"] for record in records)),
        "methods": methods,
        "best_fixed_by_ssim": best_fixed,
        "adaptive_delta_to_best_fixed_ssim": methods["adaptive"]["mean_ssim"]
        - methods[best_fixed]["mean_ssim"],
        "scenes": records,
    }


def _run_case(source: Path, engine: Engine, scratch: Path, case: Case) -> CaseResult:
    reference_path = scratch / f"{case.name}-reference.mkv"
    degraded_path = scratch / f"{case.name}-degraded.mkv"
    _make_pair(source, reference_path, degraded_path, case)
    reference_info = probe(reference_path)
    input_info = probe(degraded_path)
    indices = _frame_indices(reference_info.frames, input_info.frames)
    shot = Shot(0, 0, len(indices))

    started = time.monotonic()
    evaluation = evaluate_shot(
        degraded_path,
        input_info,
        shot,
        REFERENCE_SIZE,
        engine,
        scratch / case.name,
        _check_cancel,
    )
    analysis_seconds = time.monotonic() - started

    reference = sample_frames(reference_path, reference_info, indices, _check_cancel)
    inputs = sample_frames(degraded_path, input_info, indices, _check_cancel)
    started = time.monotonic()
    ai_frames = engine.upscale(inputs, _check_cancel, target_size=REFERENCE_SIZE)
    full_ai_seconds = time.monotonic() - started
    scores = {
        profile: score_frames(
            reference,
            [
                profile_frame(frame, ai, REFERENCE_SIZE, profile)
                for frame, ai in zip(inputs, ai_frames, strict=True)
            ],
        )
        for profile in PROFILES
    }
    winner = evaluation["recommended"]
    return {
        "scene": case.name,
        "start_seconds": case.start,
        "degradation": {"filter": case.transform, "crf": case.crf},
        "input_size": [input_info.width, input_info.height],
        "selected": winner,
        "reason": evaluation["reason"],
        "trial": evaluation["candidates"],
        "metrics": {
            "fixed_lanczos": scores["original"],
            "fixed_gentle": scores["gentle"],
            "fixed_ai": scores["restore"],
            "adaptive": scores[winner],
        },
        "analysis_seconds": round(analysis_seconds, 4),
        "full_ai_seconds": round(full_ai_seconds, 4),
    }


def _make_pair(source: Path, reference: Path, degraded: Path, case: Case) -> None:
    width, height = REFERENCE_SIZE
    geometry = (
        f"scale={width}:{height}:flags=lanczos,setsar=1,fps={RATE},format=yuv420p"
    )
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1"]
    _run(
        [
            *base,
            "-ss",
            str(case.start),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            geometry,
            "-frames:v",
            str(FRAMES),
            "-c:v",
            "ffv1",
            str(reference),
        ]
    )
    filter_args = ["-vf", case.transform] if case.transform else []
    codec_args = (
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(case.crf)]
        if case.crf
        else ["-c:v", "ffv1"]
    )
    _run([*base, "-i", str(reference), *filter_args, *codec_args, str(degraded)])


def _run(arguments: list[str]) -> None:
    subprocess.run(arguments, check=True, timeout=45)


def _frame_indices(reference_frames: int, input_frames: int) -> list[int]:
    if reference_frames != FRAMES or input_frames != FRAMES:
        raise ValueError(
            f"Expected {FRAMES} aligned frames; found {reference_frames} and {input_frames}."
        )
    return list(range(FRAMES))


def _mean_and_worst(records: list[CaseResult], method: str) -> FrameScores:
    values = [record["metrics"][method] for record in records]
    return {
        "mean_ssim": sum(metric["mean_ssim"] for metric in values) / len(values),
        "worst_ssim": min(metric["worst_ssim"] for metric in values),
        "mean_psnr": sum(metric["mean_psnr"] for metric in values) / len(values),
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_cancel() -> None:
    return None


if __name__ == "__main__":
    main()
