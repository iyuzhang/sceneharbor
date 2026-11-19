from __future__ import annotations

from collections.abc import Callable, Sequence
from math import log10
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .media import MODEL_SCALE, Engine, Shot, VideoInfo, sample_frames
from .render import PROFILES, profile_frame

if TYPE_CHECKING:
    from .models import Profile

PROFILE_LABELS: Final[dict[Profile, str]] = {
    "original": "Original",
    "gentle": "Gentle",
    "restore": "Restore",
}
EVALUATION_VERSION = 1
PROXY_MAX_EDGE = 256
PREVIEW_MAX_EDGE = 1_280
MAX_ANALYSIS_MODEL_PIXELS = 160_000_000
MIN_PROXY_MARGIN = 0.12
MIN_BLOCK_EXCESS = 0.004
MIN_BLOCK_REDUCTION = 0.20
MIN_BLOCK_POINT_GAIN = 0.0015
MAX_RGB_DISTANCE = 0.047
MAX_LUMA_SHIFT = 0.021
MAX_ADDED_FLICKER = 0.014
STATIC_LUMA_DIFFERENCE = 0.07
SSIM_WINDOW = 7
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])


class Candidate(TypedDict):
    score: float
    risk: float
    note: str
    trial_ssim: float
    trial_worst_ssim: float
    trial_psnr: float
    source_distance: float
    luma_shift: float
    added_flicker: float
    block_excess: float | None
    eligible: bool


class Evaluation(TypedDict):
    version: int
    recommended: Profile
    reason: str
    candidates: dict[Profile, Candidate]
    sample_frames: list[int]
    preview_frame: int


class FrameScores(TypedDict):
    mean_ssim: float
    worst_ssim: float
    mean_psnr: float


def evaluate_shot(
    source: Path,
    info: VideoInfo,
    shot: Shot,
    size: tuple[int, int],
    engine: Engine,
    dest: Path,
    cancel: Callable[[], None],
) -> Evaluation:
    """Evaluates rendered candidates and saves synchronized representative frames."""
    indices = _sample_indices(shot)
    actual = sample_frames(source, info, indices, cancel)
    # Use a second, smaller copy of each source frame as a pseudo-reference.
    references = [
        _resized(frame, _bounded_size(frame.size, PROXY_MAX_EDGE)) for frame in actual
    ]
    inputs = [
        _resized(frame, (max(1, frame.width // 2), max(1, frame.height // 2)))
        for frame in references
    ]
    actual_ai = _scaled_ai(actual, size, engine, cancel)
    trial_ai = _scaled_ai(inputs, references[0].size, engine, cancel)
    native = size == actual[0].size
    upscale = size[0] > actual[0].width or size[1] > actual[0].height
    preview_index = _preview_index(actual, indices, (shot.start + shot.end - 1) // 2)
    dest.mkdir(parents=True, exist_ok=True)
    candidates = _evaluate_candidates(
        actual,
        actual_ai,
        references,
        inputs,
        trial_ai,
        size,
        indices,
        preview_index,
        dest,
        cancel,
        native,
    )
    recommended, reason = _choose_candidate(candidates, native=native, upscale=upscale)
    return {
        "version": EVALUATION_VERSION,
        "recommended": recommended,
        "reason": reason,
        "candidates": candidates,
        "sample_frames": indices,
        "preview_frame": indices[preview_index],
    }


def score_frames(
    references: Sequence[Image.Image], candidates: Sequence[Image.Image]
) -> FrameScores:
    """Computes luma SSIM and PSNR against aligned known-reference images."""
    if not references or len(references) != len(candidates):
        raise ValueError("Provide matching nonempty reference and candidate frames.")
    ssim = []
    psnr = []
    for reference, candidate in zip(references, candidates, strict=True):
        reference_luma = _luma(reference)
        candidate_luma = _luma(_resized(candidate, reference.size))
        difference = reference_luma - candidate_luma
        error = float(np.mean(difference * difference))
        ssim.append(_ssim(reference_luma, candidate_luma))
        psnr.append(-10 * log10(error) if error > 1e-10 else 100.0)
    return {
        "mean_ssim": float(np.mean(ssim)),
        "worst_ssim": min(ssim),
        "mean_psnr": float(np.mean(psnr)),
    }


def _sample_indices(shot: Shot) -> list[int]:
    midpoint = (shot.start + shot.end - 1) // 2
    return sorted({shot.start, midpoint, min(midpoint + 1, shot.end - 1), shot.end - 1})


def _preview_index(
    frames: Sequence[Image.Image], indices: list[int], midpoint: int
) -> int:
    scores = []
    for image in frames:
        luma = _luma(_resized(image, _bounded_size(image.size, 96)))
        low, high = np.percentile(luma, [10, 95])
        texture = (
            float(np.mean(np.abs(np.diff(luma, axis=1)))) if luma.shape[1] > 1 else 0.0
        )
        score = float(high - low) + texture if np.mean(luma) >= 0.035 else 0.0
        scores.append(score)
    return max(
        range(len(frames)),
        key=lambda index: (scores[index], -abs(indices[index] - midpoint)),
    )


def _bounded_size(size: tuple[int, int], limit: int) -> tuple[int, int]:
    width, height = size
    factor = min(1.0, limit / max(size))
    return max(1, round(width * factor)), max(1, round(height * factor))


def _resized(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return (
        image.convert("RGB")
        if image.size == size
        else image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    )


def _scaled_ai(
    inputs: list[Image.Image],
    size: tuple[int, int],
    engine: Engine,
    cancel: Callable[[], None],
) -> list[Image.Image]:
    images = []
    frame_pixels = inputs[0].width * inputs[0].height * MODEL_SCALE**2
    capacity = max(1, MAX_ANALYSIS_MODEL_PIXELS // frame_pixels)
    for start in range(0, len(inputs), capacity):
        batch = inputs[start : start + capacity]
        rendered = engine.upscale(batch, cancel, target_size=size)
        if len(rendered) != len(batch):
            raise RuntimeError("The AI engine returned an unexpected number of frames.")
        images.extend(rendered)
    return images


def _evaluate_candidates(
    actual: list[Image.Image],
    actual_ai: list[Image.Image],
    references: list[Image.Image],
    inputs: list[Image.Image],
    trial_ai: list[Image.Image],
    size: tuple[int, int],
    indices: list[int],
    preview_index: int,
    dest: Path,
    cancel: Callable[[], None],
    native: bool,
) -> dict[Profile, Candidate]:
    candidates: dict[Profile, Candidate] = {}
    anchor = [
        _rgb(_resized(profile_frame(frame, None, size, "original"), reference.size))
        for frame, reference in zip(actual, references, strict=True)
    ]
    for profile in PROFILES:
        cancel()
        trial = [
            profile_frame(frame, ai, reference.size, profile)
            for frame, ai, reference in zip(inputs, trial_ai, references, strict=True)
        ]
        scored = score_frames(references, trial)
        assessed_frames = []
        block_values = []
        for position, (frame, ai, reference) in enumerate(
            zip(actual, actual_ai, references, strict=True)
        ):
            rendered = profile_frame(frame, ai, size, profile)
            assessed_frames.append(_rgb(_resized(rendered, reference.size)))
            if native:
                block_values.append(_block_excess(rendered))
            if position == preview_index:
                _resized(rendered, _bounded_size(size, PREVIEW_MAX_EDGE)).save(
                    dest / f"{profile}.jpg",
                    quality=91,
                    optimize=True,
                )
        block_excess = float(np.median(block_values)) if block_values else None
        candidates[profile] = _measure_candidate(
            profile, scored, anchor, assessed_frames, indices, block_excess
        )
    return candidates


def _measure_candidate(
    profile: Profile,
    scored: FrameScores,
    anchor: list[NDArray[np.float64]],
    assessed: list[NDArray[np.float64]],
    indices: list[int],
    block_excess: float | None,
) -> Candidate:
    source_distance, luma_shift = _source_deltas(anchor, assessed)
    added_flicker = _added_flicker(anchor, assessed, indices)
    components = {
        "source detail changed substantially": source_distance / MAX_RGB_DISTANCE,
        "brightness shifted substantially": luma_shift / MAX_LUMA_SHIFT,
        "added variation between neighboring frames": added_flicker / MAX_ADDED_FLICKER,
    }
    limiter, max_risk = max(components.items(), key=lambda item: item[1])
    eligible = max_risk <= 1.0
    if profile == "original":
        note = "Lanczos reference. No learned enhancement."
    elif eligible:
        note = "Within source-preservation limits on the sampled frames."
    else:
        note = f"Review recommended: {limiter}."
    return {
        "score": round(
            100 * (0.75 * scored["mean_ssim"] + 0.25 * scored["worst_ssim"]), 4
        ),
        "risk": round(100 * min(max_risk, 1.0), 2),
        "note": note,
        "trial_ssim": round(scored["mean_ssim"], 6),
        "trial_worst_ssim": round(scored["worst_ssim"], 6),
        "trial_psnr": round(scored["mean_psnr"], 3),
        "source_distance": round(source_distance, 6),
        "luma_shift": round(luma_shift, 6),
        "added_flicker": round(added_flicker, 6),
        "block_excess": round(block_excess, 6) if block_excess is not None else None,
        "eligible": eligible,
    }


def _choose_candidate(
    candidates: dict[Profile, Candidate],
    *,
    native: bool,
    upscale: bool,
) -> tuple[Profile, str]:
    enhanced = [
        profile
        for profile in PROFILES
        if profile != "original" and candidates[profile]["eligible"]
    ]
    if not enhanced:
        return (
            "original",
            "Both AI choices exceeded the source or motion limits on the sampled frames.",
        )
    if native:
        return _choose_native(candidates, enhanced)
    if not upscale:
        return (
            "original",
            "The output is smaller than the source. A resize preserves the sampled detail.",
        )
    baseline = candidates["original"]["score"]
    best: Profile = max(enhanced, key=lambda profile: candidates[profile]["score"])
    improvement = candidates[best]["score"] - baseline
    if improvement < MIN_PROXY_MARGIN:
        return (
            "original",
            "The downsample trial found too little improvement to justify changing this shot.",
        )
    return best, (
        f"{PROFILE_LABELS[best]} improved the downsample trial by {improvement:.2f} SSIM points "
        "and stayed within the source and motion limits."
    )


def _choose_native(
    candidates: dict[Profile, Candidate],
    enhanced: Sequence[Profile],
) -> tuple[Profile, str]:
    baseline = candidates["original"]["block_excess"] or 0.0
    if baseline < MIN_BLOCK_EXCESS:
        return (
            "original",
            "The shot is already at the output size and has little 8-pixel blocking in the samples.",
        )
    for profile in enhanced:
        blocks = candidates[profile]["block_excess"] or 0.0
        gain = baseline - blocks
        if gain >= max(MIN_BLOCK_POINT_GAIN, MIN_BLOCK_REDUCTION * baseline):
            return profile, (
                f"{PROFILE_LABELS[profile]} reduced the sampled 8-pixel blocking signal by "
                f"{100 * gain / baseline:.0f}% and stayed within source and motion limits."
            )
    return (
        "original",
        "No AI treatment sufficiently reduced the sampled blocking at this output size.",
    )


def _block_excess(image: Image.Image) -> float:
    width, height = image.size
    left = max(0, (width - 512) // 16 * 8)
    top = max(0, (height - 512) // 16 * 8)
    sample = _luma(
        image.crop((left, top, min(width, left + 512), min(height, top + 512)))
    )
    if min(sample.shape) < 16:
        return 0.0
    horizontal = np.abs(np.diff(sample, axis=1))
    vertical = np.abs(np.diff(sample, axis=0))
    boundaries = (np.mean(horizontal[:, 7::8]) + np.mean(vertical[7::8, :])) / 2
    near_edges = (
        np.mean(horizontal[:, 6::8])
        + np.mean(horizontal[:, 8::8])
        + np.mean(vertical[6::8, :])
        + np.mean(vertical[8::8, :])
    ) / 4
    return max(0.0, float(boundaries - near_edges))


def _source_deltas(
    anchor: list[NDArray[np.float64]],
    assessed: list[NDArray[np.float64]],
) -> tuple[float, float]:
    differences = [
        candidate - reference
        for reference, candidate in zip(anchor, assessed, strict=True)
    ]
    return (
        max(float(np.mean(np.abs(difference))) for difference in differences),
        max(
            float(abs(np.mean(difference @ LUMA_WEIGHTS))) for difference in differences
        ),
    )


def _added_flicker(
    anchor: list[NDArray[np.float64]],
    assessed: list[NDArray[np.float64]],
    indices: list[int],
) -> float:
    increments = [0.0]
    for index in range(1, len(indices)):
        if indices[index] - indices[index - 1] != 1:
            continue
        source_change = np.abs((anchor[index] - anchor[index - 1]) @ LUMA_WEIGHTS)
        output_change = np.abs((assessed[index] - assessed[index - 1]) @ LUMA_WEIGHTS)
        static = source_change <= STATIC_LUMA_DIFFERENCE
        if np.any(static):
            added = np.maximum(output_change[static] - source_change[static], 0.0)
            increments.append(float(np.mean(added)))
    return max(increments)


def _rgb(image: Image.Image) -> NDArray[np.float64]:
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def _luma(image: Image.Image) -> NDArray[np.float64]:
    return _rgb(image) @ LUMA_WEIGHTS


def _ssim(reference: NDArray[np.float64], candidate: NDArray[np.float64]) -> float:
    reference_mean = _box_mean(reference)
    candidate_mean = _box_mean(candidate)
    reference_variance = np.maximum(
        _box_mean(reference * reference) - reference_mean**2, 0.0
    )
    candidate_variance = np.maximum(
        _box_mean(candidate * candidate) - candidate_mean**2, 0.0
    )
    covariance = _box_mean(reference * candidate) - reference_mean * candidate_mean
    numerator = (2 * reference_mean * candidate_mean + 0.01**2) * (
        2 * covariance + 0.03**2
    )
    denominator = (reference_mean**2 + candidate_mean**2 + 0.01**2) * (
        reference_variance + candidate_variance + 0.03**2
    )
    return float(np.mean(numerator / denominator))


def _box_mean(values: NDArray[np.float64]) -> NDArray[np.float64]:
    radius = SSIM_WINDOW // 2
    padded = np.pad(values, radius, mode="edge")
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)))
    width = SSIM_WINDOW
    area = (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )
    return area / (width * width)
