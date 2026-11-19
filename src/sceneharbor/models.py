"""Defines validated job requests and portable restoration recipes."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Profile = Literal["original", "gentle", "restore"]
Status = Literal["queued", "analyzing", "review", "rendering", "paused", "completed", "failed"]
Stage = Literal["analyze", "render"]

PROFILES: tuple[Profile, ...] = ("original", "gentle", "restore")


class Record(BaseModel):
    """Rejects unexpected fields at every entry point."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Options(Record):
    """Validates the export target."""

    height: Literal[360, 480, 720, 1080, 1440, 2160] = 1080
    crf: int = Field(default=18, ge=12, le=28)


class NewJob(Options):
    """Adds an explicitly chosen local media file."""

    path: str = Field(min_length=1, max_length=4096)
    name: str | None = Field(default=None, min_length=1, max_length=240)


class Choice(Record):
    """Overrides a shot or restores the suggested treatment."""

    profile: Profile | None


class Progress(Record):
    """Reports the completed units for the current stage."""

    done: int = 0
    total: int = 0


class Video(Record):
    """Describes the source video timeline."""

    width: int
    height: int
    fps: str
    frames: int
    duration: float
    audio_tracks: int
    video_index: int = 0
    seek_offset: float = 0.0
    sar: str = "1"
    audio_codecs: tuple[str, ...] = ()
    subtitle_tracks: int = 0
    chapters: int = 0


class Candidate(BaseModel):
    """Stores diagnostic signals for one treatment."""

    model_config = ConfigDict(extra="allow", frozen=True, allow_inf_nan=False)

    score: float
    risk: float
    note: str


class ShotRecord(Record):
    """Retains the frame range, evidence, and optional human decision."""

    index: int
    start: int
    end: int
    recommended: Profile | None = None
    override: Profile | None = None
    reason: str = ""
    candidates: dict[Profile, Candidate] = Field(default_factory=dict)

    @property
    def effective_profile(self) -> Profile | None:
        """Returns the treatment that the renderer uses."""
        return self.override or self.recommended


class Job(Record):
    """Persists one source, its decisions, and its resumable stage."""

    id: str
    name: str
    status: Status = "queued"
    stage: Stage = "analyze"
    message: str = "Waiting to analyze"
    error: str | None = None
    path: str
    source_fingerprint: str
    uploaded: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    options: Options
    info: Video | None = None
    size: tuple[int, int] | None = None
    progress: Progress = Field(default_factory=Progress)
    shots: list[ShotRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    pause_requested: bool = False


class ShotView(ShotRecord):
    """Exposes the effective choice to interactive clients."""

    profile: Profile | None


class JobView(Record):
    """Shares job status without exposing local source paths."""

    id: str
    name: str
    status: Status
    stage: Stage
    message: str
    error: str | None
    created_at: datetime
    updated_at: datetime
    info: Video | None
    size: tuple[int, int] | None
    options: Options
    progress: Progress
    shots: list[ShotView]
    output_url: str | None
    recipe_url: str
    warnings: list[str]
    pause_requested: bool

    @classmethod
    def from_job(cls, job: Job) -> "JobView":
        """Builds client links and publicly visible shot details."""
        path = f"/api/jobs/{job.id}"
        data = job.model_dump(exclude={"path", "source_fingerprint", "uploaded", "shots"})
        return cls(
            **data,
            shots=[
                ShotView(**shot.model_dump(), profile=shot.effective_profile)
                for shot in job.shots
            ],
            output_url=f"{path}/output.mp4" if job.status == "completed" else None,
            recipe_url=f"{path}/recipe.json",
        )


class Recipe(Record):
    """Describes a source-bound shot plan that can be applied on another run."""

    format: Literal["sceneharbor.recipe.v1"] = "sceneharbor.recipe.v1"
    app_version: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    engine: str | None
    options: Options
    source: Video | None
    output_size: tuple[int, int] | None
    shots: list[ShotView]
    warnings: list[str]
