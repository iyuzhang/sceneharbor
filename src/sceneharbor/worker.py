"""Orchestrates persistent analysis and rendering without blocking requests."""

import hashlib
import logging
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from uuid import uuid4

from . import __version__
from .media import Engine, Shot, VideoInfo, detect_shots, output_size, probe
from .models import Job, Profile, Progress, Recipe, ShotRecord, Status, Video
from .quality import evaluate_shot
from .render import mux, render_shot
from .store import Store, fingerprint, write_atomic

LOGGER = logging.getLogger(__name__)


class Cancelled(Exception):
    """Stops a stage at the next safe boundary."""


class WorkflowError(ValueError):
    """Reports an invalid transition or unavailable engine."""


class Worker:
    """Runs one GPU job at a time and preserves committed work."""

    def __init__(self, store: Store, engine: Engine | None, engine_id: str) -> None:
        self.store = store
        self.engine = engine
        self.engine_id = engine_id
        self.stop = Event()
        self.wake = Event()
        self.interrupt = Event()
        self.thread = Thread(target=self._loop, name="sceneharbor-worker")

    def start(self) -> None:
        """Recovers interrupted jobs before claiming queued work."""
        with self.store.lock:
            for job in self.store.list():
                if job.status in {"analyzing", "rendering"}:
                    self.store.save(
                        job,
                        status="paused",
                        pause_requested=False,
                        message="Interrupted by service shutdown. Ready to resume.",
                    )
        self.thread.start()
        self.wake.set()

    def close(self) -> None:
        """Stops active work while retaining checkpoints."""
        self.stop.set()
        self.wake.set()
        self.thread.join()

    def pause(self, job_id: str) -> Job:
        """Requests a pause at the next safe boundary."""
        with self.store.lock:
            job = self.store.get(job_id)
            if job.status == "queued":
                return self.store.save(job, status="paused", message="Paused")
            if job.status not in {"analyzing", "rendering"}:
                raise WorkflowError("Only queued or active work can be paused.")
            self.interrupt.set()
            return self.store.save(job, pause_requested=True, message="Pausing")

    def resume(self, job_id: str) -> Job:
        """Queues a paused or failed stage with its existing work."""
        with self.store.lock:
            job = self.store.get(job_id)
            if job.status not in {"paused", "failed"}:
                raise WorkflowError("Only paused or failed jobs can be resumed.")
            return self._queue(job)

    def render(self, job_id: str) -> Job:
        """Queues export for a reviewed shot plan."""
        with self.store.lock:
            job = self.store.get(job_id)
            if job.status != "review":
                raise WorkflowError("Review the analyzed shots before rendering.")
            progress = Progress(total=job.info.frames if job.info else 0)
            return self._queue(job, stage="render", progress=progress)

    def override(self, job_id: str, index: int, profile: Profile | None) -> Job:
        """Changes one shot without invalidating the other checkpoints."""
        with self.store.lock:
            job = self.store.get(job_id)
            if job.status not in {"review", "completed", "paused", "failed"}:
                raise WorkflowError("Pause the render or wait for analysis before editing a shot.")
            if not 0 <= index < len(job.shots) or not job.shots[index].candidates:
                raise FileNotFoundError("That shot has no scored candidates.")
            shot = job.shots[index]
            if shot.override == profile:
                return job
            shots = job.shots.copy()
            shots[index] = shot.model_copy(update={"override": profile})
            if job.status == "completed":
                return self.store.save(job, shots=shots, status="review", message="Shot changed. Ready to render.")
            return self.store.save(job, shots=shots)

    def apply_recipe(self, job_id: str, recipe: Recipe) -> Job:
        """Reuses verified choices only when the source, engine and cuts match."""
        with self.store.lock:
            job = self.store.get(job_id)
            if job.status != "review":
                raise WorkflowError("A recipe can only be applied to a reviewed source.")
            if (
                recipe.source_sha256 != job.source_fingerprint
                or recipe.engine != self.engine_id
                or recipe.output_size != job.size
                or recipe.options != job.options
            ):
                raise WorkflowError("The recipe belongs to a different source, engine, or export target.")
            ranges = [(shot.index, shot.start, shot.end) for shot in job.shots]
            if ranges != [(shot.index, shot.start, shot.end) for shot in recipe.shots]:
                raise WorkflowError("The source cuts do not match the saved recipe.")
            if any(shot.profile is None for shot in recipe.shots):
                raise WorkflowError("Finish analyzing all shots before exporting a reusable recipe.")
            shots = [
                shot.model_copy(update={
                    "override": saved.profile if saved.profile != shot.recommended else None,
                })
                for shot, saved in zip(job.shots, recipe.shots, strict=True)
            ]
            return self.store.save(job, shots=shots, message="Saved recipe applied")

    def _queue(self, job: Job, **fields: object) -> Job:
        """Persists a transition before waking the worker."""
        if self.engine is None:
            raise WorkflowError("The AI engine is unavailable. Install it and restart SceneHarbor.")
        queued = self.store.save(
            job,
            **fields,
            status="queued",
            message="Waiting for the GPU",
            error=None,
            pause_requested=False,
        )
        self.wake.set()
        return queued

    def _loop(self) -> None:
        """Claims and executes the oldest queued stage."""
        while not self.stop.is_set():
            self.wake.clear()
            job = self._claim()
            if job is None:
                self.wake.wait()
                continue
            self._execute(job)

    def _claim(self) -> Job | None:
        """Transitions queued work to its running state under the workspace lock."""
        if self.engine is None:
            return None
        with self.store.lock:
            if self.stop.is_set():
                return None
            for job in reversed(self.store.list()):
                if job.status != "queued":
                    continue
                self.interrupt.clear()
                return self.store.save(
                    job,
                    status="analyzing" if job.stage == "analyze" else "rendering",
                    message="Checking the video source",
                )
        return None

    def _execute(self, job: Job) -> None:
        """Executes a stage and records its terminal transition."""
        try:
            source = Path(job.path)
            self._check_source(source, job)
            if job.stage == "analyze":
                self._analyze(job, source)
                self._finish(job.id, "review", "Ready for shot review")
            else:
                self._render(job, source)
                self._finish(job.id, "completed", "Export ready")
        except Cancelled:
            self._update(job.id, status="paused", pause_requested=False, message="Paused. Progress was saved.")
        except Exception as error:
            LOGGER.exception("Job %s failed", job.id)
            self._update(job.id, status="failed", pause_requested=False, message="Job failed", error=str(error))

    def _check_source(self, source: Path, job: Job) -> None:
        """Checks that the imported source still matches the saved plan."""
        self._cancel()
        if not source.is_file() or fingerprint(source, self._cancel) != job.source_fingerprint:
            raise WorkflowError("The source was moved or modified. Restore it at the original path or import it again.")
        self._cancel()

    def _analyze(self, job: Job, source: Path) -> None:
        """Detects cuts and commits each candidate comparison independently."""
        if self.engine is None:
            raise WorkflowError("The AI engine is unavailable.")
        directory = self.store.directory(job.id)
        analysis_stamp = directory / "analysis-engine"
        same_engine = analysis_stamp.is_file() and analysis_stamp.read_text() == self.engine_id
        if not same_engine or job.info is None or job.size is None or not job.shots:
            job = self._plan(job, source)
            write_atomic(analysis_stamp, self.engine_id.encode())
        video = _video_info(job)
        size = _target_size(job)
        shots = job.shots.copy()
        for index, shot in enumerate(shots):
            self._cancel()
            previews = directory / "previews" / str(index)
            if shot.candidates and _previews_exist(previews):
                continue
            self._update(job.id, message=f"Testing AI treatments on shot {index + 1} of {len(shots)}")
            evaluated = self._evaluate(source, video, shot, size, self.engine, previews)
            shots[index] = evaluated
            self._update(job.id, shots=shots, progress=Progress(done=index + 1, total=len(shots)))

    def _plan(self, job: Job, source: Path) -> Job:
        """Builds a complete, frame-indexed plan before candidate processing."""
        self._update(job.id, message="Detecting cuts and checking the timeline")
        video = probe(source, self._cancel)
        self._cancel()
        shots = [
            ShotRecord(index=shot.index, start=shot.start, end=shot.end)
            for shot in detect_shots(source, video, self._cancel)
        ]
        dimensions = output_size(video, job.options.height)
        warnings = [
            "Scores are diagnostics against the source. Check fine detail, text and faces before export."
        ]
        if video.subtitle_tracks:
            warnings.append(
                f"Export omits {video.subtitle_tracks} subtitle track(s). Keep the source for subtitles."
            )
        return self._update(
            job.id,
            info=Video.model_validate(asdict(video)),
            size=dimensions,
            shots=shots,
            warnings=warnings,
            progress=Progress(total=len(shots)),
        )

    def _evaluate(
        self,
        source: Path,
        video: VideoInfo,
        shot: ShotRecord,
        size: tuple[int, int],
        engine: Engine,
        previews: Path,
    ) -> ShotRecord:
        """Publishes previews once all profiles for a shot are present."""
        previews.parent.mkdir(exist_ok=True)
        temporary = previews.with_name(f".{previews.name}-{uuid4().hex}")
        temporary.mkdir()
        try:
            result = evaluate_shot(source, video, _shot(shot), size, engine, temporary, self._cancel)
            evaluated = ShotRecord.model_validate({
                **shot.model_dump(),
                "recommended": result["recommended"],
                "reason": result["reason"],
                "candidates": result["candidates"],
            })
            if not _previews_exist(temporary):
                raise WorkflowError("Candidate generation did not produce all previews.")
            self._cancel()
            if previews.exists():
                shutil.rmtree(previews)
            temporary.rename(previews)
            return evaluated
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _render(self, job: Job, source: Path) -> None:
        """Renders changed sections and muxes an atomic final export."""
        if self.engine is None:
            raise WorkflowError("The AI engine is unavailable.")
        directory = self.store.directory(job.id)
        stamp = directory / "analysis-engine"
        if not stamp.is_file() or stamp.read_text() != self.engine_id:
            raise WorkflowError("The AI engine changed since review. Restart with the same engine or import the video again.")
        parts_directory = directory / "parts"
        parts_directory.mkdir(exist_ok=True)
        video = _video_info(job)
        size = _target_size(job)
        parts: list[Path] = []
        for shot in job.shots:
            self._cancel()
            part = self._render_part(job, source, video, shot, size, parts_directory)
            parts.append(part)
            self._update(job.id, progress=Progress(done=shot.end, total=video.frames))
        self._update(job.id, message="Finalizing MP4 and preserving audio tracks")
        self._check_source(source, job)
        source_stat = source.stat()
        mux(source, parts, directory / "output.mp4", video, self._cancel)
        final_stat = source.stat()
        if (source_stat.st_ino, source_stat.st_mtime_ns, source_stat.st_size) != (
            final_stat.st_ino, final_stat.st_mtime_ns, final_stat.st_size,
        ):
            raise WorkflowError("The source changed during muxing. Restore the original file and resume.")

    def _render_part(
        self, job: Job, source: Path, video: VideoInfo, shot: ShotRecord, size: tuple[int, int], directory: Path
    ) -> Path:
        """Verifies an existing segment or commits a replacement."""
        profile = shot.effective_profile
        if self.engine is None or profile is None:
            raise WorkflowError("The shot needs a completed candidate analysis.")
        signature = f"{__version__}:{self.engine_id}:{size}:{job.options.crf}:{shot.start}:{shot.end}:{profile}"
        short_hash = hashlib.sha256(signature.encode()).hexdigest()[:16]
        part = directory / f"{shot.index:06d}-{short_hash}.mp4"
        checksum = part.with_suffix(".sha256")
        if part.is_file() and checksum.is_file() and fingerprint(part) == checksum.read_text():
            return part
        message = f"Rendering shot {shot.index + 1} of {len(job.shots)}"
        self._update(job.id, message=message)
        render_shot(
            source, video, _shot(shot), profile, size, self.engine, part, self._cancel,
            self._progress(job.id, shot.start, video.frames), crf=job.options.crf,
        )
        write_atomic(checksum, fingerprint(part).encode())
        return part

    def _progress(self, job_id: str, start: int, total: int) -> Callable[[int], None]:
        """Creates a throttled progress callback for the current segment."""
        previous = 0.0

        def update(frames: int) -> None:
            """Reports completed frames at most once per second."""
            nonlocal previous
            now = monotonic()
            if now - previous >= 1.0:
                self._update(job_id, progress=Progress(done=start + frames, total=total))
                previous = now

        return update

    def _finish(self, job_id: str, status: Status, message: str) -> None:
        """Completes a stage unless the operator already requested a pause."""
        with self.store.lock:
            self._cancel()
            self.store.save(
                self.store.get(job_id), status=status, pause_requested=False, message=message
            )

    def _update(self, job_id: str, **fields: object) -> Job:
        """Updates a job without losing concurrent pause requests."""
        with self.store.lock:
            return self.store.save(self.store.get(job_id), **fields)

    def _cancel(self) -> None:
        """Checks whether the current stage should return to its checkpoint."""
        if self.stop.is_set() or self.interrupt.is_set():
            raise Cancelled


def _shot(record: ShotRecord) -> Shot:
    """Converts a persisted cut to its media counterpart."""
    return Shot(index=record.index, start=record.start, end=record.end)


def _video_info(job: Job) -> VideoInfo:
    """Converts a checked job into media metadata."""
    if job.info is None:
        raise WorkflowError("Analyze the video before rendering.")
    return VideoInfo(**job.info.model_dump())


def _target_size(job: Job) -> tuple[int, int]:
    """Returns the planned dimensions."""
    if job.size is None:
        raise WorkflowError("No output dimensions were planned.")
    return job.size


def _previews_exist(previews: Path) -> bool:
    """Checks that all three preview candidates are readable files."""
    return all((previews / f"{profile}.jpg").is_file() for profile in ("original", "gentle", "restore"))
