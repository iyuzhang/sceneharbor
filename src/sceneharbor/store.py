"""Persists job revisions and protects the workspace from concurrent workers."""

import fcntl
import hashlib
import os
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import BinaryIO
from uuid import uuid4

from .models import Job, NewJob, Options

MEDIA_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".webm"})
HASH_CHUNK = 1024 * 1024


class WorkspaceBusy(RuntimeError):
    """Signals that a workspace already has an active worker."""


class JobActive(ValueError):
    """Signals that a queued or running job cannot be removed."""


class Store:
    """Saves revisions with fsync and atomic replacement."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock = RLock()

    @contextmanager
    def exclusive_worker(self) -> Iterator[None]:
        """Prevents two service processes from rendering into one workspace."""
        with (self.root / ".worker.lock").open("a+b") as handle:
            _lock_worker(handle)
            yield

    def directory(self, job_id: str) -> Path:
        """Resolves a generated identifier into its workspace directory."""
        return self.root / job_id

    def add(self, request: NewJob, *, upload: Path | None = None) -> Job:
        """Registers a stable source or moves an upload into the workspace."""
        path = media_path(request.path) if upload is None else upload
        digest = fingerprint(path)
        job_id = uuid4().hex
        directory = self.directory(job_id)
        directory.mkdir(mode=0o700)
        if upload is not None:
            path = directory / f"source{upload.suffix}"
            os.replace(upload, path)
        job = Job(
            id=job_id,
            name=request.name or path.name,
            path=str(path),
            uploaded=upload is not None,
            source_fingerprint=digest,
            options=Options(height=request.height, crf=request.crf),
        )
        return self.save(job)

    def get(self, job_id: str) -> Job:
        """Loads a single validated job record."""
        return Job.model_validate_json((self.directory(job_id) / "job.json").read_bytes())

    def list(self) -> list[Job]:
        """Lists jobs from newest to oldest."""
        with self.lock:
            jobs = [
                Job.model_validate_json(entry.read_bytes())
                for entry in self.root.glob("*/job.json")
                if not entry.parent.name.startswith(".")
            ]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def save(self, job: Job, **fields: object) -> Job:
        """Writes a complete revision and returns that revision."""
        result = job.model_copy(update={**fields, "updated_at": datetime.now(UTC)})
        write_atomic(self.directory(job.id) / "job.json", result.model_dump_json().encode())
        return result

    def remove(self, job_id: str) -> None:
        """Detaches an inactive job before removing its data."""
        with self.lock:
            job = self.get(job_id)
            if job.status in {"queued", "analyzing", "rendering"}:
                raise JobActive("Pause the job before removing it.")
            destination = self.root / f".deleted-{job_id}"
            self.directory(job_id).rename(destination)
        shutil.rmtree(destination)


def write_atomic(path: Path, payload: bytes) -> None:
    """Replaces a file after writing and syncing a temporary sibling."""
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)


def media_path(value: str) -> Path:
    """Accepts readable local files in supported containers."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("Enter an absolute path or choose a file to upload.")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("The source must be a regular file.")
    if path.suffix.lower() not in MEDIA_SUFFIXES:
        raise ValueError("Choose an MP4, MOV, MKV, WebM, AVI, MPEG or TS video.")
    return path


def fingerprint(path: Path, cancel: Callable[[], None] | None = None) -> str:
    """Computes a content digest and detects changes during hashing."""
    original = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            if cancel is not None:
                cancel()
            digest.update(chunk)
    final = path.stat()
    before = (original.st_dev, original.st_ino, original.st_size, original.st_mtime_ns)
    after = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
    if before != after:
        raise ValueError("The file changed while it was being read.")
    return digest.hexdigest()


def _lock_worker(handle: BinaryIO) -> None:
    """Claims an advisory lock for this service process."""
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise WorkspaceBusy("This workspace already has a running SceneHarbor service.") from error
