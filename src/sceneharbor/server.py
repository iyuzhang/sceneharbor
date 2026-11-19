"""Exposes the review workspace and protects its local process boundary."""

import hmac
import os
import re
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from . import __version__
from .media import Engine
from .models import Choice, Job, JobView, NewJob, Options, Profile
from .store import MEDIA_SUFFIXES, JobActive, Store
from .worker import Worker, WorkflowError

STATIC = Path(__file__).with_name("static")
MAX_UPLOAD = 20 * 1024**3
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; media-src 'self' blob:; "
        "frame-ancestors 'none'; base-uri 'none'; object-src 'none'; form-action 'self'"
    ),
}


def create_app(
    store: Store, engine: Engine | None, engine_id: str, engine_message: str, port: int
) -> FastAPI:
    """Creates the loopback-only API and starts its durable worker."""
    worker = Worker(store, engine, engine_id)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Holds the workspace lock until the worker stops."""
        with store.exclusive_worker():
            worker.start()
            try:
                yield
            finally:
                await run_in_threadpool(worker.close)

    app = FastAPI(title="SceneHarbor", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def local_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Enforces loopback host and same-origin browser access."""
        host = request.headers.get("host", "")
        expected = {"127.0.0.1", "localhost", "[::1]"} if port == 80 else {
            f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"
        }
        origin = request.headers.get("origin")
        fetch_site = request.headers.get("sec-fetch-site")
        if host not in expected or (origin and not hmac.compare_digest(origin, f"http://{host}")):
            return JSONResponse({"detail": "Open SceneHarbor from its local address."}, status_code=403)
        opening_home = request.method == "GET" and request.url.path == "/" and (
            request.headers.get("sec-fetch-mode") == "navigate"
        )
        if fetch_site == "cross-site" and not opening_home:
            return JSONResponse({"detail": "Open SceneHarbor in its own browser tab."}, status_code=403)
        if request.url.path == "/api/uploads":
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse({"detail": "Uploads require Content-Length."}, status_code=411)
            if not length.isdecimal() or int(length) > MAX_UPLOAD + 65536:
                return JSONResponse({"detail": "Upload limit is 20 GiB. Import larger videos by path."}, status_code=413)
        response = await call_next(request)
        response.headers.update(HEADERS)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(JobActive)
    @app.exception_handler(WorkflowError)
    async def conflict(_: Request, error: ValueError) -> JSONResponse:
        """Returns a readable invalid-transition response."""
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(FileNotFoundError)
    async def missing(_: Request, error: FileNotFoundError) -> JSONResponse:
        """Avoids disclosing workspace paths for missing jobs."""
        message = "Job, source, or asset not found." if error.filename else str(error)
        return JSONResponse({"detail": message}, status_code=404)

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        """Opens the local restoration workspace."""
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str | bool]:
        """Reports whether the GPU engine passed its startup check."""
        return {
            "version": __version__,
            "engine_ready": engine is not None,
            "engine_message": engine_message,
            "workspace": store.root.name,
        }

    @app.get("/api/jobs")
    def jobs() -> list[JobView]:
        """Lists persisted jobs and their statuses."""
        return [JobView.from_job(job) for job in store.list()]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> JobView:
        """Returns the full review plan for one job."""
        return JobView.from_job(store.get(_job_id(job_id)))

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def remove(job_id: str) -> Response:
        """Deletes the project from this workspace without touching its external source."""
        store.remove(_job_id(job_id))
        return Response(status_code=204)

    @app.post("/api/jobs", status_code=201)
    def add(request: NewJob) -> JobView:
        """Imports a local path without copying the source."""
        _require_engine(engine)
        try:
            job = store.add(request)
        except (OSError, ValueError) as error:
            raise HTTPException(422, f"Unable to import this source. {error}") from error
        worker.wake.set()
        return JobView.from_job(job)

    @app.post("/api/uploads", status_code=201)
    def upload(
        file: UploadFile, height: Annotated[int, Form()] = 1080, crf: Annotated[int, Form()] = 18
    ) -> JobView:
        """Copies the selected browser file to the private workspace."""
        _require_engine(engine)
        job = _save_upload(store, file, height, crf)
        worker.wake.set()
        return JobView.from_job(job)

    @app.post("/api/jobs/{job_id}/pause")
    def pause(job_id: str) -> JobView:
        """Pauses queued or running work."""
        return JobView.from_job(worker.pause(_job_id(job_id)))

    @app.post("/api/jobs/{job_id}/resume")
    def resume(job_id: str) -> JobView:
        """Restarts from saved progress."""
        return JobView.from_job(worker.resume(_job_id(job_id)))

    @app.post("/api/jobs/{job_id}/render")
    def render(job_id: str) -> JobView:
        """Queues the selected restoration plan for export."""
        return JobView.from_job(worker.render(_job_id(job_id)))

    @app.patch("/api/jobs/{job_id}/shots/{index}")
    def choice(job_id: str, index: int, request: Choice) -> JobView:
        """Applies a per-shot treatment override."""
        return JobView.from_job(worker.override(_job_id(job_id), index, request.profile))

    @app.get("/api/jobs/{job_id}/shots/{index}/{profile}.jpg")
    def candidate(job_id: str, index: int, profile: Profile) -> FileResponse:
        """Returns the same source frame under one candidate treatment."""
        job = store.get(_job_id(job_id))
        if not 0 <= index < len(job.shots) or profile not in job.shots[index].candidates:
            raise HTTPException(404, "This preview is still being generated.")
        path = store.directory(job.id) / "previews" / str(index) / f"{profile}.jpg"
        _require_asset(path)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/jobs/{job_id}/output.mp4")
    def output(job_id: str, download: bool = False) -> FileResponse:
        """Serves the verified render with HTTP range support."""
        job = store.get(_job_id(job_id))
        path = store.directory(job.id) / "output.mp4"
        if job.status != "completed":
            raise HTTPException(409, "The current shot plan has no completed export.")
        _require_asset(path)
        return FileResponse(
            path, media_type="video/mp4",
            filename=f"{Path(job.name).stem or 'sceneharbor'}-restored.mp4" if download else None,
        )

    @app.get("/api/jobs/{job_id}/recipe.json")
    def recipe(job_id: str) -> JSONResponse:
        """Exports the frame ranges, diagnostics, and selected treatments."""
        job = store.get(_job_id(job_id))
        view = JobView.from_job(job).model_dump(mode="json")
        return JSONResponse(
            {
                "format": "sceneharbor.recipe.v1",
                "app_version": __version__,
                "source_sha256": job.source_fingerprint,
                "engine": _analysis_engine(store, job),
                "options": view["options"],
                "source": view["info"],
                "output_size": view["size"],
                "shots": view["shots"],
                "warnings": view["warnings"],
            },
            headers={"Content-Disposition": 'attachment; filename="sceneharbor-recipe.json"'},
        )

    return app


def _job_id(value: str) -> str:
    """Rejects paths and malformed job identifiers."""
    if not JOB_ID.fullmatch(value):
        raise HTTPException(404, "Job not found.")
    return value


def _require_engine(engine: Engine | None) -> None:
    """Reports the installation requirement before accepting new work."""
    if engine is None:
        raise HTTPException(503, "Run sceneharbor install-engine and restart the service.")


def _require_asset(path: Path) -> None:
    """Checks assets before handing them off to the file server."""
    if not path.is_file():
        raise HTTPException(404, "This workspace asset is missing.")


def _save_upload(store: Store, file: UploadFile, height: int, crf: int) -> Job:
    """Validates and registers a file without trusting its client path."""
    name = Path((file.filename or "").replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    if suffix not in MEDIA_SUFFIXES or not file.size or file.size > MAX_UPLOAD:
        raise HTTPException(422, "Choose a supported video file up to 20 GiB.")
    try:
        options = Options.model_validate({"height": height, "crf": crf})
    except ValidationError as error:
        raise HTTPException(422, "Choose a supported output height and CRF between 12 and 28.") from error
    temp: Path | None = None
    try:
        with NamedTemporaryFile(dir=store.root, suffix=suffix, delete=False) as handle:
            temp = Path(handle.name)
            shutil.copyfileobj(file.file, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        return store.add(
            NewJob(path=str(temp), name=name[:240], height=options.height, crf=options.crf),
            upload=temp,
        )
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _analysis_engine(store: Store, job: Job) -> str | None:
    """Returns the identity of the engine that produced the previews."""
    path = store.directory(job.id) / "analysis-engine"
    return path.read_text() if path.is_file() else None
