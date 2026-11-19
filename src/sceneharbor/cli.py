"""Runs the local workspace, diagnostics, and unattended batch restoration."""

import argparse
import sys
from pathlib import Path
from time import sleep

import uvicorn

from . import __version__
from .install import check_engine, data_root, default_binary, install
from .models import Job, NewJob, Options, Recipe
from .server import create_app
from .store import Store
from .worker import Worker

DEFAULT_PORT = 8752


def main() -> None:
    """Dispatches validated command-line options and reports actionable errors."""
    args = _parser().parse_args()
    try:
        return_code = _dispatch(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"SceneHarbor: {error}", file=sys.stderr)
        return_code = 1
    except KeyboardInterrupt:
        print("\nPaused. Use 'sceneharbor jobs' and 'sceneharbor batch --resume JOB_ID' to continue.")
        return_code = 130
    if return_code:
        raise SystemExit(return_code)


def _parser() -> argparse.ArgumentParser:
    """Defines the workspace and supported commands."""
    parser = argparse.ArgumentParser(prog="sceneharbor", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--workspace", type=Path, default=data_root() / "workspace")
    parser.add_argument("--engine", type=Path, default=default_binary())
    parser.add_argument("--gpu", type=int, help="Vulkan GPU index")
    parser.add_argument("--tile", type=int, default=0, help="0 for automatic or a positive tile size")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Open the local browser workspace")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    installer = commands.add_parser("install-engine", help="Install the pinned upstream Real-ESRGAN release")
    installer.add_argument("--archive", type=Path, help="Install a local copy of the pinned release ZIP")
    commands.add_parser("doctor", help="Check FFmpeg and run a small GPU inference")
    commands.add_parser("jobs", help="List project IDs and statuses")
    batch = commands.add_parser("batch", help="Analyze and export using the recommended profiles")
    batch.add_argument("videos", nargs="*", type=Path)
    batch.add_argument("--height", type=int, default=1080, choices=[360, 480, 720, 1080, 1440, 2160])
    batch.add_argument("--crf", type=int, default=18)
    batch.add_argument("--resume", action="append", default=[], metavar="JOB_ID")
    batch.add_argument("--recipe", type=Path, help="Apply a saved recipe to its matching source")
    return parser


def _dispatch(args: argparse.Namespace) -> int:
    """Runs the requested command with shared engine configuration."""
    if args.command == "install-engine":
        print(install(args.archive))
        return 0
    store = Store(args.workspace)
    if args.command == "jobs":
        for job in store.list():
            print(f"{job.id}\t{job.status}\t{job.name}")
        return 0
    if args.tile != 0 and args.tile < 32:
        raise ValueError("Choose an automatic tile size (0) or a tile of at least 32 pixels.")
    if args.gpu is not None and args.gpu < 0:
        raise ValueError("Choose a nonnegative Vulkan GPU index.")
    check = check_engine(args.engine, args.gpu, args.tile)
    if args.command == "doctor":
        print(check.message)
        return int(check.engine is None)
    if args.command == "serve":
        if not 1024 <= args.port <= 65535:
            raise ValueError("Choose a port from 1024 to 65535.")
        print(f"SceneHarbor is available at http://127.0.0.1:{args.port}", flush=True)
        app = create_app(store, check.engine, check.identity, check.message, args.port)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, proxy_headers=False))
        server.run()
        return int(not server.started)
    if check.engine is None:
        raise RuntimeError(check.message)
    return _batch(store, Worker(store, check.engine, check.identity), args)


def _batch(store: Store, worker: Worker, args: argparse.Namespace) -> int:
    """Registers batch jobs and waits for their resumable exports."""
    if not args.videos and not args.resume:
        raise ValueError("Pass a video path or --resume JOB_ID.")
    recipe = _read_recipe(args)
    options = recipe.options if recipe else Options.model_validate({"height": args.height, "crf": args.crf})
    requests = [
        NewJob(path=str(path.expanduser().absolute()), height=options.height, crf=options.crf)
        for path in args.videos
    ]
    with store.exclusive_worker():
        existing = {job.id for job in store.list()}
        if missing := set(args.resume) - existing:
            raise ValueError(f"Unknown job ID: {', '.join(sorted(missing))}")
        if recipe is not None and args.resume:
            _check_replay_resume(store.get(args.resume[0]), recipe, worker.engine_id)
        added = [store.add(request) for request in requests]
        ids = dict.fromkeys([*(job.id for job in added), *args.resume])
        worker.start()
        try:
            for job_id in args.resume:
                if store.get(job_id).status in {"paused", "failed"}:
                    worker.resume(job_id)
            return _watch(store, worker, list(ids), recipe)
        finally:
            worker.close()


def _read_recipe(args: argparse.Namespace) -> Recipe | None:
    """Reads a bounded exported recipe for one source."""
    path: Path | None = args.recipe
    if path is None:
        return None
    if len(args.videos) + len(args.resume) != 1:
        raise ValueError("Pass exactly one source or one job ID when using --recipe.")
    with path.open("rb") as handle:
        payload = handle.read(16 * 1024**2 + 1)
    if len(payload) > 16 * 1024**2:
        raise ValueError("The recipe exceeds 16 MiB.")
    return Recipe.model_validate_json(payload)


def _check_replay_resume(job: Job, recipe: Recipe, engine_id: str) -> None:
    """Resumes export only if the job already uses the supplied recipe."""
    if job.source_fingerprint != recipe.source_sha256 or job.options != recipe.options or engine_id != recipe.engine:
        raise ValueError("The saved recipe belongs to a different source, engine, or export target.")
    if job.stage == "render" and job.status != "review":
        current = [(shot.start, shot.end, shot.effective_profile) for shot in job.shots]
        saved = [(shot.start, shot.end, shot.profile) for shot in recipe.shots]
        if current != saved or job.size != recipe.output_size:
            raise ValueError("This render uses a different recipe. Apply the recipe to a fresh import.")


def _watch(store: Store, worker: Worker, ids: list[str], recipe: Recipe | None) -> int:
    """Advances reviewed jobs to export and prints results."""
    previous: dict[str, str] = {}
    while True:
        jobs = [store.get(job_id) for job_id in ids]
        for job in jobs:
            _report(job, previous)
            if job.status == "review":
                if recipe is not None:
                    worker.apply_recipe(job.id, recipe)
                worker.render(job.id)
        if all(job.status in {"completed", "failed", "paused"} for job in jobs):
            break
        sleep(0.5)
    for job in jobs:
        if job.status == "completed":
            print(store.directory(job.id) / "output.mp4")
        elif job.error:
            print(f"{job.name}: {job.error}", file=sys.stderr)
    return int(any(job.status != "completed" for job in jobs))


def _report(job: Job, previous: dict[str, str]) -> None:
    """Reports state transitions without flooding a terminal."""
    key = f"{job.status} {job.message}"
    if previous.get(job.id) != key:
        print(f"{job.id} {job.name} · {job.message}", file=sys.stderr, flush=True)
        previous[job.id] = key


if __name__ == "__main__":
    main()
