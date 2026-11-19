# Contributing

Use Linux x86-64, Python 3.11+, FFmpeg with libx264, and a Vulkan GPU for the
inference path.

```sh
uv sync
uv run sceneharbor install-engine
uv run sceneharbor doctor
uv run sceneharbor serve
```

Use a separate --workspace when developing against a running instance.
Use publicly shareable or generated video when reporting a bug. Include the
SceneHarbor version, FFmpeg version, operating system, GPU, and the export
recipe when possible. Recipes include the source file's SHA-256. Check for any
private information you do not want to share.

A media change should be verified from import through export with real frames,
including the reported case. See [the benchmark](docs/benchmark.md) for a
repeatable reference comparison.
