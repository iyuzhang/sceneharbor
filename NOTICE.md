# Third-party assets

SceneHarbor source code is MIT-licensed.

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) provides the x4plus model.
  [Real-ESRGAN NCNN Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
  provides the GPU executable. The executable source is BSD-3-Clause. The
  installer downloads the model and executable from the original project, with
  a pinned checksum. The SceneHarbor distribution contains neither.
- [FFmpeg](https://ffmpeg.org/) and ffprobe are system dependencies. Your FFmpeg
  build's license applies to that build and its enabled codecs.
- Public benchmark footage is the *Sintel* teaser © Blender Foundation,
  [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), hosted by the
  [W3C](https://media.w3.org/2010/05/sintel/trailer.mp4). The benchmark excerpts
  resample and degrade copies to create known comparisons. SceneHarbor's MIT
  license does not apply to this footage or derived images.
- The release's `showcase.mp4` and the workbench screenshot use excerpts edited
  from the same teaser under CC BY 3.0. See the
  [showcase processing steps](docs/benchmark.md#showcase-clip).
