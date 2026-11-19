# Measuring scene selection

SceneHarbor compares three treatments for each shot. **Original** resizes with Lanczos. **Gentle** blends 35% Real-ESRGAN output with the Lanczos image. **Restore** uses the full Real-ESRGAN output resized to the same dimensions. Preview frames and exports use the same treatments.

## How the app decides

SceneHarbor samples the first, middle, and last frames of a shot, including a neighboring pair near the middle when available. For the downsample trial, it uses a smaller version of each source frame as a temporary reference, reduces it again, renders all three treatments and compares their luma SSIM. The score puts 75% weight on the mean and 25% on the worst sampled frame.

The app also renders each treatment from the original sampled frames. It measures per-channel RGB deviation from the Lanczos candidate, a global brightness shift, and added temporal variation in pixels that changed little between neighboring source frames. The risk indicator shows the highest fraction of these limits. For upscaling, SceneHarbor picks an eligible AI treatment when the trial improves by at least 0.12 SSIM points on the 0–100 scale. At the source dimensions, it first looks for measurable blocking at 8-pixel boundaries. An AI treatment must reduce that signal by at least 20% and 0.0015 in normalized luma, and pass the same preservation limits. Review the previews and override the recommendation for any shot.

The trial estimates how the renderer handles an extra reduction of this shot. Real degradation can differ from that reduction. A regular pixel pattern can also resemble blocking. A high trial score provides no information about genuine detail that was already absent from the source, and sparse motion samples can miss flicker elsewhere in the shot.

## Reproduce the comparison

Use the same Real-ESRGAN NCNN Vulkan installation that powers the app and run from the repository root:

```bash
uv run python scripts/benchmark.py \
  --engine /path/to/realesrgan-ncnn-vulkan \
  --report benchmark.json
```

The script downloads the 4.2 MB [Sintel teaser hosted by W3C](https://media.w3.org/2010/05/sintel/trailer.mp4). Pass `--source ./trailer.mp4` to reuse a local copy. Sintel is © Blender Foundation and is distributed under [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/). The video also appears in the [W3C media demonstration](https://www.w3.org/2010/05/video/mediaevents.html). SceneHarbor distributes the measurement script and regenerates the clip derivatives locally.

The holdout run uses four disjoint one-second windows at 12 fps and 384×216: clean, Lanczos-downscaled by 2, H.264 CRF 40, and downscaled with seeded temporal noise. The optional `--phase pilot` uses two earlier windows as a development check. The source is a compressed teaser. Its decoded, resized frames serve as the paired references for these controlled transformations.

For a shot, selection sees only the degraded video. The benchmark then runs all three treatments on every frame and compares them against the retained reference. It reports mean luma SSIM, worst frame SSIM, mean luma PSNR, selected treatments and processing times alongside source, executable and model hashes. Exact frame matches use a 100 dB reporting ceiling. The adaptive column uses the selected treatment's fully rendered frames. The explicit fixed-treatment comparisons also let you see what the same model can do without the selector.

The paired reference measures reconstruction under these controlled transformations. Report findings with the tested footage, models, and settings. Real footage may have different degradation and needs visual review.

## Recorded holdout run

Mean luma SSIM on 48 frames of the public Sintel teaser. An RTX 4060 Ti ran the official Real-ESRGAN NCNN Vulkan `realesrgan-x4plus` model.

| Source condition | Selected | Fixed Lanczos | Fixed Gentle | Fixed Restore | Adaptive |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean | Original | 1.000000 | 0.979993 | 0.926830 | 1.000000 |
| Downscaled | Gentle | 0.904666 | 0.907885 | 0.862551 | 0.907885 |
| Compressed | Original | 0.768128 | 0.788299 | 0.792012 | 0.768128 |
| Noisy and downscaled | Gentle | 0.788351 | 0.880922 | 0.939780 | 0.880922 |
| **Mean** | | **0.865286** | **0.889275** | **0.880293** | **0.889234** |

Adaptive kept the clean clip unchanged and improved the two reduced-resolution clips over Lanczos. Fixed Gentle had the highest mean by 0.000041, and fixed Restore led on the two heaviest degradations. The source-sized compression guard missed a case where enhancement improved SSIM. The worst-frame SSIM was 0.731703 for Adaptive, 0.751792 for fixed Gentle, and 0.757524 for fixed Restore. This run shows when the selector preserves known source detail and where its conservative decision loses reconstruction quality. The analysis has its own cost, so this test cannot establish an end-to-end speed advantage.

## Showcase clip

The public walkthrough uses four consecutive three-second excerpts from the Sintel teaser. Each clip keeps its original audio. The source offsets are 13.2, 18.5, 29.0, and 40.2 seconds. The excerpts share a 384×216 output grid at 24 fps. The second was encoded at H.264 CRF 38 before assembly. The third was reduced to 192×108, treated with seeded temporal noise (`noise=all_seed=5234:alls=10:allf=t+u`), then resized to the common grid. The joined clip was encoded at H.264 CRF 13 with one AAC audio track. This separate visual example has no retained reference in the app.
