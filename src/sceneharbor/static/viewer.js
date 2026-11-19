import { PROFILE_NAMES, button, element, frameTime, localUrl, previewUrl, timecode } from "./ui.js";

/** Shows either the image comparison or the rendered clip. */
export function viewer(job, shot, state, actions) {
  const output = localUrl(job.output_url);
  const start = frameTime(shot.start, job.info?.fps);
  const end = frameTime(shot.end, job.info?.fps);
  const heading = element("div", { class: "surface-heading" },
    element("div", { class: "viewer-heading-main" },
      element("h2", {}, state.mode === "video" && output ? "Rendered playback" : "Shot comparison"),
      element("span", { class: "shot-time" }, timecode(start, 2) + " – " + timecode(end, 2)),
    ),
  );
  if (output) {
    heading.append(element("div", { class: "mode-switch", "aria-label": "Viewer mode" },
      button("Compare", () => actions.mode("compare"), "button-quiet", {
        "aria-pressed": String(state.mode !== "video"), "data-focus-key": "mode-compare",
      }),
      button("Playback", () => actions.mode("video"), "button-quiet", {
        "aria-pressed": String(state.mode === "video"), "data-focus-key": "mode-video",
      }),
    ));
  }
  const playing = state.mode === "video" && output;
  const media = playing ? renderedVideo(output, start) : comparison(job, shot, state, actions.split);
  media.dataset.mediaKey = JSON.stringify([
    job.id, shot.index, playing ? output : state.preview, Object.keys(shot.candidates ?? {}),
  ]);
  const help = playing
    ? "Playback starts at the selected shot."
    : state.preview === "original" ? "Original frame"
      : "Drag the divider or use the arrow keys to compare.";
  const resolution = Array.isArray(job.size) ? job.size.join(" × ") + " output" : "";
  return element("section", { class: "surface viewer", "aria-label": "Video viewer" },
    heading, media, element("div", { class: "viewer-footer" }, element("span", {}, help),
      resolution ? element("span", {}, resolution) : null),
  );
}

/** Lists detected shot boundaries with their source preview thumbnails. */
export function timeline(job, activeIndex, onSelect) {
  const strip = element("div", { class: "shot-strip", "aria-label": "Detected shots" });
  for (const [position, shot] of job.shots.entries()) {
    const label = "Shot " + String(position + 1).padStart(2, "0");
    const start = frameTime(shot.start, job.info?.fps);
    const tile = element("button", {
      type: "button",
      class: "shot-card",
      "aria-pressed": String(shot.index === activeIndex),
      "aria-label": label + " at " + timecode(start, 2),
      "data-focus-key": "shot-" + shot.index,
    },
    shot.candidates?.original
      ? element("img", { class: "shot-thumb", src: previewUrl(job, shot, "original"), alt: "", loading: "lazy" })
      : element("span", { class: "shot-thumb", "aria-hidden": "true" }),
    element("span", { class: "shot-card-body" },
      element("span", { class: "shot-card-name" }, label),
      element("span", { class: "shot-card-time" }, timecode(start)),
    ));
    tile.addEventListener("click", () => onSelect(shot.index));
    strip.append(tile);
  }
  return element("section", { class: "surface timeline", "aria-label": "Shot timeline" },
    element("header", { class: "surface-heading" }, element("h2", {}, "Shot timeline"),
      element("span", { class: "shot-time" }, job.shots.length + (job.shots.length === 1 ? " shot" : " shots"))),
    strip,
  );
}

function comparison(job, shot, state, setSplit) {
  const stage = element("div", { class: "media-stage" });
  if (!shot.candidates?.original) {
    stage.append(element("p", { class: "preview-error" }, "Preparing this shot's preview."));
    return stage;
  }
  const source = element("img", {
    src: previewUrl(job, shot, "original"), alt: "Source frame from the selected shot",
  });
  stage.append(source);
  watchMedia(source, stage);
  stage.append(element("span", { class: "compare-label compare-label-source" }, "Source"));
  if (state.preview === "original" || !shot.candidates[state.preview]) return stage;

  const candidate = element("img", {
    class: "compare-candidate", src: previewUrl(job, shot, state.preview),
    alt: PROFILE_NAMES[state.preview] + " restoration of the selected frame",
  });
  const slider = element("input", {
    type: "range", class: "compare-range", min: 0, max: 100, value: state.split,
    "aria-label": "Reveal source frame", "data-focus-key": "compare-divider",
  });
  watchMedia(candidate, stage);
  setPosition();
  slider.addEventListener("input", setPosition);
  stage.append(candidate, element("span", { class: "compare-label compare-label-candidate" }, PROFILE_NAMES[state.preview]),
    element("span", { class: "compare-divider", "aria-hidden": "true" }), slider);
  return stage;

  function setPosition() {
    const position = Number(slider.value);
    stage.style.setProperty("--split", position + "%");
    slider.setAttribute("aria-valuetext", position + "% source frame");
    setSplit(position);
  }
}

function renderedVideo(url, start) {
  const stage = element("div", { class: "media-stage" });
  const player = element("video", {
    src: url, controls: true, preload: "metadata", playsinline: true,
    "aria-label": "Rendered video playback",
  });
  player.addEventListener("loadedmetadata", () => {
    if (start > 0 && start < player.duration) player.currentTime = start;
  }, { once: true });
  watchMedia(player, stage);
  stage.append(player);
  return stage;
}

function watchMedia(media, stage) {
  media.addEventListener("error", () => {
    if (stage.querySelector(".preview-error")) return;
    stage.append(element("p", { class: "preview-error", role: "status" }, "Media preview unavailable."));
  }, { once: true });
}
