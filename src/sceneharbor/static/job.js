import { PROFILE_NAMES, STATUS_NAMES, button, element, frameRate, localUrl, timecode } from "./ui.js";
import { timeline, viewer } from "./viewer.js";

const SCORE_FORMAT = new Intl.NumberFormat("en", { maximumFractionDigits: 3 });

/** Renders the selected job and its review tools. */
export function jobView(job, state, actions) {
  const shot = findShot(job, state.shot);
  const body = shot
    ? element("div", { class: "studio-grid" },
      element("div", { class: "studio-left" }, viewer(job, shot, state, actions),
        timeline(job, shot.index, actions.selectShot)),
      inspector(job, shot, state, actions),
    )
    : element("div", { class: "surface pending-viewer" }, pendingText(job));

  return [
    jobHeader(job, actions, state.busy),
    statusOverview(job),
    body,
    Array.isArray(job.warnings) && job.warnings.length ? warningList(job.warnings) : null,
  ].filter(Boolean);
}

/** Finds the selected shot or moves to the first available shot. */
export function findShot(job, index) {
  const shots = Array.isArray(job.shots) ? job.shots : [];
  return shots.find((shot) => shot.index === index) ?? shots[0] ?? null;
}

/** Resolves a known render profile from the selected shot. */
export function shotProfile(shot) {
  if (Object.hasOwn(PROFILE_NAMES, shot?.profile)) return shot.profile;
  if (Object.hasOwn(PROFILE_NAMES, shot?.recommended)) return shot.recommended;
  return "original";
}

/** Shows a status label shared by the library and active job. */
export function statusLabel(status) {
  return element("span", { class: "status", "data-status": status }, STATUS_NAMES[status] ?? status);
}

function jobHeader(job, actions, busy) {
  const metadata = [statusLabel(job.status)];
  if (job.info) {
    metadata.push(
      element("span", {}, job.info.width + " × " + job.info.height),
      element("span", {}, frameRate(job.info.fps)),
      element("span", {}, timecode(job.info.duration)),
    );
    if (Number.isInteger(job.info.audio_tracks)) {
      const audio = job.info.audio_tracks;
      metadata.push(element("span", {}, audio + (audio === 1 ? " audio track" : " audio tracks")));
    }
  }
  const output = localUrl(job.output_url);
  const recipe = localUrl(job.recipe_url);
  const jobActions = element("div", { class: "job-actions" });
  if (recipe && job.shots?.length) {
    jobActions.append(element("a", { class: "button", href: recipe, download: true }, "Recipe JSON"));
  }
  if (output) {
    const download = new URL(output, location.href);
    download.searchParams.set("download", "true");
    jobActions.append(element("a", { class: "button", href: download.pathname + download.search, download: true }, "Download video"));
  }
  if (!["queued", "analyzing", "rendering"].includes(job.status)) {
    jobActions.append(button("Delete project", actions.remove, "button-quiet button-danger", {
      disabled: busy,
      "data-focus-key": "delete-job",
    }));
  }
  const action = job.status === "review" ? ["render", "Render video"]
    : job.status === "paused" ? ["resume", "Resume"]
      : ["queued", "analyzing", "rendering"].includes(job.status) ? ["pause", "Pause"] : null;
  if (action) {
    const pausing = action[0] === "pause" && job.pause_requested;
    jobActions.append(button(pausing ? "Pausing…" : action[1], () => actions.task(action[0]), "button-primary", {
      disabled: busy || pausing,
      "data-focus-key": "job-action",
    }));
  }
  return element("header", { class: "workspace-header" },
    element("div", { class: "title-region" },
      element("h1", { class: "job-title" }, job.name),
      element("div", { class: "job-meta" }, ...metadata),
    ),
    jobActions,
  );
}

function statusOverview(job) {
  const progress = job.progress;
  const showProgress = ["analyzing", "rendering", "paused"].includes(job.status) && progress?.total > 0;
  const message = job.error || job.message;
  if (!message && !showProgress) return null;
  const progressBar = showProgress ? element("div", { class: "progress-stack" },
    element("progress", {
      value: progress.done, max: progress.total, "aria-label": STATUS_NAMES[job.status] + " progress",
    }),
    element("span", {}, progress.done + " / " + progress.total),
  ) : null;
  return element("div", { class: "status-overview", "data-status": job.status },
    element("span", {}, message ?? ""),
    progressBar,
  );
}

function inspector(job, shot, state, actions) {
  const actual = shotProfile(shot);
  const recommended = Object.hasOwn(PROFILE_NAMES, shot.recommended) ? shot.recommended : actual;
  const candidates = element("div", { class: "inspector-section", "aria-label": "Comparison profiles" });
  for (const profile of Object.keys(PROFILE_NAMES)) {
    const measurements = shot.candidates?.[profile];
    if (!measurements) continue;
    candidates.append(profileCard(profile, measurements, state.preview, actions.preview));
  }
  const editable = ["review", "completed"].includes(job.status);
  const hasChanges = state.preview !== actual;
  const changes = element("div", { class: "profile-selection" },
    element("span", {}, "Selected ", element("strong", {}, PROFILE_NAMES[actual])),
  );
  if (editable) {
    if (shot.override !== null && shot.override !== undefined && shot.override !== false) {
      changes.append(button("Reset", () => actions.apply(shot.index, null), "button-quiet", {
        disabled: state.busy, "data-focus-key": "profile-reset",
      }));
    }
    if (hasChanges) {
      changes.append(button("Use " + PROFILE_NAMES[state.preview], () => actions.apply(shot.index, state.preview),
        "button-primary", { disabled: state.busy, "data-focus-key": "profile-apply" }));
    }
  }
  const shotNumber = job.shots.indexOf(shot) + 1;
  return element("aside", { class: "surface inspector", "aria-label": "Restoration profiles" },
    element("div", { class: "surface-heading" },
      element("h2", {}, "Restoration profiles"),
      element("span", { class: "shot-time" }, "Shot " + String(shotNumber).padStart(2, "0")),
    ),
    shot.recommended ? element("div", { class: "recommendation" },
      element("strong", {}, "Recommended " + PROFILE_NAMES[recommended]),
      shot.reason ? element("span", {}, shot.reason) : null,
    ) : element("p", { class: "recommendation" }, "Preparing candidate profiles."),
    candidates,
    changes,
  );
}

function profileCard(profile, measurements, preview, choose) {
  const score = Number.isFinite(measurements.score) ? SCORE_FORMAT.format(measurements.score) : null;
  const view = element("button", {
    type: "button", class: "profile-card",
    "aria-pressed": String(profile === preview),
    "data-focus-key": "profile-" + profile,
  },
  element("span", { class: "profile-card-header" },
    element("span", { class: "profile-name" }, PROFILE_NAMES[profile]),
    score === null ? null : element("span", { class: "profile-score", title: "Candidate quality score" }, "Score " + score),
  ),
  measurements.note ? element("span", { class: "profile-note" }, measurements.note) : null,
  Number.isFinite(measurements.risk)
    ? element("span", { class: "profile-note" }, "Risk " + SCORE_FORMAT.format(measurements.risk)) : null);
  view.addEventListener("click", () => choose(profile));
  return view;
}

function warningList(warnings) {
  return element("details", { class: "surface warning-list" },
    element("summary", {}, warnings.length === 1 ? "1 render note" : warnings.length + " render notes"),
    element("ul", {}, ...warnings.map((warning) => element("li", {}, warning))),
  );
}

function pendingText(job) {
  if (job.status === "failed") return "Choose another video to start a new job.";
  if (job.status === "paused") return "Resume analysis to generate shot previews.";
  if (job.status === "queued") return "This video is waiting for the engine.";
  return "Shot previews will appear as analysis progresses.";
}
