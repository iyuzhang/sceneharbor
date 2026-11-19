import { bindImporter } from "./import.js";
import { findShot, jobView, shotProfile, statusLabel } from "./job.js";
import { PROFILE_NAMES, button, element, jobPath, request } from "./ui.js";

const WORKING = new Set(["queued", "analyzing", "rendering"]);
const ERROR_WAIT = 5000;
const QUICK_WAIT = 1100;
const IDLE_WAIT = 4500;
const workspace = document.getElementById("workspace");
const library = document.getElementById("jobs");
const notice = document.getElementById("notice");
const initialId = new URL(location.href).searchParams.get("job");

const state = {
  job: null, jobs: [], shot: null, preview: "original", mode: "compare",
  split: 52, busy: false, loading: true, error: null,
};

let activeId = initialId;
let readVersion = 0;
let pollHandle = null;
let librarySnapshot = "";
let jobSnapshot = "";

const openImport = bindImporter(acceptImport);
const actions = {
  selectShot,
  preview,
  mode: setMode,
  split: (value) => { state.split = value; },
  task: updateTask,
  apply: updateProfile,
  remove: removeJob,
};

document.getElementById("dismiss-notice").addEventListener("click", () => { notice.hidden = true; });
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) schedule(0);
});
loadHealth();
refresh();

async function loadHealth() {
  const engine = document.getElementById("engine-state");
  try {
    const info = await request("/health", { signal: AbortSignal.timeout(8000) });
    engine.dataset.ready = String(info.engine_ready);
    engine.textContent = info.engine_ready ? "Local engine ready" : "Engine needs setup";
    if (info.engine_message) engine.title = info.engine_message;
    if (!info.engine_ready && info.engine_message) showNotice(info.engine_message, true);
  } catch (error) {
    engine.textContent = "Engine status unavailable";
    engine.title = error.message;
  }
}

async function refresh() {
  if (state.busy) {
    schedule(QUICK_WAIT);
    return;
  }
  const version = ++readVersion;
  try {
    const jobs = await request("/jobs", { signal: AbortSignal.timeout(8000) });
    const id = jobs.find((job) => String(job.id) === activeId)?.id ?? jobs[0]?.id ?? null;
    if (version !== readVersion) return;
    const selected = id === null ? null : await request(jobPath(id), { signal: AbortSignal.timeout(8000) });
    if (version !== readVersion) return;
    state.jobs = jobs;
    state.loading = false;
    state.error = null;
    recordSelection(id);
    saveJob(selected);
    renderLibrary();
    renderWorkspace();
    schedule(jobs.some((job) => WORKING.has(job.status)) ? QUICK_WAIT : IDLE_WAIT);
  } catch (error) {
    if (version !== readVersion) return;
    applyReadError(error);
    schedule(ERROR_WAIT);
  }
}

async function selectJob(id) {
  if (String(id) === activeId && state.job) return;
  recordSelection(id);
  renderLibrary();
  const version = ++readVersion;
  try {
    const job = await request(jobPath(id), { signal: AbortSignal.timeout(8000) });
    if (version !== readVersion) return;
    state.error = null;
    saveJob(job);
    renderWorkspace();
  } catch (error) {
    if (version === readVersion) applyReadError(error);
  } finally {
    schedule(QUICK_WAIT);
  }
}

function acceptImport(job) {
  readVersion++;
  state.jobs = [job, ...state.jobs.filter(({ id }) => id !== job.id)];
  state.loading = false;
  state.error = null;
  recordSelection(job.id);
  saveJob(job);
  renderLibrary();
  renderWorkspace();
  schedule(QUICK_WAIT);
}

function recordSelection(id) {
  const selection = id === null ? null : String(id);
  if (selection === activeId) return;
  activeId = selection;
  const url = new URL(location.href);
  if (selection === null) url.searchParams.delete("job");
  else url.searchParams.set("job", selection);
  history.replaceState(null, "", url);
}

function saveJob(job) {
  const changed = job?.id !== state.job?.id;
  const previous = state.job ? findShot(state.job, state.shot) : null;
  const followsPrevious = state.preview === shotProfile(previous);
  state.job = job;
  if (!job) {
    state.shot = null;
    return;
  }
  const shot = findShot(job, changed ? null : state.shot);
  if (changed || shot?.index !== state.shot) {
    state.shot = shot?.index ?? null;
    state.preview = shotProfile(shot);
    state.mode = "compare";
  } else if (followsPrevious) {
    state.preview = shotProfile(shot);
  }
  if (!job.output_url) state.mode = "compare";
}

function selectShot(index) {
  state.shot = index;
  state.preview = shotProfile(findShot(state.job, index));
  renderWorkspace();
}

function preview(profile) {
  state.preview = profile;
  state.mode = "compare";
  renderWorkspace();
}

function setMode(mode) {
  state.mode = mode;
  renderWorkspace();
}

function updateTask(action) {
  return mutate(jobPath(state.job.id) + "/" + action, { method: "POST" });
}

function updateProfile(index, profile) {
  if (profile !== null && !Object.hasOwn(PROFILE_NAMES, profile)) return;
  const path = jobPath(state.job.id) + "/shots/" + encodeURIComponent(index);
  return mutate(path, { method: "PATCH", data: { profile } }, index);
}

async function removeJob() {
  const job = state.job;
  const confirmed = window.confirm(
    'Delete "' + job.name + '" and its workspace? This removes uploaded copies, previews and exports. Files imported by local path stay on disk.',
  );
  if (confirmed) await mutate(jobPath(job.id), { method: "DELETE" });
}

async function mutate(path, options, editedIndex = null) {
  if (state.busy || activeId !== String(state.job?.id)) return;
  const id = state.job?.id;
  state.busy = true;
  readVersion++;
  renderWorkspace();
  try {
    const job = await request(path, { ...options, signal: AbortSignal.timeout(15000) });
    if (options.method === "DELETE") {
      state.jobs = state.jobs.filter((item) => item.id !== id);
      if (activeId === String(id)) {
        recordSelection(null);
        saveJob(null);
      }
    } else if (job?.id && activeId === String(id)) {
      state.jobs = state.jobs.map((item) => item.id === id ? job : item);
      saveJob(job);
      if (editedIndex === state.shot) {
        state.preview = shotProfile(findShot(job, state.shot));
        state.mode = "compare";
      }
    }
  } catch (error) {
    showNotice(error.message, true);
  } finally {
    state.busy = false;
    renderLibrary();
    renderWorkspace();
    schedule(0);
  }
}

function applyReadError(error) {
  state.loading = false;
  if (state.error !== error.message) showNotice(error.message, true);
  state.error = error.message;
  renderLibrary();
  renderWorkspace();
}

function renderLibrary() {
  const snapshot = JSON.stringify([activeId, state.jobs.map(({ id, name, status }) => [id, name, status])]);
  if (snapshot === librarySnapshot) return;
  librarySnapshot = snapshot;
  const focused = document.activeElement?.dataset?.libraryId;
  if (!state.jobs.length) {
    library.replaceChildren(element("p", { class: "library-empty" }, "No videos yet."));
    return;
  }
  const entries = state.jobs.map((job) => {
    const entry = element("button", {
      type: "button", class: "job-entry", "aria-current": String(String(job.id) === activeId),
      "data-library-id": job.id,
    }, element("span", { class: "job-name", title: job.name }, job.name), statusLabel(job.status));
    entry.addEventListener("click", () => selectJob(job.id));
    return entry;
  });
  library.replaceChildren(...entries);
  if (focused) {
    entries.find((item) => item.dataset.libraryId === focused)?.focus({ preventScroll: true });
  }
}

function renderWorkspace() {
  const snapshot = JSON.stringify([state.job, state.shot, state.preview, state.mode, state.busy, state.loading, state.error]);
  if (snapshot === jobSnapshot) return;
  jobSnapshot = snapshot;
  const focused = workspace.contains(document.activeElement) ? document.activeElement.dataset.focusKey : null;
  const scrollLeft = workspace.querySelector(".shot-strip")?.scrollLeft ?? 0;
  if (state.job) {
    const panels = jobView(state.job, state, actions);
    keepMedia(panels);
    workspace.replaceChildren(...panels);
    const strip = workspace.querySelector(".shot-strip");
    if (strip) strip.scrollLeft = scrollLeft;
    document.title = state.job.name + " · SceneHarbor";
  } else {
    workspace.replaceChildren(emptyWorkspace());
    document.title = "SceneHarbor";
  }
  if (focused) {
    workspace.querySelector('[data-focus-key="' + CSS.escape(focused) + '"]')?.focus({ preventScroll: true });
  }
}

function keepMedia(panels) {
  const current = workspace.querySelector(".media-stage[data-media-key]");
  if (!current || current.querySelector(".preview-error")) return;
  const replacement = panels.map((panel) => panel.querySelector(".media-stage[data-media-key]")).find(Boolean);
  if (replacement?.dataset.mediaKey === current.dataset.mediaKey) replacement.replaceWith(current);
}

function emptyWorkspace() {
  if (state.loading) return element("p", { class: "startup", role: "status" }, "Opening workspace…");
  return element("section", { class: "empty-workspace" },
    element("h1", {}, state.error ? "Workspace unavailable" : "Restore video, shot by shot."),
    element("p", {}, state.error || "Import a video to detect cuts, compare restoration profiles, and render your choices."),
    state.error ? button("Retry", () => schedule(0), "button-primary")
      : button("Import video", openImport, "button-primary"),
  );
}

function showNotice(message, isError) {
  document.getElementById("notice-message").textContent = message;
  notice.dataset.error = String(isError);
  notice.hidden = false;
}

function schedule(delay) {
  clearTimeout(pollHandle);
  pollHandle = setTimeout(refresh, document.hidden ? Math.max(delay, IDLE_WAIT) : delay);
}
