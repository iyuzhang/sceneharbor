export const PROFILE_NAMES = Object.freeze({
  original: "Original",
  gentle: "Gentle",
  restore: "Restore",
});

export const STATUS_NAMES = Object.freeze({
  queued: "Queued",
  analyzing: "Analyzing",
  review: "Ready to review",
  rendering: "Rendering",
  paused: "Paused",
  completed: "Rendered",
  failed: "Failed",
});

/** Creates an element with text or existing nodes as children. */
export function element(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value !== null && value !== undefined && value !== false) {
      node.setAttribute(name, value === true ? "" : String(value));
    }
  }
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

/** Creates a keyboard-accessible action. */
export function button(label, action, classes = "", attributes = {}) {
  const node = element("button", { type: "button", class: "button " + classes, ...attributes }, label);
  node.addEventListener("click", action);
  return node;
}

/** Returns an API result or raises a readable request error. */
export async function request(path, options = {}) {
  const data = options.data;
  const isForm = data instanceof FormData;
  const response = await fetch("/api" + path, {
    method: options.method ?? "GET",
    headers: data && !isForm ? { "Content-Type": "application/json" } : undefined,
    body: isForm ? data : data ? JSON.stringify(data) : undefined,
    signal: options.signal,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    const detail = error?.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).filter(Boolean).join(" ")
      : detail;
    throw new Error(typeof message === "string" && message ? message : "Request failed (" + response.status + ").");
  }
  return response.status === 204 ? null : response.json();
}

/** Formats a clip timestamp for transport controls. */
export function timecode(seconds, precision = 0) {
  if (!Number.isFinite(seconds) || seconds < 0) return "–";
  const scale = 10 ** precision;
  const rounded = Math.floor(seconds * scale + 0.000001);
  const minutes = Math.floor(rounded / (60 * scale));
  const wholeSeconds = Math.floor(rounded / scale) % 60;
  const fraction = precision ? "." + String(rounded % scale).padStart(precision, "0") : "";
  return String(minutes).padStart(2, "0") + ":" + String(wholeSeconds).padStart(2, "0") + fraction;
}

/** Formats rational and decimal frame rates from the video probe. */
export function frameRate(value) {
  const rate = rateNumber(value);
  return rate ? Number(rate.toFixed(2)) + " fps" : "";
}

/** Converts a frame offset to the corresponding timestamp. */
export function frameTime(frame, fps) {
  const rate = rateNumber(fps);
  return rate ? frame / rate : NaN;
}

/** Resolves a service-owned media URL. */
export function localUrl(value) {
  if (typeof value !== "string" || !value.startsWith("/")) return null;
  const url = new URL(value, location.href);
  return url.origin === location.origin ? url.pathname + url.search : null;
}

/** Returns the endpoint for the selected job. */
export function jobPath(id) {
  return "/jobs/" + encodeURIComponent(id);
}

/** Returns an image URL that has an existing shot profile. */
export function previewUrl(job, shot, profile) {
  return "/api" + jobPath(job.id) + "/shots/" + encodeURIComponent(shot.index) + "/" + profile + ".jpg";
}

function rateNumber(value) {
  const parts = String(value ?? "").split("/").map(Number);
  const rate = parts.length === 2 ? parts[0] / parts[1] : parts[0];
  return Number.isFinite(rate) && rate > 0 ? rate : 0;
}
