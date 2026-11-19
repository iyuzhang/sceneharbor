import { request } from "./ui.js";

/** Connects the video import dialog to the workspace. */
export function bindImporter(onImported) {
  const dialog = document.getElementById("import-dialog");
  const form = document.getElementById("import-form");
  const inputFile = document.getElementById("video-file");
  const inputPath = document.getElementById("video-path");
  const submit = document.getElementById("submit-import");
  const errorLine = document.getElementById("import-error");
  let uploading = false;

  document.getElementById("close-import").addEventListener("click", close);
  document.getElementById("cancel-import").addEventListener("click", close);
  dialog.addEventListener("cancel", (event) => {
    if (uploading) event.preventDefault();
  });
  document.getElementById("import-video").addEventListener("click", open);
  form.addEventListener("change", updateSource);
  form.addEventListener("submit", send);
  updateSource();
  return open;

  function open() {
    form.reset();
    errorLine.hidden = true;
    updateSource();
    if (!dialog.open) dialog.showModal();
  }

  function close() {
    if (!uploading) dialog.close();
  }

  function updateSource() {
    const usePath = new FormData(form).get("source") === "path";
    document.getElementById("path-field").hidden = !usePath;
    document.getElementById("upload-field").hidden = usePath;
    inputPath.required = usePath;
    inputFile.required = !usePath;
  }

  async function send(event) {
    event.preventDefault();
    if (uploading) return;
    const values = new FormData(form);
    const source = values.get("source");
    const height = values.get("height");
    const crf = Number(values.get("crf"));
    if (!Number.isInteger(crf) || crf < 12 || crf > 28) {
      putError("Enter a CRF between 12 and 28.");
      return;
    }
    if (source === "path" && !String(values.get("path")).trim()) {
      putError("Enter the path to a local video.");
      return;
    }
    if (source === "file" && !inputFile.files?.length) {
      putError("Choose a video file to upload.");
      return;
    }
    setBusy(true);
    errorLine.hidden = true;
    try {
      const job = source === "path" ? await submitPath(values, height, crf) : await submitFile(inputFile, height, crf);
      dialog.close();
      onImported(job);
    } catch (error) {
      putError(error.message);
    } finally {
      setBusy(false);
    }
  }

  function putError(message) {
    errorLine.textContent = message;
    errorLine.hidden = false;
  }

  function setBusy(value) {
    uploading = value;
    for (const control of form.elements) control.disabled = value;
    document.getElementById("close-import").disabled = value;
    submit.textContent = value ? "Importing…" : "Analyze video";
  }
}

function submitFile(inputFile, height, crf) {
  const data = new FormData();
  data.set("file", inputFile.files[0]);
  if (height) data.set("height", height);
  data.set("crf", String(crf));
  return request("/uploads", { method: "POST", data });
}

function submitPath(values, height, crf) {
  const data = { path: String(values.get("path")).trim(), crf };
  const name = String(values.get("name")).trim();
  if (name) data.name = name;
  if (height) data.height = Number(height);
  return request("/jobs", { method: "POST", data });
}
