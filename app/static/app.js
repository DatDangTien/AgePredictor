"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  tabUpload: $("tab-upload"),
  tabCamera: $("tab-camera"),
  panelUpload: $("panel-upload"),
  panelCamera: $("panel-camera"),
  dropZone: $("drop-zone"),
  fileInput: $("file-input"),
  video: $("video"),
  btnStart: $("btn-start"),
  btnCapture: $("btn-capture"),
  btnSwitch: $("btn-switch"),
  status: $("status"),
  resultSection: $("result-section"),
  canvas: $("canvas"),
  faces: $("faces"),
  captureCanvas: $("capture-canvas"),
};

let stream = null;
let facingMode = "user";
let busy = false;

// ---------------------------------------------------------------- status
function setStatus(msg, kind = "info") {
  els.status.className = `status ${kind}`;
  els.status.innerHTML =
    kind === "loading" ? `<span class="spinner"></span>${msg}` : msg;
}
function clearStatus() {
  els.status.className = "status hidden";
  els.status.textContent = "";
}

// ---------------------------------------------------------------- tabs
function selectTab(which) {
  const upload = which === "upload";
  els.tabUpload.classList.toggle("active", upload);
  els.tabCamera.classList.toggle("active", !upload);
  els.panelUpload.classList.toggle("hidden", !upload);
  els.panelCamera.classList.toggle("hidden", upload);
  if (upload) stopCamera();
}
els.tabUpload.addEventListener("click", () => selectTab("upload"));
els.tabCamera.addEventListener("click", () => selectTab("camera"));

// ---------------------------------------------------------------- upload
els.fileInput.addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) handleFile(f);
});

["dragenter", "dragover"].forEach((ev) =>
  els.dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  els.dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropZone.classList.remove("dragover");
  })
);
els.dropZone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) handleFile(f);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    setStatus("That file is not an image.", "error");
    return;
  }
  sendBlob(file, file.name);
}

// ---------------------------------------------------------------- camera
function cameraSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function startCamera() {
  // getUserMedia requires a secure context (https) or localhost.
  if (!window.isSecureContext) {
    setStatus(
      "Camera needs a secure connection (https) or localhost. " +
        "Open the site over HTTPS, then try again.",
      "error"
    );
    return;
  }
  if (!cameraSupported()) {
    setStatus("This browser does not support camera capture.", "error");
    return;
  }
  stopCamera();
  setStatus("Requesting camera permission…", "loading");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    els.video.srcObject = stream;
    await els.video.play();
    els.btnCapture.disabled = false;
    els.btnSwitch.disabled = false;
    els.btnStart.textContent = "Restart camera";
    clearStatus();
  } catch (err) {
    handleCameraError(err);
  }
}

function handleCameraError(err) {
  const name = err && err.name ? err.name : "";
  let msg;
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      msg =
        "Camera permission denied. Allow camera access in your browser " +
        "settings and try again.";
      break;
    case "NotFoundError":
    case "OverconstrainedError":
      msg = "No camera found on this device.";
      break;
    case "NotReadableError":
      msg = "Camera is already in use by another app.";
      break;
    default:
      msg = "Could not start the camera: " + (err.message || name || "unknown error");
  }
  setStatus(msg, "error");
  stopCamera();
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  els.video.srcObject = null;
  els.btnCapture.disabled = true;
  els.btnSwitch.disabled = true;
}

els.btnStart.addEventListener("click", startCamera);
els.btnSwitch.addEventListener("click", () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  startCamera();
});
els.btnCapture.addEventListener("click", () => {
  if (!stream) return;
  const v = els.video;
  const c = els.captureCanvas;
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  c.toBlob(
    (blob) => {
      if (blob) sendBlob(blob, "capture.jpg");
      else setStatus("Failed to capture frame.", "error");
    },
    "image/jpeg",
    0.92
  );
});

// ---------------------------------------------------------------- request
async function sendBlob(blob, name) {
  if (busy) return;
  busy = true;
  setStatus("Detecting faces and ages…", "loading");
  els.resultSection.classList.add("hidden");

  const form = new FormData();
  form.append("file", blob, name || "image.jpg");

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    let payload;
    try {
      payload = await res.json();
    } catch {
      payload = null;
    }
    if (!res.ok) {
      const detail = (payload && payload.detail) || `HTTP ${res.status}`;
      setStatus("Error: " + detail, "error");
      return;
    }
    await drawResult(blob, payload);
    if (payload.count === 0) setStatus("No faces detected.", "info");
    else
      setStatus(
        `Detected ${payload.count} face${payload.count > 1 ? "s" : ""}.`,
        "info"
      );
  } catch (err) {
    setStatus("Network error: " + (err.message || "request failed"), "error");
  } finally {
    busy = false;
  }
}

// ---------------------------------------------------------------- draw
function drawResult(blob, payload) {
  return new Promise((resolve) => {
    const img = new Image();
    // Prefer the server's enhanced image (what the models saw); fall back to
    // the original upload blob if the response omitted it.
    const objectUrl = payload.image ? null : URL.createObjectURL(blob);
    img.onload = () => {
      const canvas = els.canvas;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      const lw = Math.max(2, Math.round(canvas.width / 250));
      const fontSize = Math.max(14, Math.round(canvas.width / 28));
      ctx.lineWidth = lw;
      ctx.strokeStyle = "#29c98b";
      ctx.font = `bold ${fontSize}px sans-serif`;
      ctx.textBaseline = "top";

      els.faces.innerHTML = "";
      (payload.faces || []).forEach((f, i) => {
        const [x1, y1, x2, y2] = f.box;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const gShort = f.gender === "Female" ? "Female" : "Male";
        const label = `${Math.round(f.age)} ${gShort}`;
        const tw = ctx.measureText(label).width;
        const ty = Math.max(0, y1 - fontSize - 4);
        ctx.fillStyle = "#29c98b";
        ctx.fillRect(x1, ty, tw + 10, fontSize + 4);
        ctx.fillStyle = "#04130d";
        ctx.fillText(label, x1 + 5, ty + 2);

        const chip = document.createElement("div");
        chip.className = "face-chip";
        chip.innerHTML =
          `<span class="age">${f.age}</span> yrs · ` +
          `<span class="gender">${f.gender}</span> ` +
          `<span class="meta">#${i + 1} · face ${(f.score * 100).toFixed(0)}%` +
          `${f.gender_score != null ? " · gender " + (f.gender_score * 100).toFixed(0) + "%" : ""}</span>`;
        els.faces.appendChild(chip);
      });

      els.resultSection.classList.remove("hidden");
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      resolve();
    };
    img.onerror = () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      resolve();
    };
    img.src = payload.image || objectUrl;
  });
}

// camera tab disabled hint if unsupported
if (!cameraSupported()) {
  els.tabCamera.title = "Camera not supported in this browser";
}
