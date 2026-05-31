const app = document.querySelector(".app");
const world = document.querySelector("#world");
const fileInput = document.querySelector("#fileInput");
const webcam = document.querySelector("#webcam");
const handCursor = document.querySelector("#handCursor");
const statusText = document.querySelector("#statusText");
const statusDot = document.querySelector("#statusDot");
const gestureChip = document.querySelector("#gestureChip");
const gestureLabel = document.querySelector("#gestureLabel");
const emptyNote = document.querySelector("#emptyNote");

const VISION_TASKS_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";
const VISION_WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const MODEL_EXTENSIONS = new Set(["obj", "stl", "glb", "gltf"]);

const state = {
  pan: { x: window.innerWidth / 2, y: window.innerHeight / 2 },
  mediaCount: 0,
  lastVideoTime: -1,
  hand: {
    active: false,
    grabbing: false,
    lastPoint: null,
    heldBlock: null
  },
  pointer: {
    active: false,
    lastPoint: null
  }
};

const landmarkTips = [8, 12, 16, 20];
const landmarkPips = [6, 10, 14, 18];

renderWorld();
startCamera();

fileInput.addEventListener("change", (event) => {
  addFiles([...event.target.files]);
  fileInput.value = "";
});

window.addEventListener("dragover", (event) => {
  event.preventDefault();
});

window.addEventListener("drop", (event) => {
  event.preventDefault();
  addFiles([...event.dataTransfer.files], {
    x: event.clientX,
    y: event.clientY
  });
});

window.addEventListener("resize", renderWorld);

app.addEventListener("pointerdown", (event) => {
  if (event.target.closest(".upload-fab") || event.target.closest(".media-block")) return;
  state.pointer.active = true;
  state.pointer.lastPoint = { x: event.clientX, y: event.clientY };
  app.classList.add("is-pointer-down");
  app.setPointerCapture(event.pointerId);
});

app.addEventListener("pointermove", (event) => {
  if (!state.pointer.active || !state.pointer.lastPoint) return;
  const current = { x: event.clientX, y: event.clientY };
  panBy(current.x - state.pointer.lastPoint.x, current.y - state.pointer.lastPoint.y);
  state.pointer.lastPoint = current;
});

app.addEventListener("pointerup", endPointerPan);
app.addEventListener("pointercancel", endPointerPan);

function endPointerPan() {
  state.pointer.active = false;
  state.pointer.lastPoint = null;
  app.classList.remove("is-pointer-down");
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Mouse canvas", "warn");
    return;
  }

  try {
    setStatus("Camera starting", "warn");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 960 },
        height: { ideal: 720 },
        facingMode: "user"
      },
      audio: false
    });
    webcam.srcObject = stream;
    await webcam.play();
    await startHandLandmarker();
  } catch {
    setStatus("Mouse canvas", "warn");
  }
}

async function startHandLandmarker() {
  try {
    setStatus("Loading hand AI", "warn");
    const { HandLandmarker, FilesetResolver } = await import(VISION_TASKS_URL);
    const vision = await FilesetResolver.forVisionTasks(VISION_WASM_PATH);
    const landmarker = await createLandmarker(HandLandmarker, vision, "GPU").catch(() => {
      return createLandmarker(HandLandmarker, vision, "CPU");
    });

    setStatus("Camera live", "live");
    scanFrame(landmarker);
  } catch {
    setStatus("Mouse canvas", "warn");
  }
}

function createLandmarker(HandLandmarker, vision, delegate) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL_URL,
      delegate
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.64,
    minHandPresenceConfidence: 0.58,
    minTrackingConfidence: 0.58
  });
}

function scanFrame(landmarker) {
  if (
    webcam.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    webcam.currentTime !== state.lastVideoTime
  ) {
    state.lastVideoTime = webcam.currentTime;
    const results = landmarker.detectForVideo(webcam, performance.now());
    handleHandLandmarkerResults(results);
  }
  requestAnimationFrame(() => scanFrame(landmarker));
}

function handleHandLandmarkerResults(results) {
  const hand = results.landmarks?.[0];

  if (!hand) {
    resetHandState();
    handCursor.classList.remove("is-live", "is-grabbing");
    app.classList.remove("is-grabbing");
    updateGesture("Open palm", false);
    return;
  }

  const point = handPointToScreen(hand[9]);
  const grabbing = isClosedFist(hand);
  const wasGrabbing = state.hand.grabbing;

  state.hand.active = true;
  moveHandCursor(point, grabbing);

  if (grabbing && !wasGrabbing) {
    holdBlockAtPoint(point);
  }

  if (grabbing && wasGrabbing && state.hand.lastPoint) {
    const dx = point.x - state.hand.lastPoint.x;
    const dy = point.y - state.hand.lastPoint.y;

    if (state.hand.heldBlock) {
      moveBlockBy(state.hand.heldBlock, dx, dy);
    } else {
      panBy(dx * 1.18, dy * 1.18);
    }
  }

  if (!grabbing && wasGrabbing) {
    releaseHeldBlock();
  }

  state.hand.grabbing = grabbing;
  state.hand.lastPoint = point;
  app.classList.toggle("is-grabbing", grabbing);
  updateGesture(getGestureLabel(grabbing), grabbing);
}

function handPointToScreen(landmark) {
  return {
    x: (1 - landmark.x) * window.innerWidth,
    y: landmark.y * window.innerHeight
  };
}

function holdBlockAtPoint(point) {
  const block = document.elementFromPoint(point.x, point.y)?.closest(".media-block");
  state.hand.heldBlock = block || null;
  state.hand.heldBlock?.classList.add("is-held");
}

function releaseHeldBlock() {
  state.hand.heldBlock?.classList.remove("is-held");
  state.hand.heldBlock = null;
}

function resetHandState() {
  releaseHeldBlock();
  state.hand.active = false;
  state.hand.grabbing = false;
  state.hand.lastPoint = null;
}

function getGestureLabel(grabbing) {
  if (!grabbing) return "Open palm";
  return state.hand.heldBlock ? "Holding media" : "Closed fist";
}

function moveBlockBy(block, dx, dy) {
  const left = parseFloat(block.style.left) + dx;
  const top = parseFloat(block.style.top) + dy;
  block.style.left = `${left}px`;
  block.style.top = `${top}px`;
}

function isClosedFist(hand) {
  const palmSize = distance(hand[0], hand[9]) || 0.1;
  const folded = landmarkTips.reduce((score, tipIndex, index) => {
    const pip = hand[landmarkPips[index]];
    const tip = hand[tipIndex];
    const tipNearPalm = distance(tip, hand[0]) < palmSize * 1.08;
    const belowKnuckle = tip.y > pip.y - 0.012;
    return score + (tipNearPalm || belowKnuckle ? 1 : 0);
  }, 0);

  const thumbTucked = distance(hand[4], hand[9]) < palmSize * 0.92;
  return folded >= 3 && thumbTucked;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function moveHandCursor(point, grabbing) {
  handCursor.style.setProperty("--cursor-x", `${point.x}px`);
  handCursor.style.setProperty("--cursor-y", `${point.y}px`);
  handCursor.style.transform = `translate3d(${point.x}px, ${point.y}px, 0) scale(${grabbing ? 0.68 : 1})`;
  handCursor.classList.add("is-live");
  handCursor.classList.toggle("is-grabbing", grabbing);
}

function updateGesture(label, grabbing) {
  gestureLabel.textContent = label;
  gestureChip.classList.toggle("is-grabbing", grabbing);
}

function setStatus(label, tone = "idle") {
  statusText.textContent = label;
  statusDot.classList.toggle("is-live", tone === "live");
  statusDot.classList.toggle("is-warn", tone === "warn");
}

function panBy(dx, dy) {
  state.pan.x += dx;
  state.pan.y += dy;
  renderWorld();
}

function renderWorld() {
  world.style.transform = `translate3d(${state.pan.x}px, ${state.pan.y}px, 0)`;
}

function addFiles(files, screenPoint) {
  const mediaFiles = files
    .map((file) => ({ file, kind: getFileKind(file) }))
    .filter((entry) => entry.kind);

  if (!mediaFiles.length) return;

  mediaFiles.forEach(({ file, kind }, index) => {
    const position = nextBlockPosition(index, screenPoint);
    const block = kind === "model"
      ? createModelBlock(file, position)
      : createImageBlock(file, URL.createObjectURL(file), position);

    world.append(block);
    state.mediaCount += 1;
  });

  emptyNote.classList.toggle("is-hidden", state.mediaCount > 0);
}

window.spatialNotebook = {
  addFiles
};

function getFileKind(file) {
  const extension = getFileExtension(file);
  const type = file.type || "";
  if (type.startsWith("image/")) return "image";
  if (MODEL_EXTENSIONS.has(extension) || type.startsWith("model/")) return "model";
  return "";
}

function getFileExtension(file) {
  return file.name.split(".").pop()?.toLowerCase() || "";
}

function nextBlockPosition(index, screenPoint) {
  const center = screenPoint || {
    x: window.innerWidth / 2 + Math.cos(state.mediaCount * 1.7) * 110,
    y: window.innerHeight / 2 + Math.sin(state.mediaCount * 1.3) * 70
  };

  const staggerX = (index % 3) * 44;
  const staggerY = Math.floor(index / 3) * 44;

  return {
    x: center.x - state.pan.x - 160 + staggerX,
    y: center.y - state.pan.y - 120 + staggerY
  };
}

function createImageBlock(file, url, position) {
  const block = document.createElement("article");
  block.className = "media-block";
  block.style.left = `${position.x}px`;
  block.style.top = `${position.y}px`;
  block.dataset.objectUrl = url;

  const media = document.createElement("img");
  media.src = url;
  media.alt = file.name;

  const caption = document.createElement("div");
  caption.className = "media-caption";

  const name = document.createElement("span");
  name.textContent = file.name;

  const kind = document.createElement("span");
  kind.textContent = "image";

  caption.append(name, kind);
  block.append(createDeleteButton(), media, caption);

  enableBlockDrag(block);
  return block;
}

function createModelBlock(file, position) {
  const block = document.createElement("article");
  block.className = "media-block model-block";
  block.style.left = `${position.x}px`;
  block.style.top = `${position.y}px`;

  const canvas = document.createElement("canvas");
  canvas.className = "model-preview";
  canvas.width = 760;
  canvas.height = 500;

  const caption = document.createElement("div");
  caption.className = "media-caption";

  const name = document.createElement("span");
  name.textContent = file.name;

  const kind = document.createElement("span");
  kind.textContent = getFileExtension(file) || "3d";

  caption.append(name, kind);
  block.append(createDeleteButton(), canvas, caption);

  drawModelPlaceholder(canvas, kind.textContent);
  loadModelPreview(file, canvas);
  enableBlockDrag(block);
  return block;
}

function createDeleteButton() {
  const button = document.createElement("button");
  button.className = "delete-block";
  button.type = "button";
  button.setAttribute("aria-label", "Remove media");
  button.textContent = "×";
  return button;
}

function removeMediaBlock(block) {
  if (block.dataset.objectUrl) URL.revokeObjectURL(block.dataset.objectUrl);
  block.remove();
  state.mediaCount = Math.max(0, state.mediaCount - 1);
  if (state.hand.heldBlock === block) releaseHeldBlock();
  emptyNote.classList.toggle("is-hidden", state.mediaCount > 0);
}

async function loadModelPreview(file, canvas) {
  const extension = getFileExtension(file);

  try {
    if (extension === "obj") {
      drawModelPreview(canvas, parseObj(await file.text()));
      return;
    }

    if (extension === "stl") {
      drawModelPreview(canvas, parseStl(await file.arrayBuffer()));
      return;
    }
  } catch {
    drawModelPlaceholder(canvas, extension || "3d");
    return;
  }

  drawModelPlaceholder(canvas, extension || "3d");
}

function parseObj(text) {
  const vertices = [];
  const edges = new Set();

  text.split(/\r?\n/).forEach((line) => {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === "v" && parts.length >= 4) {
      vertices.push(parts.slice(1, 4).map(Number));
    }
    if (parts[0] === "f" && parts.length >= 4) {
      const ids = parts.slice(1)
        .map((part) => Number(part.split("/")[0]) - 1)
        .filter((id) => Number.isFinite(id) && id >= 0 && id < vertices.length);

      ids.forEach((id, index) => {
        const next = ids[(index + 1) % ids.length];
        edges.add([Math.min(id, next), Math.max(id, next)].join("-"));
      });
    }
  });

  return {
    vertices,
    edges: [...edges].slice(0, 7000).map((edge) => edge.split("-").map(Number))
  };
}

function parseStl(buffer) {
  const text = decodeAscii(buffer);
  if (text.trimStart().startsWith("solid") && text.includes("vertex")) {
    return parseAsciiStl(text);
  }
  return parseBinaryStl(buffer);
}

function decodeAscii(buffer) {
  return new TextDecoder("utf-8", { fatal: false }).decode(buffer.slice(0, 2000000));
}

function parseAsciiStl(text) {
  const vertices = [];
  const edges = [];
  const current = [];

  text.split(/\r?\n/).forEach((line) => {
    const parts = line.trim().split(/\s+/);
    if (parts[0] !== "vertex" || parts.length < 4) return;

    vertices.push(parts.slice(1, 4).map(Number));
    current.push(vertices.length - 1);

    if (current.length === 3) {
      edges.push([current[0], current[1]], [current[1], current[2]], [current[2], current[0]]);
      current.length = 0;
    }
  });

  return { vertices, edges: edges.slice(0, 9000) };
}

function parseBinaryStl(buffer) {
  if (buffer.byteLength < 84) return { vertices: [], edges: [] };

  const view = new DataView(buffer);
  const triangleCount = view.getUint32(80, true);
  const vertices = [];
  const edges = [];
  const safeCount = Math.min(triangleCount, Math.floor((buffer.byteLength - 84) / 50), 3000);

  for (let triangle = 0; triangle < safeCount; triangle += 1) {
    const offset = 84 + triangle * 50 + 12;
    const start = vertices.length;

    for (let vertex = 0; vertex < 3; vertex += 1) {
      const pointOffset = offset + vertex * 12;
      vertices.push([
        view.getFloat32(pointOffset, true),
        view.getFloat32(pointOffset + 4, true),
        view.getFloat32(pointOffset + 8, true)
      ]);
    }

    edges.push([start, start + 1], [start + 1, start + 2], [start + 2, start]);
  }

  return { vertices, edges };
}

function drawModelPreview(canvas, geometry) {
  if (!geometry.vertices.length || !geometry.edges.length) {
    drawModelPlaceholder(canvas, "3d");
    return;
  }

  const ctx = canvas.getContext("2d");
  const normalized = normalizeGeometry(geometry.vertices);
  let frame = 0;

  function render() {
    if (!canvas.isConnected) return;

    frame += 0.012;
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    drawModelBackground(ctx, width, height);
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim() || "#141414";
    ctx.globalAlpha = 0.78;

    geometry.edges.forEach(([a, b]) => {
      const start = projectPoint(normalized[a], frame, width, height);
      const end = projectPoint(normalized[b], frame, width, height);
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(end.x, end.y);
      ctx.stroke();
    });

    ctx.globalAlpha = 1;
    requestAnimationFrame(render);
  }

  render();
}

function normalizeGeometry(vertices) {
  const bounds = vertices.reduce((box, point) => ({
    min: point.map((value, index) => Math.min(value, box.min[index])),
    max: point.map((value, index) => Math.max(value, box.max[index]))
  }), {
    min: [Infinity, Infinity, Infinity],
    max: [-Infinity, -Infinity, -Infinity]
  });

  const center = bounds.min.map((value, index) => (value + bounds.max[index]) / 2);
  const size = Math.max(...bounds.max.map((value, index) => value - bounds.min[index])) || 1;

  return vertices.map((point) => point.map((value, index) => (value - center[index]) / size));
}

function projectPoint(point, angle, width, height) {
  const [x, y, z] = point;
  const cosY = Math.cos(angle);
  const sinY = Math.sin(angle);
  const cosX = Math.cos(angle * 0.55);
  const sinX = Math.sin(angle * 0.55);
  const rx = x * cosY - z * sinY;
  const rz = x * sinY + z * cosY;
  const ry = y * cosX - rz * sinX;
  const depth = y * sinX + rz * cosX + 2.2;
  const scale = Math.min(width, height) * 0.78 / depth;

  return {
    x: width / 2 + rx * scale,
    y: height / 2 - ry * scale
  };
}

function drawModelPlaceholder(canvas, label) {
  const ctx = canvas.getContext("2d");
  drawModelBackground(ctx, canvas.width, canvas.height);

  const width = canvas.width;
  const height = canvas.height;
  const ink = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim() || "#141414";
  const muted = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim() || "#6e6e67";

  ctx.strokeStyle = ink;
  ctx.lineWidth = 3;
  ctx.globalAlpha = 0.72;
  ctx.beginPath();
  ctx.moveTo(width / 2, height / 2 - 88);
  ctx.lineTo(width / 2 + 92, height / 2 - 36);
  ctx.lineTo(width / 2 + 92, height / 2 + 70);
  ctx.lineTo(width / 2, height / 2 + 122);
  ctx.lineTo(width / 2 - 92, height / 2 + 70);
  ctx.lineTo(width / 2 - 92, height / 2 - 36);
  ctx.closePath();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(width / 2, height / 2 - 88);
  ctx.lineTo(width / 2, height / 2 + 18);
  ctx.lineTo(width / 2 + 92, height / 2 + 70);
  ctx.moveTo(width / 2, height / 2 + 18);
  ctx.lineTo(width / 2 - 92, height / 2 + 70);
  ctx.stroke();

  ctx.globalAlpha = 1;
  ctx.fillStyle = muted;
  ctx.font = "700 28px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label.toUpperCase(), width / 2, height / 2 + 170);
}

function drawModelBackground(ctx, width, height) {
  const bg = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim() || "#fff";
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(125, 125, 116, 0.18)";
  ctx.lineWidth = 1;

  for (let x = 0; x <= width; x += 48) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  for (let y = 0; y <= height; y += 48) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function enableBlockDrag(block) {
  let dragging = false;
  let lastPoint = null;

  block.addEventListener("pointerdown", (event) => {
    if (event.target.closest(".delete-block")) return;
    dragging = true;
    lastPoint = { x: event.clientX, y: event.clientY };
    block.setPointerCapture(event.pointerId);
  });

  block.querySelector(".delete-block").addEventListener("click", (event) => {
    event.stopPropagation();
    removeMediaBlock(block);
  });

  block.addEventListener("pointermove", (event) => {
    if (!dragging || !lastPoint) return;
    const current = { x: event.clientX, y: event.clientY };
    const left = parseFloat(block.style.left) + current.x - lastPoint.x;
    const top = parseFloat(block.style.top) + current.y - lastPoint.y;
    block.style.left = `${left}px`;
    block.style.top = `${top}px`;
    lastPoint = current;
  });

  block.addEventListener("pointerup", () => {
    dragging = false;
    lastPoint = null;
  });

  block.addEventListener("pointercancel", () => {
    dragging = false;
    lastPoint = null;
  });
}
