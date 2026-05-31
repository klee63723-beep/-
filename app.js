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
const uploadFab = document.querySelector(".upload-fab");

const VISION_TASKS_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";
const VISION_WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const MODEL_EXTENSIONS = new Set(["obj", "stl", "glb", "gltf"]);
const LONG_PRESS_MS = 600;
const LONG_PRESS_MOVE_TOLERANCE = 10;
const STACK_OVERLAP_THRESHOLD = 0.32;
const HAND_PINCH_MOVE_DEADZONE = 3;
const HAND_PINCH_ZOOM_SENSITIVITY = 0.006;
const SHAKE_WINDOW_MS = 820;
const SHAKE_MIN_STEP = 10;
const SHAKE_REVERSALS_TO_DETACH = 3;
const SHAKE_MIN_TRAVEL = 120;
const STACK_DETACH_COOLDOWN_MS = 900;

const state = {
  pan: { x: window.innerWidth / 2, y: window.innerHeight / 2 },
  scale: 1,
  mediaCount: 0,
  nextLayer: 1,
  nextStackId: 1,
  lastVideoTime: -1,
  hand: {
    active: false,
    grabbing: false,
    pinching: false,
    pinchDirection: "",
    lastPoint: null,
    heldBlock: null,
    movedHeldBlock: false,
    shake: null
  },
  pointer: {
    active: false,
    lastPoint: null,
    points: new Map(),
    pinch: null
  },
  selectedBar: null
};

const landmarkTips = [8, 12, 16, 20];
const landmarkPips = [6, 10, 14, 18];

renderWorld();
initLongPressSelection(uploadFab);
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

document.addEventListener("pointerdown", (event) => {
  if (!event.target.closest(".selectable-bar")) deselectBar();
}, true);

app.addEventListener("pointerdown", (event) => {
  if (event.target.closest(".upload-fab") || event.target.closest(".media-block")) return;

  if (event.pointerType === "touch") {
    event.preventDefault();
    trackCanvasPointer(event);
    app.setPointerCapture(event.pointerId);

    if (state.pointer.points.size === 2) {
      startCanvasPinch();
      state.pointer.active = false;
      app.classList.remove("is-pointer-down");
      return;
    }
  }

  state.pointer.active = true;
  state.pointer.lastPoint = { x: event.clientX, y: event.clientY };
  app.classList.add("is-pointer-down");
  app.setPointerCapture(event.pointerId);
});

app.addEventListener("pointermove", (event) => {
  if (event.pointerType === "touch" && state.pointer.points.has(event.pointerId)) {
    event.preventDefault();
    trackCanvasPointer(event);

    if (state.pointer.pinch && state.pointer.points.size >= 2) {
      updateCanvasPinch();
      return;
    }
  }

  if (!state.pointer.active || !state.pointer.lastPoint) return;
  const current = { x: event.clientX, y: event.clientY };
  panBy(current.x - state.pointer.lastPoint.x, current.y - state.pointer.lastPoint.y);
  state.pointer.lastPoint = current;
});

app.addEventListener("pointerup", endCanvasPointer);
app.addEventListener("pointercancel", endCanvasPointer);

function endCanvasPointer(event) {
  if (event.pointerType === "touch") {
    state.pointer.points.delete(event.pointerId);
    state.pointer.pinch = null;

    const remaining = [...state.pointer.points.values()][0];
    state.pointer.lastPoint = remaining || null;
    state.pointer.active = Boolean(remaining);
    app.classList.toggle("is-pointer-down", Boolean(remaining));
    return;
  }

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
  const pinching = isPinching(hand) && !grabbing;
  const wasGrabbing = state.hand.grabbing;
  const wasPinching = state.hand.pinching;

  state.hand.active = true;
  moveHandCursor(point, grabbing, pinching);

  if (pinching) {
    if (wasGrabbing) releaseHeldBlock();
    updateHandPinchZoom(point, wasPinching);
    state.hand.pinching = true;
    state.hand.grabbing = false;
    state.hand.lastPoint = point;
    app.classList.remove("is-grabbing");
    updateGesture(getPinchGestureLabel(), false, true);
    return;
  }

  if (!pinching && wasPinching) {
    state.hand.pinching = false;
    state.hand.pinchDirection = "";
  }

  if (grabbing && !wasGrabbing) {
    holdBlockAtPoint(point);
  }

  if (grabbing && wasGrabbing && state.hand.lastPoint) {
    const dx = point.x - state.hand.lastPoint.x;
    const dy = point.y - state.hand.lastPoint.y;

    if (state.hand.heldBlock) {
      trackStackShake(state.hand.heldBlock, state.hand.shake, dx, dy);
      moveBlockBy(state.hand.heldBlock, dx, dy);
      updateStackHover(state.hand.heldBlock);
      state.hand.movedHeldBlock = true;
    } else {
      panBy(dx * 1.18, dy * 1.18);
    }
  }

  if (!grabbing && wasGrabbing) {
    releaseHeldBlock();
  }

  state.hand.grabbing = grabbing;
  state.hand.pinching = false;
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
  state.hand.movedHeldBlock = false;
  state.hand.shake = block ? createShakeTracker() : null;
  state.hand.heldBlock?.classList.add("is-held");
}

function releaseHeldBlock() {
  if (state.hand.heldBlock && state.hand.movedHeldBlock) {
    settleImageStack(state.hand.heldBlock);
  } else if (state.hand.heldBlock) {
    clearStackHover(state.hand.heldBlock);
  }
  state.hand.heldBlock?.classList.remove("is-held");
  state.hand.heldBlock = null;
  state.hand.movedHeldBlock = false;
  state.hand.shake = null;
}

function resetHandState() {
  releaseHeldBlock();
  state.hand.active = false;
  state.hand.grabbing = false;
  state.hand.pinching = false;
  state.hand.pinchDirection = "";
  state.hand.lastPoint = null;
  state.hand.movedHeldBlock = false;
  state.hand.shake = null;
}

function getGestureLabel(grabbing) {
  if (!grabbing) return "Open palm";
  return state.hand.heldBlock ? "Holding media" : "Closed fist";
}

function getPinchGestureLabel() {
  if (state.hand.pinchDirection === "in") return "Pinch zoom in";
  if (state.hand.pinchDirection === "out") return "Pinch zoom out";
  return "Pinch ready";
}

function updateHandPinchZoom(point, wasPinching) {
  if (!wasPinching || !state.hand.lastPoint) {
    state.hand.pinchDirection = "";
    return;
  }

  const dx = point.x - state.hand.lastPoint.x;
  if (Math.abs(dx) < HAND_PINCH_MOVE_DEADZONE) {
    state.hand.pinchDirection = "";
    return;
  }

  const factor = 1 + Math.min(Math.abs(dx) * HAND_PINCH_ZOOM_SENSITIVITY, 0.08);
  state.hand.pinchDirection = dx > 0 ? "in" : "out";
  zoomCanvasBy(point, dx > 0 ? factor : 1 / factor);
}

function moveBlockBy(block, dx, dy) {
  getMoveTargets(block).forEach((target) => moveSingleBlockBy(target, dx, dy));
}

function moveSingleBlockBy(block, dx, dy) {
  const left = parseFloat(block.style.left) + dx / state.scale;
  const top = parseFloat(block.style.top) + dy / state.scale;
  block.style.left = `${left}px`;
  block.style.top = `${top}px`;
}

function setTopLayer(block) {
  state.nextLayer += 1;
  block.style.zIndex = String(state.nextLayer);
  block.dataset.layer = String(state.nextLayer);
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

function isPinching(hand) {
  const palmSize = distance(hand[0], hand[9]) || 0.1;
  const pinchDistance = distance(hand[4], hand[8]);
  const middleExtended = hand[12].y < hand[10].y;
  return pinchDistance < palmSize * 0.34 && middleExtended;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function moveHandCursor(point, grabbing, pinching = false) {
  handCursor.style.setProperty("--cursor-x", `${point.x}px`);
  handCursor.style.setProperty("--cursor-y", `${point.y}px`);
  handCursor.style.transform = `translate3d(${point.x}px, ${point.y}px, 0) scale(${grabbing ? 0.68 : pinching ? 1.28 : 1})`;
  handCursor.classList.add("is-live");
  handCursor.classList.toggle("is-grabbing", grabbing);
  handCursor.classList.toggle("is-pinching", pinching);
}

function updateGesture(label, grabbing, pinching = false) {
  gestureLabel.textContent = label;
  gestureChip.classList.toggle("is-grabbing", grabbing);
  gestureChip.classList.toggle("is-pinching", pinching);
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
  world.style.transform = `translate3d(${state.pan.x}px, ${state.pan.y}px, 0) scale(${state.scale})`;
}

function trackCanvasPointer(event) {
  state.pointer.points.set(event.pointerId, { x: event.clientX, y: event.clientY });
}

function startCanvasPinch() {
  const points = getFirstTwoPoints(state.pointer.points);
  if (!points) return;

  const center = getMidpoint(points[0], points[1]);
  state.pointer.pinch = {
    startDistance: getDistance(points[0], points[1]),
    startScale: state.scale,
    worldPoint: {
      x: (center.x - state.pan.x) / state.scale,
      y: (center.y - state.pan.y) / state.scale
    }
  };
}

function updateCanvasPinch() {
  const points = getFirstTwoPoints(state.pointer.points);
  if (!points || !state.pointer.pinch) return;

  const center = getMidpoint(points[0], points[1]);
  const ratio = getDistance(points[0], points[1]) / state.pointer.pinch.startDistance;
  const nextScale = clamp(state.pointer.pinch.startScale * ratio, 1, 4);

  zoomCanvasAt(center, nextScale, state.pointer.pinch.worldPoint);
}

function zoomCanvasBy(center, factor) {
  const worldPoint = {
    x: (center.x - state.pan.x) / state.scale,
    y: (center.y - state.pan.y) / state.scale
  };
  zoomCanvasAt(center, state.scale * factor, worldPoint);
}

function zoomCanvasAt(center, nextScale, worldPoint) {
  const scale = clamp(nextScale, 1, 4);
  state.scale = scale;
  state.pan.x = center.x - worldPoint.x * scale;
  state.pan.y = center.y - worldPoint.y * scale;
  renderWorld();
}

function getFirstTwoPoints(points) {
  const values = [...points.values()];
  return values.length >= 2 ? values.slice(0, 2) : null;
}

function getDistance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function getMidpoint(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function initLongPressSelection(element) {
  if (!element) return;

  element.classList.add("selectable-bar");

  let timer = null;
  let origin = null;
  let activePointerId = null;
  let suppressNextClick = false;

  const clearTimer = () => {
    if (timer) clearTimeout(timer);
    timer = null;
    origin = null;
    activePointerId = null;
  };

  element.addEventListener("pointerdown", (event) => {
    if (event.button && event.button !== 0) return;
    if (state.pointer.pinch || event.target.closest(".media-block")?.classList.contains("is-held")) return;

    origin = { x: event.clientX, y: event.clientY };
    activePointerId = event.pointerId;
    timer = window.setTimeout(() => {
      suppressNextClick = true;
      selectBar(element);
      timer = null;
    }, LONG_PRESS_MS);
  });

  element.addEventListener("pointermove", (event) => {
    if (event.pointerId !== activePointerId || !origin) return;
    if (getDistance(origin, { x: event.clientX, y: event.clientY }) > LONG_PRESS_MOVE_TOLERANCE) {
      clearTimer();
    }
  });

  element.addEventListener("pointerup", clearTimer);
  element.addEventListener("pointercancel", clearTimer);
  element.addEventListener("lostpointercapture", clearTimer);

  element.addEventListener("click", (event) => {
    if (!suppressNextClick) return;
    suppressNextClick = false;
    event.preventDefault();
    event.stopPropagation();
  }, true);
}

function selectBar(element) {
  if (state.selectedBar === element) return;
  deselectBar();
  state.selectedBar = element;
  element.classList.add("is-bar-selected");
}

function deselectBar() {
  state.selectedBar?.classList.remove("is-bar-selected");
  state.selectedBar = null;
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
    x: (center.x - state.pan.x) / state.scale - 160 + staggerX,
    y: (center.y - state.pan.y) / state.scale - 120 + staggerY
  };
}

function createImageBlock(file, url, position) {
  const block = document.createElement("article");
  block.className = "media-block image-block";
  block.style.left = `${position.x}px`;
  block.style.top = `${position.y}px`;
  block.dataset.objectUrl = url;
  setTopLayer(block);

  const media = document.createElement("img");
  media.src = url;
  media.alt = file.name;
  initZoomTarget(media);

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
  initZoomTarget(canvas);

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
  initLongPressSelection(button);
  return button;
}

function removeMediaBlock(block) {
  const stackId = block.dataset.stackId;
  if (block.dataset.objectUrl) URL.revokeObjectURL(block.dataset.objectUrl);
  if (block.contains(state.selectedBar)) deselectBar();
  block.remove();
  cleanupStack(stackId);
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

function initZoomTarget(element) {
  element.dataset.zoomScale = "1";
  element.dataset.zoomX = "0";
  element.dataset.zoomY = "0";
  applyMediaZoom(element);
}

function getZoomTarget(target) {
  return target.closest?.(".media-block img, .model-preview") || null;
}

function getMediaZoom(element) {
  return {
    scale: Number(element.dataset.zoomScale || 1),
    x: Number(element.dataset.zoomX || 0),
    y: Number(element.dataset.zoomY || 0)
  };
}

function setMediaZoom(element, zoom) {
  const scale = clamp(zoom.scale, 1, 4);
  const x = scale === 1 ? 0 : zoom.x;
  const y = scale === 1 ? 0 : zoom.y;

  element.dataset.zoomScale = String(scale);
  element.dataset.zoomX = String(x);
  element.dataset.zoomY = String(y);
  element.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
}

function applyMediaZoom(element) {
  setMediaZoom(element, getMediaZoom(element));
}

function startMediaPinch(block, points, element) {
  const pair = getFirstTwoPoints(points);
  if (!pair || !element) return null;

  const center = getMidpoint(pair[0], pair[1]);
  const zoom = getMediaZoom(element);
  const rect = getMediaViewportRect(block);

  return {
    element,
    startDistance: getDistance(pair[0], pair[1]),
    startScale: zoom.scale,
    localPoint: {
      x: (center.x - rect.left - zoom.x) / zoom.scale,
      y: (center.y - rect.top - zoom.y) / zoom.scale
    }
  };
}

function updateMediaPinch(block, points, pinch) {
  const pair = getFirstTwoPoints(points);
  if (!pair || !pinch) return;

  const center = getMidpoint(pair[0], pair[1]);
  const rect = getMediaViewportRect(block);
  const ratio = getDistance(pair[0], pair[1]) / pinch.startDistance;
  const scale = clamp(pinch.startScale * ratio, 1, 4);

  setMediaZoom(pinch.element, {
    scale,
    x: center.x - rect.left - pinch.localPoint.x * scale,
    y: center.y - rect.top - pinch.localPoint.y * scale
  });
}

function panMediaBy(element, dx, dy) {
  const zoom = getMediaZoom(element);
  if (zoom.scale <= 1) return;
  setMediaZoom(element, {
    scale: zoom.scale,
    x: zoom.x + dx,
    y: zoom.y + dy
  });
}

function updateStackHover(block) {
  if (!isStackableImageBlock(block)) return;
  if (isStackDetachCoolingDown(block)) return;

  const target = findStackTarget(block);
  clearStackHover(block);
  target?.classList.add("is-stack-target");
}

function settleImageStack(block) {
  if (!isStackableImageBlock(block)) return;
  if (isStackDetachCoolingDown(block)) {
    clearStackHover(block);
    return;
  }

  const target = findStackTarget(block);
  clearStackHover(block);
  if (!target) return;

  mergeImageStacks(block, target);
}

function clearStackHover(exceptBlock) {
  document.querySelectorAll(".image-block.is-stack-target").forEach((block) => {
    if (block !== exceptBlock) block.classList.remove("is-stack-target");
  });
}

function findStackTarget(block) {
  const draggedRect = block.getBoundingClientRect();
  let bestTarget = null;
  let bestRatio = 0;

  document.querySelectorAll(".image-block").forEach((candidate) => {
    if (candidate === block) return;
    if (candidate.dataset.stackId && candidate.dataset.stackId === block.dataset.stackId) return;

    const ratio = getOverlapRatio(draggedRect, candidate.getBoundingClientRect());
    if (ratio > bestRatio) {
      bestRatio = ratio;
      bestTarget = candidate;
    }
  });

  return bestRatio >= STACK_OVERLAP_THRESHOLD ? bestTarget : null;
}

function getOverlapRatio(a, b) {
  const width = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
  const height = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
  const overlapArea = width * height;
  const smallerArea = Math.min(a.width * a.height, b.width * b.height) || 1;
  return overlapArea / smallerArea;
}

function isStackableImageBlock(block) {
  return block.classList.contains("image-block");
}

function createShakeTracker() {
  return {
    samples: [],
    detached: false
  };
}

function trackStackShake(block, tracker, dx, dy) {
  if (!tracker || tracker.detached || !block.dataset.stackId) return false;

  const absX = Math.abs(dx);
  const absY = Math.abs(dy);
  const axis = absX >= absY ? "x" : "y";
  const amount = axis === "x" ? dx : dy;

  if (Math.abs(amount) < SHAKE_MIN_STEP) return false;

  const now = performance.now();
  tracker.samples.push({
    axis,
    direction: amount > 0 ? 1 : -1,
    travel: Math.abs(amount),
    time: now
  });
  tracker.samples = tracker.samples.filter((sample) => now - sample.time <= SHAKE_WINDOW_MS);

  const stats = getShakeStats(tracker.samples);
  if (stats.reversals < SHAKE_REVERSALS_TO_DETACH || stats.travel < SHAKE_MIN_TRAVEL) return false;

  tracker.detached = detachBlockFromStack(block);
  tracker.samples = [];
  return tracker.detached;
}

function getShakeStats(samples) {
  let reversals = 0;
  let travel = 0;
  let previous = null;

  samples.forEach((sample) => {
    travel += sample.travel;
    if (previous && previous.axis === sample.axis && previous.direction !== sample.direction) {
      reversals += 1;
    }
    previous = sample;
  });

  return { reversals, travel };
}

function detachBlockFromStack(block) {
  const stackId = block.dataset.stackId;
  if (!stackId) return false;

  delete block.dataset.stackId;
  block.classList.remove("is-stacked", "is-stack-target");
  block.classList.add("is-unstacking");
  block.dataset.detachedAt = String(performance.now());
  setTopLayer(block);
  cleanupStack(stackId);

  window.setTimeout(() => {
    block.classList.remove("is-unstacking");
    delete block.dataset.detachedAt;
  }, STACK_DETACH_COOLDOWN_MS);

  return true;
}

function isStackDetachCoolingDown(block) {
  const detachedAt = Number(block.dataset.detachedAt || 0);
  return Boolean(detachedAt) && performance.now() - detachedAt < STACK_DETACH_COOLDOWN_MS;
}

function mergeImageStacks(block, target) {
  const stackId = target.dataset.stackId || block.dataset.stackId || `stack-${state.nextStackId}`;
  if (!target.dataset.stackId && !block.dataset.stackId) state.nextStackId += 1;

  const members = new Set([
    ...getStackMembers(block),
    ...getStackMembers(target),
    block,
    target
  ]);

  members.forEach((member) => {
    member.dataset.stackId = stackId;
    member.classList.add("is-stacked");
  });

  bringStackToFront(stackId, block);
}

function getStackMembers(block) {
  const stackId = block.dataset.stackId;
  if (!stackId) return [block];
  return [...document.querySelectorAll(`.image-block[data-stack-id="${stackId}"]`)];
}

function getMoveTargets(block) {
  if (!isStackableImageBlock(block) || !block.dataset.stackId) return [block];
  return getStackMembers(block);
}

function bringStackToFront(stackId, topBlock) {
  const members = [...document.querySelectorAll(`.image-block[data-stack-id="${stackId}"]`)]
    .sort((a, b) => Number(a.dataset.layer || 0) - Number(b.dataset.layer || 0))
    .filter((member) => member !== topBlock);

  members.push(topBlock);
  members.forEach(setTopLayer);
}

function cleanupStack(stackId) {
  if (!stackId) return;
  const members = [...document.querySelectorAll(`.image-block[data-stack-id="${stackId}"]`)];
  if (members.length > 1) return;

  members.forEach((member) => {
    delete member.dataset.stackId;
    member.classList.remove("is-stacked");
  });
}

function getMediaViewportRect(block) {
  const blockRect = block.getBoundingClientRect();
  const captionHeight = block.querySelector(".media-caption")?.getBoundingClientRect().height || 0;

  return {
    left: blockRect.left,
    top: blockRect.top,
    width: blockRect.width,
    height: blockRect.height - captionHeight
  };
}

function enableBlockDrag(block) {
  let dragging = false;
  let didDrag = false;
  let lastPoint = null;
  let shake = null;
  const touch = {
    points: new Map(),
    pinch: null,
    pan: null
  };

  block.addEventListener("pointerdown", (event) => {
    if (event.target.closest(".delete-block")) return;

    if (event.pointerType === "touch") {
      event.preventDefault();
      touch.points.set(event.pointerId, { x: event.clientX, y: event.clientY });
      block.setPointerCapture(event.pointerId);

      if (touch.points.size === 2) {
        touch.pinch = startMediaPinch(block, touch.points, getZoomTarget(event.target) || block.querySelector(".model-preview, img"));
        touch.pan = null;
        dragging = false;
        return;
      }

      const target = getZoomTarget(event.target);
      if (target && getMediaZoom(target).scale > 1) {
        touch.pan = { element: target, lastPoint: { x: event.clientX, y: event.clientY } };
        dragging = false;
        return;
      }
    }

    dragging = true;
    didDrag = false;
    lastPoint = { x: event.clientX, y: event.clientY };
    shake = createShakeTracker();
    block.setPointerCapture(event.pointerId);
  });

  block.querySelector(".delete-block").addEventListener("click", (event) => {
    event.stopPropagation();
    removeMediaBlock(block);
  });

  block.addEventListener("pointermove", (event) => {
    if (event.pointerType === "touch" && touch.points.has(event.pointerId)) {
      event.preventDefault();
      const current = { x: event.clientX, y: event.clientY };
      touch.points.set(event.pointerId, current);

      if (touch.pinch && touch.points.size >= 2) {
        updateMediaPinch(block, touch.points, touch.pinch);
        return;
      }

      if (touch.pan) {
        panMediaBy(touch.pan.element, current.x - touch.pan.lastPoint.x, current.y - touch.pan.lastPoint.y);
        touch.pan.lastPoint = current;
        return;
      }
    }

    if (!dragging || !lastPoint) return;
    const current = { x: event.clientX, y: event.clientY };
    const dx = current.x - lastPoint.x;
    const dy = current.y - lastPoint.y;
    trackStackShake(block, shake, dx, dy);
    moveBlockBy(block, dx, dy);
    updateStackHover(block);
    didDrag = true;
    lastPoint = current;
  });

  block.addEventListener("pointerup", (event) => {
    if (event.pointerType === "touch") {
      touch.points.delete(event.pointerId);
      if (touch.points.size < 2) touch.pinch = null;
      if (!touch.points.size) touch.pan = null;
    }

    dragging = false;
    lastPoint = null;
    shake = null;
    if (didDrag) {
      settleImageStack(block);
    } else {
      clearStackHover(block);
    }
    didDrag = false;
  });

  block.addEventListener("pointercancel", (event) => {
    if (event.pointerType === "touch") {
      touch.points.delete(event.pointerId);
      touch.pinch = null;
      touch.pan = null;
    }

    dragging = false;
    lastPoint = null;
    shake = null;
    didDrag = false;
    clearStackHover(block);
  });
}
