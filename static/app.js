const video = document.querySelector("#camera");
const overlay = document.querySelector("#overlay");
const capture = document.querySelector("#capture");
const welcome = document.querySelector("#welcome");
const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const statusLabel = document.querySelector("#status");
const pingLabel = document.querySelector("#ping");
const robotStateLabel = document.querySelector("#robotState");
const reasonLabel = document.querySelector("#reason");
const routeLabel = document.querySelector("#route");
const sourceLabel = document.querySelector("#source");

const captureContext = capture.getContext("2d", { alpha: false });
const overlayContext = overlay.getContext("2d");
const INFERENCE_INTERVAL_MS = 333;

let socket;
let waitingForMac = false;
let shouldReconnect = false;
let lastFrameSentAt = 0;
let latestResult;
let cameraPitch = 20;
let motionListenerInstalled = false;

async function enableMotionSensor() {
  if (motionListenerInstalled || typeof DeviceMotionEvent === "undefined") return;

  if (typeof DeviceMotionEvent.requestPermission === "function") {
    try {
      const permission = await DeviceMotionEvent.requestPermission();
      if (permission !== "granted") return;
    } catch {
      return;
    }
  }

  window.addEventListener("devicemotion", (event) => {
    const gravity = event.accelerationIncludingGravity;
    if (!gravity || gravity.z == null || gravity.x == null || gravity.y == null) return;
    const parallelGravity = Math.hypot(gravity.x, gravity.y);
    const measured = Math.abs(Math.atan2(gravity.z, parallelGravity) * 180 / Math.PI);
    if (Number.isFinite(measured)) cameraPitch = cameraPitch * 0.82 + measured * 0.18;
  });
  motionListenerInstalled = true;
}

function setStatus(text, connected = false) {
  statusLabel.textContent = text;
  statusLabel.classList.toggle("connected", connected);
}

function displayedVideoRect() {
  const viewportWidth = overlay.clientWidth;
  const viewportHeight = overlay.clientHeight;
  const videoWidth = video.videoWidth || viewportWidth;
  const videoHeight = video.videoHeight || viewportHeight;
  const videoAspect = videoWidth / videoHeight;
  const viewportAspect = viewportWidth / viewportHeight;

  if (viewportAspect > videoAspect) {
    const height = viewportHeight;
    const width = height * videoAspect;
    return { x: (viewportWidth - width) / 2, y: 0, width, height };
  }

  const width = viewportWidth;
  const height = width / videoAspect;
  return { x: 0, y: (viewportHeight - height) / 2, width, height };
}

function resizeOverlay() {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.round(overlay.clientWidth * dpr);
  const height = Math.round(overlay.clientHeight * dpr);
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
  overlayContext.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawLabel(text, x, y, color) {
  overlayContext.font = "bold 13px ui-monospace, monospace";
  const width = overlayContext.measureText(text).width + 12;
  const top = Math.max(0, y - 23);
  overlayContext.fillStyle = color;
  overlayContext.fillRect(x, top, width, 23);
  overlayContext.fillStyle = "#020302";
  overlayContext.fillText(text, x + 6, top + 16);
}

function mixPoint(a, b, amount) {
  return {
    x: a.x + (b.x - a.x) * amount,
    y: a.y + (b.y - a.y) * amount,
  };
}

function normalizedPoint(point, rect) {
  return {
    x: rect.x + point[0] * rect.width,
    y: rect.y + point[1] * rect.height,
  };
}

function drawDepthObstacles(result, rect) {
  const obstacles = result.depth && result.depth.obstacles;
  if (!obstacles || !obstacles.length) return;
  overlayContext.setLineDash([9, 6]);
  for (const obstacle of obstacles) {
    const [x1, y1, x2, y2] = obstacle.box;
    const x = rect.x + x1 * rect.width;
    const y = rect.y + y1 * rect.height;
    const boxWidth = (x2 - x1) * rect.width;
    const boxHeight = (y2 - y1) * rect.height;
    overlayContext.strokeStyle = "#ffb020";
    overlayContext.lineWidth = 3;
    overlayContext.strokeRect(x, y, boxWidth, boxHeight);
    drawLabel(`DEPTH ${obstacle.score}`, x, y + boxHeight + 23, "#ffb020");
  }
  overlayContext.setLineDash([]);
}

function drawResult(result) {
  resizeOverlay();
  const width = overlay.clientWidth;
  const height = overlay.clientHeight;
  overlayContext.clearRect(0, 0, width, height);

  const rect = displayedVideoRect();
  const topLeft = normalizedPoint(result.corridor.top_left, rect);
  const topRight = normalizedPoint(result.corridor.top_right, rect);
  const bottomRight = normalizedPoint(result.corridor.bottom_right, rect);
  const bottomLeft = normalizedPoint(result.corridor.bottom_left, rect);
  const laneNames = ["left", "center", "right"];
  for (let index = 0; index < laneNames.length; index += 1) {
    const lane = laneNames[index];
    const start = index / 3;
    const end = (index + 1) / 3;
    const laneTopLeft = mixPoint(topLeft, topRight, start);
    const laneTopRight = mixPoint(topLeft, topRight, end);
    const laneBottomRight = mixPoint(bottomLeft, bottomRight, end);
    const laneBottomLeft = mixPoint(bottomLeft, bottomRight, start);
    const occupied = result.lanes[lane];
    overlayContext.fillStyle = occupied
      ? "rgba(255, 40, 40, 0.10)"
      : "rgba(56, 255, 117, 0.045)";
    overlayContext.strokeStyle = occupied
      ? "rgba(255, 69, 69, 0.90)"
      : "rgba(56, 255, 117, 0.55)";
    overlayContext.lineWidth = 2;
    overlayContext.beginPath();
    overlayContext.moveTo(laneTopLeft.x, laneTopLeft.y);
    overlayContext.lineTo(laneTopRight.x, laneTopRight.y);
    overlayContext.lineTo(laneBottomRight.x, laneBottomRight.y);
    overlayContext.lineTo(laneBottomLeft.x, laneBottomLeft.y);
    overlayContext.closePath();
    overlayContext.fill();
    overlayContext.stroke();
  }

  for (const detection of result.detections) {
    const [x1, y1, x2, y2] = detection.box;
    const x = rect.x + x1 * rect.width;
    const y = rect.y + y1 * rect.height;
    const boxWidth = (x2 - x1) * rect.width;
    const boxHeight = (y2 - y1) * rect.height;
    const color = detection.dangerous ? "#ff4545" : "#38ff75";

    overlayContext.strokeStyle = color;
    overlayContext.lineWidth = detection.dangerous ? 4 : 3;
    overlayContext.strokeRect(x, y, boxWidth, boxHeight);
    drawLabel(
      `${detection.display_name} ${Math.round(detection.confidence * 100)}%`,
      x,
      y,
      color,
    );
  }

  drawDepthObstacles(result, rect);

  const routeLane = { LEFT: 0, STRAIGHT: 1, RIGHT: 2 }[result.route];
  const startX = (bottomLeft.x + bottomRight.x) / 2;
  const startY = bottomLeft.y - 18;
  if (routeLane !== undefined) {
    const targetX = mixPoint(topLeft, topRight, (routeLane + 0.5) / 3).x;
    const targetY = topLeft.y + 22;
    const controlX = startX + (targetX - startX) * 0.28;
    const controlY = startY - rect.height * 0.34;
    const angle = Math.atan2(targetY - controlY, targetX - controlX);
    overlayContext.strokeStyle = "#ffd428";
    overlayContext.fillStyle = "#ffd428";
    overlayContext.lineWidth = 7;
    overlayContext.beginPath();
    overlayContext.moveTo(startX, startY);
    overlayContext.quadraticCurveTo(controlX, controlY, targetX, targetY);
    overlayContext.stroke();
    overlayContext.beginPath();
    overlayContext.moveTo(targetX, targetY);
    overlayContext.lineTo(
      targetX - 22 * Math.cos(angle - 0.55),
      targetY - 22 * Math.sin(angle - 0.55),
    );
    overlayContext.lineTo(
      targetX - 22 * Math.cos(angle + 0.55),
      targetY - 22 * Math.sin(angle + 0.55),
    );
    overlayContext.closePath();
    overlayContext.fill();
  } else {
    overlayContext.strokeStyle = "#ff3838";
    overlayContext.lineWidth = 8;
    overlayContext.beginPath();
    overlayContext.moveTo(startX - 28, startY - 58);
    overlayContext.lineTo(startX + 28, startY);
    overlayContext.moveTo(startX + 28, startY - 58);
    overlayContext.lineTo(startX - 28, startY);
    overlayContext.stroke();
  }

  document.body.classList.toggle("stop", result.state === "STOP");
  document.body.classList.toggle("go", result.state === "GO");
  robotStateLabel.textContent = result.state;
  reasonLabel.textContent = result.reason_display || "путь свободен";
  routeLabel.textContent = `маршрут ${result.route}`;
  sourceLabel.textContent = result.source
    ? `источник ${result.source.toUpperCase()}`
    : "источник —";
}

async function sendFrame() {
  if (!socket || socket.readyState !== WebSocket.OPEN || waitingForMac) return;
  if (!video.videoWidth || !video.videoHeight) return;

  const maxWidth = 640;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  capture.width = Math.round(video.videoWidth * scale);
  capture.height = Math.round(video.videoHeight * scale);
  captureContext.drawImage(video, 0, 0, capture.width, capture.height);

  waitingForMac = true;
  capture.toBlob(
    async (blob) => {
      if (!blob || !socket || socket.readyState !== WebSocket.OPEN) {
        waitingForMac = false;
        return;
      }
      socket.send(JSON.stringify({
        type: "orientation",
        pitch_degrees: Math.round(cameraPitch * 10) / 10,
      }));
      socket.send(await blob.arrayBuffer());
      lastFrameSentAt = performance.now();
    },
    "image/jpeg",
    0.68,
  );
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);

  socket.onopen = () => {
    setStatus("Mac подключён", true);
    sendFrame();
  };

  socket.onmessage = (event) => {
    const result = JSON.parse(event.data);
    if (result.error) {
      waitingForMac = false;
      setStatus("ошибка кадра");
      window.setTimeout(sendFrame, INFERENCE_INTERVAL_MS);
      return;
    }
    const roundTripMs = Math.round(performance.now() - lastFrameSentAt);
    pingLabel.textContent = `PING ${roundTripMs} MS`;
    latestResult = result;
    drawResult(result);

    waitingForMac = false;
    const elapsed = performance.now() - lastFrameSentAt;
    const delay = Math.max(0, INFERENCE_INTERVAL_MS - elapsed);
    window.setTimeout(sendFrame, delay);
  };

  socket.onclose = () => {
    waitingForMac = false;
    if (shouldReconnect) {
      setStatus("соединение потеряно");
      pingLabel.textContent = "PING —";
      window.setTimeout(connect, 1200);
    }
  };

  socket.onerror = () => socket.close();
}

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  setStatus("запрос камеры…");

  try {
    await enableMotionSensor();
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    await video.play();
    shouldReconnect = true;
    stopButton.hidden = false;
    welcome.classList.add("hidden");
    resizeOverlay();
    connect();
  } catch (error) {
    startButton.disabled = false;
    setStatus("камера недоступна");
    alert(`Не удалось открыть камеру: ${error.message}`);
  }
});

stopButton.addEventListener("click", () => {
  shouldReconnect = false;
  waitingForMac = false;

  if (socket) {
    socket.onclose = null;
    socket.close();
    socket = undefined;
  }

  if (video.srcObject) {
    for (const track of video.srcObject.getTracks()) track.stop();
    video.srcObject = null;
  }

  latestResult = undefined;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  document.body.classList.remove("stop", "go");
  robotStateLabel.textContent = "—";
  reasonLabel.textContent = "ожидание";
  routeLabel.textContent = "маршрут —";
  sourceLabel.textContent = "источник —";
  pingLabel.textContent = "PING —";
  stopButton.hidden = true;
  startButton.disabled = false;
  welcome.classList.remove("hidden");
  setStatus("трансляция остановлена");
});

window.addEventListener("resize", () => {
  resizeOverlay();
  if (latestResult) drawResult(latestResult);
});
window.addEventListener("orientationchange", () => {
  window.setTimeout(() => {
    resizeOverlay();
    if (latestResult) drawResult(latestResult);
  }, 150);
});
