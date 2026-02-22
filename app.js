import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15";

const startCameraBtn = document.getElementById("startCameraBtn");
const inputVideo = document.getElementById("inputVideo");
const overlayCanvas = document.getElementById("overlayCanvas");
const cameraStatus = document.getElementById("cameraStatus");
const stabilityScore = document.getElementById("stabilityScore");
const coordinationScore = document.getElementById("coordinationScore");
const hipExtensionScore = document.getElementById("hipExtensionScore");
const tipsList = document.getElementById("tipsList");

const ctx = overlayCanvas.getContext("2d");
let poseLandmarker;
let running = false;
let rafId;

async function initPoseModel() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function toPercent(value) {
  return `${Math.max(0, Math.min(100, Math.round(value)))}%`;
}

function updateSuggestions({ stable, coordinated, hipOpen }) {
  const tips = [];
  if (stable < 65) tips.push("重心偏移較明顯，嘗試將髖部保持在雙腳支撐區域中央。\n");
  if (coordinated < 65) tips.push("手腳切換不同步，建議先做『腳先到位、手再發力』的節奏練習。\n");
  if (hipOpen < 65) tips.push("髖部延展不足，嘗試在踩高點時先轉髖再伸手，減少手臂硬拉。\n");
  if (!tips.length) {
    tips.push("節奏與重心表現良好，可嘗試更小支點路線提升精準度。");
  }

  tipsList.innerHTML = "";
  tips.forEach((tip) => {
    const li = document.createElement("li");
    li.textContent = tip;
    tipsList.appendChild(li);
  });
}

function evaluatePose(landmarks) {
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];
  const leftWrist = landmarks[15];
  const rightWrist = landmarks[16];
  const leftAnkle = landmarks[27];
  const rightAnkle = landmarks[28];

  const shoulderCenter = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2,
  };

  const hipCenter = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2,
  };

  const feetCenterX = (leftAnkle.x + rightAnkle.x) / 2;
  const centerOffset = Math.abs(hipCenter.x - feetCenterX);
  const stable = 100 - centerOffset * 180;

  const handSpread = distance(leftWrist, rightWrist);
  const footSpread = distance(leftAnkle, rightAnkle);
  const spreadDiff = Math.abs(handSpread - footSpread);
  const coordinated = 100 - spreadDiff * 260;

  const hipToShoulder = Math.abs(hipCenter.y - shoulderCenter.y);
  const hipOpen = 100 - Math.abs(0.2 - hipToShoulder) * 240;

  stabilityScore.textContent = toPercent(stable);
  coordinationScore.textContent = toPercent(coordinated);
  hipExtensionScore.textContent = toPercent(hipOpen);

  updateSuggestions({ stable, coordinated, hipOpen });
}

function drawPose(result) {
  const drawingUtils = new DrawingUtils(ctx);
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  for (const landmarks of result.landmarks || []) {
    drawingUtils.drawLandmarks(landmarks, { radius: 3, color: "#5ad8ff" });
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
      color: "#73ffbf",
      lineWidth: 2,
    });
    evaluatePose(landmarks);
  }
}

async function predictLoop() {
  if (!running) return;
  if (inputVideo.readyState >= 2) {
    overlayCanvas.width = inputVideo.videoWidth;
    overlayCanvas.height = inputVideo.videoHeight;

    const now = performance.now();
    const result = poseLandmarker.detectForVideo(inputVideo, now);
    drawPose(result);
  }

  rafId = requestAnimationFrame(predictLoop);
}

async function startCamera() {
  if (!poseLandmarker) {
    cameraStatus.textContent = "模型載入中...";
    await initPoseModel();
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 },
    audio: false,
  });

  inputVideo.srcObject = stream;
  await inputVideo.play();

  running = true;
  cameraStatus.textContent = "分析中：請面向鏡頭模擬攀爬動作。";
  startCameraBtn.textContent = "分析進行中";
  startCameraBtn.disabled = true;
  predictLoop();
}

startCameraBtn.addEventListener("click", async () => {
  try {
    await startCamera();
  } catch (error) {
    console.error(error);
    cameraStatus.textContent = "無法啟動攝影機或模型，請確認瀏覽器權限與網路。";
  }
});

window.addEventListener("beforeunload", () => {
  running = false;
  cancelAnimationFrame(rafId);
  const stream = inputVideo.srcObject;
  if (stream && stream.getTracks) {
    stream.getTracks().forEach((track) => track.stop());
  }
});
