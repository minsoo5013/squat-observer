const $ = (selector) => document.querySelector(selector);

const state = {
  stream: null,
  recorder: null,
  recordChunks: [],
  recordTimer: null,
  sourceUrl: null,
  sourceName: null,
  sourceKind: null,
  poseLandmarker: null,
  pyodide: null,
  frames: null,
  result: null,
  analysisFps: 30,
  analysisElapsedSeconds: null,
  frameWidth: 0,
  frameHeight: 0,
  selectedRep: 0,
  evidenceToken: 0,
  frameDrawQueue: Promise.resolve(),
  repThumbnailObserver: null,
  comparisonMetric: null,
  recommendations: null,
  retrySession: null,
  comparison: null,
  samplePair: null,
  samplePhase: 0,
  captureMode: 'solo',
  facingMode: 'user',
  capturePhase: 'idle',
  captureStatusKey: null,
  captureMonitorTimer: null,
  captureMonitorActive: false,
  captureMonitorBusy: false,
  monitorCanvas: null,
  monitorContext: null,
  monitorFootSamples: [],
  readinessReadySince: null,
  lastReadiness: null,
  countdownToken: 0,
  poseTimestampMs: 0,
  poseLandmarkerPromise: null,
  audioEnabled: true,
  audioContext: null,
  wakeLock: null,
  recordingStartedAt: null,
  stoppingRecording: false,
  analysisEndSeconds: null,
  captureRecordMeta: null,
  roughRepCount: 0,
  inSquat: false,
  standBaselineSamples: [],
  standBaseline: null,
  finishStandSince: null,
  lastGoodCaptureSeconds: 0,
  invalidExitSamples: 0,
  cameraRequestedAt: null,
  cameraReadyAt: null,
  readinessCompletedAt: null,
  countdownCancelCount: 0,
  cameraSettings: null,
};

const video = $('#preview-video');
const cameraPlaceholder = $('#camera-placeholder');
const cameraButton = $('#camera-button');
const recordButton = $('#record-button');
const stopButton = $('#stop-button');
const fileInput = $('#video-file');
const countdown = $('#countdown');
const recordBadge = $('#record-badge');
const recordTime = $('#record-time');
const captureStatus = $('#capture-status');
const cameraStage = $('#camera-stage');
const previewOverlay = $('#preview-overlay');
const facingControl = $('#facing-control');
const facingSelect = $('#facing-select');
const soundToggle = $('#sound-toggle');
const fileButton = $('#file-button');

const CONNECTIONS = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
  [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
];

const MAX_VIDEO_SECONDS = 90;

// 촬영 보조용 잠정값. run63의 한 영상에 맞추지 않도록 몸 높이·정지 범위를 넓게 두었다.
// 자세 판정에는 사용하지 않으며 실기기 로그를 모은 뒤 별도로 재검토한다.
const SOLO_CAPTURE_CONFIG = Object.freeze({
  previewFps: 8,
  edgeMargin: 0.06,
  minVisibility: 0.65,
  minBodyHeight: 0.42,
  maxBodyHeight: 0.78,
  footStableWindowMs: 1000,
  footStableMaxMove: 0.018,
  readyHoldMs: 800,
  countdownSeconds: 3,
  initialStandMs: 2000,
  finishStandMs: 2000,
  maxRecordingMs: MAX_VIDEO_SECONDS * 1000,
  minAutoStopReps: 6,
  minExitStopReps: 4,
  squatEnterRatio: 0.80,
  squatExitRatio: 0.90,
  exitFootMove: 0.035,
  invalidExitSamples: 2,
});

function setHidden(element, hidden) {
  element.hidden = hidden;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function once(target, event, errorEvent = 'error', timeoutMs = 12000) {
  return new Promise((resolve, reject) => {
    let timer;
    const cleanup = () => {
      clearTimeout(timer);
      target.removeEventListener(event, onSuccess);
      target.removeEventListener(errorEvent, onError);
    };
    const onSuccess = (value) => { cleanup(); resolve(value); };
    const onError = () => { cleanup(); reject(new Error(`미디어에서 ${event} 이벤트를 기다리는 중 오류가 발생했습니다.`)); };
    target.addEventListener(event, onSuccess, { once: true });
    target.addEventListener(errorEvent, onError, { once: true });
    timer = setTimeout(() => {
      cleanup();
      reject(new Error(`미디어 준비 시간이 ${Math.round(timeoutMs / 1000)}초를 넘었습니다.`));
    }, timeoutMs);
  });
}

function formatClock(seconds) {
  const value = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`;
}

function formatRemaining(seconds) {
  if (!finite(seconds) || seconds < 0) return '남은 시간 계산 중';
  if (seconds < 2) return '마무리 중';
  return `약 ${formatClock(Math.ceil(seconds))} 남음`;
}

function finite(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function percent(value) {
  return finite(value) ? `${Math.round(value * 100)}%` : '확인 어려움';
}

function multiple(value) {
  return finite(value) ? `${value.toFixed(1)}배` : '확인 어려움';
}

function depthDrop(value) {
  return finite(value) ? 1 - value : null;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function primaryFeedback(result = state.result) {
  return (result?.feedback || []).find((item) => item.primary) || null;
}

function updateStep(step) {
  document.querySelectorAll('.stepper li').forEach((item, index) => {
    item.classList.toggle('active', index === step - 1);
  });
}

function showAnalysis(title, detail, progress = 4, count = '') {
  setHidden($('#capture-section'), true);
  setHidden($('#results-section'), true);
  setHidden($('#analysis-section'), false);
  $('#analysis-title').textContent = title;
  $('#analysis-detail').textContent = detail;
  const safeProgress = Math.max(4, Math.min(100, progress));
  $('#analysis-progress').style.width = `${safeProgress}%`;
  $('.progress').setAttribute('aria-valuenow', String(Math.round(safeProgress)));
  $('.progress').setAttribute('aria-valuetext', count || `${Math.round(safeProgress)}% 진행`);
  $('#analysis-count').textContent = count;
  updateStep(2);
  $('#analysis-section').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function showAnalysisError(error) {
  const rawMessage = error instanceof Error ? error.message : String(error);
  console.error(error);
  const message = /Traceback|PythonError|\/pyodide\//i.test(rawMessage)
    ? '영상 분석 계산 중 오류가 발생했습니다. 같은 영상을 다시 선택해 주세요. 같은 오류가 반복되면 이 화면을 캡처해 알려 주세요.'
    : rawMessage;
  showAnalysis('분석을 마치지 못했습니다.', message, 100);
  document.querySelector('.analysis-retry')?.remove();
  $('#analysis-detail').insertAdjacentHTML(
    'afterend',
    '<button class="button button-quiet analysis-retry">촬영 화면으로 돌아가기</button>',
  );
  $('.analysis-retry').addEventListener('click', () => restart({ preserveRetry: Boolean(state.retrySession) }));
}

function setCaptureStatus(key, message, { speak = false, metrics = null } = {}) {
  const changed = state.captureStatusKey !== key;
  if (changed) {
    console.info('[solo-capture]', {
      at: new Date().toISOString(),
      from: state.captureStatusKey,
      to: key,
      metrics,
    });
    state.captureStatusKey = key;
  }
  state.capturePhase = key;
  captureStatus.textContent = message;
  setHidden(captureStatus, false);
  if (changed && speak) speakCapture(message);
}

function hideCaptureStatus() {
  state.captureStatusKey = null;
  state.capturePhase = 'idle';
  captureStatus.textContent = '';
  setHidden(captureStatus, true);
}

async function ensureAudioReady() {
  if (!state.audioEnabled) return;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return;
  state.audioContext ||= new AudioContextClass();
  if (state.audioContext.state === 'suspended') await state.audioContext.resume().catch(() => {});
}

function playCountdownBeep() {
  if (!state.audioEnabled || !state.audioContext) return;
  const context = state.audioContext;
  const oscillator = context.createOscillator();
  const gain = context.createGain();
  oscillator.frequency.value = 880;
  gain.gain.setValueAtTime(0.0001, context.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.22, context.currentTime + 0.015);
  gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.16);
  oscillator.connect(gain).connect(context.destination);
  oscillator.start();
  oscillator.stop(context.currentTime + 0.18);
}

function speakCapture(message) {
  if (!state.audioEnabled || !('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(message);
  utterance.lang = 'ko-KR';
  utterance.rate = 1;
  window.speechSynthesis.speak(utterance);
}

async function requestWakeLock() {
  if (!('wakeLock' in navigator) || state.wakeLock) return;
  try {
    state.wakeLock = await navigator.wakeLock.request('screen');
    state.wakeLock.addEventListener('release', () => { state.wakeLock = null; }, { once: true });
  } catch (_) {
    // 지원하지 않거나 권한이 없으면 촬영은 그대로 진행한다.
  }
}

async function releaseWakeLock() {
  const lock = state.wakeLock;
  state.wakeLock = null;
  if (lock) await lock.release().catch(() => {});
}

function clearPreviewOverlay() {
  const context = previewOverlay.getContext('2d');
  context?.clearRect(0, 0, previewOverlay.width, previewOverlay.height);
}

function stopCaptureMonitor() {
  state.captureMonitorActive = false;
  state.captureMonitorBusy = false;
  clearTimeout(state.captureMonitorTimer);
  state.captureMonitorTimer = null;
  state.countdownToken += 1;
  setHidden(countdown, true);
}

function stopStream() {
  stopCaptureMonitor();
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  video.srcObject = null;
  cameraStage.classList.remove('mirrored');
  clearPreviewOverlay();
  void releaseWakeLock();
}

function chooseRecorderMime() {
  if (!window.MediaRecorder) return '';
  const choices = [
    'video/mp4;codecs=avc1.42E01E',
    'video/mp4',
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  return choices.find((type) => MediaRecorder.isTypeSupported(type)) || '';
}

function selectCaptureMode(mode, { preserveStream = false } = {}) {
  if (!['solo', 'assisted', 'upload'].includes(mode)) return;
  if (state.recorder?.state === 'recording') return;
  if (!preserveStream) stopStream();
  state.captureMode = mode;
  state.facingMode = mode === 'solo' ? 'user' : facingSelect.value;
  document.querySelectorAll('[data-capture-mode]').forEach((button) => {
    const active = button.dataset.captureMode === mode;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });
  const descriptions = {
    solo: '전면 카메라로 준비 상태를 확인한 뒤 자동으로 촬영을 시작하고 마칩니다.',
    assisted: '촬영자가 전면 또는 후면 카메라를 고르고 시작·종료 버튼을 누릅니다.',
    upload: '앞뒤로 걸어가거나 휴대폰을 만지는 부분을 잘라낸 기존 영상을 선택합니다.',
  };
  $('#capture-mode-description').textContent = descriptions[mode];
  setHidden(cameraButton, mode === 'upload');
  setHidden(facingControl, mode !== 'assisted');
  setHidden(recordButton, true);
  setHidden(stopButton, true);
  setHidden(soundToggle, mode === 'upload');
  setHidden(fileButton, mode !== 'upload');
  setHidden($('#mobile-upload-note'), mode !== 'upload');
  setHidden($('#capture-helper'), mode === 'upload');
  setHidden(cameraPlaceholder, false);
  hideCaptureStatus();
  const placeholder = {
    solo: ['혼자 촬영 준비', '휴대폰을 허리 높이에 세우고 2~3걸음 물러나 주세요.'],
    assisted: ['촬영 준비', '카메라 방향을 고르고 촬영자가 시작해 주세요.'],
    upload: ['기존 영상 선택', '앞뒤 이동 장면을 잘라낸 영상을 선택해 주세요.'],
  }[mode];
  $('#camera-placeholder-title').textContent = placeholder[0];
  $('#camera-placeholder-detail').textContent = placeholder[1];
  cameraButton.textContent = mode === 'solo' ? '전면 카메라 시작' : '카메라 켜기';
}

async function enableCamera() {
  cameraButton.disabled = true;
  try {
    await ensureAudioReady();
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('이 브라우저에서는 카메라 촬영을 지원하지 않습니다. 저장된 영상을 선택해 주세요.');
    }
    stopStream();
    state.cameraRequestedAt = performance.now();
    state.cameraReadyAt = null;
    state.readinessCompletedAt = null;
    state.countdownCancelCount = 0;
    state.cameraSettings = null;
    const facingMode = state.captureMode === 'solo' ? 'user' : facingSelect.value;
    state.facingMode = facingMode;
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1080 },
        height: { ideal: 1920 },
        frameRate: { ideal: 30, max: 30 },
      },
    });
    video.removeAttribute('src');
    video.srcObject = state.stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
    state.cameraReadyAt = performance.now();
    state.cameraSettings = state.stream.getVideoTracks()[0]?.getSettings?.() || null;
    cameraStage.classList.toggle('mirrored', facingMode === 'user');
    setHidden(cameraPlaceholder, true);
    setHidden(recordButton, state.captureMode !== 'assisted');
    cameraButton.textContent = '카메라 다시 연결';
    await requestWakeLock();
    if (state.captureMode === 'solo') {
      setCaptureStatus('loading_detector', '준비 감지 도구를 불러오고 있습니다');
      await startCaptureMonitor();
    } else {
      setCaptureStatus('assisted_ready', '촬영자가 시작 버튼을 눌러 주세요');
    }
  } catch (error) {
    stopStream();
    alert(error instanceof Error ? error.message : String(error));
  } finally {
    cameraButton.disabled = false;
  }
}

function monitorPointUsable(point, minVisibility = SOLO_CAPTURE_CONFIG.minVisibility) {
  return Boolean(point) && finite(point.x) && finite(point.y) && (point.visibility ?? 1) >= minVisibility;
}

function averagePoint(points) {
  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
  };
}

function readinessFromLandmarks(landmarks, now) {
  const get = (index) => landmarks?.[index];
  const nose = get(0);
  const shoulders = [get(11), get(12)];
  const hips = [get(23), get(24)];
  const knees = [get(25), get(26)];
  const ankles = [get(27), get(28)];
  const toes = [get(31), get(32)];
  const torsoSeen = [...shoulders, ...hips].filter((point) => monitorPointUsable(point)).length >= 3;
  if (!torsoSeen) return { ready: false, key: 'no_person', message: '화면 안으로 들어와 주세요', frameComplete: false };
  if (!monitorPointUsable(nose)) return { ready: false, key: 'head_cut', message: '머리까지 보이도록 조금 뒤로 가 주세요', frameComplete: false };
  if (!ankles.every((point) => monitorPointUsable(point))) return { ready: false, key: 'feet_cut', message: '발끝까지 보이도록 조금 뒤로 가 주세요', frameComplete: false };
  const core = [nose, ...shoulders, ...hips, ...knees, ...ankles];
  if (!core.every((point) => monitorPointUsable(point))) {
    return { ready: false, key: 'weak_tracking', message: '밝은 곳에서, 창문이나 조명을 등지지 않게 서 주세요', frameComplete: false };
  }
  const feet = toes.every((point) => monitorPointUsable(point)) ? toes : ankles;
  if (Math.max(...feet.map((point) => point.y)) > 0.98) {
    return { ready: false, key: 'feet_cut', message: '발끝까지 보이도록 조금 뒤로 가 주세요', frameComplete: false };
  }
  if (nose.y < 0.018) return { ready: false, key: 'head_cut', message: '머리까지 보이도록 조금 뒤로 가 주세요', frameComplete: false };
  const allX = [...core, ...feet].map((point) => point.x);
  const allY = [...core, ...feet].map((point) => point.y);
  const bodyHeight = Math.max(...allY) - Math.min(...allY);
  const hipCenter = averagePoint(hips);
  const ankleCenter = averagePoint(ankles);
  const hipAnkle = ankleCenter.y - hipCenter.y;
  const frameComplete = Math.min(...allX) > 0.01 && Math.max(...allX) < 0.99
    && Math.min(...allY) > 0.005 && Math.max(...allY) < 0.995;
  const footPair = feet.map((point) => ({ x: point.x, y: point.y }));
  state.monitorFootSamples.push({ at: now, feet: footPair });
  state.monitorFootSamples = state.monitorFootSamples.filter((sample) => now - sample.at <= SOLO_CAPTURE_CONFIG.footStableWindowMs * 1.7);
  const prior = state.monitorFootSamples.find((sample) => now - sample.at >= SOLO_CAPTURE_CONFIG.footStableWindowMs);
  const footMotion = prior ? Math.max(...footPair.map((point, index) => Math.hypot(
    point.x - prior.feet[index].x,
    point.y - prior.feet[index].y,
  ))) : Infinity;
  if (bodyHeight > SOLO_CAPTURE_CONFIG.maxBodyHeight) {
    return { ready: false, key: 'too_close', message: '조금 뒤로 가 주세요', bodyHeight, hipAnkle, footMotion, frameComplete };
  }
  if (bodyHeight < SOLO_CAPTURE_CONFIG.minBodyHeight) {
    return { ready: false, key: 'too_far', message: '조금 앞으로 와 주세요', bodyHeight, hipAnkle, footMotion, frameComplete };
  }
  if (Math.min(...allX) < SOLO_CAPTURE_CONFIG.edgeMargin || Math.max(...allX) > 1 - SOLO_CAPTURE_CONFIG.edgeMargin) {
    return { ready: false, key: 'edge', message: '화면 가운데로 와 주세요', bodyHeight, hipAnkle, footMotion, frameComplete };
  }
  const ready = finite(footMotion) && footMotion <= SOLO_CAPTURE_CONFIG.footStableMaxMove;
  return {
    ready,
    key: ready ? 'ready' : 'moving',
    message: ready ? '준비되었습니다' : '그 자리에서 가만히 서 주세요',
    bodyHeight,
    hipAnkle,
    footMotion,
    frameComplete,
  };
}

function drawMonitorOverlay(landmarks) {
  if (!video.videoWidth || !video.videoHeight) return;
  if (previewOverlay.width !== video.videoWidth || previewOverlay.height !== video.videoHeight) {
    previewOverlay.width = video.videoWidth;
    previewOverlay.height = video.videoHeight;
  }
  const context = previewOverlay.getContext('2d');
  context.clearRect(0, 0, previewOverlay.width, previewOverlay.height);
  drawSkeleton(context, landmarks, previewOverlay.width, previewOverlay.height);
}

function medianNumber(values) {
  const usable = values.filter(finite).sort((a, b) => a - b);
  if (!usable.length) return null;
  const middle = Math.floor(usable.length / 2);
  return usable.length % 2 ? usable[middle] : (usable[middle - 1] + usable[middle]) / 2;
}

function cancelSoloCountdown(readiness) {
  state.countdownToken += 1;
  state.countdownCancelCount += 1;
  setHidden(countdown, true);
  state.readinessReadySince = null;
  setCaptureStatus(readiness.key, readiness.message, { metrics: readiness });
}

async function beginSoloCountdown() {
  if (state.capturePhase === 'countdown' || !state.lastReadiness?.ready) return;
  const token = ++state.countdownToken;
  setCaptureStatus('countdown', '준비되었습니다', { speak: true, metrics: state.lastReadiness });
  setHidden(countdown, false);
  for (let value = SOLO_CAPTURE_CONFIG.countdownSeconds; value >= 1; value -= 1) {
    if (token !== state.countdownToken || !state.captureMonitorActive) return;
    if (!state.lastReadiness?.ready) {
      cancelSoloCountdown(state.lastReadiness || { key: 'moving', message: '그 자리에서 가만히 서 주세요' });
      return;
    }
    countdown.textContent = String(value);
    playCountdownBeep();
    await sleep(1000);
  }
  if (token !== state.countdownToken || !state.lastReadiness?.ready) {
    cancelSoloCountdown(state.lastReadiness || { key: 'moving', message: '그 자리에서 가만히 서 주세요' });
    return;
  }
  setHidden(countdown, true);
  await beginRecording({ automatic: true });
}

function handleRecordingMonitor(readiness, now) {
  const elapsedMs = now - state.recordingStartedAt;
  const elapsedSeconds = Math.max(0, elapsedMs / 1000);
  if (readiness.frameComplete && finite(readiness.hipAnkle)) {
    state.invalidExitSamples = 0;
    if (elapsedMs <= SOLO_CAPTURE_CONFIG.initialStandMs + 400) {
      state.standBaselineSamples.push(readiness.hipAnkle);
    }
    if (!finite(state.standBaseline) && elapsedMs >= SOLO_CAPTURE_CONFIG.initialStandMs) {
      state.standBaseline = medianNumber(state.standBaselineSamples);
      console.info('[solo-capture] standing baseline', { value: state.standBaseline, samples: state.standBaselineSamples.length });
    }
    const ratio = finite(state.standBaseline) && state.standBaseline > 0
      ? readiness.hipAnkle / state.standBaseline
      : null;
    const leaving = state.roughRepCount >= SOLO_CAPTURE_CONFIG.minExitStopReps
      && !state.inSquat
      && finite(readiness.footMotion)
      && readiness.footMotion > SOLO_CAPTURE_CONFIG.exitFootMove;
    if (leaving) {
      void stopRecording({ reason: 'walk_away', analysisEndSeconds: state.lastGoodCaptureSeconds });
      return;
    }
    state.lastGoodCaptureSeconds = elapsedSeconds;
    if (elapsedMs < SOLO_CAPTURE_CONFIG.initialStandMs) {
      setCaptureStatus('recording_initial_stand', '그대로 2초 서 계세요');
      return;
    }
    if (finite(ratio)) {
      if (!state.inSquat && ratio < SOLO_CAPTURE_CONFIG.squatEnterRatio) {
        state.inSquat = true;
        state.finishStandSince = null;
        console.info('[solo-capture] descent detected', { elapsedSeconds, ratio });
      } else if (state.inSquat && ratio > SOLO_CAPTURE_CONFIG.squatExitRatio) {
        state.inSquat = false;
        state.roughRepCount += 1;
        state.finishStandSince = now;
        console.info('[solo-capture] repetition completed', { rep: state.roughRepCount, elapsedSeconds, ratio });
      }
      const finishReady = state.roughRepCount >= SOLO_CAPTURE_CONFIG.minAutoStopReps
        && !state.inSquat
        && ratio > SOLO_CAPTURE_CONFIG.squatExitRatio
        && finite(readiness.footMotion)
        && readiness.footMotion <= SOLO_CAPTURE_CONFIG.footStableMaxMove;
      if (finishReady && state.finishStandSince && now - state.finishStandSince >= SOLO_CAPTURE_CONFIG.finishStandMs) {
        void stopRecording({ reason: 'standing_complete' });
        return;
      }
    }
    const message = state.roughRepCount >= SOLO_CAPTURE_CONFIG.minAutoStopReps
      ? `${state.roughRepCount}회 · 다 하셨으면 제자리에서 2초 서 계세요`
      : state.roughRepCount > 0
        ? `${state.roughRepCount}회 · 8회 권장 (최소 6회)`
        : '시작하세요 · 8회 권장 (최소 6회)';
    setCaptureStatus(`recording_${state.roughRepCount}`, message);
  } else if (state.roughRepCount >= SOLO_CAPTURE_CONFIG.minExitStopReps) {
    state.invalidExitSamples += 1;
    if (state.invalidExitSamples >= SOLO_CAPTURE_CONFIG.invalidExitSamples) {
      void stopRecording({ reason: 'left_frame', analysisEndSeconds: state.lastGoodCaptureSeconds });
    }
  }
}

function handleMonitorLandmarks(landmarks, now) {
  const readiness = readinessFromLandmarks(landmarks, now);
  state.lastReadiness = readiness;
  if (state.recorder?.state === 'recording') {
    handleRecordingMonitor(readiness, now);
    return;
  }
  if (state.capturePhase === 'countdown') {
    if (!readiness.ready) cancelSoloCountdown(readiness);
    return;
  }
  if (!readiness.ready) {
    state.readinessReadySince = null;
    setCaptureStatus(readiness.key, readiness.message, { metrics: readiness });
    return;
  }
  state.readinessReadySince ||= now;
  state.readinessCompletedAt ||= now;
  setCaptureStatus('ready', '준비되었습니다', { metrics: readiness });
  if (now - state.readinessReadySince >= SOLO_CAPTURE_CONFIG.readyHoldMs) void beginSoloCountdown();
}

async function startCaptureMonitor() {
  state.captureMonitorActive = true;
  state.monitorFootSamples = [];
  state.readinessReadySince = null;
  try {
    const landmarker = await initPoseLandmarker({ quiet: true });
    if (!state.captureMonitorActive || !state.stream) return;
    const [width, height] = analysisCanvasSize(video.videoWidth, video.videoHeight, 320);
    state.monitorCanvas ||= document.createElement('canvas');
    state.monitorCanvas.width = width;
    state.monitorCanvas.height = height;
    state.monitorContext = state.monitorCanvas.getContext('2d', { alpha: false });
    console.info('[solo-capture] provisional config', SOLO_CAPTURE_CONFIG);
    const intervalMs = 1000 / SOLO_CAPTURE_CONFIG.previewFps;
    const tick = async () => {
      if (!state.captureMonitorActive || !state.stream || state.captureMonitorBusy) return;
      const started = performance.now();
      state.captureMonitorBusy = true;
      try {
        if (video.readyState >= 2) {
          state.monitorContext.drawImage(video, 0, 0, width, height);
          state.poseTimestampMs += intervalMs;
          const result = landmarker.detectForVideo(state.monitorCanvas, state.poseTimestampMs);
          const landmarks = result.landmarks?.[0] || null;
          drawMonitorOverlay(landmarks);
          handleMonitorLandmarks(landmarks, performance.now());
        }
      } catch (error) {
        console.error('[solo-capture] preview detection failed', error);
      } finally {
        state.captureMonitorBusy = false;
      }
      const delay = Math.max(0, intervalMs - (performance.now() - started));
      if (state.captureMonitorActive) state.captureMonitorTimer = setTimeout(tick, delay);
    };
    void tick();
  } catch (error) {
    console.error('[solo-capture] detector setup failed', error);
    setCaptureStatus('detector_error', '준비 감지를 시작할 수 없습니다 · 다른 사람이 촬영을 선택해 주세요');
  }
}

async function beginRecording({ automatic = false } = {}) {
  if (!state.stream) return;
  const mimeType = chooseRecorderMime();
  if (!window.MediaRecorder || !mimeType) {
    alert('이 브라우저에서는 영상 녹화를 시작할 수 없습니다. 저장된 영상을 선택해 주세요.');
    return;
  }

  state.recordChunks = [];
  state.recorder = new MediaRecorder(state.stream, { mimeType });
  state.recorder.addEventListener('dataavailable', (event) => {
    if (event.data?.size) state.recordChunks.push(event.data);
  });
  state.recorder.start(250);
  state.recordingStartedAt = performance.now();
  state.analysisEndSeconds = null;
  state.roughRepCount = 0;
  state.inSquat = false;
  state.standBaselineSamples = [];
  state.standBaseline = null;
  state.finishStandSince = null;
  state.lastGoodCaptureSeconds = 0;
  state.invalidExitSamples = 0;
  state.captureRecordMeta = {
    mode: state.captureMode,
    facingMode: state.facingMode,
    mimeType,
    stopReason: null,
    analysisEndSeconds: null,
    readinessSeconds: finite(state.cameraRequestedAt) && finite(state.readinessCompletedAt)
      ? (state.readinessCompletedAt - state.cameraRequestedAt) / 1000
      : null,
    countdownCancelCount: state.countdownCancelCount,
    cameraSettings: state.cameraSettings,
  };
  state.recordTimer = setInterval(() => {
    const elapsedMs = performance.now() - state.recordingStartedAt;
    recordTime.textContent = formatClock(elapsedMs / 1000);
    if (elapsedMs >= SOLO_CAPTURE_CONFIG.maxRecordingMs) {
      void stopRecording({ reason: 'maximum', analysisEndSeconds: MAX_VIDEO_SECONDS });
    }
  }, 250);
  recordTime.textContent = '00:00';
  setHidden(recordBadge, false);
  setHidden(recordButton, true);
  setHidden(cameraButton, true);
  setHidden(facingControl, true);
  setHidden(stopButton, false);
  setCaptureStatus('recording_initial_stand', automatic ? '그대로 2초 서 계세요' : '촬영 중 · 준비자세부터 시작해 주세요', { speak: automatic });
}

async function startRecording() {
  await beginRecording({ automatic: false });
}

async function stopRecording({ reason = 'manual', analysisEndSeconds = null } = {}) {
  if (state.stoppingRecording || !state.recorder || state.recorder.state === 'inactive') return;
  state.stoppingRecording = true;
  stopButton.disabled = true;
  stopCaptureMonitor();
  const elapsedSeconds = finite(state.recordingStartedAt)
    ? Math.max(0, (performance.now() - state.recordingStartedAt) / 1000)
    : null;
  const endSeconds = finite(analysisEndSeconds)
    ? Math.max(0, Math.min(analysisEndSeconds, elapsedSeconds ?? analysisEndSeconds))
    : null;
  state.analysisEndSeconds = endSeconds;
  if (state.captureRecordMeta) {
    state.captureRecordMeta.stopReason = reason;
    state.captureRecordMeta.analysisEndSeconds = endSeconds;
  }
  setCaptureStatus('capture_complete', '촬영 완료 · 휴대폰으로 와 주세요', { speak: true });
  try {
    const stopped = once(state.recorder, 'stop', 'error', 10000);
    state.recorder.stop();
    await stopped;
    const mimeType = state.recorder.mimeType || state.recordChunks[0]?.type || 'video/webm';
    const blob = new Blob(state.recordChunks, { type: mimeType });
    stopStream();
    await sleep(650);
    await loadVideoAndAnalyze(
      blob,
      `browser-recording.${mimeType.includes('mp4') ? 'mp4' : 'webm'}`,
      'camera',
      { analysisEndSeconds: endSeconds },
    );
  } catch (error) {
    stopStream();
    showAnalysisError(error);
  } finally {
    clearInterval(state.recordTimer);
    setHidden(recordBadge, true);
    setHidden(stopButton, true);
    stopButton.disabled = false;
    state.stoppingRecording = false;
  }
}

async function loadVideoSource(blob) {
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.sourceUrl = URL.createObjectURL(blob);
  video.pause();
  video.srcObject = null;
  video.removeAttribute('src');
  video.load();
  video.src = state.sourceUrl;
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  const ready = once(video, 'loadedmetadata', 'error', 15000);
  video.load();
  await ready;
  if (!finite(video.duration) || video.duration <= 0 || !video.videoWidth || !video.videoHeight) {
    throw new Error('영상의 길이나 화면 크기를 읽지 못했습니다. 브라우저에서 재생 가능한 MP4/WebM 영상을 사용해 주세요.');
  }
  const effectiveDuration = finite(state.analysisEndSeconds)
    ? Math.min(video.duration, state.analysisEndSeconds)
    : video.duration;
  if (effectiveDuration > MAX_VIDEO_SECONDS) {
    throw new Error(`영상이 ${MAX_VIDEO_SECONDS}초를 넘습니다. 전체 세트를 임의로 잘라 분석하지 않도록 중단했습니다. 사진 앱이나 편집 도구에서 한 세트만 ${MAX_VIDEO_SECONDS}초 이하로 잘라 다시 올려 주세요.`);
  }
  if (video.duration > 60) {
    $('#capture-deck').textContent = `${Math.ceil(video.duration)}초 영상입니다. 분석 화면에서 진행률과 예상 남은 시간을 확인할 수 있습니다.`;
  }
  setHidden(cameraPlaceholder, true);
}

async function loadVideoAndAnalyze(blob, name, kind, { analysisEndSeconds = null } = {}) {
  try {
    state.sourceName = name;
    state.sourceKind = kind;
    state.analysisEndSeconds = finite(analysisEndSeconds) ? analysisEndSeconds : null;
    if (kind !== 'camera') {
      state.captureRecordMeta = {
        mode: kind === 'file' ? 'upload' : kind,
        facingMode: null,
        mimeType: blob.type || null,
        stopReason: null,
        analysisEndSeconds: state.analysisEndSeconds,
      };
    }
    await loadVideoSource(blob);
    await analyzeVideo();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (kind === 'file' && /loadedmetadata|미디어 준비|화면 크기/.test(message)) {
      console.error('영상 열기 실패', error);
      showAnalysisError(new Error("이 영상을 브라우저에서 열 수 없어요. 카메라 설정의 '높은 호환성'(H.264)으로 찍은 영상이나 MP4 파일을 선택해 주세요."));
    } else {
      showAnalysisError(error);
    }
  }
}

async function loadSampleSet(phase) {
  const key = phase === 2 ? 'set2_url' : 'set1_url';
  const url = state.samplePair?.[key];
  if (!url) throw new Error('샘플 영상 경로가 준비되지 않았습니다.');
  showAnalysis('샘플 영상을 준비하고 있습니다.', '승인된 예시 세트를 브라우저 안에서 불러옵니다.', 4, '샘플 불러오는 중');
  const response = await fetch(url);
  if (!response.ok) throw new Error('샘플 영상을 불러오지 못했습니다.');
  state.samplePhase = phase;
  await loadVideoAndAnalyze(await response.blob(), `샘플 세트 ${phase}`, 'sample');
}

async function initSampleExperience() {
  const response = await fetch('./data/sample_pair.json', { cache: 'no-store' });
  if (!response.ok) return;
  const pair = await response.json();
  if (!pair.enabled || !pair.set1_url || !pair.set2_url) return;
  state.samplePair = pair;
  const button = $('#sample-button');
  button.textContent = pair.label || '샘플로 전체 흐름 보기';
  setHidden(button, false);
}

async function startSampleExperience() {
  if (!state.samplePair) return;
  restart();
  try {
    await loadSampleSet(1);
  } catch (error) {
    showAnalysisError(error);
  }
}

async function seekVideo(time) {
  const safeTime = Math.max(0, Math.min(time, Math.max(0, video.duration - 0.002)));
  if (Math.abs(video.currentTime - safeTime) < 0.003 && video.readyState >= 2) return;
  const event = once(video, 'seeked', 'error', 10000);
  video.currentTime = safeTime;
  await event;
}

async function initPoseLandmarker({ quiet = false } = {}) {
  if (state.poseLandmarker) return state.poseLandmarker;
  if (!quiet) showAnalysis('자세 인식 도구를 준비하고 있습니다.', '처음 한 번만 내려받고 이후에는 브라우저 캐시를 사용합니다.', 8, '도구 준비 중 · 남은 시간 계산 중');
  state.poseLandmarkerPromise ||= (async () => {
    const { FilesetResolver, PoseLandmarker } = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm'
    );
    const fileset = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm',
    );
    state.poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: new URL('./models/pose_landmarker_lite.task', window.location.href).href,
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    return state.poseLandmarker;
  })();
  try {
    return await state.poseLandmarkerPromise;
  } catch (error) {
    state.poseLandmarkerPromise = null;
    throw error;
  }
}

function analysisCanvasSize(width, height, maxSide = 960) {
  const scale = Math.min(1, maxSide / Math.max(width, height));
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
}

async function extractLandmarks() {
  const landmarker = await initPoseLandmarker();
  const [width, height] = analysisCanvasSize(video.videoWidth, video.videoHeight);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { alpha: false, willReadFrequently: false });
  const duration = finite(state.analysisEndSeconds)
    ? Math.min(video.duration, state.analysisEndSeconds)
    : video.duration;
  const frameCount = Math.max(1, Math.floor(duration * state.analysisFps));
  const frames = [];
  const startedAt = performance.now();

  state.frameWidth = width;
  state.frameHeight = height;
  for (let index = 0; index < frameCount; index += 1) {
    const time = index / state.analysisFps;
    await seekVideo(time);
    context.drawImage(video, 0, 0, width, height);
    state.poseTimestampMs += 1000 / state.analysisFps;
    const result = landmarker.detectForVideo(canvas, state.poseTimestampMs);
    const landmarks = result.landmarks?.[0];
    frames.push(
      landmarks
        ? landmarks.map((point) => ({ x: point.x, y: point.y, visibility: point.visibility ?? 1 }))
        : null,
    );
    if (index % 4 === 0 || index === frameCount - 1) {
      const processed = index + 1;
      const ratio = processed / frameCount;
      const elapsedSeconds = (performance.now() - startedAt) / 1000;
      const remainingSeconds = processed >= 8
        ? (elapsedSeconds / processed) * (frameCount - processed)
        : Number.NaN;
      showAnalysis(
        '반복과 관절 움직임을 읽고 있습니다.',
        '영상은 이 기기 안에서 프레임 단위로 처리됩니다.',
        15 + ratio * 66,
        `${processed} / ${frameCount} 프레임 · ${formatRemaining(remainingSeconds)}`,
      );
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }
  return frames;
}

async function initPyodide() {
  if (state.pyodide) return state.pyodide;
  if (typeof window.loadPyodide !== 'function') {
    throw new Error('브라우저 분석 엔진을 내려받지 못했습니다. 인터넷 연결을 확인해 주세요.');
  }
  showAnalysis('반복 분석 엔진을 준비하고 있습니다.', '검증된 정면 분석 코드를 브라우저 안에서 불러옵니다.', 83, '분석 도구 불러오는 중');
  const pyodide = await window.loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.28.2/full/',
  });
  await pyodide.loadPackage(['numpy', 'scipy', 'pandas']);
  const manifest = await fetch('./python/manifest.json').then((response) => {
    if (!response.ok) throw new Error('분석 코드 목록을 읽지 못했습니다.');
    return response.json();
  });
  for (const file of manifest) {
    const response = await fetch(`./python/${file}`);
    if (!response.ok) throw new Error(`분석 코드 ${file}을 읽지 못했습니다.`);
    const target = `/home/pyodide/${file}`;
    pyodide.FS.mkdirTree(target.slice(0, target.lastIndexOf('/')));
    pyodide.FS.writeFile(target, await response.text());
  }
  state.pyodide = pyodide;
  return pyodide;
}

function requestedAdjMerge() {
  const raw = new URLSearchParams(window.location.search).get('adjMerge');
  if (raw === null || raw === '') return null;
  const value = Number(raw);
  return Number.isInteger(value) && value > 0 ? value : null;
}

async function runEngine(calibration) {
  const pyodide = await initPyodide();
  const payload = {
    frames: state.frames,
    width: state.frameWidth,
    height: state.frameHeight,
    fps: state.analysisFps,
    calibration,
    // Pyodide 0.28 may expose JavaScript null as JsNull rather than Python None.
    // Zero is not a valid adjMerge query value, so it is safe as the unset sentinel.
    adj_merge: requestedAdjMerge() ?? 0,
  };
  const proxy = pyodide.toPy(payload);
  pyodide.globals.set('WEB_INPUT', proxy);
  try {
    const output = pyodide.runPython(`
import dataclasses, json, math, sys
import numpy as np
sys.path.insert(0, '/home/pyodide')
from frontal import engine, feedback_rules

def clean(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    elif hasattr(value, '_asdict'):
        value = value._asdict()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

cal_raw = WEB_INPUT['calibration']
calibration = None if cal_raw is None else tuple(float(x) for x in cal_raw)
adj_raw = int(WEB_INPUT['adj_merge'])
adj_merge = None if adj_raw == 0 else adj_raw
result = engine.analyze_mediapipe(
    WEB_INPUT['frames'], int(WEB_INPUT['width']), int(WEB_INPUT['height']),
    float(WEB_INPUT['fps']), mirrored=None, adj_merge=adj_merge,
    calibration=calibration,
)
selected = {
    'n_frames': result['n_frames'],
    'n_reps': result['n_reps'],
    'bottoms': result['bottoms'],
    'bottom_on_interpolated_frame': result['bottom_on_interpolated_frame'],
    'per_rep': result['per_rep'],
    'features': result['features'],
    'gap_fill': result['gap_fill'],
    'detection': result['detection'],
    'standing': result['standing'],
    'capture_qc': result['capture_qc'],
    'qc': result['qc'],
    'reps_usable': result['reps_usable'],
    'warnings': result['warnings'],
    'result_usable': result['result_usable'],
    'side_note': result['side_note'],
    'rep_consistency': result['rep_consistency'],
    'rhythm': result['rhythm'],
    'schema': result['schema'],
    'note': result['note'],
}
selected['feedback'] = feedback_rules.evaluate(selected)
selected['retry_targets'] = feedback_rules.retry_targets(selected['feedback'])
json.dumps(clean(selected), ensure_ascii=False)
`);
    return JSON.parse(output);
  } finally {
    proxy.destroy();
    pyodide.runPython('del WEB_INPUT');
  }
}

async function compareSets(previous, current, targets) {
  const pyodide = await initPyodide();
  const proxy = pyodide.toPy({ previous, current, targets });
  pyodide.globals.set('WEB_COMPARE', proxy);
  try {
    const output = pyodide.runPython(`
import dataclasses, json, math
import numpy as np
from frontal import compare_sets

def clean_compare(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): clean_compare(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_compare(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_compare(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

rows = compare_sets.compare_many(
    WEB_COMPARE['previous'], WEB_COMPARE['current'], WEB_COMPARE['targets'])
payload = []
for row in rows:
    item = clean_compare(row)
    item['display'] = compare_sets.DISPLAY[row['verdict']]
    item['summary'] = compare_sets.summary_text(row)
    payload.append(item)
json.dumps(payload, ensure_ascii=False)
`);
    return JSON.parse(output);
  } finally {
    proxy.destroy();
    pyodide.runPython('del WEB_COMPARE');
  }
}

async function analyzeVideo() {
  const startedAt = performance.now();
  showAnalysis('영상을 준비하고 있습니다.', '브라우저가 재생할 수 있는 형식인지 확인했습니다.', 4, '분석 준비 중');
  state.frames = await extractLandmarks();
  showAnalysis('반복별 값을 계산하고 있습니다.', '처음 2초의 준비자세를 기준으로 비교합니다.', 90, '마무리 계산 중');
  state.result = await runEngine([0, 2]);
  showAnalysis('결과를 정리하고 있습니다.', '근거 장면과 관련 콘텐츠를 연결합니다.', 98, '곧 완료됩니다');
  state.recommendations ||= await fetch('./data/recommendations.json').then((response) => response.json());
  state.comparison = state.retrySession
    ? await compareSets(state.retrySession.previous, state.result, state.retrySession.targets)
    : null;
  state.analysisElapsedSeconds = (performance.now() - startedAt) / 1000;
  await renderResults();
}

function repUsable(index) {
  return state.result.reps_usable?.[index] !== false;
}

function metricValues(key) {
  return state.result.per_rep
    .map((rep, index) => (repUsable(index) ? rep[key] : null))
    .filter(finite);
}

function extremeRep(key, mode = 'min') {
  let bestIndex = -1;
  let bestValue = mode === 'min' ? Infinity : -Infinity;
  state.result.per_rep.forEach((rep, index) => {
    if (!repUsable(index)) return;
    const value = rep[key];
    if (!finite(value)) return;
    if ((mode === 'min' && value < bestValue) || (mode === 'max' && value > bestValue)) {
      bestValue = value;
      bestIndex = index;
    }
  });
  return { index: bestIndex, value: bestValue };
}

function renderSummary() {
  const knee = extremeRep('A2_knee_w_rel_stand', 'min');
  const deep = extremeRep('D1_hip_ankle_rel', 'min');
  const usableCount = state.result.per_rep.filter((_, index) => repUsable(index)).length;
  $('#result-title').textContent = `${state.result.n_reps}회의 저점 장면을 나누어 보았습니다.`;
  $('#result-summary').innerHTML = `
    <div class="summary-chip"><small>구분된 반복</small><strong>${state.result.n_reps}회 · 값에 사용 ${usableCount}회</strong></div>
    <div class="summary-chip"><small>준비자세 대비 가장 좁은 무릎 간격</small><strong>${knee.index >= 0 ? `${knee.index + 1}회차 · ${multiple(knee.value)}` : '확인 어려움'}</strong></div>
    <div class="summary-chip"><small>세트 안 깊이 순서</small><strong>${deep.index >= 0 ? `${deep.index + 1}회차가 가장 깊음` : '확인 어려움'}</strong></div>
  `;
  setHidden($('#low-rep-note'), !(state.result.n_reps >= 4 && state.result.n_reps <= 5));
  return { knee, deep };
}

function median(values) {
  if (!values.length) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function orderedFeedback() {
  return [...(state.result.feedback || [])].sort((a, b) => Number(Boolean(b.primary)) - Number(Boolean(a.primary)));
}

function feedbackForRep(index) {
  const repNumber = index + 1;
  return orderedFeedback().filter((item) => (item.reps || []).map(Number).includes(repNumber));
}

function feedbackTagsForRep(index) {
  const tags = [];
  for (const item of feedbackForRep(index)) {
    if (item.rule_id?.startsWith('KNEE') && !tags.includes('무릎 간격 좁음')) tags.push('무릎 간격 좁음');
    if (item.rule_id?.startsWith('DEPTH') && !tags.includes('얕음')) tags.push('얕음');
    if (item.rule_id?.endsWith('_LATE') && !tags.includes('후반부')) tags.push('후반부');
  }
  return tags;
}

function feedbackMatchesChart(item, family) {
  return family === 'knee' ? item.rule_id?.startsWith('KNEE') : item.rule_id?.startsWith('DEPTH');
}

function feedbackSentence(item) {
  if (!item) return '';
  const reps = (item.reps || []).map(Number).filter(Number.isFinite);
  const repText = reps.map((rep) => `${rep}회차`).join('·');
  if (item.rule_id === 'KNEE_LATE') return `후반 반복(${repText})에서 무릎 간격이 초반보다 좁아지는 경향이 보였습니다.`;
  if (item.rule_id === 'KNEE_REP') return `${repText}에서 무릎 간격이 다른 반복보다 눈에 띄게 좁았습니다.`;
  if (item.rule_id === 'DEPTH_LATE') return `후반 반복(${repText})에서 골반이 내려간 정도가 초반보다 작아지는 경향이 보였습니다.`;
  if (item.rule_id === 'DEPTH_REP') return `${repText}에서 골반이 내려간 정도가 다른 반복보다 작았습니다.`;
  return item.text || '';
}

function chartEarlyReps(family) {
  const reps = new Set();
  for (const item of orderedFeedback()) {
    if (!feedbackMatchesChart(item, family) || !item.rule_id?.endsWith('_LATE')) continue;
    const early = family === 'knee' ? item.evidence?.A2?.early_reps : item.evidence?.early_reps;
    (early || []).forEach((rep) => reps.add(Number(rep)));
  }
  return reps;
}

function renderBarChart({
  family,
  title,
  detail,
  metric,
  values,
  summary = '',
  help = '',
  valueFormatter = percent,
  medianFormatter = valueFormatter,
  reference = false,
  missingLabel = '제외',
}) {
  const usableValues = values.filter((value, index) => repUsable(index) && finite(value));
  const middle = median(usableValues);
  const maximum = Math.max(0, ...usableValues, finite(middle) ? middle : 0);
  const scaleMaximum = maximum > 0 ? maximum * 1.12 : 1;
  const barAreaHeight = 128;
  const medianBottom = 27 + Math.max(0, Math.min(1, middle / scaleMaximum)) * barAreaHeight;
  const earlyReps = reference ? new Set() : chartEarlyReps(family);
  const feedback = reference ? [] : orderedFeedback().filter((item) => feedbackMatchesChart(item, family));
  const bars = values.map((value, index) => {
    const repNumber = index + 1;
    const unavailable = !finite(value);
    const excluded = !repUsable(index) || unavailable;
    const flagged = feedback.some((item) => (item.reps || []).map(Number).includes(repNumber));
    const early = !flagged && earlyReps.has(repNumber);
    const missingTag = typeof missingLabel === 'function' ? missingLabel(index) : missingLabel;
    const tag = excluded ? (unavailable ? missingTag : '제외') : flagged ? (family === 'knee' ? '좁음' : '얕음') : early ? '초반' : '';
    const height = excluded ? 0 : Math.max(2, Math.min(barAreaHeight, (Math.max(0, value) / scaleMaximum) * barAreaHeight));
    const stateClass = excluded ? 'excluded' : flagged ? 'flagged' : early ? 'early' : '';
    const valueText = excluded ? '—' : valueFormatter(value);
    const content = `
        <span class="rep-bar-tag">${escapeHtml(tag)}</span>
        <strong>${escapeHtml(valueText)}</strong>
        <span class="rep-bar-space">${excluded ? '' : `<span class="rep-bar-fill" style="height:${height}px"></span>`}</span>
        <small>${repNumber}회</small>`;
    if (reference) {
      return `<div class="rep-bar-button ${stateClass}" aria-label="${repNumber}회차 ${escapeHtml(title)} ${escapeHtml(valueText)}${tag ? `, ${escapeHtml(tag)}` : ''}">${content}</div>`;
    }
    return `
      <button class="rep-bar-button ${stateClass} ${index === state.selectedRep ? 'active' : ''}" data-rep="${index}" data-metric="${metric}" aria-current="${index === state.selectedRep}" aria-label="${repNumber}회차 ${escapeHtml(title)} ${escapeHtml(valueText)}${tag ? `, ${escapeHtml(tag)}` : ''}">
        ${content}
      </button>`;
  }).join('');
  return `
    <article class="rep-bar-chart ${reference ? 'reference' : ''}">
      <div class="rep-bar-chart-title"><h4>${escapeHtml(title)}</h4>${reference ? '<b class="reference-badge">참고</b>' : ''}<span>${escapeHtml(detail)}</span>${help ? `<details class="chart-help"><summary aria-label="${escapeHtml(title)} 읽는 법">ⓘ 읽는 법</summary><p>${escapeHtml(help)}</p></details>` : ''}</div>
      ${summary ? `<p class="rep-bar-summary">${escapeHtml(summary)}</p>` : ''}
      <div class="rep-bar-stage">
        <span class="rep-chart-zero">0</span>
        ${finite(middle) ? `<span class="rep-chart-median" style="bottom:${medianBottom}px"><em>세트 중앙값 ${escapeHtml(medianFormatter(middle))}</em></span>` : ''}
        <div class="rep-bar-grid" style="grid-template-columns:repeat(${values.length},minmax(0,1fr))">${bars}</div>
      </div>
    </article>`;
}

function transformedExtreme(values, mode) {
  let bestIndex = -1;
  let bestValue = mode === 'min' ? Infinity : -Infinity;
  values.forEach((value, index) => {
    if (!repUsable(index) || !finite(value)) return;
    if ((mode === 'min' && value < bestValue) || (mode === 'max' && value > bestValue)) {
      bestIndex = index;
      bestValue = value;
    }
  });
  return { index: bestIndex, value: bestValue };
}

function rhythmValues() {
  const intervals = state.result.rhythm?.rep_interval_sec || [];
  return state.result.per_rep.map((_, index) => (index === 0 ? null : intervals[index - 1] ?? null));
}

function rhythmSummary(values) {
  const usable = values.filter(finite);
  if (usable.length < 3) return '반복 시간은 앞 회차와 이번 회차의 저점 사이 시간입니다.';
  const early = usable.slice(0, 3);
  const late = usable.slice(-3);
  const earlyMean = early.reduce((sum, value) => sum + value, 0) / early.length;
  const lateMean = late.reduce((sum, value) => sum + value, 0) / late.length;
  return `앞 3회 평균 ${earlyMean.toFixed(1)}초, 뒤 3회 평균 ${lateMean.toFixed(1)}초로 ${Math.abs(lateMean - earlyMean).toFixed(1)}초 차이가 있었습니다.`;
}

function renderReferenceCharts() {
  const trunkValues = state.result.per_rep.map((rep) => (finite(rep.C1_trunk_span_rel) ? 1 - rep.C1_trunk_span_rel : null));
  const timingValues = rhythmValues();
  const trunk = transformedExtreme(trunkValues, 'max');
  $('#reference-chart-inner').style.minWidth = state.result.per_rep.length > 12 ? `${state.result.per_rep.length * 30}px` : '';
  $('#reference-chart-inner').innerHTML = [
    renderBarChart({
      family: 'trunk',
      title: '상체 숙임',
      detail: '준비자세보다 정면 상체 길이가 짧아 보인 정도',
      metric: '',
      values: trunkValues,
      summary: trunk.index >= 0 ? `${trunk.index + 1}회차에서 상체가 가장 많이 숙여 보였습니다.` : '어깨가 보인 회차에서만 상체 길이를 확인합니다.',
      valueFormatter: (value) => `${percent(value)} 짧아 보임`,
      medianFormatter: (value) => `${percent(value)} 짧아 보임`,
      reference: true,
      missingLabel: '어깨 가림',
    }),
    renderBarChart({
      family: 'rhythm',
      title: '반복 시간',
      detail: '앞 회차 저점에서 이번 회차 저점까지',
      metric: '',
      values: timingValues,
      summary: rhythmSummary(timingValues),
      valueFormatter: (value) => `${value.toFixed(1)}초`,
      medianFormatter: (value) => `${value.toFixed(1)}초`,
      reference: true,
      missingLabel: (index) => (index === 0 ? '첫 회차' : '확인 어려움'),
    }),
  ].join('');
  const disclosure = $('#reference-charts');
  if (window.matchMedia('(max-width: 680px)').matches) disclosure.removeAttribute('open');
  else disclosure.setAttribute('open', '');
}

function renderRepCharts() {
  const kneeValues = state.result.per_rep.map((rep) => rep.A2_knee_w_rel_stand);
  const depthValues = state.result.per_rep.map((rep) => (
    finite(rep.D1_hip_ankle_rel) ? 1 - rep.D1_hip_ankle_rel : null
  ));
  const count = state.result.per_rep.length;
  const kneeFeedback = orderedFeedback().find((item) => feedbackMatchesChart(item, 'knee'));
  const kneeMedian = median(kneeValues.filter((value, index) => repUsable(index) && finite(value)));
  const deepest = transformedExtreme(depthValues, 'max');
  const shallowest = transformedExtreme(depthValues, 'min');
  const scroll = $('#rep-chart-scroll');
  scroll.classList.toggle('scrollable', count > 12);
  $('#rep-chart-inner').style.minWidth = count > 12 ? `${count * 30}px` : '';
  $('#rep-chart-inner').innerHTML = [
    renderBarChart({
      family: 'knee',
      title: '무릎 간격',
      detail: 'A2 · 준비자세 대비',
      metric: 'A2_knee_w_rel_stand',
      values: kneeValues,
      summary: kneeFeedback ? feedbackSentence(kneeFeedback) : finite(kneeMedian)
        ? `모든 회차에서 무릎이 서 있을 때의 약 ${multiple(kneeMedian)}로 벌어졌습니다.`
        : '무릎 간격을 비교할 수 있는 회차가 없습니다.',
      help: '1배는 준비자세의 무릎 간격입니다. 막대가 낮을수록 준비자세보다 무릎 간격이 좁게 보인 회차입니다.',
      valueFormatter: multiple,
      medianFormatter: (value) => `약 ${multiple(value)}`,
    }),
    renderBarChart({
      family: 'depth',
      title: '골반이 내려간 정도',
      detail: '100% − 골반–발목 세로거리 · 높을수록 깊음',
      metric: 'D1_hip_ankle_rel',
      values: depthValues,
      summary: deepest.index >= 0 && shallowest.index >= 0 ? `가장 깊게 앉은 회차는 ${deepest.index + 1}회차, 가장 얕은 회차는 ${shallowest.index + 1}회차입니다.` : '',
      help: '준비자세에서 골반과 발목 사이의 세로거리를 기준으로, 저점에서 줄어든 정도입니다. 막대가 높을수록 골반이 더 내려간 회차입니다.',
      valueFormatter: (value) => `${percent(value)} 내려감`,
      medianFormatter: (value) => `${percent(value)} 내려감`,
    }),
  ].join('');
  renderReferenceCharts();
  $('#rep-chart-caption').textContent = feedbackSentence(primaryFeedback())
    || '반복 사이 큰 변화가 두드러지지 않았습니다.';
  document.querySelectorAll('button.rep-bar-button').forEach((button) => {
    button.addEventListener('click', () => {
      void selectRep(Number(button.dataset.rep), button.dataset.metric);
    });
  });
}

function metricPosition(value, values) {
  if (!finite(value) || !values.length) return 50;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (Math.abs(maximum - minimum) < 1e-9) return 50;
  const padding = (maximum - minimum) * 0.08;
  return ((value - (minimum - padding)) / (maximum - minimum + padding * 2)) * 100;
}

function medianBar(value, values, excluded, label, formatter = percent) {
  if (excluded || !finite(value) || !values.length) {
    return '<span class="rep-median-track empty" aria-hidden="true"></span>';
  }
  const middle = median(values);
  const valuePosition = metricPosition(value, values);
  const medianPosition = metricPosition(middle, values);
  const left = Math.min(valuePosition, medianPosition);
  const width = Math.max(2, Math.abs(valuePosition - medianPosition));
  return `
    <span class="rep-median-track" aria-label="${escapeHtml(label)} ${escapeHtml(formatter(value))}, 세트 중앙값 ${escapeHtml(formatter(middle))}">
      <span class="rep-median-span" style="left:${left}%;width:${width}%"></span>
      <span class="rep-median-marker" style="left:${medianPosition}%"></span>
      <span class="rep-value-marker" style="left:${valuePosition}%"></span>
    </span>`;
}

function updateRepSelection() {
  document.querySelectorAll('.rep-overview-card, .rep-bar-button').forEach((card) => {
    const active = Number(card.dataset.rep) === state.selectedRep;
    card.classList.toggle('active', active);
    card.setAttribute('aria-current', active ? 'true' : 'false');
  });
}

function observeRepThumbnails() {
  state.repThumbnailObserver?.disconnect();
  const canvases = [...document.querySelectorAll('.rep-thumbnail')];
  const load = (canvas) => {
    if (canvas.dataset.queued === 'true') return;
    canvas.dataset.queued = 'true';
    void drawThumbnail(canvas, Number(canvas.dataset.frame)).catch((error) => console.error(error));
  };
  if (!('IntersectionObserver' in window)) {
    canvases.forEach(load);
    return;
  }
  state.repThumbnailObserver = new IntersectionObserver((entries, observer) => {
    entries.filter((entry) => entry.isIntersecting).forEach((entry) => {
      observer.unobserve(entry.target);
      load(entry.target);
    });
  }, { rootMargin: '180px 0px' });
  canvases.forEach((canvas) => state.repThumbnailObserver.observe(canvas));
}

function renderRepOverview() {
  const kneeValues = metricValues('A2_knee_w_rel_stand');
  const depthValues = state.result.per_rep
    .map((rep, index) => (repUsable(index) ? depthDrop(rep.D1_hip_ankle_rel) : null))
    .filter(finite);
  const depthByRep = state.result.per_rep.map((rep) => depthDrop(rep.D1_hip_ankle_rel));
  const deepest = transformedExtreme(depthByRep, 'max');
  const shallowest = transformedExtreme(depthByRep, 'min');
  $('#rep-grid').innerHTML = state.result.per_rep.map((rep, index) => {
    const excluded = !repUsable(index);
    const tags = feedbackTagsForRep(index);
    const related = feedbackForRep(index)[0];
    const notes = [];
    if (excluded) notes.push('값 비교에서 제외');
    else if (related) notes.push(feedbackSentence(related));
    else notes.push('다른 반복과 큰 차이 없음');
    if (!excluded && index === deepest.index) notes.push('세트에서 가장 깊음');
    if (!excluded && index === shallowest.index) notes.push('세트에서 가장 얕음');
    const depthValue = depthByRep[index];
    return `
      <button class="rep-overview-card ${tags.length ? 'flagged' : ''} ${excluded ? 'excluded' : ''} ${index === state.selectedRep ? 'active' : ''}" data-rep="${index}" role="listitem" aria-current="${index === state.selectedRep}">
        <span class="rep-thumbnail-wrap">
          <canvas class="rep-thumbnail" data-frame="${state.result.bottoms[index]}" aria-label="${index + 1}회차 가장 낮은 장면"></canvas>
          ${tags.length ? `<span class="rep-tags">${tags.map((tag) => `<span>${tag}</span>`).join('')}</span>` : ''}
        </span>
        <span class="rep-overview-body">
          <span class="rep-overview-head"><strong>${index + 1}회차</strong>${excluded ? '<em>값 제외</em>' : ''}</span>
          <span class="rep-overview-metric"><span><small>무릎 간격</small><b>${excluded ? '—' : multiple(rep.A2_knee_w_rel_stand)}</b></span>${medianBar(rep.A2_knee_w_rel_stand, kneeValues, excluded, '무릎 간격', multiple)}</span>
          <span class="rep-overview-metric"><span><small>골반이 내려간 정도</small><b>${excluded ? '—' : `${percent(depthValue)} 내려감`}</b></span>${medianBar(depthValue, depthValues, excluded, '골반이 내려간 정도', (value) => `${percent(value)} 내려감`)}</span>
          <span class="rep-overview-note">${notes.map(escapeHtml).join(' · ')}</span>
        </span>
      </button>`;
  }).join('');
  document.querySelectorAll('.rep-overview-card').forEach((button) => {
    button.addEventListener('click', () => { void selectRep(Number(button.dataset.rep)); });
  });
  observeRepThumbnails();
}

function renderObservation(index) {
  const rep = state.result.per_rep[index];
  $('#observation-title').textContent = `${index + 1}회차`;
  if (!repUsable(index)) {
    $('#observation-lines').innerHTML = '<div class="metric-line"><small>값에서 제외</small><strong>확인 어려움</strong><span>가장 낮은 자세 부근에서 몸이 가려져 이 반복의 값은 사용하지 않았습니다.</span></div>';
    $('#uncertainty-note').innerHTML = '<strong>근거 장면은 확인 가능</strong><br>아래 원본 장면은 표시하지만 반복 비교에는 포함하지 않습니다.';
    setHidden($('#uncertainty-note'), false);
    return;
  }
  const lines = [
    {
      label: '저점의 무릎 간격',
      value: multiple(rep.A2_knee_w_rel_stand),
      note: '준비자세의 무릎 간격을 1배로 본 값입니다.',
    },
    {
      label: '골반이 내려간 정도',
      value: `${percent(depthDrop(rep.D1_hip_ankle_rel))} 내려감`,
      note: '준비자세의 골반–발목 세로거리를 기준으로 계산했습니다. 값이 클수록 골반이 더 내려간 회차입니다.',
    },
  ];
  if (finite(rep.C1_trunk_span_rel)) {
    lines.push({
      label: '정면에서 본 상체 길이',
      value: `${percent(1 - rep.C1_trunk_span_rel)} 짧아 보임`,
      note: '준비자세를 100%로 본 정면 상체 길이입니다. 앞으로 숙일수록 짧게 보입니다.',
    });
  }
  $('#observation-lines').innerHTML = lines.map((line) => `
    <div class="metric-line"><small>${line.label}</small><strong>${line.value}</strong><span>${line.note}</span></div>
  `).join('');
  const interpolated = state.result.bottom_on_interpolated_frame?.[index];
  setHidden($('#uncertainty-note'), !interpolated);
  $('#uncertainty-note').innerHTML = interpolated
    ? '<strong>확인 메모</strong><br>이 저점 프레임에는 미검출 관절을 이은 값이 포함돼 있습니다.'
    : '';
}

function drawSkeleton(context, landmarks, width, height) {
  if (!landmarks) return;
  context.save();
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.lineWidth = Math.max(3, Math.min(width, height) * 0.006);
  context.strokeStyle = '#8fc0e4';
  context.shadowColor = 'rgba(10,30,24,.55)';
  context.shadowBlur = 5;
  for (const [a, b] of CONNECTIONS) {
    const p = landmarks[a];
    const q = landmarks[b];
    if (!p || !q || (p.visibility ?? 1) < 0.5 || (q.visibility ?? 1) < 0.5) continue;
    context.beginPath();
    context.moveTo(p.x * width, p.y * height);
    context.lineTo(q.x * width, q.y * height);
    context.stroke();
  }
  for (const index of [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]) {
    const point = landmarks[index];
    if (!point || (point.visibility ?? 1) < 0.5) continue;
    context.beginPath();
    context.fillStyle = index === 25 || index === 26 ? '#ff6b3d' : '#f7fff9';
    context.arc(point.x * width, point.y * height, Math.max(4, Math.min(width, height) * 0.009), 0, Math.PI * 2);
    context.fill();
  }
  context.restore();
}

function landmarkVisible(point) {
  return Boolean(point) && (point.visibility ?? 1) >= 0.5 && finite(point.x) && finite(point.y);
}

function drawReferenceGuides(context, landmarks, width, height) {
  const standingIndex = state.result?.standing?.index;
  const standing = Number.isInteger(standingIndex) ? state.frames?.[standingIndex] : null;
  const currentKnees = [landmarks?.[25], landmarks?.[26]];
  const standingKnees = [standing?.[25], standing?.[26]];
  const standingHips = [standing?.[23], standing?.[24]];
  if (!currentKnees.every(landmarkVisible)) return;

  const currentY = ((currentKnees[0].y + currentKnees[1].y) / 2) * height;
  const currentX = currentKnees.map((point) => point.x * width);
  const centerX = (currentX[0] + currentX[1]) / 2;
  const lineWidth = Math.max(2, Math.min(width, height) * 0.004);
  const label = (text, x, y, color) => {
    if (Math.min(width, height) < 230) return;
    context.font = `700 ${Math.max(9, Math.round(Math.min(width, height) * 0.018))}px sans-serif`;
    context.textAlign = 'left';
    context.textBaseline = 'bottom';
    context.lineWidth = 3;
    context.strokeStyle = 'rgba(10,18,24,.78)';
    context.strokeText(text, x, y);
    context.fillStyle = color;
    context.fillText(text, x, y);
  };

  context.save();
  context.shadowBlur = 0;
  context.lineCap = 'round';
  context.lineWidth = lineWidth;
  context.strokeStyle = '#ffb14a';
  context.setLineDash([]);
  context.beginPath();
  context.moveTo(Math.min(...currentX), currentY);
  context.lineTo(Math.max(...currentX), currentY);
  context.stroke();
  label('이 회차 무릎', Math.max(6, Math.min(...currentX)), currentY - 4, '#ffd18b');

  if (standingKnees.every(landmarkVisible)) {
    const standingWidth = Math.abs(standingKnees[1].x - standingKnees[0].x) * width;
    const guideY = currentY + Math.max(6, height * 0.012);
    context.strokeStyle = '#a9d5f2';
    context.setLineDash([lineWidth * 2.2, lineWidth * 1.8]);
    context.beginPath();
    context.moveTo(centerX - standingWidth / 2, guideY);
    context.lineTo(centerX + standingWidth / 2, guideY);
    context.stroke();
    label('준비 무릎', Math.max(6, centerX - standingWidth / 2), guideY + Math.max(17, height * 0.03), '#d3ebfa');
  }

  if (standingHips.every(landmarkVisible)) {
    const pelvisY = ((standingHips[0].y + standingHips[1].y) / 2) * height;
    context.strokeStyle = '#d7dde1';
    context.setLineDash([lineWidth * 2.2, lineWidth * 1.8]);
    context.beginPath();
    context.moveTo(width * 0.18, pelvisY);
    context.lineTo(width * 0.82, pelvisY);
    context.stroke();
    label('준비 골반 높이', width * 0.19, pelvisY - 4, '#f2f5f7');
  }
  context.restore();
}

function queueFrameDraw(task) {
  const queued = state.frameDrawQueue.catch(() => {}).then(task);
  state.frameDrawQueue = queued;
  return queued;
}

async function paintFrame(canvas, frameIndex, maxSide = null, guard = () => true) {
  return queueFrameDraw(async () => {
    await seekVideo(frameIndex / state.analysisFps);
    if (!guard()) return;
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    const scale = maxSide ? Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight)) : 1;
    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(1, Math.round(sourceHeight * scale));
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, width, height);
    const landmarks = state.frames[frameIndex];
    drawSkeleton(context, landmarks, width, height);
    drawReferenceGuides(context, landmarks, width, height);
  });
}

async function drawThumbnail(canvas, frameIndex) {
  if (canvas.dataset.rendered === 'true') return;
  await paintFrame(canvas, frameIndex, 360);
  canvas.dataset.rendered = 'true';
}

async function drawFrame(canvas, frameIndex) {
  const token = ++state.evidenceToken;
  await paintFrame(canvas, frameIndex, null, () => token === state.evidenceToken);
}

async function renderEvidence(index) {
  const bottom = state.result.bottoms[index];
  const isPrimaryEvidence = Boolean(primaryFeedback()) && index === primaryEvidenceRep();
  $('#evidence-title').textContent = `${isPrimaryEvidence ? '우선 확인 · ' : ''}${index + 1}회차의 가장 낮은 장면`;
  $('#frame-time').textContent = `${(bottom / state.analysisFps).toFixed(1)}초`;
  $('#evidence-caption').textContent = repUsable(index)
    ? '원본 장면 위에 MediaPipe 2D 관절과 기준선을 표시했습니다. 실선은 이 회차 무릎 간격, 점선은 준비자세의 무릎 간격과 골반 높이입니다.'
    : '이 반복은 가림 때문에 값에서 제외했습니다. 원본 장면과 검출된 2D 관절만 확인할 수 있습니다.';
  await drawFrame($('#evidence-canvas'), bottom);
}

async function selectRep(index, metric = null) {
  state.selectedRep = index;
  const related = feedbackForRep(index)[0];
  state.comparisonMetric = metric || related?.metric || state.comparisonMetric || 'A2_knee_w_rel_stand';
  updateRepSelection();
  renderObservation(index);
  await renderEvidence(index);
  await renderComparison();
}

function medianReferenceIndex(metric, targetIndex) {
  const candidates = state.result.per_rep
    .map((rep, index) => ({ index, value: rep[metric] }))
    .filter((item) => item.index !== targetIndex && repUsable(item.index) && finite(item.value));
  if (!candidates.length) return -1;
  const middle = median(metricValues(metric));
  return candidates.reduce((best, item) => (
    Math.abs(item.value - middle) < Math.abs(best.value - middle) ? item : best
  )).index;
}

function comparisonMetricMeta(metric) {
  if (metric === 'D1_hip_ankle_rel') {
    return { label: '골반이 내려간 정도', detail: '100% − 골반–발목 세로거리', value: (raw) => 1 - raw, format: (value) => `${percent(value)} 내려감` };
  }
  return { label: '무릎 간격', detail: '준비자세 대비 무릎 간격', value: (raw) => raw, format: multiple };
}

async function renderComparison() {
  const targetIndex = state.selectedRep;
  const metric = state.comparisonMetric || primaryFeedback()?.metric || 'A2_knee_w_rel_stand';
  const referenceIndex = medianReferenceIndex(metric, targetIndex);
  if (referenceIndex < 0) {
    setHidden($('#compare-card'), true);
    return;
  }
  setHidden($('#compare-card'), false);
  const target = state.result.per_rep[targetIndex];
  const reference = state.result.per_rep[referenceIndex];
  const meta = comparisonMetricMeta(metric);
  const targetValue = finite(target[metric]) ? meta.value(target[metric]) : null;
  const referenceValue = finite(reference[metric]) ? meta.value(reference[metric]) : null;
  $('#compare-card').innerHTML = `
    <div class="compare-heading"><div><p class="card-label">회차 비교</p><h3>나란히 비교</h3></div><p>기준은 ${escapeHtml(meta.detail)}이 세트 중앙값에 가장 가까운 회차입니다.</p></div>
    <div class="compare-grid">
      <div class="compare-item compare-shot selected"><small>선택 · ${targetIndex + 1}회차 저점</small><canvas id="compare-target"></canvas><strong>${meta.label} ${repUsable(targetIndex) ? meta.format(targetValue) : '확인 어려움'}</strong></div>
      <div class="compare-item compare-shot"><small>평소 · ${referenceIndex + 1}회차 저점</small><canvas id="compare-reference"></canvas><strong>${meta.label} ${meta.format(referenceValue)}</strong></div>
    </div>
    <p class="evidence-caption">${targetIndex + 1}회차 ${meta.label} ${repUsable(targetIndex) ? meta.format(targetValue) : '확인 어려움'} · 평소(${referenceIndex + 1}회차) ${meta.format(referenceValue)}</p>
  `;
  await drawFrame($('#compare-target'), state.result.bottoms[targetIndex]);
  await drawFrame($('#compare-reference'), state.result.bottoms[referenceIndex]);
}

function renderRecommendations() {
  const section = $('#recommend-section');
  const primary = primaryFeedback();
  const steady = !primary;
  const selections = primary ? [primary] : [{ content_group: 'steady_set', text: '' }];
  $('#recommend-eyebrow').textContent = steady ? '선택형 루틴' : '다음 세트 전 선택사항';
  $('#recommend-title').textContent = steady
    ? '다음 세트 전·운동 후 루틴 (국민체력100)'
    : '관련 부위를 가볍게 준비해 보세요.';
  $('#recommend-disclaimer').textContent = steady
    ? '다음 세트 전 또는 운동을 마친 뒤 가볍게 움직여 볼 수 있는 국민체력100 콘텐츠입니다.'
    : '관찰된 변화와 관련된 부위를 가볍게 움직여 볼 수 있는 국민체력100 콘텐츠입니다.';
  const cards = [];
  const seen = new Set();
  for (const selection of selections) {
    if (selection.content_group === 'before_next_set') continue;
    const group = state.recommendations.rules[selection.content_group];
    if (!group) continue;
    for (const item of group.items) {
      if (seen.has(item.catalog_id)) continue;
      seen.add(item.catalog_id);
      const reason = item.reason || (selection.text ? `${selection.text} → ${item.relation}` : item.relation);
      cards.push(`
        <article class="recommend-card">
          <div class="recommend-media"><video controls playsinline preload="metadata" src="${escapeHtml(item.video_url)}"${item.thumbnail_url ? ` poster="${escapeHtml(item.thumbnail_url)}"` : ''}></video></div>
          <div class="recommend-body">
            <span class="recommend-tag">${escapeHtml(item.relation)}</span>
            <h3>${escapeHtml(item.title)}</h3>
            <p class="recommend-reason"><b>왜 이 영상인가</b><br>${escapeHtml(reason)}</p>
            <p>${Math.round(item.duration_seconds)}초 · 준비물 ${escapeHtml(item.equipment)}</p>
            <p>출처: 서울올림픽기념국민체육진흥공단, 국민체력100 동영상 정보(공공누리 제1유형)</p>
            <a class="recommend-link" href="${escapeHtml(item.video_url)}" target="_blank" rel="noopener noreferrer">재생이 안 되면 새 탭에서 보기</a>
          </div>
        </article>
      `);
    }
  }
  setHidden(section, cards.length === 0);
  $('#recommend-grid').innerHTML = cards.join('');
  section.querySelectorAll('video').forEach((media) => {
    media.addEventListener('error', () => media.closest('.recommend-card')?.classList.add('media-error'));
  });
}

function renderNotices() {
  const warnings = state.result.warnings || [];
  const labels = { info: '안내', warn: '확인', error: '분석 보류' };
  $('#result-notices').innerHTML = warnings.map((warning) => `
    <div class="result-notice ${escapeHtml(warning.severity)}">
      <b>${labels[warning.severity] || '안내'}</b>
      <span>${escapeHtml(warning.message)}</span>
    </div>
  `).join('');
  setHidden($('#result-notices'), warnings.length === 0);
}

function renderFeedback() {
  const feedback = orderedFeedback().slice(0, 3);
  setHidden($('#feedback-card'), false);
  $('#feedback-card').classList.toggle('neutral', feedback.length === 0);
  if (!feedback.length) {
    $('#feedback-list').innerHTML = '<p class="feedback-neutral">이번 세트에서는 반복 사이 큰 변화가 두드러지지 않았습니다.</p>';
    return;
  }
  const titles = {
    KNEE_LATE: '후반부의 무릎 간격 변화',
    KNEE_REP: '다른 반복보다 무릎이 모였던 회차',
    DEPTH_LATE: '후반부의 깊이 변화',
    DEPTH_REP: '가장 얕았던 회차',
  };
  $('#feedback-list').innerHTML = feedback.map((item, index) => `
    <button class="feedback-item ${item.primary ? 'primary' : ''}" data-feedback="${index}">
      <span><b>${item.primary ? '먼저 보기' : '함께 보기'}</b><strong>${escapeHtml(titles[item.rule_id] || '세트 안 반복 변화')}</strong></span>
      <small>${escapeHtml(item.text)}</small>
    </button>
  `).join('');
  document.querySelectorAll('.feedback-item').forEach((button) => {
    button.addEventListener('click', async () => {
      const item = feedback[Number(button.dataset.feedback)];
      const requested = Math.max(0, Number(item.reps?.[0] || 1) - 1);
      const index = nearestUsableRep(state.result, requested);
      if (index < 0) return;
      await selectRep(index, item.metric);
      $('#evidence-detail').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

function standingLabel() {
  const standing = state.result.standing || {};
  const method = String(standing.method || '');
  let label = '확인 어려움';
  if (method === 'calibration') label = '영상 앞 2초';
  else if (method.startsWith('early_search:')) label = '영상 앞부분의 가만히 선 구간(자동)';
  else if (method.startsWith('search:')) label = '영상 중 가만히 선 구간(자동)';
  if (standing.reliability === 'low') label += ' · 기준 안정성 낮음';
  return label;
}

function nearestUsableRep(result, preferred) {
  const total = result.per_rep?.length || 0;
  if (!total) return -1;
  const usable = result.reps_usable || Array(total).fill(true);
  const safe = Math.max(0, Math.min(total - 1, preferred));
  if (usable[safe] !== false) return safe;
  for (let distance = 1; distance < total; distance += 1) {
    for (const candidate of [safe - distance, safe + distance]) {
      if (candidate >= 0 && candidate < total && usable[candidate] !== false) return candidate;
    }
  }
  return -1;
}

function primaryEvidenceRep() {
  const primary = primaryFeedback();
  const requested = Math.max(0, Number(primary?.reps?.[0] || 1) - 1);
  return nearestUsableRep(state.result, requested);
}

async function startRetry() {
  const targets = state.result.retry_targets || [];
  if (!targets.length) return;
  const evidence = {};
  for (const [metric, , , source] of targets) {
    const feedback = (state.result.feedback || []).find((item) => item.metric === (source || metric));
    const preferred = Math.max(0, Number(feedback?.reps?.[0] || 1) - 1);
    const repIndex = nearestUsableRep(state.result, preferred);
    if (repIndex < 0) continue;
    const canvas = document.createElement('canvas');
    await drawFrame(canvas, state.result.bottoms[repIndex]);
    evidence[metric] = { repIndex, dataUrl: canvas.toDataURL('image/jpeg', 0.88) };
  }
  state.retrySession = {
    previous: JSON.parse(JSON.stringify(state.result)),
    targets: JSON.parse(JSON.stringify(targets)),
    evidence,
    sourceName: state.sourceName,
  };
  state.comparison = null;
  const continueSample = Boolean(state.samplePair && state.samplePhase === 1);
  restart({ preserveRetry: true });
  if (continueSample) {
    try {
      await loadSampleSet(2);
    } catch (error) {
      showAnalysisError(error);
    }
  }
}

function renderRetry() {
  const targets = state.result.retry_targets || [];
  const show = state.result.result_usable !== false && targets.length > 0 && !state.retrySession;
  setHidden($('#retry-section'), !show);
  if (!show) return;
  const primary = primaryFeedback();
  $('#retry-context').textContent = primary
    ? `${primary.text} 같은 조건으로 한 세트를 더 촬영하면 직전 세트와 변화 방향을 비교합니다.`
    : '같은 조건으로 한 세트를 더 촬영하면 직전 세트와 변화 방향을 비교합니다.';

  const warmup = state.recommendations?.rules?.before_next_set;
  const item = warmup?.items?.[0];
  const container = $('#retry-warmup');
  if (!item) {
    container.replaceChildren();
    setHidden(container, true);
    return;
  }
  setHidden(container, false);
  container.innerHTML = `
    <div class="retry-warmup-media">
      <video controls playsinline preload="metadata" src="${escapeHtml(item.video_url)}"${item.thumbnail_url ? ` poster="${escapeHtml(item.thumbnail_url)}"` : ''}></video>
    </div>
    <div class="retry-warmup-body">
      <p class="retry-warmup-label">다음 세트 전 가볍게 몸 풀기(선택)</p>
      <h4>${escapeHtml(item.title)}</h4>
      <p>${escapeHtml(item.reason)}</p>
      <p>${Math.round(item.duration_seconds)}초 · 준비물 ${escapeHtml(item.equipment)}</p>
      <p>출처: 서울올림픽기념국민체육진흥공단, 국민체력100 동영상 정보(공공누리 제1유형)</p>
      <a class="recommend-link" href="${escapeHtml(item.video_url)}" target="_blank" rel="noopener noreferrer">재생이 안 되면 새 탭에서 보기</a>
    </div>
  `;
  container.querySelector('video')?.addEventListener('error', () => container.classList.add('media-error'));
}

async function renderVerify() {
  const section = $('#verify-section');
  if (!state.retrySession || !state.comparison) {
    setHidden(section, true);
    return;
  }
  setHidden(section, false);
  const rows = state.comparison;
  $('#verify-results').innerHTML = rows.map((row, index) => {
    const previousEvidence = state.retrySession.evidence[row.metric];
    const currentIndex = nearestUsableRep(state.result, previousEvidence?.repIndex ?? 0);
    const hasEvidence = previousEvidence && currentIndex >= 0;
    const verdictClass = row.verdict === '악화' ? 'worse' : (row.verdict === '비교불가' ? 'unavailable' : '');
    const reasons = row.reasons || [];
    const cautions = row.cautions || [];
    return `
      <article class="verify-result">
        <div class="verify-result-head"><h3>${escapeHtml(row.label)}</h3><span class="verify-verdict ${verdictClass}">${escapeHtml(row.display)}</span></div>
        <p class="verify-summary">${escapeHtml(row.summary)}</p>
        ${hasEvidence ? `<div class="verify-media">
          <div class="verify-shot"><small>직전 세트 · ${previousEvidence.repIndex + 1}회차 저점</small><img src="${previousEvidence.dataUrl}" alt="직전 세트의 저점 근거 장면"><strong>직전 세트 근거</strong></div>
          <div class="verify-shot"><small>새 세트 · ${currentIndex + 1}회차 저점</small><canvas id="verify-new-${index}"></canvas><strong>새 세트 근거</strong></div>
        </div>` : ''}
        ${reasons.map((text) => `<p class="verify-caution verify-reason"><b>비교 불가 사유</b><br>${escapeHtml(text)}</p>`).join('')}
        ${cautions.map((text) => `<p class="verify-caution"><b>확인 메모</b><br>${escapeHtml(text)}</p>`).join('')}
      </article>
    `;
  }).join('');
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const canvas = $(`#verify-new-${index}`);
    if (!canvas) continue;
    const previousEvidence = state.retrySession.evidence[row.metric];
    const currentIndex = nearestUsableRep(state.result, previousEvidence?.repIndex ?? 0);
    if (currentIndex >= 0) await drawFrame(canvas, state.result.bottoms[currentIndex]);
  }
}

function renderTechnical() {
  const gap = state.result.gap_fill || {};
  const warnings = state.result.warnings || [];
  const qc = state.result.capture_qc || {};
  const tracking = state.result.qc || {};
  const record = state.captureRecordMeta || {};
  const camera = record.cameraSettings || {};
  const modeLabel = { solo: '혼자 촬영', assisted: '다른 사람이 촬영', upload: '기존 영상 업로드', sample: '샘플' }[record.mode] || '확인 어려움';
  const stopLabel = { standing_complete: '2초 서기 감지', walk_away: '발 이동 감지', left_frame: '화면 이탈 감지', maximum: '최대 촬영 시간', manual: '수동 종료' }[record.stopReason] || '해당 없음';
  $('#technical-content').innerHTML = `
    <p><strong>입력:</strong> ${escapeHtml(state.sourceName)} · ${state.analysisFps}fps로 분석 · ${state.result.n_frames}프레임</p>
    <p><strong>기기 내 분석 시간:</strong> ${finite(state.analysisElapsedSeconds) ? `${state.analysisElapsedSeconds.toFixed(1)}초` : '확인 어려움'} · MediaPipe Lite · 최대 960px</p>
    <p><strong>준비자세:</strong> ${standingLabel()}</p>
    <p><strong>촬영 방식:</strong> ${modeLabel}${record.mimeType ? ` · ${escapeHtml(record.mimeType)}` : ''}${record.stopReason ? ` · ${stopLabel}` : ''}${finite(record.analysisEndSeconds) ? ` · 분석 종료 ${record.analysisEndSeconds.toFixed(1)}초` : ''}</p>
    ${record.mode === 'solo' ? `<p><strong>자동 촬영 기록:</strong> 권한 요청부터 준비 완료 ${finite(record.readinessSeconds) ? `${record.readinessSeconds.toFixed(1)}초` : '확인 어려움'} · 카운트다운 취소 ${record.countdownCancelCount ?? 0}회 · 카메라 ${camera.width ?? '?'}×${camera.height ?? '?'} ${finite(camera.frameRate) ? `${camera.frameRate.toFixed(0)}fps` : 'fps 확인 어려움'}</p>` : ''}
    <p><strong>미검출 연결:</strong> ${gap.n_frames_interpolated ?? 0}프레임 · 가장 긴 연속 공백 ${gap.longest_gap_frames ?? 0}프레임</p>
    <p><strong>촬영 기록:</strong> 화면 점유율 ${finite(qc.frame_fill_ratio) ? qc.frame_fill_ratio.toFixed(2) : '확인 어려움'} · 좌우 방향 표기 ${qc.side_labels_usable ? '사용 가능' : '사용하지 않음'}</p>
    <p><strong>추적 기록:</strong> 관절 튐 제외 ${tracking.jump_frames ?? 0}프레임 · 핵심 관절 미검출 비율 ${finite(tracking.core_missing_frac) ? `${Math.round(tracking.core_missing_frac * 100)}%` : '확인 어려움'}</p>
    ${warnings.length ? `<p><strong>안내 기록:</strong> ${warnings.map((warning) => escapeHtml(`${warning.code}: ${warning.message}`)).join(' ')}</p>` : ''}
    <p><strong>해석 범위:</strong> 정면 영상에 보이는 같은 세트 안의 반복 변화만 관찰하며, 변화의 원인·의학적 상태·실제 3D 관절각은 판단하지 않습니다.</p>
    <p><strong>안전 안내:</strong> 개인의 체력 수준에 맞춰 무리하지 않는 범위에서 수행하고, 필요한 경우 안전한 환경과 지지물을 확보하세요.</p>
    <p>촬영 품질 합격선과 연속 영상의 반복 병합 간격은 아직 확정되지 않았습니다. URL의 <code>adjMerge</code> 값이 제공된 경우에만 명시값을 사용합니다.</p>
  `;
}

async function renderResults() {
  setHidden($('#analysis-section'), true);
  setHidden($('#capture-section'), true);
  setHidden($('#results-section'), false);
  updateStep(3);
  renderNotices();
  const usable = state.result.result_usable !== false;
  setHidden($('#unusable-result'), usable);
  setHidden($('#usable-result'), !usable);
  if (!usable) {
    $('#result-title').textContent = '촬영 상태를 확인해 주세요.';
    const errors = (state.result.warnings || []).filter((warning) => warning.severity === 'error');
    $('#unusable-messages').innerHTML = errors.length
      ? errors.map((warning) => `<p>${escapeHtml(warning.message)}</p>`).join('')
      : '<p>이 영상에서는 반복별 값을 안정적으로 확인하기 어렵습니다.</p>';
    await renderVerify();
    $('#results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    return;
  }
  state.selectedRep = primaryEvidenceRep();
  state.comparisonMetric = primaryFeedback()?.metric || 'A2_knee_w_rel_stand';
  renderSummary();
  renderRepCharts();
  renderRepOverview();
  renderFeedback();
  renderObservation(state.selectedRep);
  await renderEvidence(state.selectedRep);
  await renderComparison();
  renderRecommendations();
  renderRetry();
  renderTechnical();
  await renderVerify();
  $('#results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function restart({ preserveRetry = false } = {}) {
  document.querySelector('.analysis-retry')?.remove();
  stopStream();
  clearInterval(state.recordTimer);
  state.repThumbnailObserver?.disconnect();
  state.repThumbnailObserver = null;
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.sourceUrl = null;
  state.sourceName = null;
  state.sourceKind = null;
  state.frames = null;
  state.result = null;
  state.analysisElapsedSeconds = null;
  state.analysisEndSeconds = null;
  state.captureRecordMeta = null;
  state.recordingStartedAt = null;
  state.stoppingRecording = false;
  state.recorder = null;
  state.recordChunks = [];
  state.cameraRequestedAt = null;
  state.cameraReadyAt = null;
  state.readinessCompletedAt = null;
  state.countdownCancelCount = 0;
  state.cameraSettings = null;
  state.comparisonMetric = null;
  state.comparison = null;
  if (!preserveRetry) {
    state.retrySession = null;
    state.samplePhase = 0;
  }
  fileInput.value = '';
  video.pause();
  video.removeAttribute('src');
  video.load();
  setHidden($('#analysis-section'), true);
  setHidden($('#results-section'), true);
  setHidden($('#capture-section'), false);
  selectCaptureMode(state.captureMode);
  $('#capture-title').textContent = preserveRetry
    ? '같은 촬영 조건으로 한 세트를 더 진행하세요.'
    : '분석할 스쿼트 영상을 준비해 주세요.';
  $('#capture-deck').textContent = preserveRetry
    ? '첫 세트와 같은 폰 위치, 같은 발 위치·발 간격을 유지하고 준비자세 2초부터 시작합니다.'
    : '모바일은 영상을 선택하거나 카메라로 바로 촬영하고, 데스크톱은 촬영하거나 파일을 선택할 수 있습니다.';
  setHidden($('#retry-position-note'), !preserveRetry);
  updateStep(1);
  $('#capture-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

document.querySelectorAll('[data-action="go-capture"]').forEach((button) => {
  button.addEventListener('click', () => {
    selectCaptureMode('solo');
    $('#capture-section').scrollIntoView({ behavior: 'smooth' });
  });
});
document.querySelectorAll('[data-action="open-upload"]').forEach((button) => {
  button.addEventListener('click', () => {
    selectCaptureMode('upload');
    $('#capture-section').scrollIntoView({ behavior: 'smooth' });
    fileInput.click();
  });
});
document.querySelectorAll('[data-capture-mode]').forEach((button) => {
  button.addEventListener('click', () => selectCaptureMode(button.dataset.captureMode));
});
cameraButton.addEventListener('click', enableCamera);
recordButton.addEventListener('click', startRecording);
stopButton.addEventListener('click', () => { void stopRecording({ reason: 'manual' }); });
facingSelect.addEventListener('change', () => { state.facingMode = facingSelect.value; });
soundToggle.addEventListener('click', async () => {
  state.audioEnabled = !state.audioEnabled;
  soundToggle.setAttribute('aria-pressed', String(state.audioEnabled));
  soundToggle.textContent = state.audioEnabled ? '소리 안내 켜짐' : '소리 안내 꺼짐';
  if (state.audioEnabled) await ensureAudioReady();
  else window.speechSynthesis?.cancel?.();
});
$('#restart-button').addEventListener('click', () => restart());
$('#retake-button').addEventListener('click', () => restart({ preserveRetry: Boolean(state.retrySession) }));
$('#retry-button').addEventListener('click', startRetry);
$('#sample-button').addEventListener('click', startSampleExperience);
fileInput.addEventListener('change', async () => {
  const [file] = fileInput.files;
  if (!file) return;
  state.samplePhase = 0;
  stopStream();
  await loadVideoAndAnalyze(file, file.name, 'file', { analysisEndSeconds: null });
});
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && (state.stream || state.recorder?.state === 'recording')) void requestWakeLock();
});
window.addEventListener('beforeunload', () => {
  stopStream();
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.poseLandmarker?.close?.();
});
selectCaptureMode('solo');
initSampleExperience().catch(() => {});
