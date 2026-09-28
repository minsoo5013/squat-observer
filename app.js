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
  recommendations: null,
  retrySession: null,
  comparison: null,
  samplePair: null,
  samplePhase: 0,
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

const CONNECTIONS = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
  [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
];

const MAX_VIDEO_SECONDS = 90;

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
  const message = error instanceof Error ? error.message : String(error);
  showAnalysis('분석을 마치지 못했습니다.', message, 100);
  document.querySelector('.analysis-retry')?.remove();
  $('#analysis-detail').insertAdjacentHTML(
    'afterend',
    '<button class="button button-quiet analysis-retry">촬영 화면으로 돌아가기</button>',
  );
  $('.analysis-retry').addEventListener('click', () => restart({ preserveRetry: Boolean(state.retrySession) }));
}

function stopStream() {
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  video.srcObject = null;
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

async function enableCamera() {
  cameraButton.disabled = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('이 브라우저에서는 카메라 촬영을 지원하지 않습니다. 저장된 영상을 선택해 주세요.');
    }
    stopStream();
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: 'user',
        width: { ideal: 1080 },
        height: { ideal: 1920 },
        frameRate: { ideal: 30, max: 60 },
      },
    });
    video.removeAttribute('src');
    video.srcObject = state.stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
    setHidden(cameraPlaceholder, true);
    setHidden(recordButton, false);
    cameraButton.textContent = '카메라 다시 연결';
  } catch (error) {
    alert(error instanceof Error ? error.message : String(error));
  } finally {
    cameraButton.disabled = false;
  }
}

async function runCountdown() {
  setHidden(countdown, false);
  for (const label of ['2', '1', '시작']) {
    countdown.textContent = label;
    await sleep(label === '시작' ? 650 : 1000);
  }
  setHidden(countdown, true);
}

async function startRecording() {
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
  const startedAt = performance.now();
  state.recordTimer = setInterval(() => {
    recordTime.textContent = formatClock((performance.now() - startedAt) / 1000);
  }, 250);
  recordTime.textContent = '00:00';
  setHidden(recordBadge, false);
  setHidden(recordButton, true);
  setHidden(cameraButton, true);
  setHidden(stopButton, false);
  await runCountdown();
}

async function stopRecording() {
  if (!state.recorder || state.recorder.state === 'inactive') return;
  stopButton.disabled = true;
  const stopped = once(state.recorder, 'stop', 'error', 10000);
  state.recorder.stop();
  await stopped;
  clearInterval(state.recordTimer);
  setHidden(recordBadge, true);
  stopStream();

  const mimeType = state.recorder.mimeType || state.recordChunks[0]?.type || 'video/webm';
  const blob = new Blob(state.recordChunks, { type: mimeType });
  stopButton.disabled = false;
  setHidden(stopButton, true);
  await loadVideoAndAnalyze(blob, `browser-recording.${mimeType.includes('mp4') ? 'mp4' : 'webm'}`, 'camera');
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
  if (video.duration > MAX_VIDEO_SECONDS) {
    throw new Error(`영상이 ${MAX_VIDEO_SECONDS}초를 넘습니다. 전체 세트를 임의로 잘라 분석하지 않도록 중단했습니다. 사진 앱이나 편집 도구에서 한 세트만 ${MAX_VIDEO_SECONDS}초 이하로 잘라 다시 올려 주세요.`);
  }
  if (video.duration > 60) {
    $('#capture-deck').textContent = `${Math.ceil(video.duration)}초 영상입니다. 분석 화면에서 진행률과 예상 남은 시간을 확인할 수 있습니다.`;
  }
  setHidden(cameraPlaceholder, true);
}

async function loadVideoAndAnalyze(blob, name, kind) {
  try {
    state.sourceName = name;
    state.sourceKind = kind;
    await loadVideoSource(blob);
    await analyzeVideo();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (kind === 'file' && /loadedmetadata|미디어 준비|화면 크기/.test(message)) {
      showAnalysisError(new Error('브라우저가 이 영상 형식을 재생하지 못했습니다. MP4/WebM으로 변환하거나, iPhone HEVC 영상은 Safari에서 다시 시도해 주세요.'));
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

async function initPoseLandmarker() {
  if (state.poseLandmarker) return state.poseLandmarker;
  showAnalysis('자세 인식 도구를 준비하고 있습니다.', '처음 한 번만 내려받고 이후에는 브라우저 캐시를 사용합니다.', 8, '도구 준비 중 · 남은 시간 계산 중');
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
}

function analysisCanvasSize(width, height) {
  const maxSide = 960;
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
  const duration = video.duration;
  const frameCount = Math.max(1, Math.floor(duration * state.analysisFps));
  const frames = [];
  const startedAt = performance.now();

  state.frameWidth = width;
  state.frameHeight = height;
  for (let index = 0; index < frameCount; index += 1) {
    const time = index / state.analysisFps;
    await seekVideo(time);
    context.drawImage(video, 0, 0, width, height);
    const result = landmarker.detectForVideo(canvas, Math.round(index * 1000 / state.analysisFps));
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
    adj_merge: requestedAdjMerge(),
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
adj_raw = WEB_INPUT['adj_merge']
adj_merge = None if adj_raw is None else int(adj_raw)
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

function scaleWithinSet(value, values) {
  if (!finite(value) || !values.length) return 0;
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (Math.abs(max - min) < 1e-9) return 62;
  return 24 + ((value - min) / (max - min)) * 70;
}

function renderSummary() {
  const knee = extremeRep('A2_knee_w_rel_stand', 'min');
  const deep = extremeRep('D1_hip_ankle_rel', 'min');
  const shallow = extremeRep('D1_hip_ankle_rel', 'max');
  const usableCount = state.result.per_rep.filter((_, index) => repUsable(index)).length;
  $('#result-title').textContent = `${state.result.n_reps}회의 저점 장면을 나누어 보았습니다.`;
  $('#result-summary').innerHTML = `
    <div class="summary-chip"><small>구분된 반복</small><strong>${state.result.n_reps}회 · 값에 사용 ${usableCount}회</strong></div>
    <div class="summary-chip"><small>준비자세 대비 가장 좁은 무릎 간격</small><strong>${knee.index >= 0 ? `${knee.index + 1}회차 · ${percent(knee.value)}` : '확인 어려움'}</strong></div>
    <div class="summary-chip"><small>세트 안 깊이 순서</small><strong>${deep.index >= 0 ? `${deep.index + 1}회차가 가장 깊음` : '확인 어려움'}</strong></div>
  `;
  return { knee, deep, shallow };
}

function renderRepStrip() {
  const kneeValues = metricValues('A2_knee_w_rel_stand');
  const depthValues = metricValues('D1_hip_ankle_rel');
  const trunkValues = metricValues('C1_trunk_span_rel');
  $('#rep-strip').innerHTML = state.result.per_rep.map((rep, index) => {
    const excluded = !repUsable(index);
    return `
    <button class="rep-card ${index === state.selectedRep ? 'active' : ''} ${excluded ? 'excluded' : ''}" data-rep="${index}" role="listitem">
      <strong>${index + 1}회차</strong>
      ${excluded ? '<span class="excluded-label">가려져 제외</span>' : ''}
      <span class="mini-metric"><span>무릎</span><span class="mini-track"><i style="width:${scaleWithinSet(rep.A2_knee_w_rel_stand, kneeValues)}%"></i></span><b>${percent(rep.A2_knee_w_rel_stand)}</b></span>
      <span class="mini-metric"><span>깊이</span><span class="mini-track"><i style="width:${scaleWithinSet(rep.D1_hip_ankle_rel, depthValues)}%"></i></span><b>${percent(rep.D1_hip_ankle_rel)}</b></span>
      <span class="mini-metric"><span>상체</span><span class="mini-track"><i style="width:${scaleWithinSet(rep.C1_trunk_span_rel, trunkValues)}%"></i></span><b>${percent(rep.C1_trunk_span_rel)}</b></span>
    </button>
  `;
  }).join('');
  document.querySelectorAll('.rep-card').forEach((button) => {
    button.addEventListener('click', () => selectRep(Number(button.dataset.rep)));
  });
}

function renderObservation(index) {
  const rep = state.result.per_rep[index];
  $('#observation-title').textContent = `${index + 1}회차`;
  if (!repUsable(index)) {
    $('#observation-lines').innerHTML = '<div class="metric-line"><small>값에서 제외</small><strong>확인 어려움</strong><span>가장 낮은 자세 부근에서 몸이 가려져 이 반복의 값은 사용하지 않았습니다.</span></div>';
    $('#uncertainty-note').innerHTML = '<strong>근거 장면은 확인 가능</strong><br>아래 원본 장면은 표시하지만 반복 비교에는 포함하지 않습니다.';
    return;
  }
  const lines = [
    {
      label: '저점의 무릎 간격',
      value: percent(rep.A2_knee_w_rel_stand),
      note: '준비자세의 무릎 간격을 100%로 본 값입니다.',
    },
    {
      label: '저점의 골반–발목 세로거리',
      value: percent(rep.D1_hip_ankle_rel),
      note: '준비자세를 100%로 보며, 세트 안에서 낮을수록 더 깊은 반복입니다.',
    },
  ];
  if (finite(rep.C1_trunk_span_rel)) {
    lines.push({
      label: '정면에서 본 상체 길이',
      value: percent(rep.C1_trunk_span_rel),
      note: '준비자세를 100%로 본 정면 투영값이며 실제 3D 상체각이 아닙니다.',
    });
  }
  $('#observation-lines').innerHTML = lines.map((line) => `
    <div class="metric-line"><small>${line.label}</small><strong>${line.value}</strong><span>${line.note}</span></div>
  `).join('');
  const interpolated = state.result.bottom_on_interpolated_frame?.[index];
  $('#uncertainty-note').innerHTML = interpolated
    ? '<strong>확인 메모</strong><br>이 저점 프레임에는 미검출 관절을 이은 값이 포함돼 있습니다.'
    : '<strong>확인 메모</strong><br>좌우 방향의 원인은 정면 영상만으로 판단하지 않습니다.';
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

async function drawFrame(canvas, frameIndex) {
  const token = ++state.evidenceToken;
  await seekVideo(frameIndex / state.analysisFps);
  if (token !== state.evidenceToken) return;
  const width = video.videoWidth;
  const height = video.videoHeight;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, width, height);
  drawSkeleton(context, state.frames[frameIndex], width, height);
}

async function renderEvidence(index) {
  const bottom = state.result.bottoms[index];
  const isPrimaryEvidence = Boolean(primaryFeedback()) && index === primaryEvidenceRep();
  $('#evidence-title').textContent = `${isPrimaryEvidence ? '우선 확인 · ' : ''}${index + 1}회차의 가장 낮은 장면`;
  $('#frame-time').textContent = `${(bottom / state.analysisFps).toFixed(1)}초`;
  $('#evidence-caption').textContent = repUsable(index)
    ? '원본 장면 위에 MediaPipe 2D 관절을 표시했습니다. 강조된 점은 양쪽 무릎입니다.'
    : '이 반복은 가림 때문에 값에서 제외했습니다. 원본 장면과 검출된 2D 관절만 확인할 수 있습니다.';
  await drawFrame($('#evidence-canvas'), bottom);
}

async function selectRep(index) {
  state.selectedRep = index;
  renderRepStrip();
  renderObservation(index);
  await renderEvidence(index);
}

async function renderComparison() {
  const primary = primaryFeedback();
  const usableIndices = state.result.per_rep
    .map((_, index) => index)
    .filter((index) => repUsable(index));
  if (!primary?.rule_id?.endsWith('_LATE') || usableIndices.length < 2) {
    setHidden($('#compare-card'), true);
    return;
  }
  setHidden($('#compare-card'), false);
  const firstIndex = usableIndices[0];
  const lastIndex = usableIndices[usableIndices.length - 1];
  const first = state.result.per_rep[firstIndex];
  const last = state.result.per_rep[lastIndex];
  const isKnee = primary.metric === 'A2_knee_w_rel_stand';
  const metricLabel = isKnee ? '준비자세 대비 무릎 간격' : '가장 낮은 자세의 골반 높이';
  $('#compare-card').innerHTML = `
    <p class="card-label">한 가지 포인트의 근거</p>
    <div class="compare-grid">
      <div class="compare-item compare-shot"><small>${firstIndex + 1}회차 저점</small><canvas id="compare-first"></canvas><strong>${metricLabel} ${percent(first[primary.metric])}</strong></div>
      <div class="compare-item compare-shot"><small>${lastIndex + 1}회차 저점</small><canvas id="compare-last"></canvas><strong>${metricLabel} ${percent(last[primary.metric])}</strong></div>
    </div>
    <p class="evidence-caption">첫 반복과 마지막 반복의 같은 값만 나란히 보여드립니다.</p>
  `;
  await drawFrame($('#compare-first'), state.result.bottoms[firstIndex]);
  await drawFrame($('#compare-last'), state.result.bottoms[lastIndex]);
}

function renderRecommendations() {
  const section = $('#recommend-section');
  const selections = primaryFeedback() ? [primaryFeedback()] : [];
  const cards = [];
  const seen = new Set();
  for (const selection of selections) {
    if (selection.content_group === 'before_next_set') continue;
    const group = state.recommendations.rules[selection.content_group];
    if (!group) continue;
    for (const item of group.items) {
      if (seen.has(item.catalog_id)) continue;
      seen.add(item.catalog_id);
      const reason = item.reason || `${selection.text} → ${item.relation}`;
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
  const primary = primaryFeedback();
  setHidden($('#feedback-card'), false);
  $('#feedback-card').classList.toggle('neutral', !primary);
  if (!primary) {
    $('#feedback-title').textContent = '반복 사이 큰 변화가 두드러지지 않았습니다.';
    $('#feedback-text').textContent = '이번 세트에서는 반복 사이 큰 변화가 보이지 않았습니다.';
    return;
  }
  const titles = {
    KNEE_LATE: '후반 반복의 무릎 간격 변화',
    KNEE_REP: '특정 반복의 무릎 간격 변화',
    DEPTH_LATE: '후반 반복의 깊이 변화',
    DEPTH_REP: '특정 반복의 깊이 변화',
  };
  $('#feedback-title').textContent = titles[primary.rule_id] || '세트 안 반복 변화';
  $('#feedback-text').textContent = primary.text;
}

function standingLabel() {
  const standing = state.result.standing || {};
  const method = String(standing.method || '');
  let label = '확인 어려움';
  if (method === 'calibration') label = '영상 앞 2초';
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
  for (const [metric] of targets) {
    const feedback = (state.result.feedback || []).find((item) => item.metric === metric);
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
  $('#technical-content').innerHTML = `
    <p><strong>입력:</strong> ${escapeHtml(state.sourceName)} · ${state.analysisFps}fps로 분석 · ${state.result.n_frames}프레임</p>
    <p><strong>기기 내 분석 시간:</strong> ${finite(state.analysisElapsedSeconds) ? `${state.analysisElapsedSeconds.toFixed(1)}초` : '확인 어려움'} · MediaPipe Lite · 최대 960px</p>
    <p><strong>준비자세:</strong> ${standingLabel()}</p>
    <p><strong>미검출 보완:</strong> ${gap.n_frames_interpolated ?? 0}프레임 · 가장 긴 연속 공백 ${gap.longest_gap_frames ?? 0}프레임</p>
    <p><strong>촬영 기록:</strong> 화면 점유율 ${finite(qc.frame_fill_ratio) ? qc.frame_fill_ratio.toFixed(2) : '확인 어려움'} · 좌우 방향 표기 ${qc.side_labels_usable ? '사용 가능' : '사용하지 않음'}</p>
    <p><strong>추적 기록:</strong> 관절 튐 제외 ${tracking.jump_frames ?? 0}프레임 · 핵심 관절 미검출 비율 ${finite(tracking.core_missing_frac) ? `${Math.round(tracking.core_missing_frac * 100)}%` : '확인 어려움'}</p>
    ${warnings.length ? `<p><strong>안내 기록:</strong> ${warnings.map((warning) => escapeHtml(`${warning.code}: ${warning.message}`)).join(' ')}</p>` : ''}
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
  renderSummary();
  renderFeedback();
  renderRepStrip();
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
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.sourceUrl = null;
  state.sourceName = null;
  state.sourceKind = null;
  state.frames = null;
  state.result = null;
  state.analysisElapsedSeconds = null;
  state.comparison = null;
  if (!preserveRetry) {
    state.retrySession = null;
    state.samplePhase = 0;
  }
  fileInput.value = '';
  video.pause();
  video.removeAttribute('src');
  video.load();
  setHidden(cameraPlaceholder, false);
  setHidden($('#analysis-section'), true);
  setHidden($('#results-section'), true);
  setHidden($('#capture-section'), false);
  setHidden(recordButton, true);
  setHidden(stopButton, true);
  setHidden(cameraButton, false);
  cameraButton.textContent = '카메라 켜기';
  $('#capture-title').textContent = preserveRetry
    ? '같은 촬영 조건으로 한 세트를 더 진행하세요.'
    : '분석할 스쿼트 영상을 준비해 주세요.';
  $('#capture-deck').textContent = preserveRetry
    ? '첫 세트와 같은 폰 위치, 같은 발 위치·발 간격을 유지하고 준비자세 2초부터 시작합니다.'
    : '모바일은 갤러리에서 영상을 선택하고, 데스크톱은 촬영하거나 파일을 선택할 수 있습니다.';
  setHidden($('#retry-position-note'), !preserveRetry);
  updateStep(1);
  $('#capture-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

document.querySelectorAll('[data-action="go-capture"]').forEach((button) => {
  button.addEventListener('click', () => $('#capture-section').scrollIntoView({ behavior: 'smooth' }));
});
document.querySelectorAll('[data-action="open-upload"]').forEach((button) => {
  button.addEventListener('click', () => fileInput.click());
});
cameraButton.addEventListener('click', enableCamera);
recordButton.addEventListener('click', startRecording);
stopButton.addEventListener('click', stopRecording);
$('#restart-button').addEventListener('click', () => restart());
$('#retake-button').addEventListener('click', () => restart({ preserveRetry: Boolean(state.retrySession) }));
$('#retry-button').addEventListener('click', startRetry);
$('#sample-button').addEventListener('click', startSampleExperience);
fileInput.addEventListener('change', async () => {
  const [file] = fileInput.files;
  if (!file) return;
  state.samplePhase = 0;
  stopStream();
  await loadVideoAndAnalyze(file, file.name, 'file');
});
window.addEventListener('beforeunload', () => {
  stopStream();
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.poseLandmarker?.close?.();
});
initSampleExperience().catch(() => {});
