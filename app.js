const $ = id => document.getElementById(id);
const video = $('video');
const canvas = $('overlay');
const ctx = canvas.getContext('2d');
const stage = $('stage');
const statusEl = $('status');
const ocrCanvas = document.createElement('canvas');
const ocrCtx = ocrCanvas.getContext('2d');

const VEHICLES = new Set(['car', 'truck', 'bus', 'motorcycle']);
const COLOR = { person: '#00ffa6', vehicle: '#ff2bd6', plate: '#ffee00' };
const HUMAN = [[0.10, 0.17], [0.34, 0.20], [0.47, 0.50], [0.38, 0.52], [0.29, 0.34], [0.25, 0.52], [0.25, 1], [0.04, 1], [0.03, 0.64]];

let model, worker, workerInit, stream, running = false;
let facing = 'environment';
let tracks = [];
let nextId = 1;
let ocrBusy = false;
let lastOcr = 0;

const VERSION = '1.0.1';
$('ver').textContent = `DetectWorld v${VERSION}`;

const setStatus = s => { statusEl.textContent = s; };

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});

async function openCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  stage.style.aspectRatio = `${canvas.width}/${canvas.height}`;
  tracks = [];
}

function initOcr() {
  workerInit ||= (async () => {
    try {
      worker = await Tesseract.createWorker('eng');
      await worker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
        tessedit_pageseg_mode: '11',
      });
    } catch {
      worker = null;
    }
  })();
}

async function start() {
  $('start').hidden = true;
  setStatus('開啟相機…');
  try {
    await openCamera();
  } catch (e) {
    setStatus('無法開啟相機：' + e.message + '（需 HTTPS 或 localhost）');
    $('start').hidden = false;
    return;
  }
  $('flip').hidden = false;
  setStatus('載入偵測模型…');
  initOcr();
  model ||= await cocoSsd.load({ base: 'lite_mobilenet_v2' });
  running = true;
  setStatus('偵測中');
  loop();
}

async function loop() {
  if (!running) return;
  if (video.readyState >= 2) {
    const dets = await model.detect(video, 20, 0.5);
    update(dets);
    draw();
    maybeOcr();
  }
  requestAnimationFrame(loop);
}

function iou(a, b) {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / (a.w * a.h + b.w * b.h - inter);
}

function update(dets) {
  const now = performance.now();
  const used = new Set();
  for (const d of dets) {
    const kind = d.class === 'person' ? 'person' : VEHICLES.has(d.class) ? 'vehicle' : null;
    if (!kind) continue;
    const box = { x: d.bbox[0], y: d.bbox[1], w: d.bbox[2], h: d.bbox[3] };
    let best = null;
    let bestIou = 0.3;
    for (const t of tracks) {
      if (t.kind !== kind || used.has(t)) continue;
      const v = iou(t.box, box);
      if (v > bestIou) { bestIou = v; best = t; }
    }
    if (best) {
      for (const k of ['x', 'y', 'w', 'h']) best.box[k] += (box[k] - best.box[k]) * 0.6;
      best.score = d.score;
      best.seen = now;
      used.add(best);
    } else {
      const t = { id: nextId++, kind, box, score: d.score, seen: now, votes: {}, plate: null, ocrAt: 0 };
      tracks.push(t);
      used.add(t);
    }
  }
  tracks = tracks.filter(t => now - t.seen < 800);
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const now = performance.now();
  const lw = Math.max(2, canvas.width / 320);
  ctx.lineWidth = lw;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  for (const t of tracks) {
    if (now - t.seen > 300) continue;
    if (t.kind === 'person') drawPerson(t);
    else drawVehicle(t);
  }
}

function glow(color) {
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 14;
}

function drawPerson(t) {
  const { x, y, w, h } = t.box;
  const color = COLOR.person;
  const cx = x + w / 2;
  glow(color);
  ctx.fillStyle = color + '22';
  const ry = h * 0.08;
  const rx = Math.min(ry * 0.85, w * 0.2);
  ctx.beginPath();
  ctx.ellipse(cx, y + ry, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  const pts = [...HUMAN.map(([dx, dy]) => [cx + dx * w, y + dy * h]), [cx, y + 0.62 * h],
    ...HUMAN.map(([dx, dy]) => [cx - dx * w, y + dy * h]).reverse()];
  pts.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  label(`人 ${Math.round(t.score * 100)}%`, x, y, color);
}

function drawVehicle(t) {
  const { x, y, w, h } = t.box;
  glow(COLOR.vehicle);
  ctx.strokeRect(x, y, w, h);
  if (t.plate) {
    const r = t.plate.rel;
    glow(COLOR.plate);
    ctx.strokeRect(x + r.x * w, y + r.y * h, r.w * w, r.h * h);
  }
  label(t.plate ? `車 ${t.plate.text}` : '車', x, y, t.plate ? COLOR.plate : COLOR.vehicle);
}

function label(text, x, y, color) {
  const fs = Math.max(14, Math.round(canvas.width / 40));
  ctx.font = `bold ${fs}px system-ui, sans-serif`;
  ctx.textBaseline = 'middle';
  const pad = fs * 0.35;
  const tw = ctx.measureText(text).width + pad * 2;
  const th = fs + pad * 2;
  const ly = y - th < 0 ? y : y - th;
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#000c';
  ctx.fillRect(x, ly, tw, th);
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;
  ctx.fillStyle = color;
  ctx.fillText(text, x + pad, ly + th / 2);
  ctx.shadowBlur = 0;
}

function maybeOcr() {
  if (!worker || ocrBusy) return;
  const now = performance.now();
  if (now - lastOcr < 600) return;
  const t = tracks
    .filter(v => v.kind === 'vehicle' && !v.plate && v.box.w > canvas.width * 0.12 && now - v.seen < 300)
    .sort((a, b) => a.ocrAt - b.ocrAt)[0];
  if (!t) return;
  ocrBusy = true;
  lastOcr = now;
  t.ocrAt = now;
  readPlate(t).catch(() => {}).finally(() => { ocrBusy = false; });
}

async function readPlate(t) {
  const { x, y, w, h } = t.box;
  const sx = Math.max(0, x);
  const sy = Math.max(0, y + h * 0.35);
  const sw = Math.min(w, canvas.width - sx);
  const sh = Math.min(h * 0.65, canvas.height - sy);
  if (sw < 40 || sh < 20) return;
  const scale = Math.min(2, 800 / sw);
  ocrCanvas.width = Math.round(sw * scale);
  ocrCanvas.height = Math.round(sh * scale);
  ocrCtx.filter = 'grayscale(1) contrast(1.5)';
  ocrCtx.drawImage(video, sx, sy, sw, sh, 0, 0, ocrCanvas.width, ocrCanvas.height);
  const { data } = await worker.recognize(ocrCanvas);
  for (const line of data.lines || []) {
    if (line.confidence < 55) continue;
    const text = parsePlate(line.text);
    if (!text) continue;
    t.votes[text] = (t.votes[text] || 0) + 1;
    if (t.votes[text] >= 2) {
      const b = line.bbox;
      t.plate = {
        text,
        rel: {
          x: (sx + b.x0 / scale - x) / w,
          y: (sy + b.y0 / scale - y) / h,
          w: (b.x1 - b.x0) / scale / w,
          h: (b.y1 - b.y0) / scale / h,
        },
      };
    }
    break;
  }
}

function parsePlate(raw) {
  const s = raw.toUpperCase().replace(/[^A-Z0-9-]/g, '').replace(/^-+|-+$/g, '');
  let m = s.match(/[A-Z0-9]{2,4}-[A-Z0-9]{2,4}/);
  if (m && /\d/.test(m[0]) && /[A-Z]/.test(m[0])) return m[0];
  m = s.match(/^([A-Z]{2,3})(\d{3,4})$|^(\d{3,4})([A-Z]{2,3})$/);
  if (m) return m[1] ? `${m[1]}-${m[2]}` : `${m[3]}-${m[4]}`;
  return null;
}

$('start').addEventListener('click', start);
$('flip').addEventListener('click', async () => {
  facing = facing === 'environment' ? 'user' : 'environment';
  try { await openCamera(); } catch (e) { setStatus('切換失敗：' + e.message); }
});
