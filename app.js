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
let plates = [];
let nextId = 1;
let ocrBusy = false;
let lastOcr = 0;

const VERSION = '1.0.3';
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
  plates = [];
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
      const t = { id: nextId++, kind, box, score: d.score, seen: now };
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
  drawPlates();
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

function plateOf(box) {
  const now = performance.now();
  return plates.find(p => plateShown(p) && now - p.seen < 4000 &&
    p.box.x + p.box.w / 2 >= box.x && p.box.x + p.box.w / 2 <= box.x + box.w &&
    p.box.y + p.box.h / 2 >= box.y && p.box.y + p.box.h / 2 <= box.y + box.h);
}

function drawVehicle(t) {
  const { x, y, w, h } = t.box;
  glow(COLOR.vehicle);
  ctx.strokeRect(x, y, w, h);
  const p = plateOf(t.box);
  label(p ? `車 ${p.text}` : '車', x, y, p ? COLOR.plate : COLOR.vehicle);
}

function drawPlates() {
  const now = performance.now();
  for (const p of plates) {
    if (!plateShown(p) || now - p.seen > 4000) continue;
    const { x, y, w, h } = p.box;
    glow(COLOR.plate);
    ctx.strokeRect(x, y, w, h);
    const inCar = tracks.some(t => t.kind === 'vehicle' && now - t.seen < 300 && plateOf(t.box) === p);
    if (!inCar) label(`牌 ${p.text}`, x, y, COLOR.plate);
  }
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
  if (now - lastOcr < 400) return;
  ocrBusy = true;
  lastOcr = now;
  readPlates().catch(() => {}).finally(() => { ocrBusy = false; });
}

function addPlate(text, box, conf) {
  const now = performance.now();
  const p = plates.find(v => v.text === text);
  if (p) {
    p.box = box;
    p.seen = now;
    p.votes++;
    p.conf = Math.max(p.conf, conf);
  } else {
    plates.push({ text, box, seen: now, votes: 1, conf });
  }
}

function unionBox(parts, scale) {
  const x0 = Math.min(...parts.map(p => p.bbox.x0));
  const y0 = Math.min(...parts.map(p => p.bbox.y0));
  const x1 = Math.max(...parts.map(p => p.bbox.x1));
  const y1 = Math.max(...parts.map(p => p.bbox.y1));
  return { x: x0 / scale, y: y0 / scale, w: (x1 - x0) / scale, h: (y1 - y0) / scale };
}

let regionIdx = 0;

function regions() {
  const W = canvas.width;
  const H = canvas.height;
  const now = performance.now();
  const clip = (x, y, w, h) => {
    const x0 = Math.max(0, x);
    const y0 = Math.max(0, y);
    return { x: x0, y: y0, w: Math.min(W, x + w) - x0, h: Math.min(H, y + h) - y0 };
  };
  const veh = tracks
    .filter(t => t.kind === 'vehicle' && now - t.seen < 300)
    .map(({ box: { x, y, w, h } }) => clip(x - w * 0.1, y + h * 0.3, w * 1.2, h * 0.8));
  const cols = 3;
  const rows = 2;
  const tw = (W / cols) * 1.3;
  const th = (H / rows) * 1.3;
  const tiles = [];
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) tiles.push(clip(i * (W - tw) / (cols - 1), j * (H - th) / (rows - 1), tw, th));
  }
  return veh.length && regionIdx % 3 ? veh : tiles;
}

async function readPlates() {
  const list = regions();
  const r = list[regionIdx++ % list.length];
  if (r.w < 40 || r.h < 20) return;
  const scale = Math.min(3, Math.max(0.5, 1000 / r.w));
  ocrCanvas.width = Math.round(r.w * scale);
  ocrCanvas.height = Math.round(r.h * scale);
  ocrCtx.filter = 'grayscale(1) contrast(1.4)';
  ocrCtx.drawImage(video, r.x, r.y, r.w, r.h, 0, 0, ocrCanvas.width, ocrCanvas.height);
  const { data } = await worker.recognize(ocrCanvas);
  plates = plates.filter(p => performance.now() - p.seen < 8000);
  for (const line of data.lines || []) {
    const words = (line.words || []).filter(w => w.confidence > 30);
    const cands = [];
    for (let i = 0; i < words.length - 1; i++) {
      cands.push([words[i].text + '-' + words[i + 1].text, [words[i], words[i + 1]]]);
    }
    cands.push([line.text, [line]]);
    for (const w of words) cands.push([w.text, [w]]);
    for (const [raw, parts] of cands) {
      const text = parsePlate(raw);
      if (!text) continue;
      const b = unionBox(parts, scale);
      b.x += r.x;
      b.y += r.y;
      addPlate(text, b, Math.min(...parts.map(p => p.confidence)));
      break;
    }
  }
}

const plateShown = p => (p.votes >= 2 || p.conf >= 80) && performance.now() - p.seen < 4000;

const TO_DIGIT = { O: '0', Q: '0', D: '0', I: '1', L: '1', Z: '2', S: '5', G: '6', B: '8' };
const TO_LETTER = { 0: 'O', 1: 'I', 2: 'Z', 5: 'S', 6: 'G', 8: 'B' };
const convert = (s, own, map) => {
  let fixed = 0;
  const out = [...s].map(c => {
    if (own.test(c)) return c;
    fixed++;
    return map[c] || '?';
  }).join('');
  return fixed > 1 ? '?' : out;
};
const asLetters = s => convert(s, /[A-Z]/, TO_LETTER);
const asDigits = s => convert(s, /\d/, TO_DIGIT);

function fit(a, b) {
  const l = asLetters(a);
  const d = asDigits(b);
  if (/^[A-Z]{2,3}$/.test(l) && /^\d{3,4}$/.test(d)) return `${l}-${d}`;
  const d2 = asDigits(a);
  const l2 = asLetters(b);
  if (/^\d{3,4}$/.test(d2) && /^[A-Z]{2,3}$/.test(l2)) return `${d2}-${l2}`;
  return null;
}

function parsePlate(raw) {
  const s = raw.toUpperCase().replace(/[·.•_\s-]+/g, '-').replace(/[^A-Z0-9-]/g, '').replace(/^-+|-+$/g, '');
  if (s.includes('-')) {
    const parts = s.split('-').filter(Boolean);
    for (let i = 0; i < parts.length - 1; i++) {
      const r = fit(parts[i], parts[i + 1]);
      if (r) return r;
    }
    return null;
  }
  if (s.length < 5 || s.length > 7) return null;
  for (const k of [2, 3, 4]) {
    const r = fit(s.slice(0, k), s.slice(k));
    if (r) return r;
  }
  return null;
}

$('start').addEventListener('click', start);
$('flip').addEventListener('click', async () => {
  facing = facing === 'environment' ? 'user' : 'environment';
  try { await openCamera(); } catch (e) { setStatus('切換失敗：' + e.message); }
});
