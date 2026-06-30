// Lynceus — Demo B: self-supervised galaxy atlas (pan/zoom/pinch canvas).
"use strict";

const $ = (s) => document.querySelector(s);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
// format a permutation p-value as a clean reporting threshold (e.g. "p<0.0001")
const fmtP = (p) => p == null ? "" : p < 1e-4 ? "p&lt;0.0001" : p < 0.01 ? "p&lt;0.01" : "p=" + p.toFixed(3);
const CLASS_COLORS = { smooth: null, featured: null, merger: null };
const CLASS_NAME = { smooth: "Smooth", featured: "Featured / disk", merger: "Merger" };
// sky-region (deep field) colours — match the globe markers in skymap.js
const REGION_COLORS = { "GOODS-S": "#5ec27a", "COSMOS": "#ff9e64", "UDS": "#c792ea" };

const stage = $("#stage"), canvas = $("#atlas"), ctx = canvas.getContext("2d");
const tip = $("#tip"), detail = $("#detail"), statusEl = $("#status");
let atlas = null, sprites = null, pts = [];
let mode = "region";   // story starts at "where" (sky region); see the colour-by order
const WORLD = 2200;
let cam = { x: 0, y: 0, scale: 0.28 };
let hover = -1, selected = -1;
let lrdPts = [], lrdOn = false, lrdSheet = null, lrdTile = 56, lrdCols = 1;
let hoverLRD = -1, selectedLRD = -1, lrdStatHTML = "";

const CONTEXT = {
  region: "These are the <b>three deep fields</b> the galaxies live in — mapped on the globe below&nbsp;↓. " +
          "In the atlas they're well mixed: <b>where</b> a galaxy is doesn't drive the structure.",
  morph: "Colour by Galaxy Zoo morphology: <b>smooth</b>, <b>featured/disk</b>, <b>merger</b>. " +
         "These the encoder discovered on its own — it was never shown a label.",
  featured: "Brightness = the volunteer <b>featured/disk</b> vote. The disk-like galaxies glow on one side.",
  merger: "Brightness = the volunteer <b>merger</b> vote.",
};
function updateContext() {
  $("#lrdstat").innerHTML = mode === "anomaly" ? lrdStatHTML : (CONTEXT[mode] || "");
}
let dpr = Math.max(1, window.devicePixelRatio || 1);
let fitted = false, drawPending = false;

// Coalesce many state changes per tick into one draw via a microtask. (A plain
// requestAnimationFrame loop can stall when the tab/preview isn't "visible".)
function markDirty() {
  if (drawPending) return;
  drawPending = true;
  Promise.resolve().then(() => { drawPending = false; draw(); });
}

function resize() {
  dpr = Math.max(1, window.devicePixelRatio || 1);
  // clientWidth can read 0 before layout (or in headless preview); fall back so
  // the canvas always gets a real backing size and renders.
  const cw = stage.clientWidth || window.innerWidth || 1100;
  const ch = stage.clientHeight || 600;
  canvas.width = Math.round(cw * dpr);
  canvas.height = Math.round(ch * dpr);
  glcanvas.width = canvas.width; glcanvas.height = canvas.height;
  if (!fitted) { resetView(); fitted = true; }
  markDirty();
}

async function boot() {
  CLASS_COLORS.smooth = css("--c-smooth");
  CLASS_COLORS.featured = css("--c-featured");
  CLASS_COLORS.merger = css("--c-merger");
  setupGL();
  try {
    atlas = await fetch("atlas.json?v=5").then((r) => r.json());
    sprites = new Image();
    await new Promise((res, rej) => { sprites.onload = res; sprites.onerror = rej; sprites.src = "sprites.jpg?v=3"; });
    pts = atlas.points.map((p) => ({ ...p, wx: p.x * WORLD, wy: p.y * WORLD }));
    uploadPositions(); uploadColors();
    $("#count").textContent = `${atlas.count.toLocaleString()} galaxies`;
    statusEl.textContent = "drag to fly · scroll to zoom · click a galaxy";
    statusEl.style.color = css("--cyan");
    buildLegend(); updateContext();
    loadLRDs();           // M3: known Little Red Dots overlay (non-blocking)
    buildWeirdest();      // M3: strip of the most anomalous galaxies
    new ResizeObserver(resize).observe(stage);  // fits + redraws once sized
    resize();
    $("#loading").classList.add("hide");
  } catch (e) {
    $("#loading").innerHTML = "failed to load atlas: " + e.message;
    console.error(e);
  }
}

// ---- transforms -----------------------------------------------------------
const sx = (wx) => (wx - cam.x) * cam.scale + canvas.width / 2;
const sy = (wy) => (wy - cam.y) * cam.scale + canvas.height / 2;
const wx_ = (px) => (px - canvas.width / 2) / cam.scale + cam.x;
const wy_ = (py) => (py - canvas.height / 2) / cam.scale + cam.y;

function resetView() {
  cam.x = WORLD / 2; cam.y = WORLD / 2;
  cam.scale = (Math.min(canvas.width, canvas.height) / WORLD) * 0.92;
  markDirty();
}

// ---- colour ---------------------------------------------------------------
const lerp = (a, b, t) => a + (b - a) * t;
const hexToRgb = (h) => { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; };
const FAINT = [44, 52, 72];
const HEAT = [255, 90, 77];   // anomaly "hot" colour
function colorFor(p) {
  if (mode === "morph") return CLASS_COLORS[p.d] || "#888";
  if (mode === "region") return REGION_COLORS[p.r] || "#888";
  if (mode === "anomaly") {
    const v = p.a || 0;
    return `rgb(${lerp(FAINT[0], HEAT[0], v) | 0},${lerp(FAINT[1], HEAT[1], v) | 0},${lerp(FAINT[2], HEAT[2], v) | 0})`;
  }
  const v = mode === "featured" ? p.f : p.m;
  const hot = hexToRgb(mode === "featured" ? CLASS_COLORS.featured : CLASS_COLORS.merger);
  return `rgb(${lerp(FAINT[0], hot[0], v) | 0},${lerp(FAINT[1], hot[1], v) | 0},${lerp(FAINT[2], hot[2], v) | 0})`;
}

// ---- WebGL point cloud ----------------------------------------------------
// The galaxy cloud (potentially tens of thousands of points) is drawn on a
// WebGL canvas behind the 2-D one — gl.POINTS scales to 50k+ at 60fps where a
// per-point canvas arc would crawl. The 2-D canvas in front keeps doing the
// low-count work: thumbnails on zoom-in, the LRD overlay, and hover/select rings.
const glcanvas = $("#atlasgl");
let gl = null, glProg = null, glLoc = {}, posBuf = null, colBuf = null, glN = 0;

const FAINTF = FAINT.map((v) => v / 255), HEATF = HEAT.map((v) => v / 255);
const mixF = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
const hexToRgbF = (h) => hexToRgb(h).map((v) => v / 255);
function colorRGB(p) {
  if (mode === "morph") return hexToRgbF(CLASS_COLORS[p.d] || "#888888");
  if (mode === "region") return hexToRgbF(REGION_COLORS[p.r] || "#888888");
  if (mode === "anomaly") return mixF(FAINTF, HEATF, p.a || 0);
  const hot = hexToRgbF(mode === "featured" ? CLASS_COLORS.featured : CLASS_COLORS.merger);
  return mixF(FAINTF, hot, mode === "featured" ? p.f : p.m);
}

const VERT = `attribute vec2 a_world; attribute vec3 a_color;
uniform vec2 u_cam; uniform float u_scale; uniform vec2 u_res; uniform float u_psize;
varying vec3 v_color;
void main() {
  vec2 screen = (a_world - u_cam) * u_scale + u_res * 0.5;
  gl_Position = vec4(screen.x / u_res.x * 2.0 - 1.0, 1.0 - screen.y / u_res.y * 2.0, 0.0, 1.0);
  gl_PointSize = u_psize; v_color = a_color;
}`;
const FRAG = `precision mediump float;
varying vec3 v_color; uniform float u_alpha;
void main() {
  float d = length(gl_PointCoord - 0.5);
  if (d > 0.5) discard;
  gl_FragColor = vec4(v_color, smoothstep(0.5, 0.4, d) * u_alpha);
}`;

function setupGL() {
  gl = glcanvas.getContext("webgl", { alpha: true, antialias: true, premultipliedAlpha: false });
  if (!gl) return false;
  const sh = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  glProg = gl.createProgram();
  gl.attachShader(glProg, sh(gl.VERTEX_SHADER, VERT));
  gl.attachShader(glProg, sh(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(glProg);
  for (const k of ["a_world", "a_color"]) glLoc[k] = gl.getAttribLocation(glProg, k);
  for (const k of ["u_cam", "u_scale", "u_res", "u_psize", "u_alpha"]) glLoc[k] = gl.getUniformLocation(glProg, k);
  posBuf = gl.createBuffer(); colBuf = gl.createBuffer();
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  return true;
}
function uploadPositions() {
  if (!gl) return;
  glN = pts.length;
  const a = new Float32Array(glN * 2);
  for (let i = 0; i < glN; i++) { a[i * 2] = pts[i].wx; a[i * 2 + 1] = pts[i].wy; }
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, a, gl.STATIC_DRAW);
}
function uploadColors() {
  if (!gl || !glN) return;
  const a = new Float32Array(glN * 3);
  for (let i = 0; i < glN; i++) { const c = colorRGB(pts[i]); a[i * 3] = c[0]; a[i * 3 + 1] = c[1]; a[i * 3 + 2] = c[2]; }
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf); gl.bufferData(gl.ARRAY_BUFFER, a, gl.DYNAMIC_DRAW);
  markDirty();
}
function drawGL(showThumbs) {
  if (!gl) return;
  gl.viewport(0, 0, glcanvas.width, glcanvas.height);
  gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
  if (showThumbs || !glN) return;            // thumbnails cover the cloud when zoomed in
  gl.useProgram(glProg);
  gl.uniform2f(glLoc.u_cam, cam.x, cam.y);
  gl.uniform1f(glLoc.u_scale, cam.scale);
  gl.uniform2f(glLoc.u_res, glcanvas.width, glcanvas.height);
  const r = Math.max(1.3 * dpr, Math.min(4.2 * dpr, cam.scale * 6.5));
  gl.uniform1f(glLoc.u_psize, 2 * r + 1.5);
  gl.uniform1f(glLoc.u_alpha, 0.92);
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.enableVertexAttribArray(glLoc.a_world); gl.vertexAttribPointer(glLoc.a_world, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
  gl.enableVertexAttribArray(glLoc.a_color); gl.vertexAttribPointer(glLoc.a_color, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.POINTS, 0, glN);
}

// ---- render ---------------------------------------------------------------
function draw() {
  if (!atlas || !canvas.width) return;
  const tile = atlas.tile, cols = atlas.cols;
  const showThumbs = cam.scale > 0.85;
  const pad = 80 * dpr;
  drawGL(showThumbs);                          // point cloud (WebGL, behind)
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (showThumbs) {                            // thumbnails take over on zoom-in
    for (const p of pts) {
      const X = sx(p.wx), Y = sy(p.wy);
      if (X < -pad || X > canvas.width + pad || Y < -pad || Y > canvas.height + pad) continue;
      const s = Math.min(70 * dpr, cam.scale * tile * 1.05);
      ctx.globalAlpha = (mode === "featured" || mode === "merger") ? 0.3 + 0.7 * (mode === "featured" ? p.f : p.m) : 1;
      ctx.drawImage(sprites, (p.i % cols) * tile, ((p.i / cols) | 0) * tile, tile, tile, X - s / 2, Y - s / 2, s, s);
    }
  } else if (!gl) {
    // Fallback for browsers without WebGL: draw the cloud as canvas-2D dots.
    const r = Math.max(1.3 * dpr, Math.min(4.2 * dpr, cam.scale * 6.5));
    ctx.globalAlpha = 0.92;
    for (const p of pts) {
      const X = sx(p.wx), Y = sy(p.wy);
      if (X < -pad || X > canvas.width + pad || Y < -pad || Y > canvas.height + pad) continue;
      ctx.fillStyle = colorFor(p);
      ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
  if (lrdOn) {                              // known Little Red Dots
    const thumbs = showThumbs && lrdSheet && lrdSheet.complete;
    for (const L of lrdPts) {
      const X = sx(L.wx), Y = sy(L.wy);
      if (X < -pad || X > canvas.width + pad || Y < -pad || Y > canvas.height + pad) continue;
      if (thumbs) {                         // show the real cutout, red-framed
        const s = Math.min(70 * dpr, cam.scale * lrdTile * 1.05);
        ctx.drawImage(lrdSheet, (L.i % lrdCols) * lrdTile, ((L.i / lrdCols) | 0) * lrdTile,
          lrdTile, lrdTile, X - s / 2, Y - s / 2, s, s);
        ctx.strokeStyle = "#ff5a4d"; ctx.lineWidth = 2 * dpr;
        ctx.strokeRect(X - s / 2, Y - s / 2, s, s);
      } else {                              // diamond marker when zoomed out
        const s = 4.5 * dpr;
        ctx.beginPath();
        ctx.moveTo(X, Y - s); ctx.lineTo(X + s, Y); ctx.lineTo(X, Y + s); ctx.lineTo(X - s, Y); ctx.closePath();
        ctx.fillStyle = "#ff5a4d"; ctx.fill();
        ctx.strokeStyle = "#1a0c0a"; ctx.lineWidth = 1 * dpr; ctx.stroke();
      }
    }
  }
  for (const [idx, col, w] of [[hover, "#ffffff", 1.6], [selected, css("--cyan"), 2.4]]) {
    if (idx < 0) continue;
    const p = pts[idx];
    ctx.strokeStyle = col; ctx.lineWidth = w * dpr;
    ctx.beginPath(); ctx.arc(sx(p.wx), sy(p.wy), (showThumbs ? 36 : 11) * dpr, 0, 7); ctx.stroke();
  }
  for (const [idx, col, w] of [[hoverLRD, "#ffffff", 1.6], [selectedLRD, "#ff5a4d", 2.6]]) {
    if (idx < 0) continue;
    const L = lrdPts[idx];
    ctx.strokeStyle = col; ctx.lineWidth = w * dpr;
    ctx.beginPath(); ctx.arc(sx(L.wx), sy(L.wy), (showThumbs ? 36 : 11) * dpr, 0, 7); ctx.stroke();
  }
}

function buildLegend() {
  const L = $("#legend");
  if (mode === "morph") {
    L.innerHTML = [["featured", "Featured / disk"], ["smooth", "Smooth"], ["merger", "Merger"]]
      .map(([k, n]) => `<span><i style="background:${CLASS_COLORS[k]}"></i>${n}</span>`).join("");
  } else if (mode === "region") {
    L.innerHTML = Object.entries(REGION_COLORS)
      .map(([k, c]) => `<span><i style="background:${c}"></i>${k}</span>`).join("");
  } else if (mode === "anomaly") {
    L.innerHTML = `<span><i style="background:rgb(44,52,72)"></i>typical</span>` +
      `<span><i style="background:rgb(255,90,77)"></i>most unusual</span>`;
  } else {
    const c = mode === "featured" ? CLASS_COLORS.featured : CLASS_COLORS.merger;
    L.innerHTML = `<span><i style="background:rgb(44,52,72)"></i>low</span><span><i style="background:${c}"></i>high ${mode} vote</span>`;
  }
}

// ---- picking + panels -----------------------------------------------------
function pickAt(mx, my, radPx) {
  let best = -1, bestD = (radPx * dpr) ** 2;
  for (let i = 0; i < pts.length; i++) {
    const dx = sx(pts[i].wx) - mx, dy = sy(pts[i].wy) - my, d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function spriteURL(sheet, idx, cols, tile, px) {
  const c = document.createElement("canvas"); c.width = c.height = px;
  const g = c.getContext("2d"); g.imageSmoothingEnabled = true;
  g.drawImage(sheet, (idx % cols) * tile, ((idx / cols) | 0) * tile, tile, tile, 0, 0, px, px);
  return c.toDataURL();
}
const tileDataURL = (p, px) => spriteURL(sprites, p.i, atlas.cols, atlas.tile, px);

function pickLRD(mx, my, radPx) {
  let best = -1, bestD = (radPx * dpr) ** 2;
  for (let i = 0; i < lrdPts.length; i++) {
    const dx = sx(lrdPts[i].wx) - mx, dy = sy(lrdPts[i].wy) - my, d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function showLRDTip(i, cssx, cssy) {
  const L = lrdPts[i];
  const img = lrdSheet && lrdSheet.complete ? `<img src="${spriteURL(lrdSheet, L.i, lrdCols, lrdTile, 80)}">` : "";
  tip.innerHTML = img + `<div class="meta"><b style="color:#ff7a6d">Little Red Dot</b><br>` +
    `z &asymp; ${L.z || "?"} · ${L.field}<br>anomaly ${L.a}</div>`;
  tip.style.opacity = 1;
  tip.style.left = Math.min(stage.clientWidth - 158, cssx + 16) + "px";
  tip.style.top = Math.min(stage.clientHeight - 140, cssy + 16) + "px";
}

function selectLRD(i) {
  selectedLRD = i; selected = -1; markDirty();
  const L = lrdPts[i];
  const img = lrdSheet && lrdSheet.complete ? `<img src="${spriteURL(lrdSheet, L.i, lrdCols, lrdTile, 168)}">` : "";
  detail.innerHTML = `<span class="x" id="dx">×</span>${img}` +
    `<div class="dl"><span class="pill" style="background:#ff5a4d">Little Red Dot</span><br>` +
    `<b>Kokorev et&nbsp;al. 2024</b><br>z &asymp; ${L.z || "?"} · field ${L.field}<br>` +
    `anomaly ${L.a} (1 = weirdest)</div>`;
  detail.classList.add("on");
  $("#dx").onclick = (e) => { e.stopPropagation(); selectedLRD = -1; detail.classList.remove("on"); markDirty(); };
}

function showTip(i, cssx, cssy) {
  const p = pts[i];
  tip.innerHTML = `<img src="${tileDataURL(p, 80)}"><div class="meta"><b>${CLASS_NAME[p.d]}</b><br>` +
    `featured ${p.f} · smooth ${p.s} · merger ${p.m}</div>`;
  tip.style.opacity = 1;
  tip.style.left = Math.min(stage.clientWidth - 158, cssx + 16) + "px";
  tip.style.top = Math.min(stage.clientHeight - 140, cssy + 16) + "px";
}
const hideTip = () => { tip.style.opacity = 0; };

function select(i) {
  selected = i; selectedLRD = -1; markDirty();
  if (i < 0) { detail.classList.remove("on"); return; }
  const p = pts[i], col = CLASS_COLORS[p.d];
  detail.innerHTML = `<span class="x" id="dx">×</span><img src="${tileDataURL(p, 168)}">` +
    `<div class="dl"><span class="pill" style="background:${col}">${CLASS_NAME[p.d]}</span><br>` +
    `<b>Galaxy Zoo votes</b><br>featured ${p.f} · smooth ${p.s}<br>merger ${p.m} · clumpy ${p.c}</div>`;
  detail.classList.add("on");
  $("#dx").onclick = (e) => { e.stopPropagation(); select(-1); };
}

// ---- mouse ----------------------------------------------------------------
// Backing-pixels per CSS-pixel (== dpr normally; differs if a fallback size was
// used). Used to map pointer coords into canvas space correctly in all cases.
const ratio = () => { const r = canvas.getBoundingClientRect(); return [canvas.width / r.width, canvas.height / r.height]; };
let dragging = false, last = null, moved = 0;
const evPos = (e) => { const r = canvas.getBoundingClientRect(); const [kx, ky] = ratio(); return [(e.clientX - r.left) * kx, (e.clientY - r.top) * ky, e.clientX - r.left, e.clientY - r.top]; };

stage.addEventListener("mousedown", (e) => { dragging = true; moved = 0; last = [e.clientX, e.clientY]; stage.classList.add("drag"); });
window.addEventListener("mouseup", (e) => {
  if (dragging && moved < 5) {
    const [mx, my] = evPos(e);
    const hl = lrdOn ? pickLRD(mx, my, 16) : -1;
    if (hl >= 0) selectLRD(hl); else select(pickAt(mx, my, 16));
  }
  dragging = false; stage.classList.remove("drag");
});
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const [kx, ky] = ratio();
  moved += Math.abs(e.clientX - last[0]) + Math.abs(e.clientY - last[1]);
  cam.x -= (e.clientX - last[0]) * kx / cam.scale;
  cam.y -= (e.clientY - last[1]) * ky / cam.scale;
  last = [e.clientX, e.clientY]; hideTip(); markDirty();
});
stage.addEventListener("mousemove", (e) => {
  if (dragging) return;
  const [mx, my, cx, cy] = evPos(e);
  const hl = lrdOn ? pickLRD(mx, my, 13) : -1;
  if (hl >= 0) {
    if (hl !== hoverLRD || hover !== -1) { hoverLRD = hl; hover = -1; markDirty(); }
    showLRDTip(hl, cx, cy);
  } else {
    if (hoverLRD !== -1) { hoverLRD = -1; markDirty(); }
    const h = pickAt(mx, my, 13);
    if (h !== hover) { hover = h; markDirty(); }
    if (h >= 0) showTip(h, cx, cy); else hideTip();
  }
});
stage.addEventListener("mouseleave", () => { hover = -1; hoverLRD = -1; hideTip(); markDirty(); });

function zoomAt(px, py, factor) {
  const wxAt = wx_(px), wyAt = wy_(py);
  cam.scale = Math.max(0.12, Math.min(9, cam.scale * factor));
  cam.x = wxAt - (px - canvas.width / 2) / cam.scale;
  cam.y = wyAt - (py - canvas.height / 2) / cam.scale;
  markDirty();
}
stage.addEventListener("wheel", (e) => {
  e.preventDefault(); const [mx, my] = evPos(e);
  zoomAt(mx, my, Math.exp(-e.deltaY * 0.0014));
}, { passive: false });

// ---- touch (pan + pinch) --------------------------------------------------
let touchMode = null, tLast = null, tDist = 0, tMoved = 0;
stage.addEventListener("touchstart", (e) => {
  e.preventDefault();
  if (e.touches.length === 1) { touchMode = "pan"; tMoved = 0; tLast = [e.touches[0].clientX, e.touches[0].clientY]; }
  else if (e.touches.length === 2) {
    touchMode = "pinch";
    const a = e.touches[0], b = e.touches[1];
    tDist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
  }
}, { passive: false });
stage.addEventListener("touchmove", (e) => {
  e.preventDefault();
  const [kx, ky] = ratio();
  if (touchMode === "pan" && e.touches.length === 1) {
    const t = e.touches[0];
    tMoved += Math.abs(t.clientX - tLast[0]) + Math.abs(t.clientY - tLast[1]);
    cam.x -= (t.clientX - tLast[0]) * kx / cam.scale;
    cam.y -= (t.clientY - tLast[1]) * ky / cam.scale;
    tLast = [t.clientX, t.clientY]; markDirty();
  } else if (touchMode === "pinch" && e.touches.length === 2) {
    const a = e.touches[0], b = e.touches[1];
    const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    const r = canvas.getBoundingClientRect();
    const mx = ((a.clientX + b.clientX) / 2 - r.left) * kx, my = ((a.clientY + b.clientY) / 2 - r.top) * ky;
    zoomAt(mx, my, d / (tDist || d)); tDist = d;
  }
}, { passive: false });
stage.addEventListener("touchend", (e) => {
  if (touchMode === "pan" && tMoved < 8 && tLast) {
    const r = canvas.getBoundingClientRect(); const [kx, ky] = ratio();
    select(pickAt((tLast[0] - r.left) * kx, (tLast[1] - r.top) * ky, 20));
  }
  touchMode = null;
}, { passive: false });

// ---- controls -------------------------------------------------------------
function setMode(m) {
  mode = m;
  document.querySelectorAll("#colorseg button").forEach((x) => x.classList.toggle("on", x.dataset.mode === m));
  $("#colorseg").className = "seg" + (m === "merger" ? " on-rose" : "");
  lrdOn = (m === "anomaly");                 // the Little Red Dots ride along with anomaly
  $("#lrdToggle").checked = lrdOn;
  buildLegend(); updateContext(); uploadColors(); markDirty();
}
$("#colorseg").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (b) setMode(b.dataset.mode);
});
$("#reset").addEventListener("click", () => { resetView(); select(-1); });
window.addEventListener("resize", resize);

// ---- M3: Little Red Dots overlay + weirdest strip -------------------------
function loadLRDs() {
  fetch("lrds.json?v=5").then((r) => r.json()).then((d) => {
    lrdTile = d.tile; lrdCols = d.cols;
    lrdSheet = new Image(); lrdSheet.onload = markDirty; lrdSheet.src = "lrd_sprites.jpg?v=3";
    lrdPts = d.points.map((p) => ({ ...p, wx: p.x * WORLD, wy: p.y * WORLD }));
    const ci = d.enrichment_ci ? ` (95% CI ${d.enrichment_ci[0]}–${d.enrichment_ci[1]}×, ${fmtP(d.p_value)})` : "";
    lrdStatHTML =
      `<b>All ${d.n} known Little Red Dots</b> in these fields (Kokorev et&nbsp;al. 2024) — a ` +
      `<b>separate published catalogue</b>, laid over the ${(atlas ? atlas.count.toLocaleString() : "reference")} ` +
      `atlas galaxies and embedded with the same encoder — are <b>${d.enrichment}×</b>${ci} over-represented ` +
      `among the top-10% most anomalous galaxies (the median dot lands at the ` +
      `${Math.round(d.median_pct * 100)}th percentile of weirdness). They pile into the hot zones the ` +
      `encoder flagged with no labels — zoom in to see them. (More galaxies sharpen this; the 216 ` +
      `known dots are fixed.)`;
    updateContext();
  }).catch(() => {});
}

$("#lrdToggle").addEventListener("change", (e) => {
  if (e.target.checked && mode !== "anomaly") { setMode("anomaly"); return; }
  lrdOn = e.target.checked;            // allow hiding LRDs while staying in anomaly
  markDirty();
});

function buildWeirdest() {
  const top = [...pts].sort((a, b) => (b.a || 0) - (a.a || 0)).slice(0, 18);
  const host = $("#weird");
  host.innerHTML = "<h3>The 18 weirdest galaxies the encoder found</h3>";
  const row = document.createElement("div"); row.className = "weirdrow";
  for (const p of top) {
    const im = document.createElement("img");
    im.src = tileDataURL(p, 54); im.title = "anomaly " + (p.a || 0).toFixed(2);
    im.onclick = () => { select(p.i); centreOn(p); };
    row.appendChild(im);
  }
  host.appendChild(row);
}

function centreOn(p) {
  cam.x = p.wx; cam.y = p.wy; cam.scale = Math.max(cam.scale, 1.6); markDirty();
}

boot();
