// Lynceus — Demo B: self-supervised galaxy atlas (pan/zoom/pinch canvas).
"use strict";

const $ = (s) => document.querySelector(s);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const CLASS_COLORS = { smooth: null, featured: null, merger: null };
const CLASS_NAME = { smooth: "Smooth", featured: "Featured / disk", merger: "Merger" };

const stage = $("#stage"), canvas = $("#atlas"), ctx = canvas.getContext("2d");
const tip = $("#tip"), detail = $("#detail"), statusEl = $("#status");
let atlas = null, sprites = null, pts = [];
let mode = "morph";
const WORLD = 2200;
let cam = { x: 0, y: 0, scale: 0.28 };
let hover = -1, selected = -1;
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
  if (!fitted) { resetView(); fitted = true; }
  markDirty();
}

async function boot() {
  CLASS_COLORS.smooth = css("--c-smooth");
  CLASS_COLORS.featured = css("--c-featured");
  CLASS_COLORS.merger = css("--c-merger");
  try {
    atlas = await fetch("atlas.json").then((r) => r.json());
    sprites = new Image();
    await new Promise((res, rej) => { sprites.onload = res; sprites.onerror = rej; sprites.src = "sprites.jpg"; });
    pts = atlas.points.map((p) => ({ ...p, wx: p.x * WORLD, wy: p.y * WORLD }));
    $("#count").textContent = `${atlas.count.toLocaleString()} galaxies`;
    statusEl.textContent = "drag to fly · scroll to zoom · click a galaxy";
    statusEl.style.color = css("--cyan");
    buildLegend();
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
function colorFor(p) {
  if (mode === "morph") return CLASS_COLORS[p.d] || "#888";
  const v = mode === "featured" ? p.f : p.m;
  const hot = hexToRgb(mode === "featured" ? CLASS_COLORS.featured : CLASS_COLORS.merger);
  return `rgb(${lerp(FAINT[0], hot[0], v) | 0},${lerp(FAINT[1], hot[1], v) | 0},${lerp(FAINT[2], hot[2], v) | 0})`;
}

// ---- render ---------------------------------------------------------------
function draw() {
  if (!atlas || !canvas.width) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const tile = atlas.tile, cols = atlas.cols;
  const showThumbs = cam.scale > 0.85;
  const r = Math.max(1.3 * dpr, Math.min(4.2 * dpr, cam.scale * 6.5));
  const pad = 80 * dpr;
  ctx.globalAlpha = showThumbs ? 1 : 0.92;
  for (const p of pts) {
    const X = sx(p.wx), Y = sy(p.wy);
    if (X < -pad || X > canvas.width + pad || Y < -pad || Y > canvas.height + pad) continue;
    if (showThumbs) {
      const s = Math.min(70 * dpr, cam.scale * tile * 1.05);
      ctx.globalAlpha = mode === "morph" ? 1 : 0.3 + 0.7 * (mode === "featured" ? p.f : p.m);
      ctx.drawImage(sprites, (p.i % cols) * tile, ((p.i / cols) | 0) * tile, tile, tile, X - s / 2, Y - s / 2, s, s);
    } else {
      ctx.fillStyle = colorFor(p);
      ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
  for (const [idx, col, w] of [[hover, "#ffffff", 1.6], [selected, css("--cyan"), 2.4]]) {
    if (idx < 0) continue;
    const p = pts[idx];
    ctx.strokeStyle = col; ctx.lineWidth = w * dpr;
    ctx.beginPath(); ctx.arc(sx(p.wx), sy(p.wy), (showThumbs ? 36 : 11) * dpr, 0, 7); ctx.stroke();
  }
}

function buildLegend() {
  const L = $("#legend");
  if (mode === "morph") {
    L.innerHTML = [["featured", "Featured / disk"], ["smooth", "Smooth"], ["merger", "Merger"]]
      .map(([k, n]) => `<span><i style="background:${CLASS_COLORS[k]}"></i>${n}</span>`).join("");
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

function tileDataURL(p, px) {
  const tile = atlas.tile, cols = atlas.cols;
  const c = document.createElement("canvas"); c.width = c.height = px;
  const g = c.getContext("2d"); g.imageSmoothingEnabled = true;
  g.drawImage(sprites, (p.i % cols) * tile, ((p.i / cols) | 0) * tile, tile, tile, 0, 0, px, px);
  return c.toDataURL();
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
  selected = i; markDirty();
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
  if (dragging && moved < 5) { const [mx, my] = evPos(e); select(pickAt(mx, my, 16)); }
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
  const h = pickAt(mx, my, 13);
  if (h !== hover) { hover = h; markDirty(); }
  if (h >= 0) showTip(h, cx, cy); else hideTip();
});
stage.addEventListener("mouseleave", () => { hover = -1; hideTip(); markDirty(); });

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
$("#colorseg").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  mode = b.dataset.mode;
  document.querySelectorAll("#colorseg button").forEach((x) => x.classList.toggle("on", x === b));
  $("#colorseg").className = "seg" + (mode === "merger" ? " on-rose" : "");
  buildLegend(); markDirty();
});
$("#reset").addEventListener("click", () => { resetView(); select(-1); });
window.addEventListener("resize", resize);

boot();
