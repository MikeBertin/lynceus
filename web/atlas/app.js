// Lynceus — Demo B: self-supervised galaxy atlas (pan/zoom canvas).
"use strict";

const $ = (s) => document.querySelector(s);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const CLASS_COLORS = { smooth: null, featured: null, merger: null };

const stage = $("#stage"), canvas = $("#atlas"), ctx = canvas.getContext("2d");
const tip = $("#tip"), statusEl = $("#status");
let atlas = null, sprites = null, pts = [];
let mode = "morph";
const WORLD = 2200;                 // world size points are laid out in
let cam = { x: 0, y: 0, scale: 0.28 };   // view transform
let hover = -1;
let dpr = Math.max(1, window.devicePixelRatio || 1);

function resize() {
  dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = stage.clientWidth * dpr;
  canvas.height = stage.clientHeight * dpr;
  draw();
}

async function boot() {
  CLASS_COLORS.smooth = css("--c-smooth");
  CLASS_COLORS.featured = css("--c-featured");
  CLASS_COLORS.merger = css("--c-merger");
  try {
    atlas = await fetch("atlas.json").then((r) => r.json());
    sprites = new Image();
    await new Promise((res, rej) => { sprites.onload = res; sprites.onerror = rej; sprites.src = "sprites.png"; });
    pts = atlas.points.map((p) => ({ ...p, wx: p.x * WORLD, wy: p.y * WORLD }));
    $("#count").textContent = `${atlas.count.toLocaleString()} galaxies`;
    statusEl.textContent = "drag to fly · scroll to zoom";
    statusEl.style.color = css("--cyan");
    buildLegend();
    resize();       // set real canvas dimensions first
    resetView();    // then fit the view to them
    draw();
  } catch (e) {
    statusEl.textContent = "failed to load atlas: " + e.message;
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
}

// ---- colour ---------------------------------------------------------------
function lerp(a, b, t) { return a + (b - a) * t; }
function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
const FAINT = [40, 48, 66];

function colorFor(p) {
  if (mode === "morph") return CLASS_COLORS[p.d] || "#888";
  const v = mode === "featured" ? p.f : p.m;
  const hot = hexToRgb(mode === "featured" ? CLASS_COLORS.featured : CLASS_COLORS.merger);
  const r = lerp(FAINT[0], hot[0], v), g = lerp(FAINT[1], hot[1], v), b = lerp(FAINT[2], hot[2], v);
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}

// ---- draw -----------------------------------------------------------------
function draw() {
  if (!atlas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const showThumbs = cam.scale > 0.9;
  const tile = atlas.tile, cols = atlas.cols;
  const r = Math.max(1.4 * dpr, Math.min(4.5 * dpr, cam.scale * 7));

  // cull to viewport
  const pad = 60 * dpr;
  for (const p of pts) {
    const X = sx(p.wx), Y = sy(p.wy);
    if (X < -pad || X > canvas.width + pad || Y < -pad || Y > canvas.height + pad) continue;
    if (showThumbs) {
      const s = Math.min(64 * dpr, cam.scale * tile * 1.1);
      ctx.globalAlpha = mode === "morph" ? 1 : 0.35 + 0.65 * (mode === "featured" ? p.f : p.m);
      ctx.drawImage(sprites, (p.i % cols) * tile, ((p.i / cols) | 0) * tile, tile, tile,
        X - s / 2, Y - s / 2, s, s);
      ctx.globalAlpha = 1;
    } else {
      ctx.fillStyle = colorFor(p);
      ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fill();
    }
  }
  if (hover >= 0) {
    const p = pts[hover];
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * dpr;
    ctx.beginPath(); ctx.arc(sx(p.wx), sy(p.wy), 10 * dpr, 0, 7); ctx.stroke();
  }
}

function buildLegend() {
  const L = $("#legend");
  if (mode === "morph") {
    L.innerHTML = [["featured", "Featured / disk"], ["smooth", "Smooth"], ["merger", "Merger"]]
      .map(([k, n]) => `<span><i style="background:${CLASS_COLORS[k]}"></i>${n}</span>`).join("");
  } else {
    const c = mode === "featured" ? CLASS_COLORS.featured : CLASS_COLORS.merger;
    L.innerHTML = `<span><i style="background:rgb(40,48,66)"></i>low</span>` +
      `<span><i style="background:${c}"></i>high ${mode} vote</span>`;
  }
}

// ---- interaction ----------------------------------------------------------
let dragging = false, last = null;
stage.addEventListener("mousedown", (e) => { dragging = true; last = [e.clientX, e.clientY]; stage.classList.add("drag"); });
window.addEventListener("mouseup", () => { dragging = false; stage.classList.remove("drag"); });
window.addEventListener("mousemove", (e) => {
  if (dragging) {
    cam.x -= (e.clientX - last[0]) * dpr / cam.scale;
    cam.y -= (e.clientY - last[1]) * dpr / cam.scale;
    last = [e.clientX, e.clientY];
    draw();
  }
});
stage.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * dpr, my = (e.clientY - rect.top) * dpr;
  if (!dragging) updateHover(mx, my, e.clientX - rect.left, e.clientY - rect.top);
});
stage.addEventListener("mouseleave", () => { hover = -1; tip.style.opacity = 0; draw(); });
stage.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * dpr, my = (e.clientY - rect.top) * dpr;
  const wxAt = wx_(mx), wyAt = wy_(my);
  cam.scale *= Math.exp(-e.deltaY * 0.0014);
  cam.scale = Math.max(0.12, Math.min(8, cam.scale));
  cam.x = wxAt - (mx - canvas.width / 2) / cam.scale;
  cam.y = wyAt - (my - canvas.height / 2) / cam.scale;
  draw();
}, { passive: false });

function updateHover(mx, my, cssx, cssy) {
  let best = -1, bestD = (14 * dpr) ** 2;
  for (let i = 0; i < pts.length; i++) {
    const dx = sx(pts[i].wx) - mx, dy = sy(pts[i].wy) - my;
    const d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  if (best !== hover) { hover = best; draw(); }
  if (best >= 0) {
    const p = pts[best];
    const cols = atlas.cols, tile = atlas.tile;
    // crop the thumbnail from the sprite sheet onto a small canvas for the tip
    const tc = tip._c || (tip._c = document.createElement("canvas"));
    tc.width = tile; tc.height = tile;
    tc.getContext("2d").drawImage(sprites, (p.i % cols) * tile, ((p.i / cols) | 0) * tile, tile, tile, 0, 0, tile, tile);
    tip.innerHTML = `<img src="${tc.toDataURL()}"><div class="meta">` +
      `<b>${p.d}</b> · GZ votes<br>featured ${p.f} · smooth ${p.s} · merger ${p.m}</div>`;
    tip.style.opacity = 1;
    const tw = 150, th = 130;
    tip.style.left = Math.min(stage.clientWidth - tw - 8, cssx + 16) + "px";
    tip.style.top = Math.min(stage.clientHeight - th - 8, cssy + 16) + "px";
  } else tip.style.opacity = 0;
}

$("#colorseg").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  mode = b.dataset.mode;
  document.querySelectorAll("#colorseg button").forEach((x) => x.classList.toggle("on", x === b));
  const seg = $("#colorseg");
  seg.className = "seg" + (mode === "featured" ? " on-cyan" : mode === "merger" ? " on-rose" : "");
  buildLegend(); draw();
});
$("#reset").addEventListener("click", () => { resetView(); draw(); });
window.addEventListener("resize", resize);

boot();
