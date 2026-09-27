// Lynceus: "where are we looking?" interactive celestial globe.
// Orthographic, drag/touch to rotate. No dependencies.
(function () {
  "use strict";
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const cv = document.getElementById("globe");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const COL = {};
  let geom = null, dpr = 1, R = 1, cx = 0, cy = 0;
  // Auto-spin is decoration: off under prefers-reduced-motion (the spin button
  // starts it), and the loop only runs while spinning, on screen and visible.
  const RM = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let yaw = -0.7, pitch = 0.45, spin = !RM, onScreen = true, raf = 0;
  const spinBtn = document.getElementById("globeSpin");
  let cssW = 360, cssH = 360;

  function resize() {
    dpr = Math.max(1, window.devicePixelRatio || 1);
    cssW = cv.clientWidth || cv.parentElement.clientWidth || 360;
    cssH = cv.clientHeight || 360;
    cv.width = Math.round(cssW * dpr); cv.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    R = Math.min(cssW, cssH) * 0.42; cx = cssW / 2; cy = cssH / 2;
    draw();
  }

  // Sun's position for a given date, as an equatorial unit vector
  // [cos(Dec)cos(RA), cos(Dec)sin(RA), sin(Dec)], the same convention as skygeom.json.
  // Low-precision solar ephemeris (good to a few arcmin; plenty for a 6px dot).
  function sunVector(date) {
    const rad = Math.PI / 180;
    const n = (date.getTime() - Date.UTC(2000, 0, 1, 12)) / 86400000;  // days since J2000.0
    const L = (280.460 + 0.9856474 * n) * rad;                          // mean longitude
    const g = (357.528 + 0.9856003 * n) * rad;                          // mean anomaly
    const lam = L + (1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * rad;  // ecliptic longitude
    const eps = 23.439 * rad;                                           // obliquity
    return [Math.cos(lam), Math.cos(eps) * Math.sin(lam), Math.sin(eps) * Math.sin(lam)];
  }
  const sunNow = sunVector(new Date());

  // rotate (yaw about vertical, pitch about horizontal), then orthographic project
  function project(v) {
    const ca = Math.cos(yaw), sa = Math.sin(yaw), cb = Math.cos(pitch), sb = Math.sin(pitch);
    const x1 = v[0] * ca + v[2] * sa, z1 = -v[0] * sa + v[2] * ca, y1 = v[1];
    const y2 = y1 * cb - z1 * sb, z2 = y1 * sb + z1 * cb;
    return [cx + R * x1, cy - R * y2, z2];   // x, y, depth(+ = toward viewer)
  }

  function ringPath(pts, color) {
    for (let i = 0; i < pts.length - 1; i++) {
      const a = project(pts[i]), b = project(pts[i + 1]);
      const front = (a[2] + b[2]) / 2 > 0;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
      ctx.strokeStyle = color; ctx.globalAlpha = front ? 0.95 : 0.16;
      ctx.lineWidth = front ? 2.2 : 1.2; ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function marker(v, color, label, opts = {}) {
    const p = project(v), front = p[2] > 0;
    if (opts.beam && front) {            // sightline from Earth (centre) outward
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(p[0], p[1]);
      ctx.strokeStyle = "#ffffff"; ctx.globalAlpha = 0.18; ctx.lineWidth = 1; ctx.stroke();
      ctx.globalAlpha = 1;
    }
    const r = opts.r || 5;
    ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 7);
    if (front) { ctx.fillStyle = color; ctx.fill(); }
    else { ctx.strokeStyle = color; ctx.globalAlpha = 0.4; ctx.lineWidth = 1.2; ctx.stroke(); ctx.globalAlpha = 1; }
    if (label && (front || opts.always)) {
      ctx.fillStyle = front ? (opts.text || "#e9ecf6") : "#5a6480";
      ctx.font = (opts.fz || 11) + "px ui-sans-serif,system-ui,sans-serif";
      ctx.textAlign = opts.align || "left";
      ctx.fillText(label, p[0] + (opts.align === "right" ? -8 : 8), p[1] + 4);
    }
  }

  function draw() {
    if (!geom) return;
    ctx.clearRect(0, 0, cssW, cssH);
    // sphere disc + limb
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7);
    ctx.fillStyle = "#070b16"; ctx.fill();
    ctx.strokeStyle = "#222a40"; ctx.lineWidth = 1; ctx.stroke();
    // back-to-front: equator (ref), ecliptic, galactic
    ringPath(geom.rings.equator, COL.eq);
    ringPath(geom.rings.ecliptic, COL.ecl);
    ringPath(geom.rings.galactic, COL.gal);
    // Earth at centre (we look out from here)
    ctx.beginPath(); ctx.arc(cx, cy, 3, 0, 7); ctx.fillStyle = "#9fb0c8"; ctx.fill();
    // markers
    for (const m of geom.markers) {
      const faint = m.name.indexOf("pole") >= 0;
      marker(m.v, faint ? "#8e9ab2" : "#ff9e64", m.name, { r: faint ? 3 : 4, fz: faint ? 10 : 11, text: faint ? "#8e9ab2" : "#ffc59e" });
    }
    marker(sunNow, COL.sun, "Sun", { r: 6, text: COL.sun });
    // sky-region colours match the atlas "sky region" mode (app.js REGION_COLORS)
    const REG = { "GOODS-S": "#5ec27a", "COSMOS": "#ff9e64", "UDS": "#c792ea" };
    for (const f of geom.fields) marker(f.v, REG[f.name] || "#fff", f.name, { r: 5, beam: true, text: REG[f.name] });
  }

  // ---- interaction ----
  let drag = false, lx = 0, ly = 0;
  const down = (x, y) => { drag = true; setSpin(false); lx = x; ly = y; };
  const move = (x, y) => {
    if (!drag) return;
    yaw += (x - lx) * 0.01; pitch += (y - ly) * 0.01;
    pitch = Math.max(-1.45, Math.min(1.45, pitch));
    lx = x; ly = y; draw();
  };
  const up = () => { drag = false; };
  cv.addEventListener("mousedown", (e) => down(e.clientX, e.clientY));
  window.addEventListener("mousemove", (e) => move(e.clientX, e.clientY));
  window.addEventListener("mouseup", up);
  cv.addEventListener("touchstart", (e) => { const t = e.touches[0]; down(t.clientX, t.clientY); }, { passive: true });
  cv.addEventListener("touchmove", (e) => { e.preventDefault(); const t = e.touches[0]; move(t.clientX, t.clientY); }, { passive: false });
  cv.addEventListener("touchend", up);

  // gentle auto-spin until first interaction
  function tick() {
    raf = 0;
    if (!spin || !onScreen || document.hidden || !geom) return;
    yaw += 0.0025; draw();
    raf = requestAnimationFrame(tick);
  }
  function kick() { if (!raf) raf = requestAnimationFrame(tick); }
  function setSpin(on) {
    spin = on;
    if (spinBtn) spinBtn.textContent = on ? "❚❚ pause spin" : "▶ spin";
    kick();
  }
  if (spinBtn) spinBtn.addEventListener("click", () => setSpin(!spin));
  setSpin(spin);
  new IntersectionObserver((es) => { onScreen = es[0].isIntersecting; kick(); }).observe(cv);
  document.addEventListener("visibilitychange", kick);

  const sd = document.getElementById("sundate");
  if (sd) sd.textContent = new Date().toLocaleDateString("en-GB",
    { day: "numeric", month: "short", year: "numeric" });

  fetch("skygeom.json").then((r) => r.json()).then((g) => {
    geom = g;
    COL.gal = css("--c-smooth") || "#e6b85a";   // gold
    COL.ecl = css("--c-featured") || "#5ad1e6";  // cyan
    COL.eq = "#5a6480"; COL.sun = "#ffd24a";
    new ResizeObserver(resize).observe(cv.parentElement);
    resize();
    kick();
  }).catch((e) => console.error("skymap:", e));
})();
