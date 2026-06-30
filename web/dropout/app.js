"use strict";
// M4 — the dropout hunter. A small photo-z MLP runs live (ONNX) on nine band
// fluxes; the page shows the Lyman break sweeping through the filters, the
// network's redshift PDF, and a Lyman-break colour-colour diagram.

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const $ = (s) => document.querySelector(s);
const dpr = Math.min(window.devicePixelRatio || 1, 2);

// NIRCam filmstrip filters (blue->red): [name, lo, hi, pivot] in microns.
const FILTERS = [
  ["F115W", 1.013, 1.283, 1.154], ["F150W", 1.331, 1.668, 1.501],
  ["F200W", 1.755, 2.226, 1.990], ["F277W", 2.423, 3.132, 2.786],
  ["F356W", 3.135, 3.981, 3.563], ["F444W", 3.881, 4.982, 4.421],
];
const LYA = 0.12157;              // Lyman-alpha rest wavelength (micron)
const RED_REF = ["f277w", "f356w", "f444w"];
const BAD = -90, SN_MIN = 2, SOFT = 0.05;

let D = null, META = null, sprite = null, session = null;
let sel = 0, curBand = 2, sweepTimer = null, ZC = null;

// ---- redshift colour ramp (low z blue -> high z red) ----------------------
const RAMP = [[0, [59,110,165]], [4, [70,194,168]], [7, [232,193,90]],
              [10, [255,122,69]], [14, [255,59,59]]];
function zColor(z) {
  z = Math.max(0, Math.min(14, z));
  for (let i = 1; i < RAMP.length; i++) {
    if (z <= RAMP[i][0]) {
      const [z0, c0] = RAMP[i - 1], [z1, c1] = RAMP[i];
      const t = (z - z0) / (z1 - z0);
      return c0.map((c, k) => Math.round(c + t * (c1[k] - c)));
    }
  }
  return RAMP[RAMP.length - 1][1];
}
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

// ---- featurise (must match core.photoz.featurize exactly) -----------------
function featurize(flux, err) {
  const B = META.bands, n = B.length;
  const valid = [], sn = [], det = [];
  for (let i = 0; i < n; i++) {
    const v = flux[i] > BAD && err[i] > 0 && isFinite(flux[i]) && isFinite(err[i]);
    valid.push(v);
    const s = v ? flux[i] / err[i] : 0; sn.push(s);
    det.push(v && s >= SN_MIN);
  }
  let ref = 0;
  for (const b of RED_REF) { const i = B.indexOf(b); if (i >= 0 && valid[i]) ref += flux[i]; }
  const scale = Math.max(ref, 1e-4);
  const f = [];
  for (let i = 0; i < n; i++) f.push(Math.asinh((det[i] ? flux[i] / scale : 0) / SOFT));
  for (let i = 0; i < n; i++) f.push(det[i] ? 1 : 0);
  for (let i = 0; i < n; i++) f.push(det[i] ? Math.log1p(sn[i]) : 0);
  f.push(Math.log10(scale));
  return Float32Array.from(f);
}

function softmax(a) {
  let m = -Infinity; for (const x of a) m = Math.max(m, x);
  let s = 0; const o = a.map((x) => { const e = Math.exp(x - m); s += e; return e; });
  return o.map((e) => e / s);
}

// PDF point estimates: refined peak (local mean around mode), spread, and a
// genuine bimodality measure — the probability mass and location of any second
// solution well away from the main peak (the low-z interloper signature).
function estimates(pdf) {
  let mi = 0; for (let i = 1; i < pdf.length; i++) if (pdf[i] > pdf[mi]) mi = i;
  const zmode = ZC[mi];
  let wn = 0, ws = 0, mean = 0, tot = 0;
  for (let i = 0; i < pdf.length; i++) {
    mean += pdf[i] * ZC[i]; tot += pdf[i];
    if (Math.abs(ZC[i] - zmode) <= 1.5) { wn += pdf[i] * ZC[i]; ws += pdf[i]; }
  }
  mean /= (tot || 1);
  const zpeak = ws > 0 ? wn / ws : zmode;
  let v = 0, sec = 0, si = -1, sp = 0;
  for (let i = 0; i < pdf.length; i++) {
    v += pdf[i] * (ZC[i] - mean) ** 2;
    if (Math.abs(ZC[i] - zpeak) > 3.5) { sec += pdf[i]; if (pdf[i] > sp) { sp = pdf[i]; si = i; } }
  }
  return { zpeak, zstd: Math.sqrt(v / (tot || 1)),
           sec: sec / (tot || 1), zsec: si >= 0 ? ZC[si] : null };
}

async function runModel(g) {
  const feat = featurize(g.flux, g.err);
  const t = new ort.Tensor("float32", feat, [1, feat.length]);
  const out = await session.run({ features: t });
  return softmax(Array.from(out.logits.data));
}

// ---- sprite helpers --------------------------------------------------------
const TILE = () => D.tile;
function drawTile(cv, galStrip, bandIdx) {
  const ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);
  if (!sprite) return;
  const t = TILE();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(sprite, bandIdx * t, galStrip * t, t, t, 0, 0, w, h);
}

// ---- filter strip + wavelength bar ----------------------------------------
function breakWavelength(z) { return LYA * (1 + z); }

function renderStrip() {
  const g = D.gallery[sel];
  const zb = breakWavelength(g.zpeak);
  const strip = $("#strip");
  strip.innerHTML = "";
  FILTERS.forEach((f, k) => {
    const dark = f[2] < zb;                       // band fully blueward of break
    const div = document.createElement("div");
    div.className = "filt" + (k === curBand ? " cur" : "") + (dark ? " dark" : "");
    const cv = document.createElement("canvas"); cv.width = cv.height = 132;
    div.appendChild(cv);
    const lab = document.createElement("div"); lab.className = "lab"; lab.textContent = f[0];
    div.appendChild(lab);
    div.onclick = () => { curBand = k; $("#scrub").value = k; renderStrip(); renderWave(); };
    strip.appendChild(div);
    drawTile(cv, g.strip, k);
  });
  const f = FILTERS[curBand];
  const dropped = f[2] < zb;
  $("#vmeta").innerHTML =
    `Filter <b>${f[0]}</b> · ${f[3].toFixed(2)} µm — ${dropped
      ? `<span style="color:#ff6b6b">below the break: the galaxy has dropped out</span>`
      : `<span style="color:#5ec27a">above the break: the galaxy shines here</span>`}.` +
    `<br>Lyman break for z = <b>${g.zpeak.toFixed(1)}</b> falls at <b>${zb.toFixed(2)} µm</b>.`;
}

function renderWave() {
  const cv = $("#wave"), ctx = cv.getContext("2d");
  const W = cv.clientWidth || 460, H = cv.clientHeight || 110;
  cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const g = D.gallery[sel];
  const lo = 0.9, hi = 5.1, pad = 8;
  const X = (um) => pad + (um - lo) / (hi - lo) * (W - 2 * pad);
  const zb = breakWavelength(g.zpeak);
  // shaded absorbed region (blueward of break)
  ctx.fillStyle = "rgba(255,60,60,.10)";
  ctx.fillRect(X(lo), 0, X(Math.min(zb, hi)) - X(lo), H - 18);
  // filter bandpasses
  FILTERS.forEach((f, k) => {
    const x0 = X(f[1]), x1 = X(f[2]);
    const on = f[2] >= zb;
    ctx.fillStyle = k === curBand ? "rgba(90,209,230,.85)" : (on ? "rgba(150,170,200,.40)" : "rgba(120,120,140,.18)");
    ctx.fillRect(x0, 18, x1 - x0, H - 40);
    ctx.fillStyle = k === curBand ? "#cfeaf2" : "rgba(190,200,220,.7)";
    ctx.font = "9px ui-monospace,monospace"; ctx.textAlign = "center";
    ctx.fillText(f[0], (x0 + x1) / 2, H - 6);
  });
  // break marker
  if (zb <= hi) {
    const xb = X(zb);
    ctx.strokeStyle = "#ff6b6b"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xb, 4); ctx.lineTo(xb, H - 20); ctx.stroke();
    ctx.fillStyle = "#ff6b6b"; ctx.font = "9px ui-monospace,monospace"; ctx.textAlign = "left";
    ctx.fillText("Lyman break", Math.min(xb + 4, W - 60), 12);
  }
}

// ---- redshift PDF ----------------------------------------------------------
function renderPDF(pdf) {
  const g = D.gallery[sel];
  const cv = $("#pdf"), ctx = cv.getContext("2d");
  const W = cv.clientWidth || 440, H = cv.clientHeight || 200;
  cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const padL = 6, padR = 6, padT = 10, padB = 20;
  const zmax = 15;
  const X = (z) => padL + z / zmax * (W - padL - padR);
  let pm = 0; for (const p of pdf) pm = Math.max(pm, p);
  const Y = (p) => H - padB - p / (pm * 1.1) * (H - padT - padB);
  // axis ticks
  ctx.fillStyle = "rgba(150,160,180,.6)"; ctx.font = "9px ui-monospace,monospace"; ctx.textAlign = "center";
  for (let z = 0; z <= zmax; z += 3) {
    ctx.strokeStyle = "rgba(120,130,150,.15)"; ctx.beginPath();
    ctx.moveTo(X(z), padT); ctx.lineTo(X(z), H - padB); ctx.stroke();
    ctx.fillText("z=" + z, X(z), H - 6);
  }
  // filled curve, tinted by redshift
  ctx.beginPath(); ctx.moveTo(X(0), Y(0));
  for (let i = 0; i < pdf.length; i++) ctx.lineTo(X(ZC[i]), Y(pdf[i]));
  ctx.lineTo(X(ZC[ZC.length - 1]), Y(0)); ctx.closePath();
  const grad = ctx.createLinearGradient(X(0), 0, X(zmax), 0);
  grad.addColorStop(0, rgb(zColor(0), .5)); grad.addColorStop(.45, rgb(zColor(6), .55));
  grad.addColorStop(1, rgb(zColor(13), .6));
  ctx.fillStyle = grad; ctx.fill();
  ctx.strokeStyle = "rgba(220,230,240,.85)"; ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < pdf.length; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(ZC[i]), Y(pdf[i]));
  ctx.stroke();
  // markers: our peak (cyan), template (amber dashed), spec (green)
  const mark = (z, col, dash, label) => {
    if (z == null || z < 0) return;
    ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(X(z), padT); ctx.lineTo(X(z), H - padB); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = col; ctx.textAlign = "left"; ctx.font = "9px ui-monospace,monospace";
    ctx.fillText(label, Math.min(X(z) + 3, W - 30), padT + 8);
  };
  mark(g.zphot, "rgba(255,200,90,.9)", [3, 3], "EAZY");
  if (g.zspec != null) mark(g.zspec, "rgba(94,194,122,.95)", [], "spec");
  mark(estimates(pdf).zpeak, "rgba(90,209,230,1)", [], "ours");
}

// ---- colour-colour ---------------------------------------------------------
function galColour(g) {
  const B = META.bands, fi = (b) => g.flux[B.indexOf(b)];
  const cx = -2.5 * Math.log10(Math.max(fi("f115w"), 1e-4) / Math.max(fi("f150w"), 1e-4));
  const cy = -2.5 * Math.log10(Math.max(fi("f200w"), 1e-4) / Math.max(fi("f444w"), 1e-4));
  return [Math.max(-2, Math.min(4, cx)), Math.max(-2, Math.min(4, cy))];
}
function renderCC() {
  const cv = $("#cc"), ctx = cv.getContext("2d");
  const W = cv.clientWidth || 440, H = cv.clientHeight || 290;
  cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const padL = 38, padR = 10, padT = 10, padB = 30;
  const xr = [-1.5, 3.5], yr = [-1.5, 3];
  const X = (x) => padL + (x - xr[0]) / (xr[1] - xr[0]) * (W - padL - padR);
  const Y = (y) => H - padB - (y - yr[0]) / (yr[1] - yr[0]) * (H - padT - padB);
  ctx.strokeStyle = "rgba(120,130,150,.18)"; ctx.fillStyle = "rgba(150,160,180,.65)";
  ctx.font = "9px ui-monospace,monospace";
  for (let x = -1; x <= 3; x++) { ctx.beginPath(); ctx.moveTo(X(x), padT); ctx.lineTo(X(x), H - padB); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText(x, X(x), H - padB + 12); }
  for (let y = -1; y <= 3; y++) { ctx.beginPath(); ctx.moveTo(padL, Y(y)); ctx.lineTo(W - padR, Y(y)); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(y, padL - 5, Y(y) + 3); }
  ctx.textAlign = "center"; ctx.fillStyle = "rgba(180,190,210,.8)";
  ctx.fillText(D.colourcolour.xlabel, (padL + W - padR) / 2, H - 3);
  ctx.save(); ctx.translate(10, (padT + H - padB) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(D.colourcolour.ylabel, 0, 0); ctx.restore();
  for (const p of D.colourcolour.points) {
    ctx.fillStyle = rgb(zColor(p.z), .55);
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 1.6, 0, 7); ctx.fill();
  }
  // selected galaxy ring
  const [cx, cy] = galColour(D.gallery[sel]);
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(X(cx), Y(cy), 6, 0, 7); ctx.stroke();
  ctx.strokeStyle = rgb(zColor(D.gallery[sel].zpeak)); ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(X(cx), Y(cy), 6, 0, 7); ctx.stroke();
}

// ---- selection -------------------------------------------------------------
async function select(i) {
  sel = i;
  document.querySelectorAll(".gal .g").forEach((el, k) => el.classList.toggle("sel", k === i));
  const g = D.gallery[i];
  renderStrip(); renderWave();
  const pdf = session ? await runModel(g) : g.pdf;
  const est = estimates(pdf);
  $("#z-ours").innerHTML = `${est.zpeak.toFixed(1)} <small>± ${est.zstd.toFixed(1)}</small>`;
  $("#z-ours").style.color = rgb(zColor(est.zpeak));
  $("#z-temp").innerHTML = `<small>z =</small> ${g.zphot.toFixed(1)}`;
  $("#z-spec").innerHTML = g.zspec != null ? `<small>z =</small> ${g.zspec.toFixed(1)}` : "<small>—</small>";
  // verdict
  const bluest = FILTERS.find((f) => f[2] >= breakWavelength(est.zpeak));
  const lo = Math.min(est.zpeak, est.zsec ?? est.zpeak), hi = Math.max(est.zpeak, est.zsec ?? est.zpeak);
  let v;
  if (est.sec > 0.15 && est.zsec != null)
    v = `<b>Two solutions.</b> The net favours z &approx; <b>${est.zpeak.toFixed(1)}</b> but keeps a ` +
        `real second peak near z &approx; <b>${est.zsec.toFixed(1)}</b> — a dusty z&nbsp;~&nbsp;${lo.toFixed(0)} ` +
        `interloper mimicking a z&nbsp;~&nbsp;${hi.toFixed(0)} dropout. This ambiguity is exactly why ` +
        `"too many bright early galaxies" needs spectroscopy.`;
  else if (est.zpeak >= 8)
    v = `Lit only from <b>${bluest ? bluest[0] : "the reddest bands"}</b> redward — a clean high-z ` +
        `<b>dropout</b> at z &approx; ${est.zpeak.toFixed(1)}, with a single confident peak.`;
  else
    v = `The break sits near <b>${(breakWavelength(est.zpeak)).toFixed(2)} µm</b>; the net reads ` +
        `z &approx; ${est.zpeak.toFixed(1)}${g.zspec != null ? ` — spectroscopy says ${g.zspec.toFixed(2)}` : ""}.`;
  $("#verdict").innerHTML = v;
  renderPDF(pdf); renderCC();
}

// ---- build gallery + metrics ----------------------------------------------
function buildGallery() {
  const gal = $("#gallery"); gal.innerHTML = "";
  D.gallery.forEach((g, i) => {
    const d = document.createElement("div"); d.className = "g";
    const cv = document.createElement("canvas"); cv.width = cv.height = 132;
    d.appendChild(cv);
    const z = document.createElement("div"); z.className = "z";
    z.textContent = "z " + g.zpeak.toFixed(1);
    d.appendChild(z);
    d.onclick = () => select(i);
    gal.appendChild(d);
    drawTile(cv, g.strip, 5);                     // F444W thumbnail (red band)
  });
}
function buildMetrics() {
  const m = D.metrics, eb = m.eazy_baseline;
  $("#metrics").innerHTML = [
    ["σ_NMAD (ours)", m.sigma_nmad.toFixed(3)],
    ["outliers", (m.outlier_frac * 100).toFixed(0) + "%"],
    ["spec-z tested", m.n_val_spec.toLocaleString()],
    ["trained on", (m.n_train / 1000).toFixed(0) + "k"],
  ].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  const nmadCI = m.sigma_nmad_ci ? `95% CI ${m.sigma_nmad_ci[0].toFixed(3)}&ndash;${m.sigma_nmad_ci[1].toFixed(3)}, ` : "";
  $("#metricnote").innerHTML =
    `Scored against <b>${m.n_val_spec.toLocaleString()}</b> real spectroscopic redshifts the net never ` +
    `saw. Our <b>σ<sub>NMAD</sub> = ${m.sigma_nmad.toFixed(3)}</b> (${nmadCI}${(m.outlier_frac * 100).toFixed(0)}% ` +
    `catastrophic outliers) sits within a hair of the <b>EAZY template</b> ceiling it distils ` +
    `(${eb.sigma_nmad.toFixed(3)}, ${(eb.outlier_frac * 100).toFixed(0)}%) — at a millionth of the ` +
    `compute, running in this browser tab.`;
}

// ---- boot ------------------------------------------------------------------
async function boot() {
  [D, META] = await Promise.all([
    fetch("dropout.json?v=2").then((r) => r.json()),
    fetch("photoz_meta.json?v=1").then((r) => r.json()),
  ]);
  ZC = D.z_centres;
  sprite = new Image();
  await new Promise((res) => { sprite.onload = res; sprite.onerror = res; sprite.src = "filmstrips.jpg?v=1"; });
  buildGallery(); buildMetrics();
  // open on the cleanest high-z dropout: zpeak >= 8 with the least second-peak mass
  let start = -1, best = 1e9;
  D.gallery.forEach((g, i) => {
    const e = estimates(g.pdf);
    if (e.zpeak >= 8 && e.sec < best) { best = e.sec; start = i; }
  });
  if (start < 0) start = D.gallery.length - 1;
  sel = start; curBand = 4;
  try {
    session = await ort.InferenceSession.create("photoz.onnx?v=1", { executionProviders: ["wasm"] });
    $("#status").textContent = "model ready — pick a galaxy.";
    $("#status").style.color = "var(--muted)";
  } catch (e) {
    $("#status").textContent = "ONNX unavailable — showing precomputed predictions.";
  }
  await select(sel);
  window.addEventListener("resize", () => { renderWave(); renderCC(); });
}

$("#scrub").addEventListener("input", (e) => { curBand = +e.target.value; renderStrip(); renderWave(); });
$("#play").addEventListener("click", () => {
  if (sweepTimer) { clearInterval(sweepTimer); sweepTimer = null; $("#play").textContent = "▶ sweep"; return; }
  $("#play").textContent = "❚❚ stop"; curBand = 0;
  sweepTimer = setInterval(() => {
    $("#scrub").value = curBand; renderStrip(); renderWave();
    if (++curBand > 5) { curBand = 0; }
  }, 650);
});

boot();
