// Lynceus — Demo A: in-browser ViT morphology classifier (onnxruntime-web).
"use strict";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const $ = (s) => document.querySelector(s);
const statusEl = $("#status");
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

let session = null, labels = null, gallery = null, metrics = null;
const N = 224;
const imgCanvas = $("#img"), imgCtx = imgCanvas.getContext("2d");
const attnCanvas = $("#attn"), attnCtx = attnCanvas.getContext("2d");

async function boot() {
  try {
    [labels, gallery, metrics] = await Promise.all([
      fetch("labels.json").then((r) => r.json()),
      fetch("gallery.json?v=2").then((r) => r.json()),
      fetch("metrics.json").then((r) => r.json()).catch(() => null),
    ]);
    renderGallery();
    renderMetrics();
    statusEl.textContent = "loading model (~20 MB)…";
    session = await ort.InferenceSession.create("model.onnx", {
      executionProviders: ["wasm"],
    });
    statusEl.textContent = "model ready — pick a galaxy.";
    statusEl.style.color = css("--cyan");
    if (gallery.length) selectGallery(gallery[0]);
  } catch (e) {
    statusEl.textContent = "failed to load: " + e.message;
    console.error(e);
  }
}

// ---- preprocessing (mirrors core/datasets.preprocess) ---------------------
function toTensor(srcCanvasOrImg) {
  imgCtx.imageSmoothingEnabled = true;
  imgCtx.clearRect(0, 0, N, N);
  imgCtx.drawImage(srcCanvasOrImg, 0, 0, N, N);
  const { data } = imgCtx.getImageData(0, 0, N, N);
  const mean = labels.mean, std = labels.std;
  const out = new Float32Array(3 * N * N);
  const plane = N * N;
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      const v = data[i * 4 + c] / 255;
      out[c * plane + i] = (v - mean[c]) / std[c];
    }
  }
  return new ort.Tensor("float32", out, [1, 3, N, N]);
}

function softmax(arr) {
  const m = Math.max(...arr);
  const ex = arr.map((v) => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / s);
}

async function classify(srcImg, meta) {
  if (!session) return;
  imgCtx.imageSmoothingEnabled = true;
  imgCtx.drawImage(srcImg, 0, 0, N, N);
  const t0 = performance.now();
  const out = await session.run({ input: toTensor(srcImg) });
  const logits = Array.from(out.logits.data);
  const probs = softmax(logits);
  renderPrediction(probs, meta, performance.now() - t0);
}

// ---- rendering ------------------------------------------------------------
function renderPrediction(probs, meta, ms) {
  const classes = labels.classes;
  const order = probs.map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]);
  const topIdx = order[0][0];
  const topName = labels.names[classes[topIdx]];

  const v = $("#verdict");
  let truth = "";
  if (meta && meta.label) {
    const ok = classes[topIdx] === meta.label;
    truth = `<span class="truth"> &middot; true: ${labels.names[meta.label]} ` +
            `${ok ? "✓" : "✗"} &middot; ${ms.toFixed(0)} ms</span>`;
  } else {
    truth = `<span class="truth"> &middot; ${ms.toFixed(0)} ms</span>`;
  }
  v.innerHTML = `Most likely: <b>${topName}</b> (${(order[0][1] * 100).toFixed(1)}%)${truth}<br>` +
                `<span class="truth">${labels.blurbs[classes[topIdx]]}</span>`;

  const bars = $("#bars");
  bars.innerHTML = "";
  for (const [i, p] of order) {
    const cl = classes[i];
    const row = document.createElement("div");
    row.className = "bar" + (i === topIdx ? " top" : "");
    row.innerHTML =
      `<span class="nm">${labels.names[cl]}</span>` +
      `<span class="track"><span class="fill" style="background:${css("--c-" + cl)}"></span></span>` +
      `<span class="pc">${(p * 100).toFixed(1)}</span>`;
    bars.appendChild(row);
    requestAnimationFrame(() => { row.querySelector(".fill").style.width = (p * 100) + "%"; });
  }
}

function renderGallery() {
  const groups = {};
  for (const g of gallery) (groups[g.label] ||= []).push(g);
  const host = $("#gallery");
  host.innerHTML = "";
  for (const cl of labels.classes) {
    if (!groups[cl]) continue;
    const wrap = document.createElement("div");
    wrap.className = "galgroup";
    wrap.innerHTML = `<h3 style="color:${css("--c-" + cl)}">${labels.names[cl]}</h3>`;
    const grid = document.createElement("div");
    grid.className = "gal";
    for (const g of groups[cl]) {
      const im = document.createElement("img");
      im.src = g.img; im.alt = cl; im.dataset.id = g.id;
      im.onclick = () => selectGallery(g);
      grid.appendChild(im);
    }
    wrap.appendChild(grid);
    host.appendChild(wrap);
  }
}

function selectGallery(g) {
  document.querySelectorAll(".gal img").forEach((e) =>
    e.classList.toggle("sel", e.dataset.id === g.id));
  $("#vm-id").textContent = g.id;
  $("#vm-z").textContent = g.redshift;
  // attention overlay for this gallery item
  const attnImg = new Image();
  attnImg.onload = () => { attnCtx.clearRect(0, 0, N, N); attnCtx.drawImage(attnImg, 0, 0, N, N); };
  attnImg.src = g.attn;
  const im = new Image();
  im.onload = () => classify(im, g);
  im.src = g.img;
}

function renderMetrics() {
  if (!metrics) return;
  const m = $("#metrics");
  const cell = (k, v, color) =>
    `<div class="stat"><span class="k">${k}</span><span class="v" style="color:${color||'var(--ink)'}">${v}</span></div>`;
  m.innerHTML =
    cell("accuracy", (metrics.accuracy * 100).toFixed(1) + "%", css("--cyan")) +
    cell("baseline", (metrics.majority_baseline * 100).toFixed(1) + "%", css("--faint")) +
    cell("macro F1", metrics.macro_f1.toFixed(3)) +
    cell("n", metrics.n);

  const C = metrics.classes, cm = metrics.confusion;
  let html = '<table class="cm"><tr><th></th>' +
    C.map((c) => `<th>${c.slice(0, 4)}</th>`).join("") + "</tr>";
  for (let i = 0; i < C.length; i++) {
    html += `<tr><th>${C[i].slice(0, 4)}</th>` +
      cm[i].map((v, j) => `<td class="${i === j ? "d" : ""}">${v}</td>`).join("") + "</tr>";
  }
  html += "</table>";
  $("#cm").innerHTML = html;
  $("#metricnote").innerHTML =
    `${metrics.folds}-fold stratified cross-validation, ${metrics.epochs} epochs/fold. ` +
    `Rows = true class, columns = prediction; the diagonal is correct.`;
}

// ---- attention toggle + user uploads --------------------------------------
$("#attnToggle").onchange = (e) =>
  $("#stage").classList.toggle("show-attn", e.target.checked);

const drop = $("#drop"), file = $("#file");
drop.onclick = () => file.click();
file.onchange = (e) => { if (e.target.files[0]) loadUserImage(e.target.files[0]); };
["dragover", "dragenter"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0]; if (f) loadUserImage(f);
});

function loadUserImage(f) {
  const url = URL.createObjectURL(f);
  const im = new Image();
  im.onload = () => {
    attnCtx.clearRect(0, 0, N, N);
    $("#attnToggle").checked = false;
    $("#stage").classList.remove("show-attn");
    $("#vm-id").textContent = f.name;
    $("#vm-z").textContent = "—";
    document.querySelectorAll(".gal img").forEach((e) => e.classList.remove("sel"));
    classify(im, null);
    URL.revokeObjectURL(url);
  };
  im.src = url;
}

boot();
