"use strict";

/* =========================================================
   Classic Perceptron -> MLP (auto-detect), drop-in app.js
   - Prefers models/mlp_p1.json (784→128→10)
   - Falls back to models/perceptron.json
   - Defensive against NaN, missing elements, stale caches
   ========================================================= */

document.addEventListener("DOMContentLoaded", () => {
  /* -------- Version tag for cache-busting (bump when you deploy) -------- */
  const APP_VERSION = "v7";

  /* -------------------- DOM helpers & element refs -------------------- */
  const $ = (id) => document.getElementById(id) || null;

  const CANVAS = $("pad");
  const CTX = CANVAS ? CANVAS.getContext("2d") : null;

  const thumbEl = $("thumb");
  const THUMB = thumbEl ? thumbEl.getContext("2d") : null;

  const STATUS = $("status");
  const PRED = $("pred");
  const TOPK = $("topk");
  const SCORES = $("scores");

  const BTN_PRED = $("predict");
  const BTN_CLEAR = $("clear");
  const SL_BRUSH = $("brush");
  const CB_INVERT = $("invert");
  const CB_CENTER = $("useCenter");
  const BTN_DL28 = $("download28");

  // Ensure Predict never submits a form
  if (BTN_PRED && !BTN_PRED.hasAttribute("type")) BTN_PRED.setAttribute("type", "button");

  console.log("[BOOT]", { href: location.href, APP_VERSION, hasPredictBtn: !!BTN_PRED });

  /* ---------------------------- State ---------------------------- */
  // Perceptron (legacy)
  let W_nb = null, b = null, mu = null;

  // MLP (phase-1: 784→128→10)
  let W1 = null, b1 = null, W2 = null, b2 = null, MU = null;
  let MLP_READY = false;

  // Drawing
  const DOWNSAMPLE = 10; // 280 -> 28
  let brushR = SL_BRUSH ? parseInt(SL_BRUSH.value, 10) : 12;
  let drawing = false;

  /* -------------------------- Utilities -------------------------- */
  const toFinite = (v) => (Number.isFinite(v) ? v : 0);

  function f32(arrLike) {
    const out = new Float32Array(arrLike.length);
    for (let i = 0; i < arrLike.length; i++) out[i] = toFinite(Number(arrLike[i]));
    return out;
  }

  function reshape2DFlatToRows(flatLike, rows, cols) {
    const flat = f32(flatLike);
    const need = rows * cols;
    if (flat.length !== need) {
      console.warn(`[reshape] length ${flat.length} != ${rows}*${cols}=${need}; padding/truncating`);
    }
    const M = new Array(rows);
    for (let r = 0; r < rows; r++) {
      const row = new Float32Array(cols);
      const base = r * cols;
      for (let c = 0; c < cols; c++) {
        const idx = base + c;
        row[c] = idx < flat.length ? flat[idx] : 0;
      }
      M[r] = row;
    }
    return M;
  }

  function matvec(W, x, bvec) {
    const out = new Float32Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let s = bvec ? bvec[i] : 0;
      const row = W[i];
      for (let k = 0; k < row.length; k++) s += row[k] * x[k];
      out[i] = toFinite(s);
    }
    return out;
  }

  function relu(v) { for (let i = 0; i < v.length; i++) v[i] = v[i] > 0 ? v[i] : 0; return v; }

  function argTopK(arr, k) {
    return arr.map((v, i) => [toFinite(v), i]).sort((A, B) => B[0] - A[0]).slice(0, k).map(p => p[1]);
  }

  /* ------------------------- Canvas logic ------------------------- */
  function setBrush(r) { brushR = r; }

  function drawCircle(x, y, color) {
    if (!CTX) return;
    CTX.fillStyle = color;
    CTX.beginPath();
    CTX.arc(x, y, brushR, 0, Math.PI * 2);
    CTX.fill();
  }

  function clearCanvas() {
    if (!CTX || !CANVAS) return;
    CTX.fillStyle = "#ffffff"; // white background
    CTX.fillRect(0, 0, CANVAS.width, CANVAS.height);
    if (PRED) PRED.textContent = "Prediction: —";
    if (TOPK) TOPK.textContent = "";
    if (SCORES) SCORES.textContent = "";
  }

  function pos(evt) {
    if (!CANVAS) return { x: 0, y: 0 };
    const rect = CANVAS.getBoundingClientRect();
    const cx = evt.touches ? evt.touches[0].clientX : evt.clientX;
    const cy = evt.touches ? evt.touches[0].clientY : evt.clientY;
    const x = Math.max(0, Math.min(rect.width, cx - rect.left)) * CANVAS.width / rect.width;
    const y = Math.max(0, Math.min(rect.height, cy - rect.top)) * CANVAS.height / rect.height;
    return { x, y };
  }

  // Downsample 280 -> 28, foreground≈1 (black stroke on white canvas becomes 1)
  function to28() {
    if (!CTX) return new Float32Array(28 * 28);
    const img = CTX.getImageData(0, 0, 280, 280).data; // RGBA
    const out = new Float32Array(28 * 28);

    for (let by = 0; by < 28; by++) {
      for (let bx = 0; bx < 28; bx++) {
        let sum = 0;
        for (let y = 0; y < DOWNSAMPLE; y++) {
          for (let x = 0; x < DOWNSAMPLE; x++) {
            const px = ((by * DOWNSAMPLE + y) * 280 + (bx * DOWNSAMPLE + x)) * 4;
            sum += img[px]; // red channel
          }
        }
        const mean = sum / (DOWNSAMPLE * DOWNSAMPLE * 255);
        out[by * 28 + bx] = 1 - mean;
      }
    }

    if (THUMB) {
      THUMB.imageSmoothingEnabled = false;
      const tmp = new ImageData(28, 28);
      for (let i = 0; i < 28 * 28; i++) {
        const v = Math.max(0, Math.min(1, out[i]));
        const g = Math.round((1 - v) * 255); // high contrast preview
        tmp.data[i * 4 + 0] = g; tmp.data[i * 4 + 1] = g; tmp.data[i * 4 + 2] = g; tmp.data[i * 4 + 3] = 255;
      }
      THUMB.putImageData(tmp, 0, 0);
      THUMB.drawImage(THUMB.canvas, 0, 0, 28, 28, 0, 0, 140, 140);
    }

    return out;
  }

  /* ---------------------------- Loaders ---------------------------- */
  async function fetchJSON(pathWithQuery) {
    const url = new URL(pathWithQuery, location.href).toString();
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) throw new Error(`${pathWithQuery} HTTP ${resp.status}`);
    return resp.json();
  }

  async function loadPerceptron() {
    try {
      const w = await fetchJSON(`models/perceptron.json?${APP_VERSION}`);

      const nC = w.meta?.n_classes ?? 10;
      const nF = w.meta?.n_features ?? 784;

      const Wflat = f32(w.W_nb);
      W_nb = Array.from({ length: nC }, (_, c) => Wflat.slice(c * nF, (c + 1) * nF));
      b = f32(w.b);

      try {
        const muj = await fetchJSON(`models/mu.json?${APP_VERSION}`);
        mu = f32(muj.mu);
      } catch {
        mu = null;
      }

      STATUS && (STATUS.textContent = `Perceptron loaded: ${nF}→${nC}. Centering: ${mu ? "available" : "none"}.`);
    } catch (err) {
      console.error("[loadPerceptron] failed:", err);
      STATUS && (STATUS.textContent = `Model load failed: ${String(err)}`);
    } finally {
      BTN_PRED && (BTN_PRED.disabled = false);
    }
  }

  async function loadMLP() {
    try {
      const j = await fetchJSON(`models/mlp_p1.json?${APP_VERSION}`);

      const D = j.meta?.n_features ?? 784;
      const H1 = j.b1.length;
      const C = j.b2.length;

      W1 = reshape2DFlatToRows(j.W1, H1, D);
      b1 = f32(j.b1);
      W2 = reshape2DFlatToRows(j.W2, C, H1);
      b2 = f32(j.b2);
      MU = j.mu ? f32(j.mu) : null;

      // quick shape sanity
      if (W1.length !== H1 || W1[0].length !== D || W2.length !== C || W2[0].length !== H1) {
        console.warn("[loadMLP] unexpected shapes; padded/truncated rows");
      }

      MLP_READY = true;
      STATUS && (STATUS.textContent = `MLP loaded: ${D}→${H1}→${C}. Centering: ${MU ? "available" : "none"}.`);
      BTN_PRED && (BTN_PRED.disabled = false);
    } catch (err) {
      console.warn("[loadMLP] falling back to perceptron:", err);
      await loadPerceptron();
    }
  }

  /* --------------------------- Inference --------------------------- */
  function forwardMLP(x784) {
    let x = Float32Array.from(x784);
    if (CB_CENTER && CB_CENTER.checked && MU && MU.length === x.length) {
      for (let i = 0; i < x.length; i++) x[i] = toFinite(x[i] - MU[i]);
    }
    const h1 = relu(matvec(W1, x, b1));   // 784→128
    const logits = matvec(W2, h1, b2);    // 128→10
    return logits;
  }

  function predict() {
    const x28 = to28();
    let scores;

    if (MLP_READY) {
      scores = forwardMLP(x28);
    } else {
      if (!W_nb || !b) { console.warn("[predict] Model not loaded yet."); return; }
      let x = Float32Array.from(x28);
      if (CB_CENTER && CB_CENTER.checked && mu && mu.length === x.length) {
        for (let i = 0; i < x.length; i++) x[i] = toFinite(x[i] - mu[i]);
      }
      scores = new Array(10).fill(0);
      for (let c = 0; c < 10; c++) {
        let s = b[c];
        const wrow = W_nb[c];
        for (let i = 0; i < wrow.length; i++) s += wrow[i] * x[i];
        scores[c] = toFinite(s);
      }
    }

    const order = argTopK(scores, 3);
    const pred = order[0];

    // softmax over top-k for display
    const sTop = order.map(i => scores[i]);
    const s0 = Math.max(...sTop.map(toFinite));
    let Z = 0; const exps = sTop.map(v => { const e = Math.exp(toFinite(v) - s0); Z += e; return e; });
    const probs = exps.map(e => (Z > 0 && Number.isFinite(Z)) ? e / Z : 0);

    if (PRED) PRED.textContent = `Prediction: ${Number.isFinite(scores[pred]) ? pred : "—"}`;
    if (TOPK) TOPK.textContent = order.map((k, i) => `${k} (~${probs[i].toFixed(2)})`).join(", ");
    if (SCORES) SCORES.textContent = scores.map((v, i) => `${i}: ${Number.isFinite(v) ? v.toFixed(3) : "NaN"}`).join("  ");
  }

  /* ---------------------------- Events ---------------------------- */
  if (SL_BRUSH) SL_BRUSH.addEventListener("input", e => setBrush(parseInt(e.target.value, 10)));

  if (CANVAS) {
    CANVAS.addEventListener("mousedown", e => { drawing = true; drawCircle(pos(e).x, pos(e).y, "#000"); });
    CANVAS.addEventListener("mousemove", e => { if (drawing) drawCircle(pos(e).x, pos(e).y, "#000"); });
    CANVAS.addEventListener("touchstart", e => { drawing = true; drawCircle(pos(e).x, pos(e).y, "#000"); e.preventDefault(); }, { passive: false });
    CANVAS.addEventListener("touchmove", e => { if (drawing) drawCircle(pos(e).x, pos(e).y, "#000"); e.preventDefault(); }, { passive: false });
  }
  window.addEventListener("mouseup", () => drawing = false);
  window.addEventListener("touchend", () => drawing = false);

  if (BTN_CLEAR) BTN_CLEAR.addEventListener("click", clearCanvas);

  if (BTN_DL28) BTN_DL28.addEventListener("click", () => {
    const x = Array.from(to28());
    const rows = [];
    for (let r = 0; r < 28; r++) rows.push(x.slice(r * 28, (r + 1) * 28).map(v => v.toFixed(6)).join(","));
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "draw_28x28.csv";
    a.click();
  });

  if (BTN_PRED) BTN_PRED.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); predict(); });
  window.addEventListener("keydown", (e) => { if (e.code === "Space") predict(); if (e.key === "c") clearCanvas(); });

  /* ----------------------------- Start ----------------------------- */
  clearCanvas();
  loadMLP(); // prefer MLP; loader falls back to perceptron if missing
});
