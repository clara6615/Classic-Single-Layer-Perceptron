"use strict";

/* =========================
   Robust, single-pass init
   ========================= */

document.addEventListener("DOMContentLoaded", () => {
  // --- Safe getters
  const $ = (id) => document.getElementById(id) || null;

  // --- Elements
  const CANVAS   = $("pad");
  const CTX      = CANVAS ? CANVAS.getContext("2d") : null;

  const thumbEl  = $("thumb");
  const THUMB    = thumbEl ? thumbEl.getContext("2d") : null;

  const STATUS   = $("status");
  const PRED     = $("pred");
  const TOPK     = $("topk");
  const SCORES   = $("scores");

  const BTN_PRED = $("predict");
  const BTN_CLEAR= $("clear");
  const SL_BRUSH = $("brush");
  const CB_INVERT= $("invert");
  const CB_CENTER= $("useCenter");
  const BTN_DL28 = $("download28");

  // --- Model state (perceptron)
  let W_nb = null, b = null, mu = null;

  // --- MLP state (phase-1: 784→128→10)
  let W1=null, b1=null, W2=null, b2=null, MU=null;
  let MLP_READY = false;

  // --- Brush & drawing state
  const DOWNSAMPLE = 10; // 280 -> 28
  let brushR = SL_BRUSH ? parseInt(SL_BRUSH.value, 10) : 12;
  let drawing = false;

  function setBrush(r){ brushR = r; }

  // ----------------------
  // Canvas utilities
  // ----------------------
  function drawCircle(x, y, color){
    if (!CTX) return;
    CTX.fillStyle = color;
    CTX.beginPath();
    CTX.arc(x, y, brushR, 0, Math.PI*2);
    CTX.fill();
  }

  function clearCanvas(){
    if (!CTX || !CANVAS) return;
    CTX.fillStyle = "#ffffff"; // white background; adjust if you want black
    CTX.fillRect(0,0,CANVAS.width,CANVAS.height);
    if (PRED)  PRED.textContent  = "Prediction: —";
    if (TOPK)  TOPK.textContent  = "";
    if (SCORES)SCORES.textContent= "";
  }

  function pos(evt){
    if (!CANVAS) return {x:0,y:0};
    const rect = CANVAS.getBoundingClientRect();
    const cx = evt.touches ? evt.touches[0].clientX : evt.clientX;
    const cy = evt.touches ? evt.touches[0].clientY : evt.clientY;
    const x = Math.max(0, Math.min(rect.width,  cx - rect.left)) * CANVAS.width  / rect.width;
    const y = Math.max(0, Math.min(rect.height, cy - rect.top )) * CANVAS.height / rect.height;
    return {x,y};
  }

  // ----------------------
  // Downsample 280 -> 28
  // ----------------------
  function to28(){
    if (!CTX) return new Float32Array(28*28);
    const img = CTX.getImageData(0,0,280,280).data; // RGBA
    const out = new Float32Array(28*28);

    for(let by=0; by<28; by++){
      for(let bx=0; bx<28; bx++){
        let sum = 0;
        for(let y=0; y<DOWNSAMPLE; y++){
          for(let x=0; x<DOWNSAMPLE; x++){
            const px = ((by*DOWNSAMPLE + y)*280 + (bx*DOWNSAMPLE + x)) * 4;
            sum += img[px]; // red channel
          }
        }
        const mean = sum / (DOWNSAMPLE*DOWNSAMPLE*255);
        // original code inverted regardless; keep behavior but respect checkbox if present
        const inv = CB_INVERT && CB_INVERT.checked;
        out[by*28 + bx] = inv ? (1 - mean) : (1 - mean); // retained existing logic
      }
    }

    // Thumbnail (guard if missing)
    if (THUMB) {
      THUMB.imageSmoothingEnabled = false;
      const tmp = new ImageData(28,28);
      for(let i=0;i<28*28;i++){
        const v = Math.max(0, Math.min(1, out[i]));
        const g = Math.round((1 - v) * 255); // preview polarity aligned with current pipeline
        tmp.data[i*4+0]=g; tmp.data[i*4+1]=g; tmp.data[i*4+2]=g; tmp.data[i*4+3]=255;
      }
      THUMB.putImageData(tmp,0,0);
      THUMB.drawImage(THUMB.canvas,0,0,28,28,0,0,140,140);
    }

    return out;
  }

  // ----------------------
  // Helpers for MLP
  // ----------------------
  function reshape2D(flat, rows, cols){
    const M = new Array(rows);
    for (let r=0; r<rows; r++){
      M[r] = new Float32Array(flat.slice(r*cols, (r+1)*cols));
    }
    return M;
  }
  function matvec(W, x, bvec){
    const out = new Float32Array(W.length);
    for (let i=0; i<W.length; i++){
      let s = bvec ? bvec[i] : 0;
      const row = W[i];
      for (let k=0; k<row.length; k++) s += row[k]*x[k];
      out[i] = s;
    }
    return out;
  }
  function relu(v){ for (let i=0;i<v.length;i++) v[i] = v[i] > 0 ? v[i] : 0; return v; }

  // ----------------------
  // Model loading (perceptron fallback)
  // ----------------------
  async function loadPerceptron(){
    try {
      const wResp = await fetch(new URL("models/perceptron.json", location.href), { cache: "no-store" });
      if (!wResp.ok) throw new Error(`perceptron.json HTTP ${wResp.status}`);
      const w = await wResp.json();

      const nC = w.meta?.n_classes ?? 10;
      const nF = w.meta?.n_features ?? 784;

      const Wflat = Float32Array.from(w.W_nb);
      W_nb = Array.from({length: nC}, (_, c) => Wflat.slice(c*nF, (c+1)*nF));
      b = Float32Array.from(w.b);

      // optional μ
      try {
        const muResp = await fetch(new URL("models/mu.json", location.href), { cache: "no-store" });
        if (muResp.ok) mu = Float32Array.from((await muResp.json()).mu);
      } catch (e) {
        console.warn("mu.json not available (optional)", e);
      }

      if (STATUS) STATUS.textContent = `Perceptron loaded: ${nF}→${nC}. Centering: ${mu ? "available" : "none"}.`;
    } catch (err) {
      console.error("[loadPerceptron] failed:", err);
      if (STATUS) STATUS.textContent = `Model load failed: ${String(err)}`;
    } finally {
      if (BTN_PRED) BTN_PRED.disabled = false;
    }
  }

  // ----------------------
  // Model loading (MLP preferred)
  // ----------------------
  async function loadMLP(){
    try {
      const resp = await fetch(new URL("models/mlp_p1.json", location.href), { cache:"no-store" });
      if (!resp.ok) throw new Error(`mlp_p1.json HTTP ${resp.status}`);
      const j = await resp.json();

      const D  = j.meta?.n_features ?? 784;
      const H1 = j.b1.length;
      const C  = j.b2.length;

      W1 = reshape2D(j.W1, H1, D);
      b1 = new Float32Array(j.b1);
      W2 = reshape2D(j.W2, C,  H1);
      b2 = new Float32Array(j.b2);
      MU = j.mu ? new Float32Array(j.mu) : null;

      MLP_READY = true;
      if (STATUS) STATUS.textContent = `MLP loaded: ${D}→${H1}→${C}. Centering: ${MU ? "available" : "none"}.`;
      if (BTN_PRED) BTN_PRED.disabled = false;
    } catch (err) {
      console.warn("[loadMLP] falling back to perceptron:", err);
      await loadPerceptron();
    }
  }

  // ----------------------
  // Inference
  // ----------------------
  function dot(a, b){ let s=0; for(let i=0;i<a.length;i++) s += a[i]*b[i]; return s; }
  function argTopK(arr, k){
    return arr.map((v,i)=>[v,i]).sort((A,B)=>B[0]-A[0]).slice(0,k).map(p=>p[1]);
  }

  function forwardMLP(x784){
    let x = x784;
    if (CB_CENTER && CB_CENTER.checked && MU && MU.length === x.length){
      const y = new Float32Array(x.length);
      for (let i=0;i<x.length;i++) y[i] = x[i] - MU[i];
      x = y;
    }
    const h1 = relu(matvec(W1, x, b1));     // 784→128
    const logits = matvec(W2, h1, b2);      // 128→10
    return logits;
  }

  function predict(){
    const x28 = to28();
    const x = Float32Array.from(x28); // 784

    let scores;
    if (MLP_READY) {
      scores = forwardMLP(x);
    } else {
      if (!W_nb || !b) { console.warn("[predict] Model not loaded yet."); return; }
      scores = new Array(10).fill(0);
      if (CB_CENTER && CB_CENTER.checked && mu && mu.length === x.length){
        for(let i=0;i<x.length;i++) x[i] = x[i] - mu[i];
      }
      for(let c=0;c<10;c++) scores[c] = dot(W_nb[c], x) + b[c];
    }

    const order = argTopK(scores, 3);
    const pred = order[0];
    const sTop = order.map(i=>scores[i]);
    const probs = sTop.map(v=>Math.exp(v - sTop[0]));
    const Z = probs.reduce((a,b)=>a+b,0);
    for(let i=0;i<probs.length;i++) probs[i]/=Z;

    if (PRED)   PRED.textContent = `Prediction: ${pred}`;
    if (TOPK)   TOPK.textContent = order.map((k,i)=>`${k} (~${probs[i].toFixed(2)})`).join(", ");
    if (SCORES) SCORES.textContent = scores.map((v,i)=>`${i}: ${v.toFixed(3)}`).join("  ");
  }

  // ----------------------
  // Event wiring (single source of truth)
  // ----------------------
  if (SL_BRUSH) SL_BRUSH.addEventListener("input", e => setBrush(parseInt(e.target.value,10)));

  if (CANVAS) {
    CANVAS.addEventListener("mousedown", e => { drawing = true; drawCircle(pos(e).x, pos(e).y, "#000"); });
    CANVAS.addEventListener("mousemove", e => { if(drawing) drawCircle(pos(e).x, pos(e).y, "#000"); });
    CANVAS.addEventListener("touchstart", e => { drawing=true; drawCircle(pos(e).x,pos(e).y,"#000"); e.preventDefault(); }, {passive:false});
    CANVAS.addEventListener("touchmove",  e => { if(drawing) drawCircle(pos(e).x,pos(e).y,"#000"); e.preventDefault(); }, {passive:false});
  }
  window.addEventListener("mouseup",   () => drawing=false);
  window.addEventListener("touchend",  () => drawing=false);

  if (BTN_CLEAR) BTN_CLEAR.addEventListener("click", clearCanvas);

  if (BTN_DL28) BTN_DL28.addEventListener("click", () => {
    const x = Array.from(to28());
    const rows = [];
    for(let r=0;r<28;r++) rows.push(x.slice(r*28,(r+1)*28).map(v=>v.toFixed(6)).join(","));
    const blob = new Blob([rows.join("\n")], {type:"text/csv"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "draw_28x28.csv";
    a.click();
  });

  if (BTN_PRED) {
    // Ensure it's a non-submit button in case it sits inside a <form>
    if (!BTN_PRED.hasAttribute("type")) BTN_PRED.setAttribute("type","button");
    BTN_PRED.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      predict();
    });
  }

  window.addEventListener("keydown", (e)=>{ if(e.code==="Space") predict(); if(e.key==="c") clearCanvas(); });

  // ----------------------
  // Kick off
  // ----------------------
  clearCanvas();
  // Prefer MLP; if not present, loader falls back to perceptron automatically.
  loadMLP();
});
