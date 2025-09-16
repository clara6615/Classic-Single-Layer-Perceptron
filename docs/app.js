"use strict";

const CANVAS = document.getElementById("pad");
const CTX = CANVAS.getContext("2d");
const THUMB = document.getElementById("thumb").getContext("2d");
const STATUS = document.getElementById("status");
const PRED = document.getElementById("pred");
const TOPK = document.getElementById("topk");
const SCORES = document.getElementById("scores");

const BTN_PRED = document.getElementById("predict");
const BTN_CLEAR = document.getElementById("clear");
const SL_BRUSH = document.getElementById("brush");
const CB_INVERT = document.getElementById("invert");
const CB_CENTER = document.getElementById("useCenter");
const BTN_DL28 = document.getElementById("download28");


document.addEventListener('DOMContentLoaded', () => {
  // --- element lookups (use your existing ids) ---
  const BTN_PRED  = document.getElementById('predict');
  const BTN_CLEAR = document.getElementById('clear');
  const SL_BRUSH  = document.getElementById('brush');
  const BTN_DL28  = document.getElementById('download28');

  // Optional: log if something’s missing
  if (!BTN_PRED)  console.error('Missing #predict');
  if (!BTN_CLEAR) console.error('Missing #clear');
  if (!SL_BRUSH)  console.error('Missing #brush');
  if (!BTN_DL28)  console.error('Missing #download28');

  // --- bind handlers (guarded) ---
  BTN_PRED && BTN_PRED.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    predict();
    try { predict(); }
    catch (err) { console.error('predict() failed:', err); }
  });

  BTN_CLEAR && BTN_CLEAR.addEventListener('click', clearCanvas);
  SL_BRUSH && SL_BRUSH.addEventListener('input', (e) => {
    setBrush(parseInt(e.target.value, 10));
  });
  BTN_DL28 && BTN_DL28.addEventListener('click', () => {
    const x = Array.from(to28());
    const rows = [];
    for (let r = 0; r < 28; r++) {
      rows.push(x.slice(r*28,(r+1)*28).map(v=>v.toFixed(6)).join(','));
    }
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'draw_28x28.csv';
    a.click();
  });


  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') predict();
    if (e.key === 'c') clearCanvas();
  });
});


const DOWNSAMPLE = 10;             // 280 -> 28
let brushR = parseInt(SL_BRUSH.value, 10);
let drawing = false;

function setBrush(r){ brushR = r; }
SL_BRUSH.addEventListener("input", e => setBrush(parseInt(e.target.value,10)));

function drawCircle(x, y, color){
  CTX.fillStyle = color;
  CTX.beginPath();
  CTX.arc(x, y, brushR, 0, Math.PI*2);
  CTX.fill();
}

function clearCanvas(){
  CTX.fillStyle = "#ffffff";
  CTX.fillRect(0,0,CANVAS.width,CANVAS.height);
  PRED.textContent = "Prediction: —";
  TOPK.textContent = "";
  SCORES.textContent = "";
}
clearCanvas();

function pos(evt){
  const rect = CANVAS.getBoundingClientRect();
  const x = (evt.touches ? evt.touches[0].clientX : evt.clientX) - rect.left;
  const y = (evt.touches ? evt.touches[0].clientY : evt.clientY) - rect.top;
  return {x: Math.max(0, Math.min(rect.width, x)) * CANVAS.width/rect.width,
          y: Math.max(0, Math.min(rect.height,y)) * CANVAS.height/rect.height};
}

CANVAS.addEventListener("mousedown", e => { drawing = true; drawCircle(pos(e).x, pos(e).y, "#000"); });
CANVAS.addEventListener("mousemove", e => { if(drawing) drawCircle(pos(e).x, pos(e).y, "#000"); });
window.addEventListener("mouseup", () => drawing=false);
CANVAS.addEventListener("touchstart", e => { drawing=true; drawCircle(pos(e).x,pos(e).y,"#000"); e.preventDefault(); }, {passive:false});
CANVAS.addEventListener("touchmove",  e => { if(drawing) drawCircle(pos(e).x,pos(e).y,"#000"); e.preventDefault(); }, {passive:false});
window.addEventListener("touchend", () => drawing=false);

BTN_CLEAR.addEventListener("click", clearCanvas);

// --- Downsample 280x280 canvas -> 28x28 grayscale in [0,1] by block-mean
function to28(){
  const img = CTX.getImageData(0,0,280,280).data; // RGBA bytes
  const out = new Float32Array(28*28);
  for(let by=0; by<28; by++){
    for(let bx=0; bx<28; bx++){
      let sum = 0;
      for(let y=0; y<DOWNSAMPLE; y++){
        for(let x=0; x<DOWNSAMPLE; x++){
          const px = ( (by*DOWNSAMPLE + y) * 280 + (bx*DOWNSAMPLE + x) ) * 4;
          // use red channel; white=255, black=0
          sum += img[px];
        }
      }
      const mean = sum / (DOWNSAMPLE*DOWNSAMPLE*255);
      out[by*28 + bx] = CB_INVERT.checked ? (1-mean) : (1-mean); // invert because white bg
    }
  }
  // thumbnail
  THUMB.imageSmoothingEnabled = false;
  const tmp = new ImageData(28,28);
  for(let i=0;i<28*28;i++){
    const v = Math.max(0, Math.min(1, out[i]));
    const g = Math.round((1-v)*255);
    tmp.data[i*4+0]=g; tmp.data[i*4+1]=g; tmp.data[i*4+2]=g; tmp.data[i*4+3]=255;
  }
  THUMB.putImageData(tmp,0,0);
  THUMB.drawImage(THUMB.canvas,0,0,28,28,0,0,140,140);
  return out;
}

// --- Model loading
let W_nb = null, b = null, mu = null;
async function loadModel(){
  const w = await fetch("models/perceptron.json").then(r=>r.json());
  const nC = w.meta?.n_classes || 10;
  const nF = w.meta?.n_features || 784;
  const Wflat = Float32Array.from(w.W_nb);
  W_nb = [];
  for(let c=0; c<nC; c++){
    W_nb.push(Wflat.slice(c*nF, (c+1)*nF)); // view per class
  }
  b = Float32Array.from(w.b);

  try {
    const m = await fetch("models/mu.json");
    if(m.ok){
      const j = await m.json();
      mu = Float32Array.from(j.mu);
    }
  } catch(e){ /* optional */ }

  STATUS.textContent = `Loaded ${nC} classes × ${W_nb[0].length} features. Centering: ${mu ? "available" : "none"}.`;
}
loadModel();

// --- Prediction
function dot(a, b){ let s=0; for(let i=0;i<a.length;i++) s += a[i]*b[i]; return s; }
function argTopK(arr, k){
  const idx = arr.map((v,i)=>[v,i]).sort((A,B)=>B[0]-A[0]).slice(0,k).map(p=>p[1]);
  return idx;
}

function predict(){
  if(!W_nb){ return; }
  const x28 = to28();
  // flatten already flat; apply centering if requested and available
  const x = Float32Array.from(x28);
  if (CB_CENTER.checked && mu && mu.length === x.length){
    for(let i=0;i<x.length;i++) x[i] = x[i] - mu[i];
  }
  // scores
  const scores = new Array(10).fill(0);
  for(let c=0;c<10;c++){
    scores[c] = dot(W_nb[c], x) + b[c];
  }
  const order = argTopK(scores, 3);
  const pred = order[0];
  const sTop = order.map(i=>scores[i]);
  const probs = sTop.map(v=>Math.exp(v - sTop[0]));
  const Z = probs.reduce((a,b)=>a+b,0); for(let i=0;i<probs.length;i++) probs[i]/=Z;

  PRED.textContent = `Prediction: ${pred}`;
  TOPK.textContent = order.map((k,i)=>`${k} (~${probs[i].toFixed(2)})`).join(", ");
  SCORES.textContent = scores.map((v,i)=>`${i}: ${v.toFixed(3)}`).join("  ");
}
BTN_PRED.addEventListener("click", predict);
window.addEventListener("keydown", (e)=>{ if(e.code==="Space") predict(); if(e.key==="c") clearCanvas(); });

BTN_DL28.addEventListener("click", ()=>{
  const x = Array.from(to28());
  const rows = [];
  for(let r=0;r<28;r++){ rows.push(x.slice(r*28,(r+1)*28).map(v=>v.toFixed(6)).join(",")); }
  const blob = new Blob([rows.join("\n")], {type:"text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "draw_28x28.csv";
  a.click();
});
