"""
Interactive digit sketchpad for MNIST perceptron.

Controls:
- Left click + drag: draw (brush)
- Right click + drag: erase
- Buttons: Predict, Clear, Save28 (saves the current 28x28 to reports/draw_28x28.png)
- Keyboard: 'p' predict, 'c' clear, 's' save

It auto-detects centering: if models/feature_center_mu.npy exists, it subtracts that mean.

Dependencies: stdlib tkinter, numpy, matplotlib (for optional save preview).
"""

import os
import time
import numpy as np
import tkinter as tk
from tkinter import ttk

# Try to import matplotlib only for saving a preview image
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

np.random.seed(42)

CANVAS_PIXELS = 280  # we draw at 280x280 then downsample by 10x to 28x28
DOWNSAMPLE = CANVAS_PIXELS // 28
BRUSH_RADIUS = 7     # in canvas pixels

MODELS_DIR = "models"
WEIGHTS_PATH = os.path.join(MODELS_DIR, "perceptron_scratch_weights.npz")
MU_PATH = os.path.join(MODELS_DIR, "feature_center_mu.npy")

# ---- Load model weights ----
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Missing {WEIGHTS_PATH}. Run: python src/train_scratch.py")

W = np.load(WEIGHTS_PATH)["W"].astype(np.float32)  # (10, 785)
W_nobias = W[:, :-1]  # (10, 784)
b = W[:, -1]          # (10,)

USE_CENTER = os.path.exists(MU_PATH)
mu = np.load(MU_PATH).astype(np.float32) if USE_CENTER else None  # (784,) or None

# ---- App state ----
buf = np.zeros((CANVAS_PIXELS, CANVAS_PIXELS), dtype=np.float32)  # drawing buffer in [0,1]

def draw_disk(x, y, val=1.0):
    """Draw a filled disk at (x,y) with radius BRUSH_RADIUS into buf, clipping to [0,1]."""
    r = BRUSH_RADIUS
    x0, x1 = max(0, x - r), min(CANVAS_PIXELS, x + r + 1)
    y0, y1 = max(0, y - r), min(CANVAS_PIXELS, y + r + 1)
    ys, xs = np.ogrid[y0:y1, x0:x1]
    mask = (xs - x)**2 + (ys - y)**2 <= r*r
    if val >= 1.0:  # draw
        buf[y0:y1, x0:x1][mask] = np.clip(buf[y0:y1, x0:x1][mask] + 1.0, 0.0, 1.0)
    else:           # erase
        buf[y0:y1, x0:x1][mask] = 0.0

def to_28x28():
    """Downsample 280x280 -> 28x28 by block mean, return float32 in [0,1]."""
    X = buf
    # exact tiling: reshape and mean over block axes
    x28 = X.reshape(28, DOWNSAMPLE, 28, DOWNSAMPLE).mean(axis=(1,3)).astype(np.float32)
    return x28

def prepare_features(x28):
    """Flatten -> (784,), normalize already in [0,1]; apply centering if available."""
    x = x28.reshape(-1).astype(np.float32)
    if USE_CENTER and mu is not None:
        x = (x - mu).astype(np.float32, copy=False)
    return x

def predict_top3(x):
    """Compute class scores and return (pred, top3_idx, top3_scores, margin)."""
    # perceptron scores = W_nobias @ x + b
    scores = (W_nobias @ x) + b  # (10,)
    pred = int(np.argmax(scores))
    # Top-3
    top3_idx = np.argsort(scores)[-3:][::-1]
    top3_scores = scores[top3_idx]
    # Margin between best and second best
    margin = float(top3_scores[0] - top3_scores[1]) if len(top3_scores) > 1 else float(top3_scores[0])
    return pred, top3_idx, top3_scores, margin

# ---- GUI ----
root = tk.Tk()
root.title("MNIST Perceptron Sketchpad")

main = ttk.Frame(root, padding=6)
main.grid(row=0, column=0, sticky="nsew")
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

canvas = tk.Canvas(main, width=CANVAS_PIXELS, height=CANVAS_PIXELS, bg="white", highlightthickness=1, highlightbackground="#888")
canvas.grid(row=0, column=0, rowspan=6, sticky="nsew", padx=(0,8), pady=(0,8))
main.rowconfigure(0, weight=1)
main.columnconfigure(0, weight=1)

pred_var = tk.StringVar(value="Prediction: —")
info_var = tk.StringVar(value=f"Feature: {'centered' if USE_CENTER else 'baseline'} | Brush: {BRUSH_RADIUS}px")

pred_label = ttk.Label(main, textvariable=pred_var, font=("TkDefaultFont", 14))
pred_label.grid(row=0, column=1, sticky="w")

info_label = ttk.Label(main, textvariable=info_var)
info_label.grid(row=1, column=1, sticky="w")

def on_draw(event):
    draw_disk(event.x, event.y, val=1.0)
    # draw a small dot for visual feedback
    r = BRUSH_RADIUS
    canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="black", outline="")
def on_draw_click(event):
    on_draw(event)
def on_erase(event):
    draw_disk(event.x, event.y, val=0.0)
    r = BRUSH_RADIUS
    # draw white dot to "erase" visually
    canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="white", outline="")
def on_erase_click(event): on_erase(event)

canvas.bind("<Button-1>", on_draw_click)
canvas.bind("<B1-Motion>", on_draw)
canvas.bind("<Button-3>", on_erase_click)
canvas.bind("<B3-Motion>", on_erase)

def do_predict(event=None):
    x28 = to_28x28()
    x = prepare_features(x28)
    pred, top_idx, top_scores, margin = predict_top3(x)
    # Softmax-ish display (not calibrated; for UX only)
    probs = np.exp(top_scores - np.max(top_scores))
    probs = probs / probs.sum()
    top_str = ", ".join([f"{int(k)}({p:.2f})" for k, p in zip(top_idx, probs)])
    pred_var.set(f"Prediction: {pred}   |   Top-3: {top_str}   |   margin={margin:.2f}")
    print(pred_var.get())

def do_clear(event=None):
    canvas.delete("all")
    buf[:] = 0.0
    pred_var.set("Prediction: —")

def do_save(event=None):
    os.makedirs("reports", exist_ok=True)
    x28 = to_28x28()
    x = prepare_features(x28)  # for reproducibility, we save the *input* too
    np.save(os.path.join("reports", "draw_28x28.npy"), x28.astype(np.float32))
    if plt is not None:
        plt.figure(figsize=(3,3))
        plt.imshow(x28, cmap="gray")
        plt.axis("off")
        out = os.path.join("reports", "draw_28x28.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        print("Saved", out)
    print("Saved reports/draw_28x28.npy (raw 28x28 float image)")

btn_predict = ttk.Button(main, text="Predict (space)", command=do_predict)
btn_predict.grid(row=2, column=1, sticky="ew", pady=(6,2))

btn_clear = ttk.Button(main, text="Clear (c)", command=do_clear)
btn_clear.grid(row=3, column=1, sticky="ew", pady=2)

btn_save = ttk.Button(main, text="Save28 (s)", command=do_save)
btn_save.grid(row=4, column=1, sticky="ew", pady=2)

# Key bindings
root.bind("<space>", do_predict)
root.bind("p", do_predict)
root.bind("c", do_clear)
root.bind("s", do_save)

root.mainloop()
