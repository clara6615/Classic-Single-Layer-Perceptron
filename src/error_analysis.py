import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data_io import load_mnist_dataset
from utils import to_float, flatten_images, stratified_train_val_split
from perceptron import OneVsRestPerceptron

np.random.seed(42)
os.makedirs("reports", exist_ok=True)

# --- Config ---
VAL_RATIO = 0.1
SUBSET_TRAIN = None
SUBSET_TEST  = None
WEIGHTS_PATH = "models/perceptron_scratch_weights.npz"  # from Step 4

# --- Load & preprocess (same as before) ---
Xtr_u8, ytr, Xte_u8, yte = load_mnist_dataset("data", subset=SUBSET_TRAIN, as_float=False, normalize=False)
Xte_u8 = Xte_u8[:SUBSET_TEST] if SUBSET_TEST else Xte_u8
yte    = yte[:SUBSET_TEST] if SUBSET_TEST else yte
Xtr = flatten_images(to_float(Xtr_u8, normalize=True))
Xte = flatten_images(to_float(Xte_u8, normalize=True))
X_tr, y_tr, X_val, y_val = stratified_train_val_split(Xtr, ytr, val_ratio=VAL_RATIO, random_state=42)

# --- Load scratch model weights ---
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Missing {WEIGHTS_PATH}. Run src/train_scratch.py first.")

W = np.load(WEIGHTS_PATH)["W"]
model = OneVsRestPerceptron(n_classes=10, max_epochs=1, shuffle=False, random_state=42)
model.set_weights(W)

def evaluate_split(X, y, split_name):
    y_pred = model.predict(X)
    acc = float((y_pred == y).mean())
    cm = confusion_matrix(y, y_pred, labels=list(range(10)))
    # Per-class accuracy = diagonal / row sum  (both 1-D)
    diag = np.diag(cm).astype(np.float64)           # (10,)
    row = cm.sum(axis=1).astype(np.float64)         # (10,)
    per_class = np.divide(diag, row,
                          out=np.zeros_like(diag, dtype=np.float64),
                          where=row > 0)            # (10,)
    row_sums = cm.sum(axis=1)

    # Save CSVs
    csv_acc = os.path.join("reports", f"per_class_acc_{split_name}.csv")
    with open(csv_acc, "w") as f:
        f.write("class,accuracy,count\n")
        for c in range(10):
            f.write(f"{c},{per_class[c]:.4f},{int(row_sums[c])}\n")

    # Confusion image
    fig = plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix ({split_name})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    out_png = os.path.join("reports", f"confusion_{split_name}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"{split_name} acc: {acc:.4f} | wrote {csv_acc} and {out_png}")
    return y_pred, cm, per_class

# --- Evaluate on val & test
y_val_pred, cm_val, per_class_val = evaluate_split(X_val, y_val, "val")
y_te_pred,  cm_te,  per_class_te  = evaluate_split(Xte,  yte,  "test")

# --- Identify top confusions (val): off-diagonal counts
def top_confusions(cm, k=5):
    pairs = []
    for t in range(10):
        for p in range(10):
            if t == p: 
                continue
            cnt = int(cm[t, p])
            if cnt > 0:
                pairs.append(((t, p), cnt))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]

top5 = top_confusions(cm_val, k=5)
txt_path = "reports/top5_confusions_val.txt"
with open(txt_path, "w") as f:
    for (t, p), cnt in top5:
        f.write(f"{t}→{p}: {cnt}\n")
print("Wrote", txt_path, "|", ", ".join([f"{t}->{p}:{c}" for (t,p),c in top5]))

# --- Make grids of most confusing examples (val)
def grid_examples(Xu8, y_true, y_pred, pair, k=16, fname="grid.png"):
    t, p = pair
    idx = np.where((y_true == t) & (y_pred == p))[0]
    idx = idx[:k]
    if len(idx) == 0:
        return None
    cols = 4
    rows = int(np.ceil(len(idx)/cols))
    fig = plt.figure(figsize=(cols*2.2, rows*2.2))
    for i, j in enumerate(idx):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(Xu8[j].reshape(28,28), cmap="gray")
        ax.set_title(f"true {t} → pred {p}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname

# Use *validation* original uint8 images to visualize
X_val_u8 = Xtr_u8.reshape(len(Xtr_u8), 28, 28)  # reshape back for visuals
# Need the indexes of val split within train:
# Recreate split indexes to map to u8 images for plotting
def stratified_indexes(y, val_ratio=0.1, random_state=42):
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    idx_train, idx_val = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        idx_val.append(idx[:n_val])
        idx_train.append(idx[n_val:])
    return np.concatenate(idx_train), np.concatenate(idx_val)

_, val_idx = stratified_indexes(ytr, val_ratio=0.1, random_state=42)
X_val_u8_imgs = X_val_u8[val_idx]  # (6000, 28, 28)

# Grids for top confusions on val
for (t, p), _ in top5:
    out = os.path.join("reports", f"examples_val_{t}_to_{p}.png")
    grid_examples(X_val_u8_imgs.reshape(len(X_val_u8_imgs), -1), y_val, y_val_pred, (t, p), k=16, fname=out)
    print("Saved", out)
