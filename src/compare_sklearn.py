import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

from data_io import load_mnist_dataset
from utils import to_float, flatten_images, stratified_train_val_split
from perceptron import OneVsRestPerceptron

np.random.seed(42)
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Config (align with scratch) ---
VAL_RATIO = 0.1
MAX_EPOCHS = 25
LEARNING_RATE = 1.0
MARGIN = 0.05
AVERAGE = True
SUBSET_TRAIN = None
SUBSET_TEST  = None

# --- Load & preprocess (same as scratch) ---
Xtr_u8, ytr, Xte_u8, yte = load_mnist_dataset("data", subset=SUBSET_TRAIN, as_float=False, normalize=False)
Xte_u8 = Xte_u8[:SUBSET_TEST] if SUBSET_TEST else Xte_u8
yte    = yte[:SUBSET_TEST] if SUBSET_TEST else yte

Xtr = flatten_images(to_float(Xtr_u8, normalize=True))
Xte = flatten_images(to_float(Xte_u8, normalize=True))
X_tr, y_tr, X_val, y_val = stratified_train_val_split(Xtr, ytr, val_ratio=VAL_RATIO, random_state=42)

def acc(pred, true): return float((pred == true).mean())

# --- Load scratch weights (or skip if not present) ---
scratch_metrics = {}
scratch_pred = {}
W_path = "models/perceptron_scratch_weights.npz"
if os.path.exists(W_path):
    W = np.load(W_path)["W"]
    scratch = OneVsRestPerceptron(n_classes=10, max_epochs=1, shuffle=False, random_state=42)
    scratch.set_weights(W)
    scratch_pred["train"] = scratch.predict(X_tr)
    scratch_pred["val"]   = scratch.predict(X_val)
    scratch_pred["test"]  = scratch.predict(Xte)
    scratch_metrics = {
        "train": acc(scratch_pred["train"], y_tr),
        "val":   acc(scratch_pred["val"],   y_val),
        "test":  acc(scratch_pred["test"],  yte),
    }
else:
    print("WARNING: weights not found at", W_path, "- run src/train_scratch.py first.")
    scratch_metrics = {"train": np.nan, "val": np.nan, "test": np.nan}

# --- Train sklearn baseline ---
sk_clf = Perceptron(
    max_iter=MAX_EPOCHS,
    tol=None,                 # run full epochs
    shuffle=True,
    random_state=42,
    fit_intercept=True,
)
sk_clf.fit(X_tr, y_tr)

# Save sklearn model (stdlib pickle to respect library constraints)
with open("models/perceptron_sklearn.pkl", "wb") as f:
    pickle.dump(sk_clf, f)

sk_pred = {
    "train": sk_clf.predict(X_tr),
    "val":   sk_clf.predict(X_val),
    "test":  sk_clf.predict(Xte),
}
sk_metrics = {k: acc(sk_pred[k], v) for k, v in [("train", y_tr), ("val", y_val), ("test", yte)]}

print("scratch:", scratch_metrics)
print("sklearn:", sk_metrics)

# --- Write a tiny CSV report ---
rows = []
for split in ["train", "val", "test"]:
    rows.append(f"scratch,{split},{scratch_metrics[split]:.4f}")
    rows.append(f"sklearn,{split},{sk_metrics[split]:.4f}")
csv_path = "reports/accuracy_comparison.csv"
with open(csv_path, "w") as f:
    f.write("method,split,accuracy\n")
    f.write("\n".join(rows) + "\n")
print("Wrote", csv_path)

# --- Bar chart ---
labels = ["train", "val", "test"]
scratch_vals = [scratch_metrics[s] for s in labels]
sk_vals      = [sk_metrics[s] for s in labels]

x = np.arange(len(labels))
w = 0.35

plt.figure(figsize=(6,4))
plt.bar(x - w/2, scratch_vals, width=w, label="scratch")
plt.bar(x + w/2, sk_vals,      width=w, label="sklearn")
plt.xticks(x, labels)
plt.ylim(0.8, 1.0)
plt.ylabel("accuracy")
plt.title("Perceptron accuracy: scratch vs sklearn")
plt.legend()
plt.tight_layout()
out_png = "reports/accuracy_comparison.png"
plt.savefig(out_png, dpi=150)
print("Saved", out_png)
