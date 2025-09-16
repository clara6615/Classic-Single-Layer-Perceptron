import os
import numpy as np
import matplotlib.pyplot as plt

from data_io import load_mnist_dataset
from utils import to_float, flatten_images
from perceptron import OneVsRestPerceptron

np.random.seed(42)
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Config (match your tuned scratch settings) ---
VAL_RATIO = 0.1
MAX_EPOCHS = 30
LEARNING_RATE = 1.0
MARGIN = 0.05
AVERAGE = False
SUBSET_TRAIN = None
SUBSET_TEST  = None

# --- Deterministic stratified indexes (copy of logic used earlier) ---
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
    idx_train = np.concatenate(idx_train)
    idx_val   = np.concatenate(idx_val)
    rng.shuffle(idx_train); rng.shuffle(idx_val)
    return idx_train, idx_val

def acc(pred, true): 
    return float((pred == true).mean())

# --- Load raw u8s and normalized baseline features ---
Xtr_u8, ytr, Xte_u8, yte = load_mnist_dataset("data", subset=SUBSET_TRAIN, as_float=False, normalize=False)
Xte_u8 = Xte_u8[:SUBSET_TEST] if SUBSET_TEST else Xte_u8
yte    = yte[:SUBSET_TEST] if SUBSET_TEST else yte

# Build the deterministic split once on labels
idx_tr, idx_val = stratified_indexes(ytr, val_ratio=VAL_RATIO, random_state=42)

# Helper to train/eval one variant
def run_variant(name, Xtr_u8, Xte_u8):
    # Build features per variant
    if name == "baseline":
        Xtr = flatten_images(to_float(Xtr_u8, normalize=True))
        Xte = flatten_images(to_float(Xte_u8, normalize=True))

    elif name == "bin_any":  # any nonzero stroke -> 1
        Xtr = flatten_images((Xtr_u8 > 0).astype(np.float32))
        Xte = flatten_images((Xte_u8 > 0).astype(np.float32))

    elif name == "bin_0_5":
        Xtr_f = to_float(Xtr_u8, normalize=True)
        Xte_f = to_float(Xte_u8, normalize=True)
        Xtr = flatten_images((Xtr_f > 0.5).astype(np.float32))
        Xte = flatten_images((Xte_f > 0.5).astype(np.float32))

    elif name == "centered":
        Xtr_f = flatten_images(to_float(Xtr_u8, normalize=True))
        # compute mean on TRAIN ONLY using the split indexes
        mu = Xtr_f[idx_tr].mean(axis=0, dtype=np.float64)
        Xtr = (Xtr_f - mu).astype(np.float32, copy=False)
        Xte = (flatten_images(to_float(Xte_u8, normalize=True)) - mu).astype(np.float32, copy=False)

    else:
        raise ValueError(f"Unknown variant: {name}")

    # Apply precomputed split
    X_tr, y_tr = Xtr[idx_tr], ytr[idx_tr]
    X_val, y_val = Xtr[idx_val], ytr[idx_val]

    # Train OVR perceptron
    model = OneVsRestPerceptron(
        n_classes=10,
        max_epochs=MAX_EPOCHS,
        shuffle=True,
        random_state=42,
        learning_rate=LEARNING_RATE,
        margin=MARGIN,
        average=AVERAGE,
    )
    model.fit(X_tr, y_tr)

    # Evaluate
    y_tr_pred  = model.predict(X_tr)
    y_val_pred = model.predict(X_val)
    y_te_pred  = model.predict(Xte)

    metrics = {
        "train": acc(y_tr_pred,  y_tr),
        "val":   acc(y_val_pred, y_val),
        "test":  acc(y_te_pred,  yte),
    }

    # Save weights per variant
    np.savez(f"models/perceptron_{name}.npz", W=model.get_weights())
    return metrics

variants = ["baseline", "bin_any", "bin_0_5", "centered"]
results = {}
for v in variants:
    print(f"=== Running {v} ===")
    results[v] = run_variant(v, Xtr_u8, Xte_u8)
    print(v, results[v])

# Write CSV report
csv_path = "reports/ablate_features.csv"
with open(csv_path, "w") as f:
    f.write("variant,train,val,test\n")
    for v in variants:
        m = results[v]
        f.write(f"{v},{m['train']:.4f},{m['val']:.4f},{m['test']:.4f}\n")
print("Wrote", csv_path)

# Plot
labels = variants
val_scores = [results[v]["val"] for v in labels]
test_scores = [results[v]["test"] for v in labels]

x = np.arange(len(labels))
plt.figure(figsize=(7,4))
plt.bar(x-0.2, val_scores, width=0.4, label="val")
plt.bar(x+0.2, test_scores, width=0.4, label="test")
plt.xticks(x, labels, rotation=15)
plt.ylim(0.0, 1.0)
plt.ylabel("accuracy")
plt.title("Feature ablations: val/test accuracy")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("reports/ablate_features.png", dpi=150)
print("Saved reports/ablate_features.png")

# Print a winner line
best = max(variants, key=lambda v: results[v]["val"])
print("Best on val:", best, results[best])
