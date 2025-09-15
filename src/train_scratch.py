import os
import numpy as np
import matplotlib.pyplot as plt
from data_io import load_mnist_dataset
from utils import to_float, flatten_images, stratified_train_val_split
from perceptron import OneVsRestPerceptron

np.random.seed(42)
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Config (tweak here) ---
VAL_RATIO = 0.1
MAX_EPOCHS = 25
LEARNING_RATE = 1.0
MARGIN = 0.05
AVERAGE = True
SUBSET_TRAIN = None
SUBSET_TEST  = None

# --- Load ---
Xtr_u8, ytr, Xte_u8, yte = load_mnist_dataset("data", subset=SUBSET_TRAIN, as_float=False, normalize=False)
Xte_u8 = Xte_u8[:SUBSET_TEST] if SUBSET_TEST else Xte_u8
yte    = yte[:SUBSET_TEST] if SUBSET_TEST else yte

# float + normalize + flatten
Xtr = flatten_images(to_float(Xtr_u8, normalize=True))
Xte = flatten_images(to_float(Xte_u8, normalize=True))

# stratified split
X_tr, y_tr, X_val, y_val = stratified_train_val_split(Xtr, ytr, val_ratio=VAL_RATIO, random_state=42)

# --- Train ---
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

# Track mistakes/epoch across classes (sum of binary mistakes)
mistakes_matrix = np.stack([np.array(c.mistakes_per_epoch_, dtype=int) for c in model.clfs_], axis=0)
# Pad to equal length (up to MAX_EPOCHS)
max_len = max(len(c.mistakes_per_epoch_) for c in model.clfs_)
padded = np.zeros((10, max_len), dtype=int)
for i, row in enumerate(mistakes_matrix):
    padded[i, :len(row)] = row
mistakes_per_epoch = padded.sum(axis=0)

# --- Evaluate ---
def acc(pred, true): return float((pred == true).mean())

y_tr_pred  = model.predict(X_tr)
y_val_pred = model.predict(X_val)
y_te_pred  = model.predict(Xte)

acc_tr  = acc(y_tr_pred,  y_tr)
acc_val = acc(y_val_pred, y_val)
acc_te  = acc(y_te_pred,  yte)

print(f"Train acc: {acc_tr:.4f}")
print(f"Val   acc: {acc_val:.4f}")
print(f"Test  acc: {acc_te:.4f}")

# --- Save weights ---
W = model.get_weights()  # (10, 784+1)
np.savez("models/perceptron_scratch_weights.npz", W=W)
print("Saved models/perceptron_scratch_weights.npz with shape", W.shape)

# --- Plots ---
plt.figure()
plt.plot(np.arange(1, len(mistakes_per_epoch)+1), mistakes_per_epoch, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Total mistakes (sum over 10 binaries)")
plt.title("Perceptron training mistakes per epoch (OVR total)")
plt.tight_layout()
plt.savefig("reports/perceptron_mistakes.png", dpi=150)
print("Saved reports/perceptron_mistakes.png")

# Quick confusion matrix on val (cheap, text)
from collections import Counter
pairs = Counter(zip(y_val.tolist(), y_val_pred.tolist()))
print("Sample val conf counts (first 10):", list(pairs.items())[:10])
