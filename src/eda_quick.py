import os
import numpy as np
import matplotlib.pyplot as plt
from data_io import load_mnist_dataset
from utils import flatten_images, to_float

np.random.seed(42)
os.makedirs("reports", exist_ok=True)

# Load a subsample for speed if you like
Xtr, ytr, Xte, yte = load_mnist_dataset("data", subset=None, as_float=False, normalize=False)

# Class frequency (train)
counts = np.bincount(ytr, minlength=10)

plt.figure(figsize=(6,3))
plt.bar(range(10), counts)
plt.xlabel("Digit")
plt.ylabel("Count")
plt.title("Train label distribution")
plt.tight_layout()
plt.savefig("reports/label_hist_train.png", dpi=150)
print("Saved reports/label_hist_train.png")

# Quick flatten check
Xtr_flat = flatten_images(Xtr)
print("Flat shape:", Xtr_flat.shape)  # Expect (60000, 784)
