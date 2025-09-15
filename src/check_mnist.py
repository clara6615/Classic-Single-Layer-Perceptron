import os
import numpy as np
import matplotlib.pyplot as plt
from data_io import load_mnist_dataset

np.random.seed(42)

Xtr, ytr, Xte, yte = load_mnist_dataset("data", subset=None, as_float=False, normalize=False)

print("Train:", Xtr.shape, ytr.shape)
print("Test :", Xte.shape, yte.shape)
print("Dtypes:", Xtr.dtype, ytr.dtype)
print("Label set:", sorted(set(ytr.tolist())))

# Quick acceptance checks
assert Xtr.shape == (60000, 28, 28)
assert Xte.shape == (10000, 28, 28)
assert ytr.shape == (60000,)
assert yte.shape == (10000,)
assert set(ytr.tolist()) == set(range(10))
assert Xtr.max() <= 255 and Xtr.min() >= 0

# Plot a tiny montage to reports/
os.makedirs("reports", exist_ok=True)
fig = plt.figure(figsize=(5, 2))
for i, idx in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    ax = plt.subplot(2, 5, i+1)
    ax.imshow(Xtr[idx], cmap="gray")
    ax.set_title(int(ytr[idx]))
    ax.axis("off")
fig.tight_layout()
out = "reports/mnist_samples.png"
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
