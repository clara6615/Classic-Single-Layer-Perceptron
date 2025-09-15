import numpy as np
from data_io import load_mnist_dataset
from utils import flatten_images, to_float, stratified_train_val_split

np.random.seed(42)

# Load (uint8), then float+normalize and flatten
Xtr, ytr, Xte, yte = load_mnist_dataset("data", subset=None, as_float=False, normalize=False)
Xtr_f = to_float(Xtr, normalize=True)
Xte_f = to_float(Xte, normalize=True)
Xtr_flat = flatten_images(Xtr_f)
Xte_flat = flatten_images(Xte_f)

X_tr, y_tr, X_val, y_val = stratified_train_val_split(Xtr_flat, ytr, val_ratio=0.1, random_state=42)

print("Train/Val shapes:", X_tr.shape, y_tr.shape, X_val.shape, y_val.shape)
print("Example ranges:", (X_tr.min(), X_tr.max()))

# Acceptance checks
assert Xtr_flat.shape == (60000, 784)
assert Xte_flat.shape == (10000, 784)
assert X_tr.ndim == 2 and X_val.ndim == 2
assert X_tr.shape[1] == 784 and X_val.shape[1] == 784
assert X_tr.dtype == np.float32 and X_val.dtype == np.float32
# label balance check (rough): each class in val has >= 1 sample
import numpy as np
val_counts = np.bincount(y_val, minlength=10)
assert (val_counts > 0).all(), f"Some class missing in val: {val_counts}"

# Determinism check
X_tr2, y_tr2, X_val2, y_val2 = stratified_train_val_split(Xtr_flat, ytr, val_ratio=0.1, random_state=42)
assert np.array_equal(y_val, y_val2), "Non-deterministic split!"
print("Deterministic stratified split OK.")
