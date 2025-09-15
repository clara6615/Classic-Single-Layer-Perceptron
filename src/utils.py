import numpy as np
from typing import Tuple

np.random.seed(42)

def flatten_images(X: np.ndarray) -> np.ndarray:
    """(N, 28, 28) -> (N, 784) without copying if possible."""
    assert X.ndim == 3 and X.shape[1:] == (28, 28), "Expect (N,28,28)"
    return X.reshape(len(X), -1)

def to_float(X: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Cast to float32; if normalize, scale to [0,1].
    """
    Xf = X.astype(np.float32, copy=False)
    if normalize:
        Xf /= 255.0
    return Xf

def stratified_train_val_split(
    X: np.ndarray, y: np.ndarray, val_ratio: float = 0.1, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split by label. Preserves class balance. Shuffles deterministically.
    Returns X_train, y_train, X_val, y_val (views where possible).
    """
    assert 0.0 < val_ratio < 1.0
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    idx_train = []
    idx_val = []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        idx_val.append(idx[:n_val])
        idx_train.append(idx[n_val:])
    idx_train = np.concatenate(idx_train)
    idx_val = np.concatenate(idx_val)
    # Final shuffle for mixing classes but still deterministic
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    return X[idx_train], y[idx_train], X[idx_val], y[idx_val]
