import gzip
import io
import os
import struct
import numpy as np

# Global reproducibility
np.random.seed(42)

# ---- Internal helpers ----

def _open_maybe_gz(path: str) -> io.BufferedReader:
    """
    Open a file that might be gzipped or raw. Returns a binary file-like.
    """
    # Heuristic: prefer by extension; fall back by magic bytes if needed.
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    # If not .gz, still detect gzip header (1f 8b) and use gzip if present
    f = open(path, "rb")
    head = f.read(2)
    f.seek(0)
    if head == b"\x1f\x8b":
        f.close()
        return gzip.open(path, "rb")
    return f

def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if b is None or len(b) != n:
        raise ValueError("Unexpected EOF while reading IDX file.")
    return b

# ---- Public API ----

def load_idx_images(path: str, max_items: int | None = None, dtype=np.uint8) -> np.ndarray:
    """
    Load IDX3 images (magic=2051). Returns array shape (N, 28, 28) of dtype (default uint8).
    If max_items is given, loads only the first max_items.
    """
    with _open_maybe_gz(path) as f:
        magic, = struct.unpack(">I", _read_exact(f, 4))
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic} (expected 2051)")
        n_imgs, rows, cols = struct.unpack(">III", _read_exact(f, 12))
        if rows != 28 or cols != 28:
            raise ValueError(f"Unexpected image size {rows}x{cols}, expected 28x28")
        if max_items is not None:
            n = min(n_imgs, int(max_items))
        else:
            n = n_imgs
        buf = _read_exact(f, rows * cols * n)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)
        if dtype != np.uint8:
            arr = arr.astype(dtype, copy=False)
        return arr

def load_idx_labels(path: str, max_items: int | None = None, dtype=np.uint8) -> np.ndarray:
    """
    Load IDX1 labels (magic=2049). Returns array shape (N,) of dtype (default uint8).
    """
    with _open_maybe_gz(path) as f:
        magic, = struct.unpack(">I", _read_exact(f, 4))
        if magic != 2049:
            raise ValueError(f"Bad magic for labels: {magic} (expected 2049)")
        n_items, = struct.unpack(">I", _read_exact(f, 4))
        if max_items is not None:
            n = min(n_items, int(max_items))
        else:
            n = n_items
        buf = _read_exact(f, n)
        arr = np.frombuffer(buf, dtype=np.uint8)
        if dtype != np.uint8:
            arr = arr.astype(dtype, copy=False)
        return arr

def load_mnist_dataset(
    data_dir: str = "data",
    subset: int | None = None,
    as_float: bool = False,
    normalize: bool = False,
):
    """
    Load MNIST train/test as (X_train, y_train, X_test, y_test).
    - subset: if set (e.g., 10000), load only first N items from each split (for speed).
    - as_float: return images as float32.
    - normalize: if True, scale to [0,1] when as_float is True.
    """
    paths = {
        "train_images": os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        "train_labels": os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
        "test_images":  os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        "test_labels":  os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
    }
    Xtr = load_idx_images(paths["train_images"], max_items=subset, dtype=np.uint8)
    ytr = load_idx_labels(paths["train_labels"], max_items=subset, dtype=np.uint8)
    Xte = load_idx_images(paths["test_images"],  max_items=subset, dtype=np.uint8)
    yte = load_idx_labels(paths["test_labels"],  max_items=subset, dtype=np.uint8)

    if as_float:
        Xtr = Xtr.astype(np.float32)
        Xte = Xte.astype(np.float32)
        if normalize:
            Xtr /= 255.0
            Xte /= 255.0

    return Xtr, ytr, Xte, yte
