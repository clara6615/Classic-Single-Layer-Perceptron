from __future__ import annotations
import numpy as np
from dataclasses import dataclass

np.random.seed(42)

def _add_bias(X: np.ndarray) -> np.ndarray:
    """Append a bias feature of 1.0 to each row: (N,F)->(N,F+1)."""
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])

@dataclass
class BinaryPerceptron:
    max_epochs: int = 10
    shuffle: bool = True
    random_state: int = 42

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryPerceptron":
        """
        y must be in {-1, +1}.
        Updates: if y_i * (w·x_i + b) <= 0, then w <- w + y_i * x_i (bias included).
        """
        rng = np.random.RandomState(self.random_state)
        Xb = _add_bias(X).astype(np.float32, copy=False)
        y = y.astype(np.int8, copy=False)
        n_features = Xb.shape[1]
        self.w_ = np.zeros(n_features, dtype=np.float32)

        self.mistakes_per_epoch_ = []
        idx = np.arange(Xb.shape[0])
        for epoch in range(self.max_epochs):
            if self.shuffle:
                rng.shuffle(idx)
            mistakes = 0
            for i in idx:
                s = float(np.dot(self.w_, Xb[i]))
                if y[i] * s <= 0.0:      # misclassified or on boundary
                    self.w_ += y[i] * Xb[i]
                    mistakes += 1
            self.mistakes_per_epoch_.append(mistakes)
            if mistakes == 0:
                break  # converged (on linearly separable data)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xb = _add_bias(X).astype(np.float32, copy=False)
        return Xb @ self.w_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

class OneVsRestPerceptron:
    def __init__(self, n_classes: int = 10, max_epochs: int = 10, shuffle: bool = True, random_state: int = 42):
        self.n_classes = n_classes
        self.max_epochs = max_epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.clfs_: list[BinaryPerceptron] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneVsRestPerceptron":
        self.clfs_ = []
        for c in range(self.n_classes):
            y_bin = np.where(y == c, 1, -1)
            clf = BinaryPerceptron(max_epochs=self.max_epochs, shuffle=self.shuffle, random_state=self.random_state + c)
            clf.fit(X, y_bin)
            self.clfs_.append(clf)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return (N, n_classes) scores."""
        assert self.clfs_ is not None, "Call fit first."
        scores = [clf.decision_function(X) for clf in self.clfs_]
        return np.stack(scores, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.argmax(scores, axis=1)

    def get_weights(self) -> np.ndarray:
        """Stack weights (including bias) into shape (n_classes, F+1)."""
        assert self.clfs_ is not None
        W = [clf.w_ for clf in self.clfs_]
        return np.stack(W, axis=0)

    def set_weights(self, W: np.ndarray) -> None:
        """Load pre-trained weights (n_classes, F+1)."""
        self.clfs_ = []
        for c in range(self.n_classes):
            clf = BinaryPerceptron(max_epochs=self.max_epochs, shuffle=self.shuffle, random_state=self.random_state + c)
            clf.w_ = W[c].astype(np.float32, copy=True)
            clf.mistakes_per_epoch_ = []
            self.clfs_.append(clf)
