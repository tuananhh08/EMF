import numpy as np


class EMFPreprocessor:
    """
    Lightweight, pickle-friendly preprocessor for 9-channel EMF.

    Steps:
      1) (optional) signed-log compression: sign(x)*log1p(|x|/scale)
      2) (optional) standardization: (x - mean) / std
    """

    def __init__(self, use_signed_log: bool = False, use_standardize: bool = True, eps: float = 1e-12):
        self.use_signed_log = bool(use_signed_log)
        self.use_standardize = bool(use_standardize)
        self.eps = float(eps)

        self._fitted = False
        self.scale_ = None  # (D,) or None
        self.mean_ = None  # (D,) or None
        self.std_ = None  # (D,) or None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        Z = X

        if self.use_signed_log:
            # robust per-feature scale (median absolute)
            scale = np.median(np.abs(Z), axis=0)
            scale = np.maximum(scale, self.eps)
            self.scale_ = scale.astype(np.float64)
            Z = np.sign(Z) * np.log1p(np.abs(Z) / self.scale_)

        if self.use_standardize:
            mean = Z.mean(axis=0)
            std = Z.std(axis=0)
            std = np.maximum(std, self.eps)
            self.mean_ = mean.astype(np.float64)
            self.std_ = std.astype(np.float64)

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EMFPreprocessor must be fitted before calling transform().")

        Z = np.asarray(X, dtype=np.float64)
        if Z.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {Z.shape}")

        if self.use_signed_log:
            Z = np.sign(Z) * np.log1p(np.abs(Z) / self.scale_)

        if self.use_standardize:
            Z = (Z - self.mean_) / self.std_

        return Z.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

