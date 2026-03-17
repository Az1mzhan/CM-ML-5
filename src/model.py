import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute a numerically stable sigmoid."""
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def soft_threshold(u: np.ndarray, tau: float) -> np.ndarray:
    """Apply the proximal operator of tau * ||.||_1 coordinate-wise."""
    return np.sign(u) * np.maximum(np.abs(u) - tau, 0.0)


def forward_and_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """Return probabilities, the gradient of the smooth term, and mean loss."""
    n_samples, _ = X.shape
    z = X @ w + b
    p = sigmoid(z)
    err = p - y

    grad_w = (X.T @ err) / n_samples
    grad_b = np.sum(err) / n_samples

    eps = 1e-12
    g_value = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    return p, grad_w, grad_b, g_value


def objective_F(X, y, w, b, lam):
    """Return F(w, b) = logistic_loss(w, b) + lam * ||w||_1."""
    _, _, _, g_value = forward_and_grad(X, y, w, b)
    return g_value + lam * np.sum(np.abs(w))
