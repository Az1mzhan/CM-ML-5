import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

def soft_threshold(u: np.ndarray, tau: float) -> np.ndarray:
    # Soft-thresholding S_tau(u) (for L1 prox)
    return np.sign(u) * np.maximum(np.abs(u) - tau, 0.0)

def forward_and_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    Compute p = σ(Xw + b), gradient of smooth part g, and loss g
    Returns: p, grad_w, grad_b, g_value
    """
    N, d = X.shape
    z = X @ w + b
    p = sigmoid(z)
    err = p - y

    grad_w = (X.T @ err) / N
    grad_b = np.sum(err) / N

    # logistic loss (smooth part)
    eps = 1e-12
    g_value = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    return p, grad_w, grad_b, g_value

def objective_F(X, y, w, b, lam):
    # Full objective F = g + lam * ||w||_1
    _, _, _, g_value = forward_and_grad(X, y, w, b)
    return g_value + lam * np.sum(np.abs(w))