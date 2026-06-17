import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute a numerically stable sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


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

    g_value = np.mean(np.logaddexp(0.0, z) - y * z)

    return p, grad_w, grad_b, g_value


def objective_F(X, y, w, b, lam):
    """Return F(w, b) = logistic_loss(w, b) + lam * ||w||_1."""
    _, _, _, g_value = forward_and_grad(X, y, w, b)
    return g_value + lam * np.sum(np.abs(w))


def logistic_lipschitz_constant(X: np.ndarray) -> float:
    """Return a valid Lipschitz constant for grad g in variables (w, b)."""
    n_samples = X.shape[0]
    augmented = np.column_stack([X, np.ones(n_samples)])
    return np.linalg.norm(augmented, ord=2) ** 2 / (4.0 * n_samples)


def prox_gradient_mapping_residual(X, y, w, b, lam, step=None):
    """
    Return the norm of the proximal-gradient mapping.

    This is a numerically cleaner stationarity diagnostic for the composite
    objective than explicitly branching on whether each weight is exactly zero.
    """
    if step is None:
        lipschitz = logistic_lipschitz_constant(X)
        step = 1.0 / max(lipschitz, 1e-16)

    _, grad_w, grad_b, _ = forward_and_grad(X, y, w, b)
    prox_w = soft_threshold(w - step * grad_w, step * lam)
    prox_b = b - step * grad_b

    residual_w = (w - prox_w) / step
    residual_b = (b - prox_b) / step
    return np.sqrt(np.sum(residual_w ** 2) + residual_b ** 2)


def stationarity_residual(X, y, w, b, lam, zero_tol=1e-6):
    """
    Return the norm of the minimum-norm element of the subdifferential of F.

    For the bias term the objective is smooth, while for the weights the
    minimum-norm subgradient can be written in closed form.
    """
    _, grad_w, grad_b, _ = forward_and_grad(X, y, w, b)

    residual_w = np.empty_like(w)
    mask_nz = np.abs(w) > zero_tol
    residual_w[mask_nz] = grad_w[mask_nz] + lam * np.sign(w[mask_nz])
    residual_w[~mask_nz] = soft_threshold(grad_w[~mask_nz], lam)

    return np.sqrt(np.sum(residual_w ** 2) + grad_b ** 2)
