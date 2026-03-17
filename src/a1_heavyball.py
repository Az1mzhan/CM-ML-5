import numpy as np

from model import forward_and_grad, objective_F, soft_threshold


def heavy_ball_prox_l1_logreg(
    X,
    y,
    lam=1e-2,
    alpha=1.0,
    beta=0.9,
    max_iter=1000,
    tol=1e-6,
    verbose=False,
):
    """
    A1: heavy-ball momentum on the smooth loss followed by an L1 prox step.

    Returns: w, b, history (dict)
    """
    _, d = X.shape

    w = np.zeros(d)
    b = 0.0
    w_prev = w.copy()
    b_prev = b

    history = {
        "F": [],
        "g": [],
        "nnz": [],  # Number of non-zero weights.
    }

    for k in range(max_iter):
        _, grad_w, grad_b, g_value = forward_and_grad(X, y, w, b)

        # Apply momentum to the smooth part, then shrink the weights with the L1 prox.
        y_w = w - alpha * grad_w + beta * (w - w_prev)
        y_b = b - alpha * grad_b + beta * (b - b_prev)

        w_next = soft_threshold(y_w, alpha * lam)
        b_next = y_b

        F_val = objective_F(X, y, w_next, b_next, lam)
        history["F"].append(F_val)
        history["g"].append(g_value)
        history["nnz"].append(np.count_nonzero(w_next))

        if verbose and (k % 50 == 0 or k == max_iter - 1):
            print(f"k={k:4d}, F={F_val:.6f}, ||dw||={np.linalg.norm(w_next - w):.3e}")

        if np.linalg.norm(w_next - w) <= tol and abs(b_next - b) <= tol:
            w, b = w_next, b_next
            break

        w_prev, b_prev = w, b
        w, b = w_next, b_next

    return w, b, history
