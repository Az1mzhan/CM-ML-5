import numpy as np
from model import forward_and_grad, soft_threshold, objective_F

def heavy_ball_prox_l1_logreg(X, y,
                              lam=1e-2,
                              alpha=1.0,
                              beta=0.9,
                              max_iter=1000,
                              tol=1e-6,
                              verbose=False):
    """
    A1: Heavy ball (momentum) method on smooth part g,
        followed by proximal step for L1 term.

    Returns: w, b, history (dict)
    """
    N, d = X.shape

    # init
    w = np.zeros(d)
    b = 0.0
    w_prev = w.copy()
    b_prev = b

    history = {
        "F": [],
        "g": [],
        "nnz": [], # number of nonzero weights
    }

    for k in range(max_iter):
        # smooth gradient and value
        _, grad_w, grad_b, g_value = forward_and_grad(X, y, w, b)

        # heavy ball step on g
        y_w = w - alpha * grad_w + beta * (w - w_prev)
        y_b = b - alpha * grad_b + beta * (b - b_prev)

        # proximal step for h = lam * ||w||_1
        w_next = soft_threshold(y_w, alpha * lam)
        b_next = y_b  # bias not regularized

        # bookkeeping
        F_val = objective_F(X, y, w_next, b_next, lam)
        history["F"].append(F_val)
        history["g"].append(g_value)
        history["nnz"].append(np.count_nonzero(w_next))

        if verbose and (k % 50 == 0 or k == max_iter - 1):
            print(f"k={k:4d}, F={F_val:.6f}, ||Δw||={np.linalg.norm(w_next - w):.3e}")

        # stopping on parameter change
        if np.linalg.norm(w_next - w) <= tol and abs(b_next - b) <= tol:
            w, b = w_next, b_next
            break

        # shift iterates
        w_prev, b_prev = w, b
        w, b = w_next, b_next

    return w, b, history