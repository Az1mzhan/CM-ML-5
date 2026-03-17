import cvxpy as cp
import numpy as np

from model import forward_and_grad, objective_F


def subgradient_l1_part(w, lam):
    """
    Return one valid subgradient of lam * ||w||_1.

    For non-zero coordinates the subgradient is fixed at lam * sign(w_i).
    At zero any value in [-lam, lam] is valid; we choose 0 for simplicity.
    """
    s = lam * np.sign(w)
    s[w == 0.0] = 0.0
    return s


def subgradient_F(X, y, w, b, lam):
    """Compute one subgradient of F at x = (w, b)."""
    _, grad_w, grad_b, _ = forward_and_grad(X, y, w, b)
    s_l1 = subgradient_l1_part(w, lam)
    g_w = grad_w + s_l1
    g_b = grad_b
    return np.concatenate([g_w, np.array([g_b])])


def proximal_bundle_l1_logreg(
    X,
    y,
    lam=1e-2,
    rho=1.0,
    gamma=0.1,
    max_iter=100,
    bundle_max=20,
    tol_pred=1e-4,
    verbose=False,
    x0=None,
):
    """
    A2: proximal bundle method for the L1-regularized logistic objective.

    X is an (N, d) design matrix, y contains binary labels,
    and x = [w; b] stores the weights and bias in one vector.
    """
    _, d = X.shape
    n = d + 1

    if x0 is None:
        x_c = np.zeros(n)
    else:
        x_c = x0.copy()

    w_c, b_c = x_c[:-1], x_c[-1]
    F_c = objective_F(X, y, w_c, b_c, lam)
    g_c = subgradient_F(X, y, w_c, b_c, lam)

    bundle = [{"x": x_c.copy(), "F": F_c, "g": g_c.copy()}]

    history = {
        "F_center": [F_c],
        "bundle_size": [1],
    }

    for it in range(max_iter):
        H = np.zeros((n + 1, n + 1))
        H[:n, :n] = rho * np.eye(n)

        f = np.zeros(n + 1)
        f[:n] = -rho * x_c
        f[-1] = 1.0

        # Each bundle element contributes a supporting hyperplane of F.
        A_rows = []
        b_vals = []
        for el in bundle:
            xj = el["x"]
            Fj = el["F"]
            gj = el["g"]
            A_j = np.concatenate([gj, np.array([-1.0])])
            bj = -Fj + gj @ xj
            A_rows.append(A_j)
            b_vals.append(bj)

        A = np.vstack(A_rows)
        b_vec = np.array(b_vals)

        z = cp.Variable(n + 1)
        obj = 0.5 * cp.quad_form(z, H) + f @ z
        constraints = [A @ z <= b_vec]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.OSQP)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"QP not solved properly: status={prob.status}")

        z_bar = z.value
        x_bar = z_bar[:n]
        t_bar = z_bar[-1]

        w_bar, b_bar = x_bar[:-1], x_bar[-1]
        F_bar = objective_F(X, y, w_bar, b_bar, lam)
        g_bar = subgradient_F(X, y, w_bar, b_bar, lam)

        mc_terms = [el["F"] + el["g"] @ (x_c - el["x"]) for el in bundle]
        m_c = max(mc_terms)

        F_model = t_bar + 0.5 * rho * np.linalg.norm(x_bar - x_c) ** 2
        delta_pred = m_c - F_model
        delta_act = F_c - F_bar

        if verbose:
            print(
                f"[iter {it:3d}] F_c={F_c:.6f}, F_bar={F_bar:.6f}, "
                f"dpred={delta_pred:.3e}, dact={delta_act:.3e}, "
                f"|B|={len(bundle)}"
            )

        if delta_pred <= tol_pred:
            break

        if delta_act >= gamma * delta_pred:
            x_c = x_bar
            F_c = F_bar

        bundle.append({"x": x_bar.copy(), "F": F_bar, "g": g_bar.copy()})
        if len(bundle) > bundle_max:
            bundle.pop(0)

        history["F_center"].append(F_c)
        history["bundle_size"].append(len(bundle))

    return x_c, history
