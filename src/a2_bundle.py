from time import perf_counter

import cvxpy as cp
import numpy as np

from model import forward_and_grad, objective_F


def smooth_value_and_grad(X, y, w, b):
    """Return the smooth logistic loss and its gradient at x = (w, b)."""
    _, grad_w, grad_b, g_value = forward_and_grad(X, y, w, b)
    grad = np.concatenate([grad_w, np.array([grad_b])])
    return g_value, grad


def proximal_bundle_l1_logreg(
    X,
    y,
    lam=1e-2,
    rho=0.3,
    gamma=0.2,
    max_iter=100,
    bundle_max=20,
    tol_pred=1e-6,
    zero_tol=1e-6,
    verbose=False,
    x0=None,
):
    """
    A2: proximal bundle method for the L1-regularized logistic objective.

    The bundle linearizes only the smooth logistic loss while keeping the L1
    term exact in the master problem. This usually gives more reliable sparse
    steps than cutting-plane models of the full objective.
    """
    _, d = X.shape
    n = d + 1

    if x0 is None:
        x_c = np.zeros(n)
    else:
        x_c = x0.copy()

    w_c, b_c = x_c[:-1], x_c[-1]
    g_c, grad_c = smooth_value_and_grad(X, y, w_c, b_c)
    F_c = objective_F(X, y, w_c, b_c, lam)

    bundle = [{"x": x_c.copy(), "g": g_c, "grad": grad_c.copy()}]

    history = {
        "F_center": [F_c],
        "bundle_size": [1],
        "time": [0.0],
        "x_center": [x_c.copy()],
        "attempted_iterations": 0,
    }
    start_time = perf_counter()

    for it in range(max_iter):
        x_var = cp.Variable(n)
        t_var = cp.Variable()

        constraints = []
        for el in bundle:
            constraints.append(t_var >= el["g"] + el["grad"] @ (x_var - el["x"]))

        objective = cp.Minimize(
            t_var
            + lam * cp.norm1(x_var[:d])
            + 0.5 * rho * cp.sum_squares(x_var - x_c)
        )
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)
        history["attempted_iterations"] = it + 1

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Master problem not solved properly: status={prob.status}")

        x_bar = x_var.value
        t_bar = t_var.value

        w_bar, b_bar = x_bar[:-1], x_bar[-1]
        g_bar, grad_bar = smooth_value_and_grad(X, y, w_bar, b_bar)
        F_bar = objective_F(X, y, w_bar, b_bar, lam)

        m_c = max(el["g"] + el["grad"] @ (x_c - el["x"]) for el in bundle)
        m_c += lam * np.sum(np.abs(x_c[:d]))

        F_model = t_bar + lam * np.sum(np.abs(w_bar)) + 0.5 * rho * np.linalg.norm(x_bar - x_c) ** 2
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

        bundle.append({"x": x_bar.copy(), "g": g_bar, "grad": grad_bar.copy()})
        if len(bundle) > bundle_max:
            bundle.pop(0)

        history["F_center"].append(F_c)
        history["bundle_size"].append(len(bundle))
        history["time"].append(perf_counter() - start_time)
        history["x_center"].append(x_c.copy())

    # Bundle solvers may leave tiny coefficients instead of exact zeros.
    weight_mask = np.abs(x_c[:-1]) < zero_tol
    x_c[:-1][weight_mask] = 0.0

    return x_c, history
