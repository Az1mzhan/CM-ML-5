import numpy as np
import cvxpy as cp
from model import forward_and_grad, objective_F

def subgradient_l1_part(w, lam, eps=1e-8):
    """
    One specific subgradient of lam * ||w||_1.
    At zero, we clamp slope into [-lam, lam] using w/eps.
    """
    s = np.zeros_like(w)
    # nonzero entries
    mask_nz = np.abs(w) > eps
    s[mask_nz] = lam * np.sign(w[mask_nz])
    # near zero (just one possible choice)
    mask_z = ~mask_nz
    s[mask_z] = lam * np.clip(w[mask_z] / eps, -1.0, 1.0)
    return s

def subgradient_F(X, y, w, b, lam):
    """
    Compute one subgradient gF of F at x=(w,b).
    """
    p, grad_w, grad_b, _ = forward_and_grad(X, y, w, b)
    s_l1 = subgradient_l1_part(w, lam)
    g_w = grad_w + s_l1
    g_b = grad_b  # no L1 on bias
    return np.concatenate([g_w, np.array([g_b])])

def proximal_bundle_l1_logreg(X, y,
                              lam=1e-2,
                              rho=1.0,
                              gamma=0.1,
                              max_iter=100,
                              bundle_max=20,
                              tol_pred=1e-4,
                              verbose=False,
                              x0=None):
    """
    A2: Proximal bundle method for F (L1-logistic).

    X ∈ R^{N×d}, y ∈ {0,1}^N.
    x = [w; b] ∈ R^{d+1}.

    Returns: x_center, history (dict)
    """
    N, d = X.shape
    n = d + 1

    # init center x_c
    if x0 is None:
        x_c = np.zeros(n)
    else:
        x_c = x0.copy()

    w_c, b_c = x_c[:-1], x_c[-1]
    F_c = objective_F(X, y, w_c, b_c, lam)
    g_c = subgradient_F(X, y, w_c, b_c, lam)

    # bundle: list of dicts
    bundle = [{
        "x": x_c.copy(),
        "F": F_c,
        "g": g_c.copy()
    }]

    history = {
        "F_center": [F_c],
        "bundle_size": [1],
    }

    for it in range(max_iter):
        # --- Build QP matrices ---
        H = np.zeros((n + 1, n + 1))
        H[:n, :n] = rho * np.eye(n) # block diag

        f = np.zeros(n + 1)
        f[:n] = -rho * x_c
        f[-1] = 1.0 # coefficient of t

        # A z ≤ b constraints
        A_rows = []
        b_vals = []
        for el in bundle:
            xj = el["x"]
            Fj = el["F"]
            gj = el["g"]
            A_j = np.concatenate([gj, np.array([-1.0])])  # shape (n+1,)
            bj = -Fj + gj @ xj
            A_rows.append(A_j)
            b_vals.append(bj)
        A = np.vstack(A_rows)
        b_vec = np.array(b_vals)

        # --- Solve QP with cvxpy ---
        z = cp.Variable(n + 1)
        obj = 0.5 * cp.quad_form(z, H) + f @ z
        constraints = [A @ z <= b_vec]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.OSQP) # or another QP solver

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"QP not solved properly: status={prob.status}")

        z_bar = z.value
        x_bar = z_bar[:n]
        t_bar = z_bar[-1]

        # --- Evaluate F and subgradient at candidate ---
        w_bar, b_bar = x_bar[:-1], x_bar[-1]
        F_bar = objective_F(X, y, w_bar, b_bar, lam)
        g_bar = subgradient_F(X, y, w_bar, b_bar, lam)

        # --- Predicted & actual decrease ---
        # model value at center
        mc_terms = [el["F"] + el["g"] @ (x_c - el["x"]) for el in bundle]
        m_c = max(mc_terms)

        F_model = t_bar + 0.5 * rho * np.linalg.norm(x_bar - x_c) ** 2
        delta_pred = m_c - F_model
        delta_act = F_c - F_bar

        if verbose:
            print(f"[iter {it:3d}] F_c={F_c:.6f}, F_bar={F_bar:.6f}, "
                  f"Δpred={delta_pred:.3e}, Δact={delta_act:.3e}, "
                  f"|B|={len(bundle)}")

        # stopping: very small predicted improvement
        if delta_pred <= tol_pred:
            break

        # serious vs null step
        if delta_act >= gamma * delta_pred:
            # serious step
            x_c = x_bar
            F_c = F_bar

        # add new cut
        bundle.append({"x": x_bar.copy(), "F": F_bar, "g": g_bar.copy()})
        if len(bundle) > bundle_max:
            # simple policy: drop oldest
            bundle.pop(0)

        history["F_center"].append(F_c)
        history["bundle_size"].append(len(bundle))

    return x_c, history