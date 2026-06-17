from time import perf_counter

import cvxpy as cp
import numpy as np

from model import forward_and_grad, objective_F


def smooth_value_and_grad(X, y, w, b):
    """Return the smooth logistic loss and its gradient at x = (w, b)."""
    _, grad_w, grad_b, g_value = forward_and_grad(X, y, w, b)
    grad = np.concatenate([grad_w, np.array([grad_b])])
    return g_value, grad


def _solver_options(solver_name, solver_tol):
    """Return CVXPY options for a master-problem solver."""
    if solver_name == "CLARABEL":
        return {
            "tol_gap_abs": solver_tol,
            "tol_gap_rel": solver_tol,
            "tol_feas": solver_tol,
            "max_iter": 1000,
            "verbose": False,
        }
    if solver_name == "OSQP":
        return {
            "eps_abs": solver_tol,
            "eps_rel": solver_tol,
            "max_iter": 200000,
            "polish": True,
            "verbose": False,
        }
    if solver_name == "SCS":
        return {
            "eps": solver_tol,
            "max_iters": 200000,
            "verbose": False,
        }
    return {"verbose": False}


def _solve_master_problem(bundle, x_c, lam, rho, d, solver_sequence, solver_tol):
    """Solve one proximal bundle master problem."""
    n = d + 1
    x_var = cp.Variable(n)
    t_var = cp.Variable()

    constraints = [
        t_var >= el["g"] + el["grad"] @ (x_var - el["x"])
        for el in bundle
    ]
    objective = cp.Minimize(
        t_var
        + lam * cp.norm1(x_var[:d])
        + 0.5 * rho * cp.sum_squares(x_var - x_c)
    )
    problem = cp.Problem(objective, constraints)

    last_error = None
    for solver_name in solver_sequence:
        try:
            problem.solve(
                solver=getattr(cp, solver_name),
                **_solver_options(solver_name, solver_tol),
            )
        except Exception as exc:  # pragma: no cover - diagnostics for solver fallback
            last_error = exc
            continue

        if problem.status in ["optimal", "optimal_inaccurate"]:
            return {
                "x": np.asarray(x_var.value).ravel(),
                "t": float(t_var.value),
                "status": problem.status,
                "solver": solver_name,
                "value": float(problem.value),
            }

    raise RuntimeError(f"Master problem was not solved. Last error: {last_error}")


def _prune_bundle(bundle, bundle_max, x_c):
    """Keep the current-center cut and the most recent cuts."""
    if bundle_max is None or len(bundle) <= bundle_max:
        return bundle

    center_idx = int(np.argmin([np.linalg.norm(el["x"] - x_c) for el in bundle]))
    keep_indices = {center_idx}

    for idx in range(len(bundle) - 1, -1, -1):
        keep_indices.add(idx)
        if len(keep_indices) >= bundle_max:
            break

    return [el for idx, el in enumerate(bundle) if idx in keep_indices]


def proximal_bundle_l1_logreg(
    X,
    y,
    lam=1e-2,
    rho=0.3,
    gamma=0.2,
    max_iter=100,
    bundle_max=20,
    tol_pred=1e-10,
    zero_tol=1e-6,
    solver_sequence=("CLARABEL", "OSQP", "SCS"),
    solver_tol=1e-10,
    certificate_tol=1e-9,
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
        "iteration": [0],
        "F_center": [F_c],
        "bundle_size": [1],
        "time": [0.0],
        "x_center": [x_c.copy()],
        "attempted_iterations": 0,
        "serious_steps": 0,
        "null_steps": 0,
        "stop_reason": "max_iter",
        "delta_pred": [],
        "delta_act": [],
        "solver_status": [],
        "solver_name": [],
    }
    start_time = perf_counter()

    for it in range(max_iter):
        history["attempted_iterations"] = it + 1
        master = None
        delta_pred = None
        delta_act = None
        best_candidate = None

        for solver_name in solver_sequence:
            try:
                candidate_master = _solve_master_problem(
                    bundle,
                    x_c,
                    lam,
                    rho,
                    d,
                    (solver_name,),
                    solver_tol,
                )
            except RuntimeError:
                continue

            candidate_x = candidate_master["x"]
            candidate_t = candidate_master["t"]

            candidate_w, candidate_b = candidate_x[:-1], candidate_x[-1]
            candidate_g, candidate_grad = smooth_value_and_grad(
                X, y, candidate_w, candidate_b
            )
            candidate_F = objective_F(X, y, candidate_w, candidate_b, lam)

            model_at_candidate = (
                candidate_t
                + lam * np.sum(np.abs(candidate_w))
                + 0.5 * rho * np.linalg.norm(candidate_x - x_c) ** 2
            )
            candidate_delta_pred = float(F_c - model_at_candidate)
            candidate_delta_act = float(F_c - candidate_F)

            candidate = {
                "master": candidate_master,
                "x": candidate_x,
                "w": candidate_w,
                "b": candidate_b,
                "g": candidate_g,
                "grad": candidate_grad,
                "F": candidate_F,
                "delta_pred": candidate_delta_pred,
                "delta_act": candidate_delta_act,
            }
            if best_candidate is None or candidate_delta_pred > best_candidate["delta_pred"]:
                best_candidate = candidate

            if candidate_master["status"] == "optimal" and candidate_delta_pred >= 0.0:
                best_candidate = candidate
                break

        if best_candidate is None:
            history["stop_reason"] = "master_solver_failure"
            raise RuntimeError("No configured solver could solve the master problem.")

        master = best_candidate["master"]
        x_bar = best_candidate["x"]
        w_bar = best_candidate["w"]
        g_bar = best_candidate["g"]
        grad_bar = best_candidate["grad"]
        F_bar = best_candidate["F"]
        delta_pred = best_candidate["delta_pred"]
        delta_act = best_candidate["delta_act"]

        if delta_pred < -certificate_tol:
            history["stop_reason"] = "master_certificate_failure"
            raise RuntimeError(
                "Master problem returned a model value above the current "
                f"objective by {-delta_pred:.3e}; last solver={master['solver']} "
                f"status={master['status']}"
            )

        history["delta_pred"].append(delta_pred)
        history["delta_act"].append(delta_act)
        history["solver_status"].append(master["status"])
        history["solver_name"].append(master["solver"])

        if verbose:
            print(
                f"[iter {it:3d}] F_c={F_c:.6f}, F_bar={F_bar:.6f}, "
                f"dpred={delta_pred:.3e}, dact={delta_act:.3e}, "
                f"|B|={len(bundle)}, solver={master['solver']}:{master['status']}"
            )

        if delta_pred < 0.0:
            history["stop_reason"] = "solver_accuracy_limit"
            break

        if delta_pred <= tol_pred:
            history["stop_reason"] = (
                "predicted_decrease"
                if master["status"] == "optimal"
                else "solver_accuracy_limit"
            )
            break

        if delta_act >= gamma * delta_pred:
            x_c = x_bar
            F_c = F_bar
            history["serious_steps"] += 1
        else:
            history["null_steps"] += 1

        bundle.append({"x": x_bar.copy(), "g": g_bar, "grad": grad_bar.copy()})
        bundle = _prune_bundle(bundle, bundle_max, x_c)

        history["iteration"].append(it + 1)
        history["F_center"].append(F_c)
        history["bundle_size"].append(len(bundle))
        history["time"].append(perf_counter() - start_time)
        history["x_center"].append(x_c.copy())

    # Bundle solvers may leave tiny coefficients instead of exact zeros.
    weight_mask = np.abs(x_c[:-1]) < zero_tol
    x_c[:-1][weight_mask] = 0.0

    return x_c, history
