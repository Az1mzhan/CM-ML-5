import csv
from pathlib import Path
from time import perf_counter

import cvxpy as cp
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from a1_heavyball import heavy_ball_prox_l1_logreg
from a2_bundle import proximal_bundle_l1_logreg
from model import objective_F, stationarity_residual

RESULTS_DIR = Path("results")
SPARSITY_TOL = 1e-6
LAM = 1e-2

A1_ALPHA_GRID = [0.2, 0.5, 1.0, 1.5]
A1_BETA_GRID = [0.0, 0.5, 0.9]

A2_RHO_GRID = [0.05, 0.1, 0.3, 1.0]
A2_BUNDLE_MAX_GRID = [5, 10, 20, 40]
A2_GAMMA = 0.2


def relative_gap(value, ref_value):
    """Return the relative gap with respect to the reference objective."""
    scale = max(abs(ref_value), 1e-16)
    return abs(value - ref_value) / scale


def make_problem_instance():
    """Generate one standardized convex optimization instance."""
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=0,
        random_state=0,
    )
    X = StandardScaler().fit_transform(X)
    return X, y


def solve_reference_problem(X, y, lam):
    """Solve the convex reference problem for gap comparisons."""
    _, d = X.shape
    w = cp.Variable(d)
    b = cp.Variable()

    logits = X @ w + b
    logistic_loss = cp.sum(cp.logistic(logits) - cp.multiply(y, logits)) / X.shape[0]
    objective = cp.Minimize(logistic_loss + lam * cp.norm1(w))
    prob = cp.Problem(objective)

    start = perf_counter()
    prob.solve(solver=cp.SCS, eps=1e-7, max_iters=20000, verbose=False)
    elapsed = perf_counter() - start

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Reference problem not solved properly: status={prob.status}")

    return np.asarray(w.value).ravel(), float(b.value), float(prob.value), elapsed


def evaluate_solution(X, y, w, b, lam, ref_obj):
    """Compute the optimization metrics used in the report."""
    objective = objective_F(X, y, w, b, lam=lam)
    return {
        "objective": objective,
        "relative_gap": relative_gap(objective, ref_obj),
        "stationarity_residual": stationarity_residual(X, y, w, b, lam=lam, zero_tol=SPARSITY_TOL),
    }


def run_a1_config(X, y, lam, ref_obj, alpha, beta, max_iter=1000, tol=1e-6):
    """Run one heavy-ball configuration and return metrics plus history."""
    start = perf_counter()
    w, b, history = heavy_ball_prox_l1_logreg(
        X,
        y,
        lam=lam,
        alpha=alpha,
        beta=beta,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )
    elapsed = perf_counter() - start

    row = {
        "alpha": alpha,
        "beta": beta,
        "iterations": len(history["F"]),
        "time_seconds": elapsed,
    }
    row.update(evaluate_solution(X, y, w, b, lam, ref_obj))
    return row, history, (w, b)


def run_a2_config(
    X,
    y,
    lam,
    ref_obj,
    rho,
    bundle_max,
    gamma=A2_GAMMA,
    max_iter=100,
    tol_pred=1e-6,
    x0=None,
):
    """Run one proximal-bundle configuration and return metrics plus history."""
    start = perf_counter()
    x_star, history = proximal_bundle_l1_logreg(
        X,
        y,
        lam=lam,
        rho=rho,
        gamma=gamma,
        max_iter=max_iter,
        bundle_max=bundle_max,
        tol_pred=tol_pred,
        verbose=False,
        x0=x0,
    )
    elapsed = perf_counter() - start

    w = x_star[:-1]
    b = x_star[-1]
    row = {
        "rho": rho,
        "gamma": gamma,
        "bundle_max": bundle_max,
        "iterations": history["attempted_iterations"],
        "time_seconds": elapsed,
    }
    row.update(evaluate_solution(X, y, w, b, lam, ref_obj))
    return row, history, x_star


def build_history_rows(history_key, iterate_key, time_key, history, X, y, lam, ref_obj):
    """Convert stored iterates into report-ready history rows."""
    objectives = history[history_key]
    iterates = history[iterate_key]
    times = history[time_key]
    rows = []

    for iteration, (objective, x_value, elapsed) in enumerate(zip(objectives, iterates, times)):
        w = x_value[:-1]
        b = x_value[-1]
        rows.append(
            {
                "iteration": iteration,
                "time_seconds": elapsed,
                "objective": objective,
                "relative_gap": relative_gap(objective, ref_obj),
                "stationarity_residual": stationarity_residual(
                    X, y, w, b, lam=lam, zero_tol=SPARSITY_TOL
                ),
            }
        )

    return rows


def save_csv(path, headers, rows):
    """Save a list of dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def save_history_csv(path, rows):
    """Save iteration-wise metrics for plotting and tables."""
    headers = ["iteration", "time_seconds", "objective", "relative_gap", "stationarity_residual"]
    save_csv(path, headers, rows)


def choose_best_configuration(rows, gap_tol=1e-6):
    """
    Choose the fastest configuration among those that reach a target gap.

    If no row reaches the requested gap, choose the smallest gap and then the
    shortest runtime.
    """
    admissible = [row for row in rows if row["relative_gap"] <= gap_tol]
    if admissible:
        return min(admissible, key=lambda row: (row["time_seconds"], row["iterations"]))
    return min(rows, key=lambda row: (row["relative_gap"], row["time_seconds"]))


def format_config(parts):
    """Format a compact configuration label for summary tables."""
    return ", ".join(f"{key}={value}" for key, value in parts.items())


def print_parameter_table(title, rows, config_headers):
    """Print a compact parameter-sweep table."""
    metric_headers = ["relative_gap", "stationarity_residual", "iterations", "time_seconds"]
    headers = config_headers + metric_headers

    print(f"\n=== {title} ===")
    print(" | ".join(header.ljust(14) for header in headers))
    print("-" * (17 * len(headers)))
    for row in rows:
        formatted = []
        for header in headers:
            value = row[header]
            if header in {"relative_gap", "stationarity_residual"}:
                formatted.append(f"{value:.2e}".ljust(14))
            elif header == "time_seconds":
                formatted.append(f"{value:.4f}".ljust(14))
            else:
                formatted.append(str(value).ljust(14))
        print(" | ".join(formatted))


def print_summary_table(rows, ref_obj, ref_time):
    """Print the final same-start summary."""
    print("\n=== Reference solution (CVXPY) ===")
    print(f"Objective: {ref_obj:.12f}")
    print(f"Solve time: {ref_time:.6f} s")

    headers = ["method", "configuration", "relative_gap", "stationarity_residual", "iterations", "time_seconds"]

    print("\n=== Same-start summary ===")
    print(" | ".join(header.ljust(24 if header == "configuration" else 14) for header in headers))
    print("-" * 108)
    for row in rows:
        formatted = [
            str(row["method"]).ljust(14),
            str(row["configuration"]).ljust(24),
            f"{row['relative_gap']:.2e}".ljust(14),
            f"{row['stationarity_residual']:.2e}".ljust(14),
            str(row["iterations"]).ljust(14),
            f"{row['time_seconds']:.4f}".ljust(14),
        ]
        print(" | ".join(formatted))


def main():
    X, y = make_problem_instance()

    ref_w, ref_b, ref_obj, ref_time = solve_reference_problem(X, y, LAM)

    a1_rows = []
    a1_artifacts = {}
    for alpha in A1_ALPHA_GRID:
        for beta in A1_BETA_GRID:
            row, history, solution = run_a1_config(X, y, LAM, ref_obj, alpha=alpha, beta=beta)
            a1_rows.append(row)
            a1_artifacts[(alpha, beta)] = {"history": history, "solution": solution, "row": row}

    a2_rows = []
    a2_artifacts = {}
    for rho in A2_RHO_GRID:
        for bundle_max in A2_BUNDLE_MAX_GRID:
            row, history, x_star = run_a2_config(
                X,
                y,
                LAM,
                ref_obj,
                rho=rho,
                bundle_max=bundle_max,
                gamma=A2_GAMMA,
                x0=np.zeros(X.shape[1] + 1),
            )
            a2_rows.append(row)
            a2_artifacts[(rho, bundle_max)] = {"history": history, "solution": x_star, "row": row}

    best_a1 = choose_best_configuration(a1_rows)
    best_a2 = choose_best_configuration(a2_rows)

    best_a1_history = a1_artifacts[(best_a1["alpha"], best_a1["beta"])]["history"]
    best_a1_solution = a1_artifacts[(best_a1["alpha"], best_a1["beta"])]["solution"]

    best_a2_history = a2_artifacts[(best_a2["rho"], best_a2["bundle_max"])]["history"]
    best_a2_solution = a2_artifacts[(best_a2["rho"], best_a2["bundle_max"])]["solution"]

    warm_start_x0 = np.concatenate([best_a1_solution[0], np.array([best_a1_solution[1]])])
    warm_row, warm_history, _ = run_a2_config(
        X,
        y,
        LAM,
        ref_obj,
        rho=best_a2["rho"],
        bundle_max=best_a2["bundle_max"],
        gamma=best_a2["gamma"],
        x0=warm_start_x0,
    )

    summary_rows = [
        {
            "method": "A1",
            "configuration": format_config({"alpha": best_a1["alpha"], "beta": best_a1["beta"]}),
            "objective": best_a1["objective"],
            "relative_gap": best_a1["relative_gap"],
            "stationarity_residual": best_a1["stationarity_residual"],
            "iterations": best_a1["iterations"],
            "time_seconds": best_a1["time_seconds"],
            "notes": "same-start comparison",
        },
        {
            "method": "A2",
            "configuration": format_config(
                {"rho": best_a2["rho"], "gamma": best_a2["gamma"], "bundle_max": best_a2["bundle_max"]}
            ),
            "objective": best_a2["objective"],
            "relative_gap": best_a2["relative_gap"],
            "stationarity_residual": best_a2["stationarity_residual"],
            "iterations": best_a2["iterations"],
            "time_seconds": best_a2["time_seconds"],
            "notes": "same-start comparison",
        },
        {
            "method": "A2 warm start",
            "configuration": format_config(
                {"rho": warm_row["rho"], "gamma": warm_row["gamma"], "bundle_max": warm_row["bundle_max"]}
            ),
            "objective": warm_row["objective"],
            "relative_gap": warm_row["relative_gap"],
            "stationarity_residual": warm_row["stationarity_residual"],
            "iterations": warm_row["iterations"],
            "time_seconds": warm_row["time_seconds"],
            "notes": "auxiliary run from best A1 solution",
        },
        {
            "method": "CVXPY ref",
            "configuration": "SCS reference solve",
            "objective": ref_obj,
            "relative_gap": relative_gap(ref_obj, ref_obj),
            "stationarity_residual": stationarity_residual(X, y, ref_w, ref_b, lam=LAM, zero_tol=SPARSITY_TOL),
            "iterations": 1,
            "time_seconds": ref_time,
            "notes": "reference objective value",
        },
    ]

    a1_headers = ["alpha", "beta", "objective", "relative_gap", "stationarity_residual", "iterations", "time_seconds"]
    a2_headers = [
        "rho",
        "gamma",
        "bundle_max",
        "objective",
        "relative_gap",
        "stationarity_residual",
        "iterations",
        "time_seconds",
    ]
    summary_headers = [
        "method",
        "configuration",
        "objective",
        "relative_gap",
        "stationarity_residual",
        "iterations",
        "time_seconds",
        "notes",
    ]

    save_csv(RESULTS_DIR / "a1_parameter_sweep.csv", a1_headers, a1_rows)
    save_csv(RESULTS_DIR / "a2_parameter_sweep.csv", a2_headers, a2_rows)
    save_csv(RESULTS_DIR / "same_start_summary.csv", summary_headers, summary_rows)

    save_history_csv(
        RESULTS_DIR / "best_a1_history.csv",
        build_history_rows("F", "x", "time", best_a1_history, X, y, LAM, ref_obj),
    )
    save_history_csv(
        RESULTS_DIR / "best_a2_history.csv",
        build_history_rows("F_center", "x_center", "time", best_a2_history, X, y, LAM, ref_obj),
    )
    save_history_csv(
        RESULTS_DIR / "best_a2_warm_start_history.csv",
        build_history_rows("F_center", "x_center", "time", warm_history, X, y, LAM, ref_obj),
    )

    print_parameter_table(
        "A1 parameter sweep",
        sorted(a1_rows, key=lambda row: (row["time_seconds"], row["relative_gap"])),
        ["alpha", "beta"],
    )
    print_parameter_table(
        "A2 parameter sweep",
        sorted(a2_rows, key=lambda row: (row["time_seconds"], row["relative_gap"])),
        ["rho", "gamma", "bundle_max"],
    )
    print_summary_table(summary_rows, ref_obj, ref_time)

    print("\nSaved optimization-study CSVs to:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
