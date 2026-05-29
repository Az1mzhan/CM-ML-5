import csv
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt

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

# Original parameter-sweep grids
A1_ALPHA_GRID = [0.2, 0.5, 1.0, 1.5]
A1_BETA_GRID = [0.0, 0.5, 0.9]

A2_RHO_GRID = [0.05, 0.1, 0.3, 1.0]
A2_BUNDLE_MAX_GRID = [5, 10, 20, 40]
A2_GAMMA = 0.2

# Best configurations from the previous parameter sweeps
BEST_A1_ALPHA = 1.5
BEST_A1_BETA = 0.5
BEST_A2_RHO = 0.1
BEST_A2_BUNDLE_MAX = 5

# Extra experiments suggested by the professor
CONVERGENCE_SEEDS = [0, 1, 2, 3, 4]
EXTRA_SEEDS = [0, 1, 2]
N_SCALING_GRID = [500, 1000, 2000, 5000]
D_FIXED_FOR_N_SCALING = 20

LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
N_FIXED_FOR_LAMBDA = 2000
D_FIXED_FOR_LAMBDA = 20


def relative_gap(value, ref_value):
    """Return the relative gap with respect to the reference objective."""
    scale = max(abs(ref_value), 1e-16)
    return abs(value - ref_value) / scale


def make_problem_instance(seed=0, n_samples=2000, n_features=20):
    """Generate one standardized convex optimization instance."""
    n_informative = min(10, n_features)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        random_state=seed,
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


def count_effective_nnz(w, zero_tol=SPARSITY_TOL):
    """Count coefficients whose magnitude is above a numerical threshold."""
    return int(np.sum(np.abs(w) > zero_tol))


def evaluate_solution(X, y, w, b, lam, ref_obj):
    """Compute the optimization metrics used in the report."""
    objective = objective_F(X, y, w, b, lam=lam)
    return {
        "objective": objective,
        "relative_gap": relative_gap(objective, ref_obj),
        "stationarity_residual": stationarity_residual(
            X, y, w, b, lam=lam, zero_tol=SPARSITY_TOL
        ),
        "nnz": count_effective_nnz(w),
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
    rows = []
    for iteration, (objective, x_value, elapsed) in enumerate(
        zip(history[history_key], history[iterate_key], history[time_key])
    ):
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
                "nnz": count_effective_nnz(w),
            }
        )
    return rows


def save_csv(path, headers, rows):
    """Save a list of dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_rows = []
    for row in rows:
        cleaned_rows.append({header: row.get(header, "") for header in headers})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(cleaned_rows)


def save_history_csv(path, rows):
    """Save iteration-wise metrics for plotting and tables."""
    headers = [
        "iteration",
        "time_seconds",
        "objective",
        "relative_gap",
        "stationarity_residual",
        "nnz",
    ]
    save_csv(path, headers, rows)


def choose_best_configuration(rows, gap_tol=1e-6):
    """Choose the fastest configuration among those that reach a target gap."""
    admissible = [row for row in rows if row["relative_gap"] <= gap_tol]
    if admissible:
        return min(admissible, key=lambda row: (row["time_seconds"], row["iterations"]))
    return min(rows, key=lambda row: (row["relative_gap"], row["time_seconds"]))


def aggregate_rows(rows, group_keys):
    """Aggregate rows by group_keys and compute mean/std for numerical metrics."""
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    metrics = [
        "objective",
        "relative_gap",
        "stationarity_residual",
        "iterations",
        "time_seconds",
        "nnz",
    ]
    aggregated = []
    for key, group in grouped.items():
        out = {name: value for name, value in zip(group_keys, key)}
        for metric in metrics:
            values = np.array([row[metric] for row in group], dtype=float)
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_std"] = float(np.std(values))
        aggregated.append(out)
    return aggregated


def format_config(parts):
    """Format a compact configuration label for summary tables."""
    return ", ".join(f"{key}={value}" for key, value in parts.items())


def print_parameter_table(title, rows, config_headers):
    """Print a compact parameter-sweep table."""
    metric_headers = ["relative_gap", "stationarity_residual", "iterations", "time_seconds", "nnz"]
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

    headers = [
        "method",
        "configuration",
        "relative_gap",
        "stationarity_residual",
        "iterations",
        "time_seconds",
        "nnz",
    ]

    print("\n=== Same-start summary ===")
    print(" | ".join(header.ljust(26 if header == "configuration" else 14) for header in headers))
    print("-" * 124)
    for row in rows:
        formatted = [
            str(row["method"]).ljust(14),
            str(row["configuration"]).ljust(26),
            f"{row['relative_gap']:.2e}".ljust(14),
            f"{row['stationarity_residual']:.2e}".ljust(14),
            str(row["iterations"]).ljust(14),
            f"{row['time_seconds']:.4f}".ljust(14),
            str(row["nnz"]).ljust(14),
        ]
        print(" | ".join(formatted))


def run_original_parameter_study():
    """Run the original parameter sweeps and same-start comparison."""
    print("\n\n================ ORIGINAL PARAMETER STUDY ================")
    X, y = make_problem_instance(seed=0, n_samples=2000, n_features=20)
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

    best_a1_history = a1_artifacts[(best_a1["alpha"], best_a1["beta"])] ["history"]
    best_a1_solution = a1_artifacts[(best_a1["alpha"], best_a1["beta"])] ["solution"]
    best_a2_history = a2_artifacts[(best_a2["rho"], best_a2["bundle_max"])] ["history"]

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
            "nnz": best_a1["nnz"],
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
            "nnz": best_a2["nnz"],
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
            "nnz": warm_row["nnz"],
            "notes": "auxiliary run from best A1 solution",
        },
        {
            "method": "CVXPY ref",
            "configuration": "SCS reference solve",
            "objective": ref_obj,
            "relative_gap": relative_gap(ref_obj, ref_obj),
            "stationarity_residual": stationarity_residual(
                X, y, ref_w, ref_b, lam=LAM, zero_tol=SPARSITY_TOL
            ),
            "iterations": 1,
            "time_seconds": ref_time,
            "nnz": count_effective_nnz(ref_w),
            "notes": "reference objective value",
        },
    ]

    a1_headers = [
        "alpha", "beta", "objective", "relative_gap", "stationarity_residual", "iterations", "time_seconds", "nnz"
    ]
    a2_headers = [
        "rho", "gamma", "bundle_max", "objective", "relative_gap", "stationarity_residual", "iterations", "time_seconds", "nnz"
    ]
    summary_headers = [
        "method", "configuration", "objective", "relative_gap", "stationarity_residual", "iterations", "time_seconds", "nnz", "notes"
    ]

    save_csv(RESULTS_DIR / "a1_parameter_sweep.csv", a1_headers, a1_rows)
    save_csv(RESULTS_DIR / "a2_parameter_sweep.csv", a2_headers, a2_rows)
    save_csv(RESULTS_DIR / "same_start_summary.csv", summary_headers, summary_rows)

    best_a1_history_rows = build_history_rows(
        "F", "x", "time", best_a1_history, X, y, LAM, ref_obj
    )

    best_a2_history_rows = build_history_rows(
        "F_center", "x_center", "time", best_a2_history, X, y, LAM, ref_obj
    )

    best_a2_warm_start_history_rows = build_history_rows(
        "F_center", "x_center", "time", warm_history, X, y, LAM, ref_obj
    )

    save_history_csv(
        RESULTS_DIR / "best_a1_history.csv",
        best_a1_history_rows,
    )

    save_history_csv(
        RESULTS_DIR / "best_a2_history.csv",
        best_a2_history_rows,
    )

    save_history_csv(
        RESULTS_DIR / "best_a2_warm_start_history.csv",
        best_a2_warm_start_history_rows,
    )

    # Save convergence plots for the representative seed used in the original study.
    save_convergence_plots(
        RESULTS_DIR,
        best_a1_history_rows,
        best_a2_history_rows,
    )

    # Also save mean convergence plots over several random seeds.
    a1_mean_histories = []
    a2_mean_histories = []

    for seed in CONVERGENCE_SEEDS:
        X_seed, y_seed = make_problem_instance(seed=seed, n_samples=2000, n_features=20)
        _, _, ref_obj_seed, _ = solve_reference_problem(X_seed, y_seed, LAM)

        _, a1_history_seed, _ = run_a1_config(
            X_seed,
            y_seed,
            LAM,
            ref_obj_seed,
            alpha=best_a1["alpha"],
            beta=best_a1["beta"],
        )

        _, a2_history_seed, _ = run_a2_config(
            X_seed,
            y_seed,
            LAM,
            ref_obj_seed,
            rho=best_a2["rho"],
            bundle_max=best_a2["bundle_max"],
            gamma=best_a2["gamma"],
            x0=np.zeros(X_seed.shape[1] + 1),
        )

        a1_history_rows_seed = build_history_rows(
            "F",
            "x",
            "time",
            a1_history_seed,
            X_seed,
            y_seed,
            LAM,
            ref_obj_seed,
        )

        a2_history_rows_seed = build_history_rows(
            "F_center",
            "x_center",
            "time",
            a2_history_seed,
            X_seed,
            y_seed,
            LAM,
            ref_obj_seed,
        )

        save_history_csv(
            RESULTS_DIR / f"seed_{seed}_best_a1_history.csv",
            a1_history_rows_seed,
        )
        save_history_csv(
            RESULTS_DIR / f"seed_{seed}_best_a2_history.csv",
            a2_history_rows_seed,
        )

        a1_mean_histories.append(a1_history_rows_seed)
        a2_mean_histories.append(a2_history_rows_seed)

    save_mean_convergence_plots(
        RESULTS_DIR,
        a1_mean_histories,
        a2_mean_histories,
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


def run_problem_size_scaling():
    """Extra experiment 1: scale the number of samples N."""
    print("\n\n================ EXTRA EXPERIMENT: PROBLEM-SIZE SCALING ================")
    rows = []

    for seed in EXTRA_SEEDS:
        for n_samples in N_SCALING_GRID:
            print(f"Running N-scaling: seed={seed}, N={n_samples}, d={D_FIXED_FOR_N_SCALING}")
            X, y = make_problem_instance(seed=seed, n_samples=n_samples, n_features=D_FIXED_FOR_N_SCALING)
            ref_w, ref_b, ref_obj, ref_time = solve_reference_problem(X, y, LAM)

            a1_row, _, _ = run_a1_config(
                X, y, LAM, ref_obj, alpha=BEST_A1_ALPHA, beta=BEST_A1_BETA
            )
            a1_row.update(
                {
                    "experiment": "N_scaling",
                    "seed": seed,
                    "N": n_samples,
                    "d": D_FIXED_FOR_N_SCALING,
                    "method": "A1",
                    "configuration": format_config({"alpha": BEST_A1_ALPHA, "beta": BEST_A1_BETA}),
                    "reference_time_seconds": ref_time,
                }
            )
            rows.append(a1_row)

            a2_row, _, _ = run_a2_config(
                X,
                y,
                LAM,
                ref_obj,
                rho=BEST_A2_RHO,
                bundle_max=BEST_A2_BUNDLE_MAX,
                gamma=A2_GAMMA,
                x0=np.zeros(X.shape[1] + 1),
            )
            a2_row.update(
                {
                    "experiment": "N_scaling",
                    "seed": seed,
                    "N": n_samples,
                    "d": D_FIXED_FOR_N_SCALING,
                    "method": "A2",
                    "configuration": format_config(
                        {"rho": BEST_A2_RHO, "gamma": A2_GAMMA, "bundle_max": BEST_A2_BUNDLE_MAX}
                    ),
                    "reference_time_seconds": ref_time,
                }
            )
            rows.append(a2_row)

    headers = [
        "experiment", "seed", "N", "d", "method", "configuration", "objective", "relative_gap",
        "stationarity_residual", "iterations", "time_seconds", "nnz", "reference_time_seconds"
    ]
    save_csv(RESULTS_DIR / "scaling_N_all_seeds.csv", headers, rows)

    agg = aggregate_rows(rows, ["N", "d", "method"])
    agg_headers = [
        "N", "d", "method", "objective_mean", "objective_std", "relative_gap_mean", "relative_gap_std",
        "stationarity_residual_mean", "stationarity_residual_std", "iterations_mean", "iterations_std",
        "time_seconds_mean", "time_seconds_std", "nnz_mean", "nnz_std"
    ]
    save_csv(RESULTS_DIR / "scaling_N_mean_std.csv", agg_headers, agg)
    save_problem_size_plots(RESULTS_DIR, agg)

    print("\nSaved N-scaling results:")
    print(" -", RESULTS_DIR / "scaling_N_all_seeds.csv")
    print(" -", RESULTS_DIR / "scaling_N_mean_std.csv")


def run_lambda_sensitivity():
    """Extra experiment 2: vary the L1 regularization parameter lambda."""
    print("\n\n================ EXTRA EXPERIMENT: LAMBDA SENSITIVITY ================")
    rows = []

    for seed in EXTRA_SEEDS:
        X, y = make_problem_instance(seed=seed, n_samples=N_FIXED_FOR_LAMBDA, n_features=D_FIXED_FOR_LAMBDA)
        for lam in LAMBDA_GRID:
            print(f"Running lambda-sensitivity: seed={seed}, lambda={lam}")
            ref_w, ref_b, ref_obj, ref_time = solve_reference_problem(X, y, lam)

            a1_row, _, _ = run_a1_config(
                X, y, lam, ref_obj, alpha=BEST_A1_ALPHA, beta=BEST_A1_BETA
            )
            a1_row.update(
                {
                    "experiment": "lambda_sensitivity",
                    "seed": seed,
                    "lambda": lam,
                    "N": N_FIXED_FOR_LAMBDA,
                    "d": D_FIXED_FOR_LAMBDA,
                    "method": "A1",
                    "configuration": format_config({"alpha": BEST_A1_ALPHA, "beta": BEST_A1_BETA}),
                    "reference_time_seconds": ref_time,
                }
            )
            rows.append(a1_row)

            a2_row, _, _ = run_a2_config(
                X,
                y,
                lam,
                ref_obj,
                rho=BEST_A2_RHO,
                bundle_max=BEST_A2_BUNDLE_MAX,
                gamma=A2_GAMMA,
                x0=np.zeros(X.shape[1] + 1),
            )
            a2_row.update(
                {
                    "experiment": "lambda_sensitivity",
                    "seed": seed,
                    "lambda": lam,
                    "N": N_FIXED_FOR_LAMBDA,
                    "d": D_FIXED_FOR_LAMBDA,
                    "method": "A2",
                    "configuration": format_config(
                        {"rho": BEST_A2_RHO, "gamma": A2_GAMMA, "bundle_max": BEST_A2_BUNDLE_MAX}
                    ),
                    "reference_time_seconds": ref_time,
                }
            )
            rows.append(a2_row)

    headers = [
        "experiment", "seed", "lambda", "N", "d", "method", "configuration", "objective", "relative_gap",
        "stationarity_residual", "iterations", "time_seconds", "nnz", "reference_time_seconds"
    ]
    save_csv(RESULTS_DIR / "lambda_sensitivity_all_seeds.csv", headers, rows)

    agg = aggregate_rows(rows, ["lambda", "N", "d", "method"])
    agg_headers = [
        "lambda", "N", "d", "method", "objective_mean", "objective_std", "relative_gap_mean", "relative_gap_std",
        "stationarity_residual_mean", "stationarity_residual_std", "iterations_mean", "iterations_std",
        "time_seconds_mean", "time_seconds_std", "nnz_mean", "nnz_std"
    ]
    save_csv(RESULTS_DIR / "lambda_sensitivity_mean_std.csv", agg_headers, agg)
    save_lambda_plots(RESULTS_DIR, agg)

    print("\nSaved lambda-sensitivity results:")
    print(" -", RESULTS_DIR / "lambda_sensitivity_all_seeds.csv")
    print(" -", RESULTS_DIR / "lambda_sensitivity_mean_std.csv")

def plot_metric_vs_axis(a1_rows, a2_rows, metric, axis, ylabel, xlabel, title, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(
        [row[axis] for row in a1_rows],
        [row[metric] for row in a1_rows],
        label="A1",
    )

    plt.plot(
        [row[axis] for row in a2_rows],
        [row[metric] for row in a2_rows],
        label="A2",
    )

    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_convergence_plots(results_dir, a1_history_rows, a2_history_rows):
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_vs_axis(
        a1_history_rows,
        a2_history_rows,
        metric="relative_gap",
        axis="iteration",
        ylabel="Relative gap",
        xlabel="Iteration",
        title="Relative Gap vs Iteration",
        path=figures_dir / "seed_0_relative_gap_vs_iteration.png",
    )

    plot_metric_vs_axis(
        a1_history_rows,
        a2_history_rows,
        metric="relative_gap",
        axis="time_seconds",
        ylabel="Relative gap",
        xlabel="Time (seconds)",
        title="Relative Gap vs Time",
        path=figures_dir / "seed_0_relative_gap_vs_time.png",
    )

    plot_metric_vs_axis(
        a1_history_rows,
        a2_history_rows,
        metric="stationarity_residual",
        axis="iteration",
        ylabel="Stationarity residual",
        xlabel="Iteration",
        title="Stationarity Residual vs Iteration",
        path=figures_dir / "seed_0_residual_vs_iteration.png",
    )

    plot_metric_vs_axis(
        a1_history_rows,
        a2_history_rows,
        metric="stationarity_residual",
        axis="time_seconds",
        ylabel="Stationarity residual",
        xlabel="Time (seconds)",
        title="Stationarity Residual vs Time",
        path=figures_dir / "seed_0_residual_vs_time.png",
    )
    print("\nSaved convergence plots to:", figures_dir.resolve())

def interpolate_history(history_rows, metric, axis, grid):
    """
    Interpolate one history onto a common grid.

    Different seeds may stop after different numbers of iterations or after
    different runtimes, so interpolation is used before averaging.
    """
    x = np.array([row[axis] for row in history_rows], dtype=float)
    y = np.array([row[metric] for row in history_rows], dtype=float)

    if len(x) == 0:
        raise ValueError("Cannot interpolate an empty history.")

    if len(x) == 1:
        return np.full_like(grid, y[0], dtype=float)

    return np.interp(grid, x, y, left=y[0], right=y[-1])


def plot_mean_curve(
    a1_histories,
    a2_histories,
    metric,
    axis,
    xlabel,
    ylabel,
    title,
    path,
    yscale_log=True,
):
    """
    Plot mean convergence curves over several seeds for A1 and A2.

    The shaded bands represent one standard deviation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    all_histories = a1_histories + a2_histories
    if not all_histories:
        raise ValueError("No histories were provided for plotting.")

    if axis == "iteration":
        max_axis = max(max(row[axis] for row in hist) for hist in all_histories)
        grid = np.arange(0, int(max_axis) + 1)
    else:
        max_axis = max(max(row[axis] for row in hist) for hist in all_histories)
        grid = np.linspace(0, max_axis, 200)

    a1_values = np.array([
        interpolate_history(hist, metric, axis, grid)
        for hist in a1_histories
    ])

    a2_values = np.array([
        interpolate_history(hist, metric, axis, grid)
        for hist in a2_histories
    ])

    a1_mean = np.mean(a1_values, axis=0)
    a1_std = np.std(a1_values, axis=0)

    a2_mean = np.mean(a2_values, axis=0)
    a2_std = np.std(a2_values, axis=0)

    plt.figure(figsize=(8, 5))

    plt.plot(grid, a1_mean, label="A1 mean")
    #plt.fill_between(
    #    grid,
    #    np.maximum(a1_mean - a1_std, 1e-16),
    #    a1_mean + a1_std,
     #   alpha=0.2,
    #)

    plt.plot(grid, a2_mean, label="A2 mean")
    #plt.fill_between(
    #    grid,
    #    np.maximum(a2_mean - a2_std, 1e-16),
    #    a2_mean + a2_std,
    #    alpha=0.2,
    #)

    if yscale_log:
        plt.yscale("log")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_mean_convergence_plots(results_dir, a1_histories, a2_histories):
    """
    Save convergence plots averaged over all convergence seeds.
    """
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_mean_curve(
        a1_histories,
        a2_histories,
        metric="relative_gap",
        axis="iteration",
        xlabel="Iteration",
        ylabel="Relative gap",
        title="Mean Relative Gap vs Iteration",
        path=figures_dir / "mean_relative_gap_vs_iteration.png",
        yscale_log=True,
    )

    plot_mean_curve(
        a1_histories,
        a2_histories,
        metric="relative_gap",
        axis="time_seconds",
        xlabel="Time (seconds)",
        ylabel="Relative gap",
        title="Mean Relative Gap vs Time",
        path=figures_dir / "mean_relative_gap_vs_time.png",
        yscale_log=True,
    )

    plot_mean_curve(
        a1_histories,
        a2_histories,
        metric="stationarity_residual",
        axis="iteration",
        xlabel="Iteration",
        ylabel="Stationarity residual",
        title="Mean Stationarity Residual vs Iteration",
        path=figures_dir / "mean_residual_vs_iteration.png",
        yscale_log=True,
    )

    plot_mean_curve(
        a1_histories,
        a2_histories,
        metric="stationarity_residual",
        axis="time_seconds",
        xlabel="Time (seconds)",
        ylabel="Stationarity residual",
        title="Mean Stationarity Residual vs Time",
        path=figures_dir / "mean_residual_vs_time.png",
        yscale_log=True,
    )

    print("\nSaved mean convergence plots to:", figures_dir.resolve())


def save_mean_std_plot(
    rows,
    x_key,
    y_mean_key,
    y_std_key,
    xlabel,
    ylabel,
    title,
    path,
    xscale_log=False,
    yscale_log=False,
):
    """
    Save a mean ± std plot comparing A1 and A2 from aggregated rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    a1_rows = sorted([row for row in rows if row["method"] == "A1"], key=lambda r: r[x_key])
    a2_rows = sorted([row for row in rows if row["method"] == "A2"], key=lambda r: r[x_key])

    plt.figure(figsize=(8, 5))

    if a1_rows:
        x = [row[x_key] for row in a1_rows]
        y = [row[y_mean_key] for row in a1_rows]
        yerr = [row[y_std_key] for row in a1_rows]
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label="A1")

    if a2_rows:
        x = [row[x_key] for row in a2_rows]
        y = [row[y_mean_key] for row in a2_rows]
        yerr = [row[y_std_key] for row in a2_rows]
        plt.errorbar(x, y, yerr=yerr, marker="s", capsize=4, label="A2")

    if xscale_log:
        plt.xscale("log")
    if yscale_log:
        plt.yscale("log")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_problem_size_plots(results_dir, scaling_agg_rows):
    """
    Generate report plot:
    1) Runtime vs N for A1 and A2
    """
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_mean_std_plot(
        rows=scaling_agg_rows,
        x_key="N",
        y_mean_key="time_seconds_mean",
        y_std_key="time_seconds_std",
        xlabel="Number of samples N",
        ylabel="Runtime (seconds)",
        title="Runtime vs N",
        path=figures_dir / "runtime_vs_N.png",
        xscale_log=False,
        yscale_log=False,
    )


def save_lambda_plots(results_dir, lambda_agg_rows):
    """
    Generate report plots:
    2) Number of nonzero weights vs lambda
    3) Stationarity residual vs lambda
    """
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_mean_std_plot(
        rows=lambda_agg_rows,
        x_key="lambda",
        y_mean_key="nnz_mean",
        y_std_key="nnz_std",
        xlabel=r"Regularization parameter $\lambda$",
        ylabel="Number of nonzero weights",
        title=r"Number of Nonzero Weights vs $\lambda$",
        path=figures_dir / "nnz_vs_lambda.png",
        xscale_log=True,
        yscale_log=False,
    )

    save_mean_std_plot(
        rows=lambda_agg_rows,
        x_key="lambda",
        y_mean_key="stationarity_residual_mean",
        y_std_key="stationarity_residual_std",
        xlabel=r"Regularization parameter $\lambda$",
        ylabel="Stationarity residual",
        title=r"Stationarity Residual vs $\lambda$",
        path=figures_dir / "stationarity_residual_vs_lambda.png",
        xscale_log=True,
        yscale_log=True,
    )

    

def main():
    run_original_parameter_study()
    run_problem_size_scaling()
    run_lambda_sensitivity()
    print("\nAll experiment CSVs saved to:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
