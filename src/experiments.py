import csv
from pathlib import Path
from time import perf_counter

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from a1_heavyball import heavy_ball_prox_l1_logreg
from a2_bundle import proximal_bundle_l1_logreg
from model import (
    objective_F,
    prox_gradient_mapping_residual,
    stationarity_residual,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

SPARSITY_TOL = 1e-6
LAM = 1e-2

PRIMARY_GAP_TARGET = 1e-8
GAP_TARGETS = [1e-6, 1e-8]

A1_ALPHA_GRID = [0.2, 0.5, 1.0, 1.5]
A1_BETA_GRID = [0.0, 0.5, 0.9]
A1_MAX_ITER = 3000
A1_TOL = 1e-9

A2_RHO_GRID = [0.05, 0.1, 0.3, 1.0]
A2_BUNDLE_MAX_GRID = [5, 10, 20, 40]
A2_GAMMA = 0.2
A2_MAX_ITER = 300
A2_TOL_PRED = 1e-10
A2_SOLVER_TOL = 1e-10

CONVERGENCE_SEEDS = [0, 1, 2, 3, 4]
EXTRA_SEEDS = [0, 1, 2]
N_SCALING_GRID = [500, 1000, 2000, 5000]
D_FIXED_FOR_N_SCALING = 20

LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
N_FIXED_FOR_LAMBDA = 2000
D_FIXED_FOR_LAMBDA = 20

LEGACY_OUTPUT_FILES = (
    "summary.csv",
    "heavy_ball_history.csv",
    "heavy_ball_iterates.csv",
    "bundle_from_hb_history.csv",
    "bundle_random_history.csv",
    "figures/gap_vs_iteration.png",
    "figures/gap_vs_time.png",
    "figures/hb_contour_slice.png",
    "figures/hb_surface_slice.png",
    "figures/relative_gap_vs_iteration.png",
    "figures/relative_gap_vs_time.png",
    "figures/residual_vs_iteration.png",
    "figures/runtime_vs_N.png",
    "figures/stationarity_residual_vs_lambda.png",
    "figures/mean_residual_vs_iteration.png",
    "figures/mean_residual_vs_time.png",
    "figures/seed_0_residual_vs_iteration.png",
    "figures/seed_0_residual_vs_time.png",
)

METHOD_STYLES = {
    "A1": {
        "color": "tab:blue",
        "marker": "o",
        "linestyle": "-",
        "fillstyle": "full",
    },
    "A2": {
        "color": "tab:orange",
        "marker": "s",
        "linestyle": "--",
        "fillstyle": "none",
    },
}


def remove_legacy_outputs(results_dir=RESULTS_DIR):
    """Remove generated outputs from the superseded plotting protocol."""
    for relative_path in LEGACY_OUTPUT_FILES:
        path = results_dir / relative_path
        if path.exists():
            path.unlink()


def relative_gap(value, ref_value):
    """Return the relative objective gap with respect to the reference value."""
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


def _reference_solve_options(solver_name):
    if solver_name == "CLARABEL":
        return {
            "tol_gap_abs": 1e-11,
            "tol_gap_rel": 1e-11,
            "tol_feas": 1e-11,
            "max_iter": 1000,
            "verbose": False,
        }
    if solver_name == "SCS":
        return {
            "eps": 1e-10,
            "max_iters": 200000,
            "verbose": False,
        }
    return {"verbose": False}


def solve_reference_problem(X, y, lam):
    """Solve the convex reference problem with tight tolerances."""
    _, d = X.shape
    w = cp.Variable(d)
    b = cp.Variable()

    logits = X @ w + b
    logistic_loss = cp.sum(cp.logistic(logits) - cp.multiply(y, logits)) / X.shape[0]
    problem = cp.Problem(cp.Minimize(logistic_loss + lam * cp.norm1(w)))

    last_error = None
    start = perf_counter()
    for solver_name in ("CLARABEL", "SCS"):
        try:
            problem.solve(
                solver=getattr(cp, solver_name),
                **_reference_solve_options(solver_name),
            )
        except Exception as exc:  # pragma: no cover - solver fallback diagnostics
            last_error = exc
            continue

        if problem.status in ["optimal", "optimal_inaccurate"]:
            elapsed = perf_counter() - start
            ref_w = np.asarray(w.value).ravel()
            ref_b = float(b.value)
            ref_obj = float(objective_F(X, y, ref_w, ref_b, lam))
            return {
                "w": ref_w,
                "b": ref_b,
                "objective": ref_obj,
                "time_seconds": elapsed,
                "solver": solver_name,
                "status": problem.status,
                "prox_gradient_residual": prox_gradient_mapping_residual(
                    X, y, ref_w, ref_b, lam
                ),
                "stationarity_residual": stationarity_residual(
                    X, y, ref_w, ref_b, lam=lam, zero_tol=SPARSITY_TOL
                ),
            }

    raise RuntimeError(f"Reference problem was not solved. Last error: {last_error}")


def count_effective_nnz(w, zero_tol=SPARSITY_TOL):
    """Count coefficients whose magnitude is above a numerical threshold."""
    return int(np.sum(np.abs(w) > zero_tol))


def evaluate_solution(X, y, w, b, lam, ref_obj):
    """Compute the optimization metrics used in the report."""
    objective = float(objective_F(X, y, w, b, lam=lam))
    return {
        "objective": objective,
        "relative_gap": relative_gap(objective, ref_obj),
        "prox_gradient_residual": prox_gradient_mapping_residual(X, y, w, b, lam=lam),
        "stationarity_residual": stationarity_residual(
            X, y, w, b, lam=lam, zero_tol=SPARSITY_TOL
        ),
        "nnz": count_effective_nnz(w),
    }


def build_history_rows(history_key, iterate_key, time_key, history, X, y, lam, ref_obj):
    """Convert stored iterates into report-ready history rows."""
    rows = []
    iterations = history.get("iteration")
    if iterations is None:
        iterations = range(len(history[history_key]))

    for iteration, objective, x_value, elapsed in zip(
        iterations,
        history[history_key],
        history[iterate_key],
        history[time_key],
    ):
        w = x_value[:-1]
        b = x_value[-1]
        metrics = evaluate_solution(X, y, w, b, lam, ref_obj)
        rows.append(
            {
                "iteration": int(iteration),
                "time_seconds": float(elapsed),
                **metrics,
            }
        )
    return rows


def first_target_hit(history_rows, target_gap):
    """Return the first history row reaching the target gap, if any."""
    for row in history_rows:
        if row["relative_gap"] <= target_gap:
            return row
    return None


def add_target_metrics(row, history_rows, target_gap=PRIMARY_GAP_TARGET):
    """Attach time-to-target information to a final-result row."""
    hit = first_target_hit(history_rows, target_gap)
    row["target_gap"] = target_gap
    row["reached_target"] = hit is not None
    if hit is None:
        row["iterations_to_target"] = ""
        row["time_to_target"] = ""
        row["gap_at_target"] = ""
    else:
        row["iterations_to_target"] = hit["iteration"]
        row["time_to_target"] = hit["time_seconds"]
        row["gap_at_target"] = hit["relative_gap"]
    return row


def run_a1_config(X, y, lam, ref_obj, alpha, beta, max_iter=A1_MAX_ITER, tol=A1_TOL):
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
    history_rows = build_history_rows("F", "x", "time", history, X, y, lam, ref_obj)

    row = {
        "alpha": alpha,
        "beta": beta,
        "iterations": len(history["F"]),
        "time_seconds": elapsed,
        "stop_reason": history["stop_reason"],
    }
    row.update(evaluate_solution(X, y, w, b, lam, ref_obj))
    add_target_metrics(row, history_rows)
    return row, history, history_rows, (w, b)


def run_a2_config(
    X,
    y,
    lam,
    ref_obj,
    rho,
    bundle_max,
    gamma=A2_GAMMA,
    max_iter=A2_MAX_ITER,
    tol_pred=A2_TOL_PRED,
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
        solver_tol=A2_SOLVER_TOL,
        verbose=False,
        x0=x0,
    )
    elapsed = perf_counter() - start

    w = x_star[:-1]
    b = x_star[-1]
    history_rows = build_history_rows(
        "F_center", "x_center", "time", history, X, y, lam, ref_obj
    )

    row = {
        "rho": rho,
        "gamma": gamma,
        "bundle_max": bundle_max,
        "iterations": history["attempted_iterations"],
        "serious_steps": history["serious_steps"],
        "null_steps": history["null_steps"],
        "time_seconds": elapsed,
        "stop_reason": history["stop_reason"],
        "last_delta_pred": history["delta_pred"][-1] if history["delta_pred"] else "",
        "last_delta_act": history["delta_act"][-1] if history["delta_act"] else "",
        "last_solver": history["solver_name"][-1] if history["solver_name"] else "",
        "last_solver_status": history["solver_status"][-1] if history["solver_status"] else "",
    }
    row.update(evaluate_solution(X, y, w, b, lam, ref_obj))
    add_target_metrics(row, history_rows)
    return row, history, history_rows, x_star


def save_csv(path, headers, rows):
    """Save a list of dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def save_history_csv(path, rows):
    """Save iteration-wise metrics for plotting and tables."""
    headers = [
        "iteration",
        "time_seconds",
        "objective",
        "relative_gap",
        "prox_gradient_residual",
        "stationarity_residual",
        "nnz",
    ]
    save_csv(path, headers, rows)


def choose_best_configuration(rows, target_gap=PRIMARY_GAP_TARGET):
    """Choose the fastest configuration that reaches the shared target gap."""
    admissible = [row for row in rows if row["reached_target"]]
    if admissible:
        return min(
            admissible,
            key=lambda row: (float(row["time_to_target"]), row["iterations_to_target"]),
        )
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
        "prox_gradient_residual",
        "stationarity_residual",
        "iterations",
        "time_seconds",
        "nnz",
        "time_to_target",
        "iterations_to_target",
    ]

    aggregated = []
    for key, group in grouped.items():
        out = {name: value for name, value in zip(group_keys, key)}
        out["target_gap"] = PRIMARY_GAP_TARGET
        out["target_success_count"] = sum(bool(row["reached_target"]) for row in group)
        out["runs"] = len(group)
        for metric in metrics:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric, "") != ""
            ]
            if values:
                array = np.asarray(values, dtype=float)
                out[f"{metric}_mean"] = float(np.mean(array))
                out[f"{metric}_std"] = float(np.std(array))
            else:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
        aggregated.append(out)
    return aggregated


def format_config(parts):
    """Format a compact configuration label for summary tables."""
    return ", ".join(f"{key}={value}" for key, value in parts.items())


def print_parameter_table(title, rows, config_headers):
    """Print a compact parameter-sweep table."""
    metric_headers = [
        "relative_gap",
        "prox_gradient_residual",
        "iterations",
        "time_seconds",
        "reached_target",
        "time_to_target",
    ]
    headers = config_headers + metric_headers

    print(f"\n=== {title} ===")
    print(" | ".join(header.ljust(18) for header in headers))
    print("-" * (21 * len(headers)))
    for row in rows:
        formatted = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                if header in {"relative_gap", "prox_gradient_residual"}:
                    formatted.append(f"{value:.2e}".ljust(18))
                else:
                    formatted.append(f"{value:.4f}".ljust(18))
            else:
                formatted.append(str(value).ljust(18))
        print(" | ".join(formatted))


def print_summary_table(rows, ref):
    """Print the final same-start summary."""
    print("\n=== Reference solution ===")
    print(f"Objective: {ref['objective']:.12f}")
    print(f"Solver: {ref['solver']} ({ref['status']})")
    print(f"Solve time: {ref['time_seconds']:.6f} s")
    print(f"PG residual: {ref['prox_gradient_residual']:.3e}")

    headers = [
        "method",
        "configuration",
        "relative_gap",
        "prox_gradient_residual",
        "iterations",
        "time_seconds",
        "reached_target",
        "time_to_target",
    ]

    print("\n=== Same-start summary ===")
    print(" | ".join(header.ljust(26 if header == "configuration" else 18) for header in headers))
    print("-" * 158)
    for row in rows:
        formatted = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                if header in {"relative_gap", "prox_gradient_residual"}:
                    formatted.append(f"{value:.2e}".ljust(18))
                else:
                    formatted.append(f"{value:.4f}".ljust(18))
            else:
                formatted.append(str(value).ljust(26 if header == "configuration" else 18))
        print(" | ".join(formatted))


def plot_metric_vs_axis(a1_rows, a2_rows, metric, axis, ylabel, xlabel, title, path):
    """Plot one convergence metric against iteration or time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))

    for label, rows in {"A1": a1_rows, "A2": a2_rows}.items():
        x = [row[axis] for row in rows]
        y = [max(row[metric], 1e-16) for row in rows]
        style = METHOD_STYLES[label]
        plt.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            label=label,
        )

    plt.yscale("log")
    if metric == "relative_gap":
        plt.axhline(
            PRIMARY_GAP_TARGET,
            color="black",
            linestyle=":",
            linewidth=1.25,
            label=f"target {PRIMARY_GAP_TARGET:g}",
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_convergence_plots(results_dir, a1_history_rows, a2_history_rows, prefix="seed_0"):
    """Save representative convergence plots."""
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("relative_gap", "Relative gap", "relative_gap"),
        ("prox_gradient_residual", "Prox-gradient residual", "pg_residual"),
    ]
    axes = [
        ("iteration", "Iteration", "iteration"),
        ("time_seconds", "Time (seconds)", "time"),
    ]

    for metric, ylabel, metric_slug in plot_specs:
        for axis, xlabel, axis_slug in axes:
            plot_metric_vs_axis(
                a1_history_rows,
                a2_history_rows,
                metric=metric,
                axis=axis,
                ylabel=ylabel,
                xlabel=xlabel,
                title=f"{ylabel} vs {xlabel}",
                path=figures_dir / f"{prefix}_{metric_slug}_vs_{axis_slug}.png",
            )


def interpolate_history(history_rows, metric, axis, grid):
    """
    Interpolate one history onto a common grid.

    Values after a run terminates are left missing instead of being padded with
    the final value, so mean curves do not create artificial stalling plateaus.
    """
    x = np.asarray([row[axis] for row in history_rows], dtype=float)
    y = np.asarray([row[metric] for row in history_rows], dtype=float)

    if len(x) == 0:
        raise ValueError("Cannot interpolate an empty history.")
    if len(x) == 1:
        out = np.full_like(grid, np.nan, dtype=float)
        out[grid == x[0]] = y[0]
        return out

    return np.interp(grid, x, y, left=y[0], right=np.nan)


def plot_mean_curve(
    a1_histories,
    a2_histories,
    metric,
    axis,
    xlabel,
    ylabel,
    title,
    path,
):
    """Plot mean curves on each method's all-seeds common support."""
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for method, histories in {"A1": a1_histories, "A2": a2_histories}.items():
        common_end = min(max(row[axis] for row in hist) for hist in histories)
        if axis == "iteration":
            grid = np.arange(0, int(common_end) + 1)
        else:
            grid = np.linspace(0, common_end, 200)

        values = np.asarray([
            interpolate_history(hist, metric, axis, grid)
            for hist in histories
        ])
        mean = np.mean(values, axis=0)
        style = METHOD_STYLES[method]
        plt.plot(
            grid,
            np.maximum(mean, 1e-16),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            label=f"{method} mean",
        )

    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_mean_convergence_plots(results_dir, a1_histories, a2_histories):
    """Save convergence plots averaged over all convergence seeds."""
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("relative_gap", "Relative gap", "mean_relative_gap"),
        ("prox_gradient_residual", "Prox-gradient residual", "mean_pg_residual"),
    ]
    axes = [
        ("iteration", "Iteration", "iteration"),
        ("time_seconds", "Time (seconds)", "time"),
    ]

    for metric, ylabel, metric_slug in plot_specs:
        for axis, xlabel, axis_slug in axes:
            plot_mean_curve(
                a1_histories,
                a2_histories,
                metric=metric,
                axis=axis,
                xlabel=xlabel,
                ylabel=ylabel,
                title=f"Mean {ylabel} vs {xlabel} (all seeds active)",
                path=figures_dir / f"{metric_slug}_vs_{axis_slug}.png",
            )


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
    """Save a mean +/- std plot comparing A1 and A2 from aggregated rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))

    for method in ("A1", "A2"):
        method_rows = sorted([row for row in rows if row["method"] == method], key=lambda r: r[x_key])
        method_rows = [row for row in method_rows if row[y_mean_key] != ""]
        if not method_rows:
            continue
        x = [row[x_key] for row in method_rows]
        y = [row[y_mean_key] for row in method_rows]
        yerr = [row[y_std_key] for row in method_rows]
        style = METHOD_STYLES[method]
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            fillstyle=style["fillstyle"],
            linewidth=2,
            capsize=4,
            zorder=3 if method == "A2" else 2,
            label=method,
        )

    if xscale_log:
        plt.xscale("log")
    if yscale_log:
        plt.yscale("log")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_problem_size_plots(results_dir, scaling_agg_rows):
    """Generate scaling plots for comparable target accuracy."""
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_mean_std_plot(
        rows=scaling_agg_rows,
        x_key="N",
        y_mean_key="time_to_target_mean",
        y_std_key="time_to_target_std",
        xlabel="Number of samples N",
        ylabel=f"Time to relative gap <= {PRIMARY_GAP_TARGET:g} (s)",
        title="Time to Target Gap vs N",
        path=figures_dir / "time_to_target_vs_N.png",
        yscale_log=True,
    )


def save_lambda_plots(results_dir, lambda_agg_rows):
    """Generate regularization-sensitivity plots."""
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_mean_std_plot(
        rows=lambda_agg_rows,
        x_key="lambda",
        y_mean_key="nnz_mean",
        y_std_key="nnz_std",
        xlabel="Regularization parameter lambda",
        ylabel="Number of nonzero weights",
        title="Number of Nonzero Weights vs Lambda",
        path=figures_dir / "nnz_vs_lambda.png",
        xscale_log=True,
    )
    save_mean_std_plot(
        rows=lambda_agg_rows,
        x_key="lambda",
        y_mean_key="time_to_target_mean",
        y_std_key="time_to_target_std",
        xlabel="Regularization parameter lambda",
        ylabel=f"Time to relative gap <= {PRIMARY_GAP_TARGET:g} (s)",
        title="Time to Target Gap vs Lambda",
        path=figures_dir / "time_to_target_vs_lambda.png",
        xscale_log=True,
        yscale_log=True,
    )


def run_original_parameter_study():
    """Run parameter sweeps and the same-start comparison."""
    print("\n\n================ NORMALIZED PARAMETER STUDY ================")
    X, y = make_problem_instance(seed=0, n_samples=2000, n_features=20)
    ref = solve_reference_problem(X, y, LAM)
    ref_obj = ref["objective"]

    a1_rows = []
    a1_artifacts = {}
    for alpha in A1_ALPHA_GRID:
        for beta in A1_BETA_GRID:
            row, history, history_rows, solution = run_a1_config(
                X, y, LAM, ref_obj, alpha=alpha, beta=beta
            )
            a1_rows.append(row)
            a1_artifacts[(alpha, beta)] = {
                "history": history,
                "history_rows": history_rows,
                "solution": solution,
                "row": row,
            }

    a2_rows = []
    a2_artifacts = {}
    for rho in A2_RHO_GRID:
        for bundle_max in A2_BUNDLE_MAX_GRID:
            row, history, history_rows, x_star = run_a2_config(
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
            a2_artifacts[(rho, bundle_max)] = {
                "history": history,
                "history_rows": history_rows,
                "solution": x_star,
                "row": row,
            }

    best_a1 = choose_best_configuration(a1_rows)
    best_a2 = choose_best_configuration(a2_rows)

    best_a1_artifact = a1_artifacts[(best_a1["alpha"], best_a1["beta"])]
    best_a2_artifact = a2_artifacts[(best_a2["rho"], best_a2["bundle_max"])]

    warm_start_x0 = np.concatenate([
        best_a1_artifact["solution"][0],
        np.array([best_a1_artifact["solution"][1]]),
    ])
    warm_row, warm_history, warm_history_rows, _ = run_a2_config(
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
            "notes": "same-start comparison",
            **best_a1,
        },
        {
            "method": "A2",
            "configuration": format_config(
                {"rho": best_a2["rho"], "gamma": best_a2["gamma"], "bundle_max": best_a2["bundle_max"]}
            ),
            "notes": "same-start comparison",
            **best_a2,
        },
        {
            "method": "A2 warm start",
            "configuration": format_config(
                {"rho": warm_row["rho"], "gamma": warm_row["gamma"], "bundle_max": warm_row["bundle_max"]}
            ),
            "notes": "auxiliary run from best A1 solution",
            **warm_row,
        },
        {
            "method": "Reference",
            "configuration": f"{ref['solver']} reference solve",
            "objective": ref["objective"],
            "relative_gap": 0.0,
            "prox_gradient_residual": ref["prox_gradient_residual"],
            "stationarity_residual": ref["stationarity_residual"],
            "iterations": 1,
            "time_seconds": ref["time_seconds"],
            "nnz": count_effective_nnz(ref["w"]),
            "target_gap": PRIMARY_GAP_TARGET,
            "reached_target": True,
            "iterations_to_target": 1,
            "time_to_target": ref["time_seconds"],
            "gap_at_target": 0.0,
            "stop_reason": ref["status"],
            "notes": "reference objective value",
        },
    ]

    a1_headers = [
        "alpha", "beta", "objective", "relative_gap", "prox_gradient_residual",
        "stationarity_residual", "iterations", "time_seconds", "target_gap",
        "reached_target", "iterations_to_target", "time_to_target", "gap_at_target",
        "nnz", "stop_reason",
    ]
    a2_headers = [
        "rho", "gamma", "bundle_max", "objective", "relative_gap",
        "prox_gradient_residual", "stationarity_residual", "iterations",
        "serious_steps", "null_steps", "time_seconds", "target_gap",
        "reached_target", "iterations_to_target", "time_to_target", "gap_at_target",
        "nnz", "stop_reason", "last_delta_pred", "last_delta_act",
        "last_solver", "last_solver_status",
    ]
    summary_headers = [
        "method", "configuration", "objective", "relative_gap",
        "prox_gradient_residual", "stationarity_residual", "iterations",
        "time_seconds", "target_gap", "reached_target", "iterations_to_target",
        "time_to_target", "gap_at_target", "nnz", "stop_reason", "notes",
    ]

    save_csv(RESULTS_DIR / "a1_parameter_sweep.csv", a1_headers, a1_rows)
    save_csv(RESULTS_DIR / "a2_parameter_sweep.csv", a2_headers, a2_rows)
    save_csv(RESULTS_DIR / "same_start_summary.csv", summary_headers, summary_rows)
    save_history_csv(RESULTS_DIR / "best_a1_history.csv", best_a1_artifact["history_rows"])
    save_history_csv(RESULTS_DIR / "best_a2_history.csv", best_a2_artifact["history_rows"])
    save_history_csv(RESULTS_DIR / "best_a2_warm_start_history.csv", warm_history_rows)

    save_convergence_plots(
        RESULTS_DIR,
        best_a1_artifact["history_rows"],
        best_a2_artifact["history_rows"],
    )

    a1_mean_histories = []
    a2_mean_histories = []
    for seed in CONVERGENCE_SEEDS:
        X_seed, y_seed = make_problem_instance(seed=seed, n_samples=2000, n_features=20)
        ref_seed = solve_reference_problem(X_seed, y_seed, LAM)

        _, _, a1_history_rows_seed, _ = run_a1_config(
            X_seed,
            y_seed,
            LAM,
            ref_seed["objective"],
            alpha=best_a1["alpha"],
            beta=best_a1["beta"],
        )
        _, _, a2_history_rows_seed, _ = run_a2_config(
            X_seed,
            y_seed,
            LAM,
            ref_seed["objective"],
            rho=best_a2["rho"],
            bundle_max=best_a2["bundle_max"],
            gamma=best_a2["gamma"],
            x0=np.zeros(X_seed.shape[1] + 1),
        )

        save_history_csv(RESULTS_DIR / f"seed_{seed}_best_a1_history.csv", a1_history_rows_seed)
        save_history_csv(RESULTS_DIR / f"seed_{seed}_best_a2_history.csv", a2_history_rows_seed)
        a1_mean_histories.append(a1_history_rows_seed)
        a2_mean_histories.append(a2_history_rows_seed)

    save_mean_convergence_plots(RESULTS_DIR, a1_mean_histories, a2_mean_histories)

    print_parameter_table(
        "A1 parameter sweep",
        sorted(a1_rows, key=lambda row: (not row["reached_target"], row.get("time_to_target") or 1e99)),
        ["alpha", "beta"],
    )
    print_parameter_table(
        "A2 parameter sweep",
        sorted(a2_rows, key=lambda row: (not row["reached_target"], row.get("time_to_target") or 1e99)),
        ["rho", "gamma", "bundle_max"],
    )
    print_summary_table(summary_rows, ref)

    return {
        "best_a1": best_a1,
        "best_a2": best_a2,
    }


def run_problem_size_scaling(best_a1, best_a2):
    """Scale the number of samples N using the selected configurations."""
    print("\n\n================ NORMALIZED N-SCALING ================")
    rows = []

    for seed in EXTRA_SEEDS:
        for n_samples in N_SCALING_GRID:
            print(f"Running N-scaling: seed={seed}, N={n_samples}, d={D_FIXED_FOR_N_SCALING}")
            X, y = make_problem_instance(seed=seed, n_samples=n_samples, n_features=D_FIXED_FOR_N_SCALING)
            ref = solve_reference_problem(X, y, LAM)

            a1_row, _, _, _ = run_a1_config(
                X, y, LAM, ref["objective"], alpha=best_a1["alpha"], beta=best_a1["beta"]
            )
            a1_row.update(
                {
                    "experiment": "N_scaling",
                    "seed": seed,
                    "N": n_samples,
                    "d": D_FIXED_FOR_N_SCALING,
                    "method": "A1",
                    "configuration": format_config({"alpha": best_a1["alpha"], "beta": best_a1["beta"]}),
                    "reference_time_seconds": ref["time_seconds"],
                }
            )
            rows.append(a1_row)

            a2_row, _, _, _ = run_a2_config(
                X,
                y,
                LAM,
                ref["objective"],
                rho=best_a2["rho"],
                bundle_max=best_a2["bundle_max"],
                gamma=best_a2["gamma"],
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
                        {"rho": best_a2["rho"], "gamma": best_a2["gamma"], "bundle_max": best_a2["bundle_max"]}
                    ),
                    "reference_time_seconds": ref["time_seconds"],
                }
            )
            rows.append(a2_row)

    headers = [
        "experiment", "seed", "N", "d", "method", "configuration", "objective",
        "relative_gap", "prox_gradient_residual", "stationarity_residual",
        "iterations", "time_seconds", "target_gap", "reached_target",
        "iterations_to_target", "time_to_target", "gap_at_target", "nnz",
        "stop_reason", "reference_time_seconds",
    ]
    save_csv(RESULTS_DIR / "scaling_N_all_seeds.csv", headers, rows)

    agg = aggregate_rows(rows, ["N", "d", "method"])
    agg_headers = [
        "N", "d", "method", "target_gap", "target_success_count", "runs",
        "objective_mean", "objective_std", "relative_gap_mean", "relative_gap_std",
        "prox_gradient_residual_mean", "prox_gradient_residual_std",
        "stationarity_residual_mean", "stationarity_residual_std",
        "iterations_mean", "iterations_std", "time_seconds_mean", "time_seconds_std",
        "time_to_target_mean", "time_to_target_std",
        "iterations_to_target_mean", "iterations_to_target_std",
        "nnz_mean", "nnz_std",
    ]
    save_csv(RESULTS_DIR / "scaling_N_mean_std.csv", agg_headers, agg)
    save_problem_size_plots(RESULTS_DIR, agg)


def run_lambda_sensitivity(best_a1, best_a2):
    """Vary the L1 regularization parameter lambda."""
    print("\n\n================ NORMALIZED LAMBDA SENSITIVITY ================")
    rows = []

    for seed in EXTRA_SEEDS:
        X, y = make_problem_instance(seed=seed, n_samples=N_FIXED_FOR_LAMBDA, n_features=D_FIXED_FOR_LAMBDA)
        for lam in LAMBDA_GRID:
            print(f"Running lambda-sensitivity: seed={seed}, lambda={lam}")
            ref = solve_reference_problem(X, y, lam)

            a1_row, _, _, _ = run_a1_config(
                X, y, lam, ref["objective"], alpha=best_a1["alpha"], beta=best_a1["beta"]
            )
            a1_row.update(
                {
                    "experiment": "lambda_sensitivity",
                    "seed": seed,
                    "lambda": lam,
                    "N": N_FIXED_FOR_LAMBDA,
                    "d": D_FIXED_FOR_LAMBDA,
                    "method": "A1",
                    "configuration": format_config({"alpha": best_a1["alpha"], "beta": best_a1["beta"]}),
                    "reference_time_seconds": ref["time_seconds"],
                }
            )
            rows.append(a1_row)

            a2_row, _, _, _ = run_a2_config(
                X,
                y,
                lam,
                ref["objective"],
                rho=best_a2["rho"],
                bundle_max=best_a2["bundle_max"],
                gamma=best_a2["gamma"],
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
                        {"rho": best_a2["rho"], "gamma": best_a2["gamma"], "bundle_max": best_a2["bundle_max"]}
                    ),
                    "reference_time_seconds": ref["time_seconds"],
                }
            )
            rows.append(a2_row)

    headers = [
        "experiment", "seed", "lambda", "N", "d", "method", "configuration",
        "objective", "relative_gap", "prox_gradient_residual",
        "stationarity_residual", "iterations", "time_seconds", "target_gap",
        "reached_target", "iterations_to_target", "time_to_target",
        "gap_at_target", "nnz", "stop_reason", "reference_time_seconds",
    ]
    save_csv(RESULTS_DIR / "lambda_sensitivity_all_seeds.csv", headers, rows)

    agg = aggregate_rows(rows, ["lambda", "N", "d", "method"])
    agg_headers = [
        "lambda", "N", "d", "method", "target_gap", "target_success_count", "runs",
        "objective_mean", "objective_std", "relative_gap_mean", "relative_gap_std",
        "prox_gradient_residual_mean", "prox_gradient_residual_std",
        "stationarity_residual_mean", "stationarity_residual_std",
        "iterations_mean", "iterations_std", "time_seconds_mean", "time_seconds_std",
        "time_to_target_mean", "time_to_target_std",
        "iterations_to_target_mean", "iterations_to_target_std",
        "nnz_mean", "nnz_std",
    ]
    save_csv(RESULTS_DIR / "lambda_sensitivity_mean_std.csv", agg_headers, agg)
    save_lambda_plots(RESULTS_DIR, agg)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    remove_legacy_outputs(RESULTS_DIR)
    selected = run_original_parameter_study()
    run_problem_size_scaling(selected["best_a1"], selected["best_a2"])
    run_lambda_sensitivity(selected["best_a1"], selected["best_a2"])
    print("\nAll experiment CSVs saved to:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
