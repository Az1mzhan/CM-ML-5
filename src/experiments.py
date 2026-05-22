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
from model import objective_F, stationarity_residual

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

SPARSITY_TOL = 1e-6
LAM = 1e-2
SEEDS = [0, 1, 2, 3, 4]

A1_ALPHA_GRID = [0.2, 0.5, 1.0, 1.5]
A1_BETA_GRID = [0.0, 0.5, 0.9]

A2_RHO_GRID = [0.05, 0.1, 0.3, 1.0]
A2_BUNDLE_MAX_GRID = [5, 10, 20, 40]
A2_GAMMA = 0.2


def relative_gap(value, ref_value):
    scale = max(abs(ref_value), 1e-16)
    return abs(value - ref_value) / scale


def make_problem_instance(seed):
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=0,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X)
    return X, y


def solve_reference_problem(X, y, lam):
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
    objective = objective_F(X, y, w, b, lam=lam)
    return {
        "objective": objective,
        "relative_gap": relative_gap(objective, ref_obj),
        "stationarity_residual": stationarity_residual(
            X, y, w, b, lam=lam, zero_tol=SPARSITY_TOL
        ),
    }


def run_a1_config(X, y, lam, ref_obj, alpha, beta, max_iter=1000, tol=1e-6):
    start = perf_counter()
    w, b, history = heavy_ball_prox_l1_logreg(
        X, y, lam=lam, alpha=alpha, beta=beta,
        max_iter=max_iter, tol=tol, verbose=False,
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
    X, y, lam, ref_obj, rho, bundle_max,
    gamma=A2_GAMMA, max_iter=100, tol_pred=1e-6, x0=None,
):
    start = perf_counter()
    x_star, history = proximal_bundle_l1_logreg(
        X, y, lam=lam, rho=rho, gamma=gamma,
        max_iter=max_iter, bundle_max=bundle_max,
        tol_pred=tol_pred, verbose=False, x0=x0,
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
    rows = []
    for iteration, (objective, x_value, elapsed) in enumerate(
        zip(history[history_key], history[iterate_key], history[time_key])
    ):
        w = x_value[:-1]
        b = x_value[-1]
        rows.append({
            "iteration": iteration,
            "time_seconds": elapsed,
            "objective": objective,
            "relative_gap": relative_gap(objective, ref_obj),
            "stationarity_residual": stationarity_residual(
                X, y, w, b, lam=lam, zero_tol=SPARSITY_TOL
            ),
        })
    return rows


def save_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_rows = []
    for row in rows:
        cleaned_rows.append({header: row.get(header, "") for header in headers})

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(cleaned_rows)


def choose_best_configuration(rows, gap_tol=1e-6):
    admissible = [row for row in rows if row["relative_gap"] <= gap_tol]
    if admissible:
        return min(admissible, key=lambda r: (r["time_seconds"], r["iterations"]))
    return min(rows, key=lambda r: (r["relative_gap"], r["time_seconds"]))


def aggregate_rows(rows, config_keys):
    grouped = {}

    for row in rows:
        key = tuple(row[k] for k in config_keys)
        grouped.setdefault(key, []).append(row)

    summary = []
    metrics = ["relative_gap", "stationarity_residual", "iterations", "time_seconds"]

    for key, group in grouped.items():
        out = {k: v for k, v in zip(config_keys, key)}
        for metric in metrics:
            values = np.array([g[metric] for g in group], dtype=float)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std())
        summary.append(out)

    return summary


def plot_history(rows_a1, rows_a2, filename_prefix):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, fname in [
        ("relative_gap", "Relative gap", "relative_gap"),
        ("stationarity_residual", "Stationarity residual", "residual"),
    ]:
        plt.figure()
        plt.semilogy(
            [r["iteration"] for r in rows_a1],
            [r[metric] for r in rows_a1],
            label="A1",
        )
        plt.semilogy(
            [r["iteration"] for r in rows_a2],
            [r[metric] for r in rows_a2],
            label="A2",
        )
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{filename_prefix}_{fname}_vs_iteration.png", dpi=300)
        plt.close()

        plt.figure()
        plt.semilogy(
            [r["time_seconds"] for r in rows_a1],
            [r[metric] for r in rows_a1],
            label="A1",
        )
        plt.semilogy(
            [r["time_seconds"] for r in rows_a2],
            [r[metric] for r in rows_a2],
            label="A2",
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{filename_prefix}_{fname}_vs_time.png", dpi=300)
        plt.close()


def main():
    all_a1_rows = []
    all_a2_rows = []
    all_summary_rows = []

    for seed in SEEDS:
        print(f"\n========== Seed {seed} ==========")

        X, y = make_problem_instance(seed)
        ref_w, ref_b, ref_obj, ref_time = solve_reference_problem(X, y, LAM)

        a1_rows = []
        a1_artifacts = {}

        for alpha in A1_ALPHA_GRID:
            for beta in A1_BETA_GRID:
                row, history, solution = run_a1_config(
                    X, y, LAM, ref_obj, alpha=alpha, beta=beta
                )
                row["seed"] = seed
                a1_rows.append(row)
                all_a1_rows.append(row)
                a1_artifacts[(alpha, beta)] = {
                    "history": history,
                    "solution": solution,
                    "row": row,
                }

        a2_rows = []
        a2_artifacts = {}

        for rho in A2_RHO_GRID:
            for bundle_max in A2_BUNDLE_MAX_GRID:
                row, history, x_star = run_a2_config(
                    X, y, LAM, ref_obj,
                    rho=rho,
                    bundle_max=bundle_max,
                    gamma=A2_GAMMA,
                    x0=np.zeros(X.shape[1] + 1),
                )
                row["seed"] = seed
                a2_rows.append(row)
                all_a2_rows.append(row)
                a2_artifacts[(rho, bundle_max)] = {
                    "history": history,
                    "solution": x_star,
                    "row": row,
                }

        best_a1 = choose_best_configuration(a1_rows)
        best_a2 = choose_best_configuration(a2_rows)

        best_a1_history = a1_artifacts[(best_a1["alpha"], best_a1["beta"])]["history"]
        best_a1_solution = a1_artifacts[(best_a1["alpha"], best_a1["beta"])]["solution"]

        best_a2_history = a2_artifacts[(best_a2["rho"], best_a2["bundle_max"])]["history"]

        warm_start_x0 = np.concatenate([best_a1_solution[0], np.array([best_a1_solution[1]])])
        warm_row, warm_history, _ = run_a2_config(
            X, y, LAM, ref_obj,
            rho=best_a2["rho"],
            bundle_max=best_a2["bundle_max"],
            gamma=best_a2["gamma"],
            x0=warm_start_x0,
        )
        warm_row["seed"] = seed

        summary_rows = [
            {
                "seed": seed,
                "method": "A1",
                "configuration": f"alpha={best_a1['alpha']}, beta={best_a1['beta']}",
                **best_a1,
                "notes": "same-start comparison",
            },
            {
                "seed": seed,
                "method": "A2",
                "configuration": (
                    f"rho={best_a2['rho']}, gamma={best_a2['gamma']}, "
                    f"bundle_max={best_a2['bundle_max']}"
                ),
                **best_a2,
                "notes": "same-start comparison",
            },
            {
                "seed": seed,
                "method": "A2 warm start",
                "configuration": (
                    f"rho={warm_row['rho']}, gamma={warm_row['gamma']}, "
                    f"bundle_max={warm_row['bundle_max']}"
                ),
                **warm_row,
                "notes": "auxiliary warm-start run",
            },
            {
                "seed": seed,
                "method": "CVXPY ref",
                "configuration": "SCS reference solve",
                "objective": ref_obj,
                "relative_gap": relative_gap(ref_obj, ref_obj),
                "stationarity_residual": stationarity_residual(
                    X, y, ref_w, ref_b, lam=LAM, zero_tol=SPARSITY_TOL
                ),
                "iterations": 1,
                "time_seconds": ref_time,
                "notes": "reference objective value",
            },
        ]

        all_summary_rows.extend(summary_rows)

        rows_a1_hist = build_history_rows(
            "F", "x", "time", best_a1_history, X, y, LAM, ref_obj
        )
        rows_a2_hist = build_history_rows(
            "F_center", "x_center", "time", best_a2_history, X, y, LAM, ref_obj
        )

        save_csv(
            RESULTS_DIR / f"seed_{seed}_best_a1_history.csv",
            ["iteration", "time_seconds", "objective", "relative_gap", "stationarity_residual"],
            rows_a1_hist,
        )
        save_csv(
            RESULTS_DIR / f"seed_{seed}_best_a2_history.csv",
            ["iteration", "time_seconds", "objective", "relative_gap", "stationarity_residual"],
            rows_a2_hist,
        )

        plot_history(rows_a1_hist, rows_a2_hist, f"seed_{seed}")

    a1_headers = [
        "seed", "alpha", "beta", "objective", "relative_gap",
        "stationarity_residual", "iterations", "time_seconds",
    ]
    a2_headers = [
        "seed", "rho", "gamma", "bundle_max", "objective", "relative_gap",
        "stationarity_residual", "iterations", "time_seconds",
    ]
    summary_headers = [
        "seed",
        "method",
        "configuration",
        "alpha",
        "beta",
        "rho",
        "gamma",
        "bundle_max",
        "objective",
        "relative_gap",
        "stationarity_residual",
        "iterations",
        "time_seconds",
        "notes",
    ]

    save_csv(RESULTS_DIR / "a1_parameter_sweep_all_seeds.csv", a1_headers, all_a1_rows)
    save_csv(RESULTS_DIR / "a2_parameter_sweep_all_seeds.csv", a2_headers, all_a2_rows)
    save_csv(RESULTS_DIR / "same_start_summary_all_seeds.csv", summary_headers, all_summary_rows)

    a1_agg = aggregate_rows(all_a1_rows, ["alpha", "beta"])
    a2_agg = aggregate_rows(all_a2_rows, ["rho", "gamma", "bundle_max"])

    save_csv(
        RESULTS_DIR / "a1_parameter_sweep_mean_std.csv",
        [
            "alpha", "beta",
            "relative_gap_mean", "relative_gap_std",
            "stationarity_residual_mean", "stationarity_residual_std",
            "iterations_mean", "iterations_std",
            "time_seconds_mean", "time_seconds_std",
        ],
        a1_agg,
    )

    save_csv(
        RESULTS_DIR / "a2_parameter_sweep_mean_std.csv",
        [
            "rho", "gamma", "bundle_max",
            "relative_gap_mean", "relative_gap_std",
            "stationarity_residual_mean", "stationarity_residual_std",
            "iterations_mean", "iterations_std",
            "time_seconds_mean", "time_seconds_std",
        ],
        a2_agg,
    )

    print("\nSaved CSV files to:", RESULTS_DIR.resolve())
    print("Saved plots to:", PLOTS_DIR.resolve())


if __name__ == "__main__":
    main()