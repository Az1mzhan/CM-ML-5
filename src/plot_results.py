import csv

from experiments import (
    CONVERGENCE_SEEDS,
    RESULTS_DIR,
    remove_legacy_outputs,
    save_convergence_plots,
    save_lambda_plots,
    save_mean_convergence_plots,
    save_problem_size_plots,
)


def load_csv_rows(path):
    """Load a CSV and convert numeric and Boolean fields."""
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = ""
                elif value in {"True", "False"}:
                    row[key] = value == "True"
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return rows


def require_files(paths):
    """Fail with a useful message when experiments have not been run."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing normalized result files. Run src/experiments.py first:\n"
            f"{formatted}"
        )

def main():
    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history_paths = [
        RESULTS_DIR / f"seed_{seed}_best_{method}_history.csv"
        for seed in CONVERGENCE_SEEDS
        for method in ("a1", "a2")
    ]
    aggregate_paths = [
        RESULTS_DIR / "scaling_N_mean_std.csv",
        RESULTS_DIR / "lambda_sensitivity_mean_std.csv",
    ]
    require_files(history_paths + aggregate_paths)

    a1_histories = [
        load_csv_rows(RESULTS_DIR / f"seed_{seed}_best_a1_history.csv")
        for seed in CONVERGENCE_SEEDS
    ]
    a2_histories = [
        load_csv_rows(RESULTS_DIR / f"seed_{seed}_best_a2_history.csv")
        for seed in CONVERGENCE_SEEDS
    ]
    scaling_rows = load_csv_rows(RESULTS_DIR / "scaling_N_mean_std.csv")
    lambda_rows = load_csv_rows(RESULTS_DIR / "lambda_sensitivity_mean_std.csv")

    remove_legacy_outputs(RESULTS_DIR)
    save_convergence_plots(
        RESULTS_DIR,
        a1_histories[0],
        a2_histories[0],
        prefix="seed_0",
    )
    save_mean_convergence_plots(RESULTS_DIR, a1_histories, a2_histories)
    save_problem_size_plots(RESULTS_DIR, scaling_rows)
    save_lambda_plots(RESULTS_DIR, lambda_rows)

    print(f"Regenerated normalized figures in {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
