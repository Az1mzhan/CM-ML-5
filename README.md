# CM-ML-5

This repository compares two optimization methods for \(L_1\)-regularized
logistic regression:

- **A1:** heavy-ball proximal method
- **A2:** proximal bundle method

The project compares optimization performance, not classification accuracy.
The main comparison uses the time and iteration count needed to reach the same
relative objective gap target, \(10^{-8}\).

## File Structure

```text
src/
  model.py           Logistic loss, gradient, objective, prox/residual tools
  a1_heavyball.py    A1 heavy-ball proximal method
  a2_bundle.py       A2 proximal bundle method
  experiments.py     Full experiment pipeline
  plot_results.py    Regenerate plots from existing CSV files

results/             Generated CSV files and plots
results/figures/     Generated figures used in the report
figures/             Figures prepared for the Overleaf report
```

## Setup

From the project root:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install numpy scikit-learn cvxpy clarabel osqp scs matplotlib
```

On macOS/Linux:

```bash
python3 -m venv env
source env/bin/activate
pip install numpy scikit-learn cvxpy clarabel osqp scs matplotlib
```

## Run Experiments

Run the full pipeline:

```bash
python src/experiments.py
```

This runs:

- A1 parameter sweep
- A2 parameter sweep
- same-start comparison
- A2 warm-start check
- convergence runs over several seeds
- sample-size scaling
- regularization-sensitivity study

The full run can take some time because A2 solves a convex master problem at
each iteration.

## Regenerate Plots Only

If the CSV files already exist and you only want to rebuild the figures:

```bash
python src/plot_results.py
```

## Main Outputs

Important CSV files:

```text
results/a1_parameter_sweep.csv
results/a2_parameter_sweep.csv
results/same_start_summary.csv
results/scaling_N_mean_std.csv
results/lambda_sensitivity_mean_std.csv
```

Important figures:

```text
results/figures/mean_relative_gap_vs_iteration.png
results/figures/mean_relative_gap_vs_time.png
results/figures/mean_pg_residual_vs_iteration.png
results/figures/mean_pg_residual_vs_time.png
results/figures/time_to_target_vs_N.png
results/figures/nnz_vs_lambda.png
results/figures/time_to_target_vs_lambda.png
```

Representative seed figures:

```text
results/figures/seed_0_relative_gap_vs_iteration.png
results/figures/seed_0_relative_gap_vs_time.png
results/figures/seed_0_pg_residual_vs_iteration.png
results/figures/seed_0_pg_residual_vs_time.png
```
