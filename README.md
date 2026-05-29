# L1-Regularized Logistic Regression: Heavy-Ball Proximal Method and Proximal Bundle Method

This repository implements and compares two optimization algorithms for the convex composite problem

\[
\min_{w,b} F(w,b) = g(w,b) + \lambda \|w\|_1,
\]

where \(g(w,b)\) is the logistic regression loss and \(\lambda\|w\|_1\) is the non-smooth \(L_1\)-regularization term.

The goal of this project is **optimization performance**, not classification accuracy. The experiments compare the algorithms using objective gap, stationarity residual, number of iterations, runtime, and sparsity.

---

## Project structure

```text
src/
├── model.py            # Logistic loss, gradient, objective, soft-thresholding, stationarity residual
├── a1_heavyball.py     # Algorithm A1: heavy-ball proximal method
├── a2_bundle.py        # Algorithm A2: proximal bundle method
└── experiments.py      # Parameter sweeps, comparisons, scaling experiments, plots, and CSV outputs
```

---

## Implemented model

The model is binary logistic regression with \(L_1\)-regularization:

\[
F(w,b) =
-\frac{1}{N}\sum_{i=1}^N
\left[
y_i \log(p_i) + (1-y_i)\log(1-p_i)
\right]
+
\lambda \|w\|_1,
\]

where

\[
p_i = \sigma(w^\top x_i+b),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

The bias \(b\) is not regularized.

In `model.py`, the following are implemented:

- numerically stable sigmoid function;
- logistic loss and gradient;
- full objective \(F(w,b)\);
- soft-thresholding operator for the \(L_1\)-proximal step;
- stationarity residual based on the first-order optimality conditions.

The stationarity residual is used as an optimization-quality measure. A smaller value means the computed point is closer to satisfying the optimality conditions of the \(L_1\)-regularized problem.

---

## Algorithm A1: Heavy-ball proximal method

Algorithm A1 applies a heavy-ball/inertial step to the smooth logistic loss and then applies the proximal operator of the \(L_1\)-term.

The update has the form

\[
y_w^k = w^k - \alpha \nabla_w g(w^k,b^k)
       + \beta(w^k-w^{k-1}),
\]

\[
y_b^k = b^k - \alpha \nabla_b g(w^k,b^k)
       + \beta(b^k-b^{k-1}),
\]

followed by

\[
w^{k+1} = S_{\alpha\lambda}(y_w^k),
\qquad
b^{k+1}=y_b^k,
\]

where \(S_{\alpha\lambda}\) is the soft-thresholding operator.

The method stops when the parameter change becomes small:

\[
\|w^{k+1}-w^k\|_2 \leq \texttt{tol},
\qquad
|b^{k+1}-b^k| \leq \texttt{tol}.
\]

The implementation is in:

```text
src/a1_heavyball.py
```

---

## Algorithm A2: Proximal bundle method

Algorithm A2 is a proximal bundle method for the same convex composite objective.

The implementation exploits the structure

\[
F(w,b)=g(w,b)+\lambda\|w\|_1.
\]

Instead of linearizing the full non-smooth objective, the method linearizes only the smooth logistic loss \(g\) and keeps the \(L_1\)-regularization term exact in the master problem.

At bundle point \(x^j=(w^j,b^j)\), convexity gives the affine lower approximation

\[
g(x) \geq g(x^j)+\nabla g(x^j)^\top(x-x^j).
\]

The bundle model is built from these cuts. At each iteration, the method solves the proximal master problem

\[
\min_{x,t}
\left\{
t + \lambda\|w\|_1
+ \frac{\rho}{2}\|x-x_c\|^2
\right\}
\]

subject to

\[
t \geq g(x^j)+\nabla g(x^j)^\top(x-x^j),
\qquad j\in \mathcal{B}.
\]

The candidate point is accepted as a serious step if the actual decrease is sufficiently large compared with the predicted decrease:

\[
\Delta_{\mathrm{act}} \geq \gamma \Delta_{\mathrm{pred}}.
\]

Otherwise, it is treated as a null step and only the bundle is updated.

The implementation is in:

```text
src/a2_bundle.py
```

---

## Experiments

The main experiment driver is:

```text
src/experiments.py
```

It generates synthetic binary-classification data using `make_classification`, standardizes the features, and uses the full dataset as one optimization problem. No train-test split is used because the project compares optimization algorithms rather than predictive performance.

The script performs the following experiments:

### 1. A1 parameter sweep

A1 is tested over

\[
\alpha \in \{0.2,0.5,1.0,1.5\},
\qquad
\beta \in \{0.0,0.5,0.9\}.
\]

The sweep studies the effect of step size and momentum on runtime, iterations, relative gap, and stationarity residual.

### 2. A2 parameter sweep

A2 is tested over

\[
\rho \in \{0.05,0.1,0.3,1.0\},
\qquad
m_{\max} \in \{5,10,20,40\}.
\]

The serious-step parameter is fixed to

\[
\gamma=0.2.
\]

The sweep studies the effect of proximal stabilization and bundle size.

### 3. Same-start comparison

The best A1 and A2 configurations are compared from the same initial point:

\[
w^0=0,
\qquad
b^0=0.
\]

This is the main fair comparison between the algorithms.

### 4. Auxiliary warm-start experiment

A2 is also run from the final A1 solution. This is only an auxiliary consistency check and is not the main comparison.

### 5. Mean convergence curves

The convergence curves are averaged over five random seeds:

```python
CONVERGENCE_SEEDS = [0, 1, 2, 3, 4]
```

The script saves mean convergence plots for:

- relative gap vs iteration;
- relative gap vs time;
- stationarity residual vs iteration;
- stationarity residual vs time.

### 6. Problem-size scaling

The number of samples is varied as

\[
N \in \{500,1000,2000,5000\},
\qquad d=20.
\]

This tests how the runtime and iteration count change as the problem size increases.

### 7. Regularization sensitivity

The \(L_1\)-regularization parameter is varied as

\[
\lambda \in \{10^{-4},10^{-3},10^{-2},10^{-1}\}.
\]

This tests how stronger or weaker \(L_1\)-regularization affects sparsity and optimization performance.

---

## Output files

After running `experiments.py`, the results are saved in the `results/` directory.

Important CSV files include:

```text
results/a1_parameter_sweep.csv
results/a2_parameter_sweep.csv
results/same_start_summary.csv
results/best_a1_history.csv
results/best_a2_history.csv
results/best_a2_warm_start_history.csv
results/scaling_N_all_seeds.csv
results/scaling_N_mean_std.csv
results/lambda_sensitivity_all_seeds.csv
results/lambda_sensitivity_mean_std.csv
```

Important figure files include:

```text
results/figures/mean_relative_gap_vs_iteration.png
results/figures/mean_relative_gap_vs_time.png
results/figures/mean_residual_vs_iteration.png
results/figures/mean_residual_vs_time.png
results/figures/runtime_vs_N.png
results/figures/nnz_vs_lambda.png
results/figures/stationarity_residual_vs_lambda.png
```

Representative seed plots are also saved:

```text
results/figures/seed_0_relative_gap_vs_iteration.png
results/figures/seed_0_relative_gap_vs_time.png
results/figures/seed_0_residual_vs_iteration.png
results/figures/seed_0_residual_vs_time.png
```

---

## Installation and running

### Option 1: macOS / Linux

From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scikit-learn cvxpy matplotlib osqp scs
cd src
python3 experiments.py
```

### Option 2: Windows PowerShell

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install numpy scikit-learn cvxpy matplotlib osqp scs
cd src
python experiments.py
```

### Option 3: Windows Command Prompt

From the project root:

```cmd
python -m venv venv
venv\Scripts\activate
pip install numpy scikit-learn cvxpy matplotlib osqp scs
cd src
python experiments.py
```

---

## Dependencies

The project requires:

```text
numpy
scikit-learn
cvxpy
matplotlib
osqp
scs
```

The code was tested with recent Python 3 versions. Python 3.10 or later is recommended.

---

## Notes on solvers

- CVXPY is used to solve the reference convex problem with SCS.
- CVXPY is also used in A2 to solve the proximal bundle master problem with OSQP.
- Occasionally, OSQP may return an `optimal_inaccurate` warning. The code accepts this status and records the corresponding optimization metrics.

---

## Interpretation of the main results

The expected trade-off is:

- A1 has cheap iterations because each iteration mainly requires a gradient evaluation and soft-thresholding.
- A2 often requires fewer outer iterations because it solves a richer bundle master problem.
- However, each A2 iteration is more expensive.

The experiments show this trade-off clearly:

- A2 generally uses fewer iterations.
- A1 is faster in wall-clock time.
- A1 reaches smaller stationarity residuals in the tested instances.
- Increasing \(\lambda\) produces sparser solutions, as expected from \(L_1\)-regularization.
- A1 and A2 produce similar sparsity levels for the same \(\lambda\).

---

## Reproducibility

The synthetic datasets are generated with fixed random seeds. Running `src/experiments.py` should reproduce the CSV files and plots used in the report, up to small timing differences depending on the machine.

