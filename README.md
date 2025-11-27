# L1-regularized logistic regression with heavy ball and proximal bundle methods

This repository implements and compares **two optimization algorithms** for a **convex, non-smooth** machine learning problem:

- **Model (M1):** L1-regularized logistic regression
- **Algorithm A1:** Heavy ball (momentum) method + proximal step for L1
- **Algorithm A2:** Proximal bundle method

The code is organized into four files:

- `model.py` – logistic regression model and objective
- `a1_heavyball.py` – heavy ball + proximal gradient algorithm (A1)
- `a2_bundle.py` – proximal bundle method (A2)
- `experiments.py` – synthetic data experiment and comparison

Below is a **mathematical explanation** of what each part of the code is doing.

---

## 1. Problem formulation (model M1)

We consider a **binary classification** problem with data:
- Features: $x_i \in \mathbb{R}^d$
- Labels: $y_i \in \{0,1\}$
- Dataset: $\{(x_i, y_i)\}_{i=1}^N$

### 1.1 Logistic regression model

In `model.py`, the prediction model is:

$$
z_i = w^\top x_i + b, \space
\hat y_i = \sigma(z_i),
$$

where:
- $w \in \mathbb{R}^d$ – weight vector
- $b \in \mathbb{R}$ – bias term
- $\sigma : \mathbb{R} \to (0,1)$ is the **sigmoid** function

```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))
```

Mathematically:
$$
\sigma = \frac{1}{1 + e^{-z}}
$$

The clipping (```np.clip```) is a **numerical stabilization** to avoid overflow in ```exp(-z)```.

### 1.2 Logistic loss (smooth part $g$)

For each sample, the logistic (cross-entropy) loss is:
$$
\ell_i(w,b) = -(y_i \log{\hat{y_i}} + (1 - y_i) \log{(1 - \hat{y_i})})
$$

The **empirical loss** (smooth part) is:
$$
g(w,b) = \frac{1}{N} \sum_{i=1}^N{\ell_i(w,b)}
$$

In ```model.py```, ```forward_and_grad``` computes:
- $p_i = \hat{y_i} = \sigma(z_i)$
- Gradient of $g$ w.r.t. $w, b$
- Value of $g$

```python
def forward_and_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    Compute p = σ(Xw + b), gradient of smooth part g, and loss g.
    Returns: p, grad_w, grad_b, g_value
    """
    N, d = X.shape
    z = X @ w + b
    p = sigmoid(z)
    err = p - y

    grad_w = (X.T @ err) / N
    grad_b = np.sum(err) / N

    # logistic loss (smooth part)
    eps = 1e-12
    g_value = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    return p, grad_w, grad_b, g_value
```

**Derivation** (vector form):

Let $X \in \mathbb{R}^{N \times d}$ have rows $x_i^\intercal, w \in \mathbb{R}^d, b \in \mathbb{R}$,

$$
z = Xw + b, p = \sigma(z), err = p - y
$$

Then:

$$
\nabla_w{g(w,b)} = \frac{1}{N}X^\intercal(p-y), \frac{\partial{g}}{\partial{b}}(w,b) = \frac{1}{N}1^\intercal(p-y)
$$

This is exactly what the code computes.

### 1.3 L1 regularization (non-smooth part $h$)

We regularize only the weights $w$, not the bias $b$:

$$
h(w,b) = \lambda||w||_1 = \lambda \sum_{j=1}^d |w_j|.
$$

The full objective is:

$$
F(w,b) = g(w,b) + h(w,b) = g(w,b) + \lambda||w||_1
$$

In ```model.py```

```python
def objective_F(X, y, w, b, lam):
    """Full objective F = g + lam * ||w||_1."""
    _, _, _, g_value = forward_and_grad(X, y, w, b)
    return g_value + lam * np.sum(np.abs(w))
```

Note:
- $g$ is **smooth** and **convex**.
- $h$ is **convex**, but **non-smooth** (kinks at $w_j=0$).

So this is a **convex composite optimization** problem:

$$
\min_{x} F(x) = g(x) + h(x), \space x = (w,b)
$$

## 2. Algorithm A1 - heavy ball + proximal step

We implement a heavy ball (momentum) method on the smooth part $g$, followed by a proximal step for the L1-regularization $h$.

### 2.1 Soft-thresholding

The proximal operator of $h(w) = \lambda||w||_1$ is:

$$
prox_{\alpha h}(u_w) = \underset{v}{\mathrm{argmin}} \left\{ \frac{1}{2} ||v - u_w||^2 + \alpha \lambda ||v||_1 \right\}.
$$

This is given coordinate-wise by **soft-thresholding**:

$$
[prox_{\alpha h}(u_w)]_j = sign(u_{w,j})max \{ |u_{w,j}| - \alpha \lambda, 0 \}
$$

In ```model.py```:

```python
def soft_threshold(u: np.ndarray, tau: float) -> np.ndarray:
    """Soft-thresholding S_tau(u) (for L1 prox)."""
    return np.sign(u) * np.maximum(np.abs(u) - tau, 0.0)
```

Here ```tau =  alpha * lam```

### 2.2 Heavy ball iteration
Let $x_k = (w_k, b_k).$

We split the update into:

1. Heavy ball step on $g$ (smooth):

$$
y_{k,w} = w_k - \alpha \nabla_w g(w_k,b_k) + \beta(w_k - w_{k-1}),
$$
$$
y_{k,b} = b_k - \alpha \frac{\partial{g}}{\partial{b}}
$$

2. Proximal step for $h$ (non-smooth L1):

$$
w_{k+1} = prox_{\alpha \lambda ||\cdot||_1}(y_{k,w}) = S_{\alpha \lambda}(y_{k,w}), \space b_{k+1} = y_{k,b}
$$

(the bias is not regularized).

This is exactly what ```heavy_ball_prox_l1_logreg``` does:

```python
for k in range(max_iter):
    # smooth gradient and value
    _, grad_w, grad_b, g_value = forward_and_grad(X, y, w, b)

    # heavy ball step on g
    y_w = w - alpha * grad_w + beta * (w - w_prev)
    y_b = b - alpha * grad_b + beta * (b - b_prev)

    # proximal step for h = lam * ||w||_1
    w_next = soft_threshold(y_w, alpha * lam)
    b_next = y_b  # bias not regularized

    # bookkeeping
    F_val = objective_F(X, y, w_next, b_next, lam)
    history["F"].append(F_val)
    history["g"].append(g_value)
    history["nnz"].append(np.count_nonzero(w_next))

    if verbose and (k % 50 == 0 or k == max_iter - 1):
        print(f"k={k:4d}, F={F_val:.6f}, ||Δw||={np.linalg.norm(w_next - w):.3e}")

    # stopping on parameter change
    if np.linalg.norm(w_next - w) <= tol and abs(b_next - b) <= tol:
        w, b = w_next, b_next
        break

    # shift iterates
    w_prev, b_prev = w, b
    w, b = w_next, b_next
```

### 2.3 Stopping Criterion and History

Stopping condition:

$$
||w_{k+1} - w_k||_2 \le tol \ \text{and} \ |b_{k+1} - b_k| \le tol.
$$

```python
# stopping on parameter change
if np.linalg.norm(w_next - w) <= tol and abs(b_next - b) <= tol:
    w, b = w_next, b_next
    break
```

The history dictionary stores:
- ```F``` - full objective value $F(w_{k+1}, b_{k+1})$
- ```g``` - smooth loss value $g(w_k,b_k)$
- ```nnz``` - number of non-zero weights (sparsity pattern)

## 3. Algorithm A2 - proximal bundle method

A2 is a **proximal bundle method** for non-smooth convex optimization of $F = g + h$

We work in the parameter space:

$$
x = \begin{bmatrix} w \\ b \end{bmatrix} \in \mathbb{R}^{d + 1}.
$$

### 3.1. Subgradient of L1 and F

For the L1 term:

$$
h(w) = \lambda || w ||_1 = \lambda \sum_j{|w_j|}.
$$

A subgradient of $h$ at $w$ is:

$$
(\partial h(w))_j = \lambda \begin{cases} sign(w_j), \ w_j \ne 0, \\ t, \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ w_j = 0, |t| \le 1 \end{cases}
$$

```subgradient_l1_part``` picks a particular subgradient:

```python
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
```

- For nonzero entries: $\lambda sign(w_j)$.
- Near zero: a value in $[-\lambda, \lambda]$.

To get a subgradient of $F$

$$
\partial F(w,b) = \nabla g(w,b) + \partial h(w,b),
$$

where there is **L1 on b**

```python
def subgradient_F(X, y, w, b, lam):
    """
    Compute one subgradient gF of F at x=(w,b).
    """
    p, grad_w, grad_b, _ = forward_and_grad(X, y, w, b)
    s_l1 = subgradient_l1_part(w, lam)
    g_w = grad_w + s_l1
    g_b = grad_b  # no L1 on bias
    return np.concatenate([g_w, np.array([g_b])])
```

So ```subgradient_F``` returns one vector $g^F \in \partial F(x)$

### 3.2 Bundle model: cutting-plane underestimator

The proximal bundle method maintains a **bundle** of points and subgradients:

$$
\{ (x^j, F^j, g^j) \}_{j \in I}, \ g^j \in \partial{F(x^j)}
$$

The **model** is a piecewise linear underestimator:

$$
\hat{F}(x) = \max_{j \in I}\{F^j + (g^j)^{\intercal} (x - x^j)\}.
$$

Each term is supporting is a supporting hyperplane of $F$.

### 3.3 Proximal master problem (QP)

Given a **center $x_c$**, we solve:

$$
\min_{x,t} t + \frac{\rho}{2} ||x - x_c||^2 \ \ \text{s.t.} \ \ t \ge F^j + (g^j)^{\intercal}(x - x^j), \ \forall j \in I.
$$

- $\rho > 0$: a proximal parameter.
- Variables: $x \in \mathbb{R}^{d+1}, t \in \mathbb{R}$.

In code, we build a QP in the form:

$$
\min_{z} \frac{1}{2}z^{\intercal}Hz + f^{\intercal}z \ \ s.t. \ \ Az \le b,
$$

where $z = [x;t] \in \mathbb{R}^{n+1}, n = d + 1.$

#### QP objective
We want:

$$
t + \frac{\rho}{2} ||x - x_c||^2 = t + \frac{\rho}{2} x^{\intercal}x - \rho x_c^{\intercal} x + \text{const}.
$$

Ignoring the constant, this corresponds to:

```python
H = np.zeros((n + 1, n + 1))
H[:n, :n] = rho * np.eye(n)  # block diag

f = np.zeros(n + 1)
f[:n] = -rho * x_c
f[-1] = 1.0  # coefficient of t
```

So:
- $H = \begin{bmatrix} \rho I_n \ 0 \\ 0^{\intercal} \ 0 \end{bmatrix}$
- $f = \begin{bmatrix} -\rho x_c \\ 1 \end{bmatrix}$

#### QP constraints
From:
$$
t \ge F^j + (g^j)^{\intercal}(x - x^j)
$$

we get:
$$
t - (g^j)^{\intercal}x \ge F^j - (g^j)^{\intercal}x^j
$$
$$
-(t - (g^j)^{\intercal}x) \le -(F^j - (g^j)^{\intercal}x^j)
$$
$$
-t + (g^j)^{\intercal}x \le -F^j + (g^j)^{\intercal}x^j
$$

So in the standard form $Az \le b$ with $z = [x;t]$:

- $A_j = [(g^j)^{\intercal}, -1]$
- $b_j = -F^j + (g^j)^{\intercal}x^j$

This is implemented as:
```python
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
```

#### Solving the QP
We then solve:
```python
z = cp.Variable(n + 1)
obj = 0.5 * cp.quad_form(z, H) + f @ z
constraints = [A @ z <= b_vec]
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve(solver=cp.OSQP) # or another QP solver
```

The solution gives:
- ```z_bar = z.value```
- ```x_bar = z_bar[:n]```
- ```z_bar = z_bar[-1]```

### 3.4 Serious vs null step, predicted vs actual decrease

After solving the master QP, we evaluate:

- True objective at candidate:

$$
F(\bar{x})
$$

- Subgradient at candidate:

```python
g_bar = subgradient_F(X, y, w_bar, b_bar, lam)
```

We also compute:

- Model value at center:

$$
m_c = \hat{F}(x_c) = \max_{j \ \in \ I} \{ F^j + (g^j)^{\intercal}(x_c - x^j) \}.
$$

- Model objective at candidate:

$$
F_{\text{model}} = \bar{t} + \frac{\rho}{2}|\bar{x} - x_c|^2
$$

- Predicted decrease:

$$
\Delta_{\text{pred}} = m_c - F_{\text{model}}. 
$$

- Actual decrease:

$$
\Delta_{\text{act}} = F(x_c) - F(\bar{x}).
$$

In code:
```python
mc_terms = [el["F"] + el["g"] @ (x_c - el["x"]) for el in bundle]
m_c = max(mc_terms)

F_model = t_bar + 0.5 * rho * np.linalg.norm(x_bar - x_c) ** 2
delta_pred = m_c - F_model
delta_act = F_c - F_bar
```

We then:

1. **Stop** if predicted improvement is too small:

$$
\Delta_{\text{pred}} \le \text{tol_pred}.
$$

2. Decide **serious vs null step** using parameter $\gamma \in (0, 1)$:

Serious step, if
$$
\Delta_{\text{act}} \ge \gamma \Delta_{\text{pred}}
$$
Then we **accept** the candidate:

```python
if delta_act >= gamma * delta_pred:
    # serious step
    x_c = x_bar
    F_c = F_bar
```

Otherwise, null step: center stays ```x_c```

3. **Update bundle** with new cut:

```python
bundle.append({"x": x_bar.copy(), "F": F_bar, "g": g_bar.copy()})
if len(bundle) > bundle_max:
    # simple policy: drop oldest
    bundle.pop(0)
```

We store:

```python
history["F_center"].append(F_c)
history["bundle_size"].append(len(bundle))
```

So ```history``` records the evolution of the center objective.

## 4. Experiments and usage

### 4.1 Data Generation and Preprocessing

We generate a synthetic binary classification dataset using scikit-learn:
```python
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=0,
    random_state=0,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

- 2000 samples, 20 features, 10 informative.
- Standardization ensures nicer conditioning for optimization.

### 4.2 Running A1 (heavy ball + prox)

```python
lam = 1e-2
w_hb, b_hb, hist_hb = heavy_ball_prox_l1_logreg(
    X_train,
    y_train,
    lam=lam,
    alpha=1.0,
    beta=0.9,
    max_iter=2000,
    tol=1e-6,
    verbose=True,
)
```

Parameters:
- ```lam``` – L1 regularization coefficient $\lambda$
- ```alpha``` – step size $\alpha$
- ```beta``` – momentum parameter $\beta$
- ```tol``` – tolerance on parameter change

### 4.3 Running A2 (bundle) from two initializations

1. From heavy-ball solution:

```python
lam = 1e-2
w_hb, b_hb, hist_hb = heavy_ball_prox_l1_logreg(
    X_train,
    y_train,
    lam=lam,
    alpha=1.0,
    beta=0.9,
    max_iter=2000,
    tol=1e-6,
    verbose=True,
)

```

2. From random start:
```python
x0_hb = np.concatenate([w_hb, np.array([b_hb])])
x_star_hb, hist_bun_hb = proximal_bundle_l1_logreg(
    X_train,
    y_train,
    lam=lam,
    rho=1.0,
    gamma=0.1,
    max_iter=100,
    bundle_max=20,
    tol_pred=1e-4,
    verbose=True,
    x0=x0_hb,
)
```

### 4.4 Evaluation: accuracy and sparsity

Prediction helper:
```python
def predict_logreg(X, w, b):
    p = sigmoid(X @ w + b)
    return (p >= 0.5).astype(int)
```

We compare:

- **Test accuracy** – classification quality.
- **Number of non-zero weights** – measure of sparsity induced by L1.

```python
w_star_hb, b_star_hb = x_star_hb[:-1], x_star_hb[-1]
w_star_rand, b_star_rand = x_star_rand[:-1], x_star_rand[-1]

y_pred_hb = predict_logreg(X_test, w_hb, b_hb)
y_pred_bun_hb = predict_logreg(X_test, w_star_hb, b_star_hb)
y_pred_bun_rand = predict_logreg(X_test, w_star_rand, b_star_rand)

print("\n=== Test accuracy ===")
print("Heavy ball acc:", accuracy_score(y_test, y_pred_hb))
print("Bundle (from HB) acc:", accuracy_score(y_test, y_pred_bun_hb))
print("Bundle (random start) acc:", accuracy_score(y_test, y_pred_bun_rand))

print("\n=== Sparsity (number of nonzero weights) ===")
print("HB nnz:", np.count_nonzero(w_hb))
print("Bun (from HB) nnz:", np.count_nonzero(w_star_hb))
print("Bun (random) nnz:", np.count_nonzero(w_star_rand))

print("\n=== Iteration counts ===")
print("HB iterations:", len(hist_hb['F']))
print("Bundle (from HB) iterations:", len(hist_bun_hb['F_center']))
print("Bundle (random) iterations:", len(hist_bun_rand['F_center']))
```

This lets **empirically** compare:

- How fast A1 and A2 decrease the objective.
- Whether A1’s inertial proximal gradient finds a sparse solution.
- How robust A2 is to non-smoothness and initialization.

## 5. How to run
### Dependencies
- Python 3.11.8
- Numpy 2.3.5
- Cvxpy 1.7.3
- Scikit-learn 1.7.2

### Guide
1. Install [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html). Note that it's recommended by developers to install it via Pipx.

```pipx install virtualenv```

2. Clone the GitHub [repository](https://github.com/Az1mzhan/CM-ML-5.git)

```git clone https://github.com/Az1mzhan/CM-ML-5.git```

3. Create a virtual environment at a root of the cloned repository

```virtualenv env```

4. Activate the environment via the script below:

```env/Scripts/activate```

5. Install required dependencies.

**NOTE:** You may install dependencies in 2 ways:

a) Automatically (You should take into account version of Python on a local environment)

```
pip install numpy cvxpy scikit-learn
```

b) Manually via WHEEL files

```
pip install WHEEL_FILE
```

6. Run experiments sandbox in the environment

```
python experiments.py
```