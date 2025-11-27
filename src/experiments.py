import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from model import sigmoid
from a1_heavyball import heavy_ball_prox_l1_logreg
from a2_bundle import proximal_bundle_l1_logreg

# generate data
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

# run A1 (heavy ball + prox)
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

# run A2 starting from A1 solution
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

# run A2 from a random starting point (to see standalone behaviour)
rng = np.random.default_rng(0)
x0_rand = rng.normal(scale=0.1, size=X_train.shape[1] + 1)
x_star_rand, hist_bun_rand = proximal_bundle_l1_logreg(
    X_train,
    y_train,
    lam=lam,
    rho=1.0,
    gamma=0.1,
    max_iter=100,
    bundle_max=20,
    tol_pred=1e-4,
    verbose=True,
    x0=x0_rand,
)

# evaluation helpers
def predict_logreg(X, w, b):
    p = sigmoid(X @ w + b)
    return (p >= 0.5).astype(int)

# evaluate accuracy / sparsity
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