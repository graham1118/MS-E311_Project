import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Global hyperparameters
# -----------------------------
LAM = 1.0        # risk aversion for solve_markowitz
GAMMA = 0.15   # L2 regularization strength
DELTA = 0.15     # tracking-error regularization strength
W_MAX = 0.5    # max weight per asset (5%)
EPS_SIGMA = 1e-4 # diagonal bump to stabilize covariance

# -----------------------------
# Helper: solve with OSQP (QP)
# -----------------------------

def solve_qp(prob):
    """
    Solve a QP using OSQP if available; otherwise let CVXPY choose.
    """
    if "OSQP" in cp.installed_solvers():
        prob.solve(solver="OSQP")
    else:
        prob.solve()


# -----------------------------
# Core Markowitz solver
# -----------------------------

def solve_markowitz(mu, Sigma, lambda_risk, gamma_reg, long_only=True):
    """
    More realistic mean-variance optimization, QP-compatible:

        minimize   w^T Sigma w
                   - λ (mu^T w)
                   + gamma_reg * ||w||_2^2
                   + DELTA * ||w - w_bench||_2^2

        subject to sum(w) = 1
                   0 <= w_i <= W_MAX    (if long_only)
    """
    n = len(mu)

    # Stabilize covariance
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = Sigma + EPS_SIGMA * np.eye(n)

    w = cp.Variable(n)
    w_bench = np.ones(n) / n

    variance_term = cp.quad_form(w, Sigma)
    return_term = mu @ w
    l2_reg = cp.sum_squares(w)
    tracking_penalty = cp.sum_squares(w - w_bench)

    objective = cp.Minimize(
        variance_term
        - lambda_risk * return_term
        + gamma_reg * l2_reg
        + DELTA * tracking_penalty
    )

    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
        constraints.append(w <= W_MAX)

    prob = cp.Problem(objective, constraints)
    solve_qp(prob)

    if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("Optimization failed in solve_markowitz.")

    return np.array(w.value)


# -----------------------------
# Efficient frontier helpers
# -----------------------------

def _build_constraints(mu, long_only=True):
    """
    Common structural constraints that do NOT include the return constraint.
    Used to compute min/max achievable return.
    """
    n = len(mu)
    w = cp.Variable(n)

    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
        constraints.append(w <= W_MAX)
    return w, constraints


def compute_return_range(mu, long_only=True):
    """
    Compute the minimum and maximum achievable portfolio return under constraints.
    No covariance needed because we’re only optimizing mu^T w.
    """
    n = len(mu)

    # Min return
    w, base_constraints = _build_constraints(mu, long_only=long_only)
    obj_min = cp.Minimize(mu @ w)
    prob_min = cp.Problem(obj_min, base_constraints)
    solve_qp(prob_min)
    if w.value is None or prob_min.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("Failed to compute minimum achievable return.")
    min_ret = float(mu @ w.value)

    # Max return
    w2, base_constraints2 = _build_constraints(mu, long_only=long_only)
    obj_max = cp.Maximize(mu @ w2)
    prob_max = cp.Problem(obj_max, base_constraints2)
    solve_qp(prob_max)
    if w2.value is None or prob_max.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("Failed to compute maximum achievable return.")
    max_ret = float(mu @ w2.value)

    return min_ret, max_ret


def solve_min_variance_for_return(mu, Sigma, target_return,
                                  long_only=False,
                                  gamma_reg=GAMMA,
                                  delta_reg=DELTA):
    """
    Efficient frontier point (QP):

        minimize   w^T Sigma w
                   + gamma_reg * ||w||_2^2
                   + delta_reg * ||w - w_bench||_2^2

        subject to mu^T w >= target_return
                   sum(w) = 1
                   0 <= w_i <= W_MAX (if long_only)

    Returns None if infeasible.
    """
    n = len(mu)
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = Sigma + EPS_SIGMA * np.eye(n)

    w = cp.Variable(n)
    w_bench = np.ones(n) / n

    variance_term = cp.quad_form(w, Sigma)
    l2_reg = cp.sum_squares(w)
    tracking_penalty = cp.sum_squares(w - w_bench)

    objective = cp.Minimize(
        variance_term
        + gamma_reg * l2_reg
        + delta_reg * tracking_penalty
    )

    constraints = [
        mu @ w >= target_return,
        cp.sum(w) == 1,
    ]
    if long_only:
        constraints.append(w >= 0)
        constraints.append(w <= W_MAX)
    else:
        # allow shorts, maybe still cap leverage:
        constraints.append(w >= -0.2)  # up to 20% short each
        constraints.append(w <= 0.2)   # up to 20% long each

    prob = cp.Problem(objective, constraints)
    solve_qp(prob)

    if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
        return None

    return np.array(w.value)


def solve_global_min_variance(mu, Sigma, long_only=True,
                              gamma_reg=GAMMA,
                              delta_reg=DELTA):
    """
    Global minimum-variance portfolio (no return constraint).
    """
    n = len(mu)
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = Sigma + EPS_SIGMA * np.eye(n)

    w = cp.Variable(n)
    w_bench = np.ones(n) / n

    variance_term = cp.quad_form(w, Sigma)
    l2_reg = cp.sum_squares(w)
    tracking_penalty = cp.sum_squares(w - w_bench)

    objective = cp.Minimize(
        variance_term
        + gamma_reg * l2_reg
        + delta_reg * tracking_penalty
    )

    constraints = [
        mu @ w >= target_return,
        cp.sum(w) == 1,
    ]
    if long_only:
        constraints.append(w >= 0)
        constraints.append(w <= W_MAX)
    else:
        # allow shorts, maybe still cap leverage:
        constraints.append(w >= -0.2)  # up to 20% short each
        constraints.append(w <= 0.2)   # up to 20% long each

    prob = cp.Problem(objective, constraints)
    solve_qp(prob)

    if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("Failed to compute global minimum-variance portfolio.")

    return np.array(w.value)


# -----------------------------
# Load regime-specific μ, Σ
# -----------------------------

npzfile = np.load("Data/mu_sigma.npz")
mu_all_regimes = npzfile["mu_all_regimes"]
sigma_all_regimes = npzfile["sigma_all_regimes"]

K = mu_all_regimes.shape[0]  # number of regimes
N = mu_all_regimes.shape[1]  # number of assets

npz_whole_dataset = np.load("Data/mu_sigma_whole_dataset.npz")
mu_whole = npz_whole_dataset["mu"]
sigma_whole = npz_whole_dataset["sigma"]

# Solve Markowitz for each regime with the more complex formulation
weights_all_regimes = np.zeros((K, N))
for k in range(K):
    weights_all_regimes[k, :] = solve_markowitz(
        mu_all_regimes[k, :],
        sigma_all_regimes[k, :, :],
        LAM,
        GAMMA
    )

# -----------------------------
# Efficient Frontier (whole dataset μ, Σ)
# -----------------------------

mu = mu_whole
Sigma = sigma_whole

# Feasible return range under constraints
min_ret, max_ret = compute_return_range(mu, long_only=True)
print(f"Feasible return range: [{min_ret:.6f}, {max_ret:.6f}]")

target_returns = np.linspace(min_ret, max_ret, 80)

frontier_returns = []
frontier_risks = []

for r_target in target_returns:
    w_opt = solve_min_variance_for_return(mu, Sigma, r_target)
    if w_opt is None:
        continue
    realized_return = float(mu @ w_opt)
    realized_risk = float(np.sqrt(w_opt @ Sigma @ w_opt))
    frontier_returns.append(realized_return)
    frontier_risks.append(realized_risk)

frontier_returns = np.array(frontier_returns)
frontier_risks = np.array(frontier_risks)

plt.figure(figsize=(7, 5))
plt.plot(frontier_risks, frontier_returns, "-o", markersize=3)
plt.xlabel("Portfolio Risk (Std Dev)")
plt.ylabel("Portfolio Expected Return")
plt.title("Efficient Frontier (Stabilized & Diversified, QP)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Backtesting
# -----------------------------

# Read training data (for column names)
# If you prefer absolute path:
# sp500_train = pd.read_csv("/Users/ohmpatel/Desktop/MSE311/MS-E311_Project/Data/sp500_20yr_clean.csv",
#                           index_col="Timestamp", parse_dates=True)
sp500_train = pd.read_csv("Data/sp500_20yr_clean.csv",
                          index_col="Timestamp", parse_dates=True)
stock_tickers = sp500_train.columns.tolist()

# Read test set
test_set = pd.read_csv("Data/test_set.csv",
                       index_col="Timestamp", parse_dates=True)

# Drop 'Unnamed: 0' column if present
if "Unnamed: 0" in test_set.columns:
    test_set = test_set.drop(columns=["Unnamed: 0"])

# Align test_set with training tickers
common_stocks = [ticker for ticker in stock_tickers if ticker in test_set.columns]
test_set_aligned = test_set[common_stocks]

# Indices of these stocks in mu_whole / Sigma
stock_indices = [stock_tickers.index(ticker) for ticker in common_stocks]

mu_aligned = mu[stock_indices]
Sigma_aligned = Sigma[np.ix_(stock_indices, stock_indices)]

# Choose a realistic target return for backtest: midpoint of feasible range
min_ret_aligned, max_ret_aligned = compute_return_range(
    mu_aligned, long_only=True
)
#target_return = 0.5 * (min_ret_aligned + max_ret_aligned)
target_return = 0.002
print(f"\nBacktest target daily return (aligned universe): {target_return:.6f}")

w_backtest = solve_min_variance_for_return(mu_aligned, Sigma_aligned, target_return)
if w_backtest is None:
    print("Target-return problem infeasible for backtest; using global min-variance portfolio.")
    w_backtest = solve_global_min_variance(mu_aligned, Sigma_aligned, long_only=False)

# First and last day prices in test set
first_day_prices = test_set_aligned.iloc[0, :].values
last_day_prices = test_set_aligned.iloc[-1, :].values

initial_value = 1.0
shares = w_backtest * initial_value / first_day_prices
final_value = np.sum(shares * last_day_prices)

actual_return = (final_value - initial_value) / initial_value

print("\n" + "=" * 60)
print("BACKTESTING RESULTS")
print("=" * 60)
print(f"Target daily return (for optimization): {target_return:.6f}")
print(f"Test period: {test_set.index[0]} to {test_set.index[-1]}")
print(f"Number of days: {len(test_set)}")
print(f"Initial portfolio value: ${initial_value:.2f}")
print(f"Final portfolio value:   ${final_value:.2f}")
print(f"Actual return over test period: {actual_return:.4%}")
annualized = (1 + actual_return) ** (252 / len(test_set)) - 1
print(f"Annualized return (252 trading days): {annualized:.4%}")
print("=" * 60)

# Top holdings
top_holdings_idx = np.argsort(w_backtest)[-10:][::-1]
print("\nTop 10 Holdings:")
for i, idx in enumerate(top_holdings_idx, 1):
    print(f"{i}. {common_stocks[idx]}: {w_backtest[idx]:.4%}")
print("=" * 60)
print(f"\nNote: Portfolio constructed using {len(common_stocks)} stocks present in both training and test sets.")
