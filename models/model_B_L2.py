# ============================================================
# model_B_L2.py
# Markowitz + L2 Regularization (SCS Solver)
# ============================================================

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GAMMA = 0.05       # L2 penalty strength
EPS_SIGMA = 1e-4   # small diagonal bump for stability
W_MAX = 0.05       # max weight per asset for return-range problems


# ------------------------------------------------------------
# SCS solver wrapper
# ------------------------------------------------------------
def solve_scs(prob):
    prob.solve(solver="SCS", verbose=False)


# ------------------------------------------------------------
# Model B: L2 Regularized Markowitz
# ------------------------------------------------------------
def solve_markowitz_L2(mu, Sigma, target_return, gamma=GAMMA):
    """
    L2-regularized Markowitz:
        minimize   wᵀ Σ w + γ||w||²
        s.t.       μᵀ w >= target_return
                   sum(w) = 1
                   w >= 0          (long-only)
    """
    n = len(mu)

    # Symmetrize + stabilize
    Sigma = 0.5 * (Sigma + Sigma.T) + EPS_SIGMA * np.eye(n)

    w = cp.Variable(n)

    variance = cp.quad_form(w, Sigma)
    l2_penalty = cp.sum_squares(w)

    objective = cp.Minimize(variance + gamma * l2_penalty)
    constraints = [
        cp.sum(w) == 1,
        mu @ w >= target_return,
        w >= 0
    ]

    prob = cp.Problem(objective, constraints)
    solve_scs(prob)

    if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
        return None

    return np.array(w.value)


# ------------------------------------------------------------
# Compute min/max achievable return (bounded, long-only)
# ------------------------------------------------------------
def compute_return_range(mu):
    """
    Compute min/max achievable portfolio return under:
        sum(w) = 1
        0 <= w <= W_MAX

    This ensures the problem is bounded so SCS returns valid solutions.
    """
    n = len(mu)

    # Min return
    w = cp.Variable(n)
    constraints_min = [
        cp.sum(w) == 1,
        w >= 0,
        w <= W_MAX
    ]
    prob_min = cp.Problem(cp.Minimize(mu @ w), constraints_min)
    solve_scs(prob_min)
    if w.value is None or prob_min.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("Failed to compute minimum achievable return.")
    min_ret = float(mu @ w.value)

    # Max return
    w2 = cp.Variable(n)
    constraints_max = [
        cp.sum(w2) == 1,
        w2 >= 0,
        w2 <= W_MAX
    ]
    prob_max = cp.Problem(cp.Maximize(mu @ w2), constraints_max)
    solve_scs(prob_max)
    if w2.value is None or prob_max.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("Failed to compute maximum achievable return.")
    max_ret = float(mu @ w2.value)

    return min_ret, max_ret


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    # -------------------------
    # Load μ and Σ
    # -------------------------
    npz_whole = np.load("Data/mu_sigma_whole_dataset.npz")
    mu = npz_whole["mu"]
    Sigma = npz_whole["sigma"]

    # -------------------------
    # Efficient Frontier
    # -------------------------
    min_ret, max_ret = compute_return_range(mu)
    print(f"Return range = [{min_ret:.6f}, {max_ret:.6f}]")

    target_returns = np.linspace(min_ret, max_ret, 60)
    risks, rets = [], []

    for r in target_returns:
        w = solve_markowitz_L2(mu, Sigma, r)
        if w is None:
            continue
        risks.append(float(np.sqrt(w @ Sigma @ w)))
        rets.append(float(mu @ w))

    plt.figure(figsize=(7, 5))
    plt.plot(risks, rets, "-o", markersize=3)
    plt.grid(True)
    plt.xlabel("Risk (Std Dev)")
    plt.ylabel("Expected Return")
    plt.title("Model B — L2 Regularized Frontier (Long-only)")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Backtest
    # -------------------------
    sp500_train = pd.read_csv(
        "Data/sp500_20yr_clean.csv",
        index_col="Timestamp",
        parse_dates=True
    )
    test = pd.read_csv(
        "Data/test_set.csv",
        index_col="Timestamp",
        parse_dates=True
    )
    if "Unnamed: 0" in test.columns:
        test = test.drop(columns=["Unnamed: 0"])

    tickers = sp500_train.columns.tolist()
    common = [t for t in tickers if t in test.columns]
    test = test[common]
    idxs = [tickers.index(t) for t in common]

    muB = mu[idxs]
    SigmaB = Sigma[np.ix_(idxs, idxs)]

    target = 0.002  # daily target used for backtest
    w_bt = solve_markowitz_L2(muB, SigmaB, target)

    if w_bt is None:
        raise RuntimeError("Model B infeasible for backtest with this target.")

    p0 = test.iloc[0].values
    p1 = test.iloc[-1].values

    initial_value = 1.0
    shares = w_bt * initial_value / p0
    final_value = np.sum(shares * p1)
    actual = (final_value - initial_value) / initial_value

    print("\n========== Backtest Results (Model B) ==========")
    print(f"Target daily return: {target:.6f}")
    print(f"Backtest actual return: {actual:.4%}")
    annualized = (1 + actual)**(252 / len(test)) - 1
    print(f"Annualized return: {annualized:.4%}")
    print("=================================================")

    top = np.argsort(w_bt)[-10:][::-1]
    print("\nTop 10 Holdings (Model B):")
    for i, ix in enumerate(top, 1):
        print(f"{i}. {common[ix]}: {w_bt[ix]:.4%}")
