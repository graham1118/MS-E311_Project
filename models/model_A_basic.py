# ============================================================
# model_A_basic.py
# Classical Markowitz Efficient Frontier + Backtest (SCS Solver)
# ============================================================

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Global hyperparameters
# -----------------------------
EPS_SIGMA = 1e-4   # stability bump
W_MAX = 0.05       # max weight per asset (5% cap for return-range problems)


# ------------------------------------------------------------
# Solver helper (SCS only)
# ------------------------------------------------------------
def solve_scs(prob):
    prob.solve(solver="SCS", verbose=False)


# ------------------------------------------------------------
# Model A: Classical Markowitz (with long-only constraints)
# ------------------------------------------------------------
def solve_markowitz_basic(mu, Sigma, target_return):
    """
    Classical Markowitz:
        minimize   wᵀ Σ w
        s.t.       μᵀ w >= target_return
                   sum(w) = 1
                   w >= 0       (long-only)
    """
    n = len(mu)

    Sigma = 0.5 * (Sigma + Sigma.T) + EPS_SIGMA * np.eye(n)

    w = cp.Variable(n)
    variance = cp.quad_form(w, Sigma)

    constraints = [
        cp.sum(w) == 1,
        mu @ w >= target_return,
        w >= 0
    ]

    prob = cp.Problem(cp.Minimize(variance), constraints)
    solve_scs(prob)

    if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
        return None

    return np.array(w.value)


# ------------------------------------------------------------
# Compute feasible range of returns (bounded, long-only)
# ------------------------------------------------------------
def compute_return_range(mu):
    """
    Compute min/max achievable portfolio return with:
        sum(w) = 1,  0 <= w <= W_MAX
    so that the problem is bounded and numerically stable.
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
# MAIN SCRIPT
# ------------------------------------------------------------
if __name__ == "__main__":

    # -----------------------------------------
    # Load μ and Σ for whole dataset
    # -----------------------------------------
    npz_whole = np.load("Data/mu_sigma_whole_dataset.npz")
    mu = npz_whole["mu"]
    Sigma = npz_whole["sigma"]

    # -----------------------------------------
    # Efficient Frontier
    # -----------------------------------------
    min_ret, max_ret = compute_return_range(mu)
    print(f"Return range = [{min_ret:.6f}, {max_ret:.6f}]")

    target_returns = np.linspace(min_ret, max_ret, 60)
    risks, rets = [], []

    for r in target_returns:
        w = solve_markowitz_basic(mu, Sigma, r)
        if w is None:
            continue
        risks.append(float(np.sqrt(w @ Sigma @ w)))
        rets.append(float(mu @ w))

    # Plot frontier
    plt.figure(figsize=(7, 5))
    plt.plot(risks, rets, "-o", markersize=3)
    plt.grid(True)
    plt.xlabel("Risk (Std Dev)")
    plt.ylabel("Expected Return")
    plt.title("Model A — Classical Markowitz Frontier (Long-only)")
    plt.tight_layout()
    plt.show()

    # -----------------------------------------
    # Backtesting Setup
    # -----------------------------------------
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

    muA = mu[idxs]
    SigmaA = Sigma[np.ix_(idxs, idxs)]

    # You can set this however you want; here we just use a fixed target
    target = 0.002  # daily return target
    w_bt = solve_markowitz_basic(muA, SigmaA, target)

    if w_bt is None:
        raise RuntimeError("Model A infeasible for backtest with this target.")

    # -----------------------------------------
    # Compute backtest return
    # -----------------------------------------
    p0 = test.iloc[0].values
    p1 = test.iloc[-1].values

    initial_value = 1.0
    shares = w_bt * initial_value / p0
    final_value = np.sum(shares * p1)
    actual = (final_value - initial_value) / initial_value

    print("\n========== Backtest Results (Model A) ==========")
    print(f"Target daily return: {target:.6f}")
    print(f"Backtest actual return: {actual:.4%}")
    annualized = (1 + actual)**(252/len(test)) - 1
    print(f"Annualized return: {annualized:.4%}")
    print("=================================================")

    # Show largest holdings
    top = np.argsort(w_bt)[-10:][::-1]
    print("\nTop 10 Holdings (Model A):")
    for i, ix in enumerate(top, 1):
        print(f"{i}. {common[ix]}: {w_bt[ix]:.4%}")
