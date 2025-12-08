# ============================================================
# model_C_L2_tracking.py
# Markowitz + L2 + Tracking Error Regularization (SCS Solver)
# ============================================================

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GAMMA = 0.05       # L2 strength
DELTA = 0.05       # Tracking error strength
EPS_SIGMA = 1e-4
W_MAX = 0.05       # constraint for return-range only


# ------------------------------------------------------------
# SCS solver wrapper
# ------------------------------------------------------------
def solve_scs(prob):
    prob.solve(solver="SCS", verbose=False)


# ------------------------------------------------------------
# Model C: L2 + Tracking Error Regularization
# ------------------------------------------------------------
def solve_markowitz_L2_TE(mu, Sigma, target_return, gamma=GAMMA, delta=DELTA):
    """
    minimize   wᵀ Σ w + γ||w||² + δ||w - w_bench||²
    s.t.       μᵀw >= target_return, sum(w)=1, w>=0
    """
    n = len(mu)

    # Stabilized covariance
    Sigma = 0.5*(Sigma + Sigma.T) + EPS_SIGMA*np.eye(n)

    w = cp.Variable(n)
    w_bench = np.ones(n) / n

    variance = cp.quad_form(w, Sigma)
    l2_pen  = cp.sum_squares(w)
    te_pen  = cp.sum_squares(w - w_bench)

    objective = cp.Minimize(variance + gamma*l2_pen + delta*te_pen)

    constraints = [
        cp.sum(w) == 1,
        mu @ w >= target_return,
        w >= 0
    ]

    prob = cp.Problem(objective, constraints)
    solve_scs(prob)

    if w.value is None:
        return None
    return np.array(w.value)


# ------------------------------------------------------------
# Min/max return for frontier construction
# ------------------------------------------------------------
def compute_return_range(mu):
    """
    Return min/max achievable return with:
        sum(w)=1, 0 <= w <= W_MAX
    """
    n = len(mu)

    # Min return
    w = cp.Variable(n)
    cons = [cp.sum(w)==1, w>=0, w<=W_MAX]
    prob = cp.Problem(cp.Minimize(mu @ w), cons)
    solve_scs(prob)
    min_ret = float(mu @ w.value)

    # Max return
    w2 = cp.Variable(n)
    cons2 = [cp.sum(w2)==1, w2>=0, w2<=W_MAX]
    prob2 = cp.Problem(cp.Maximize(mu @ w2), cons2)
    solve_scs(prob2)
    max_ret = float(mu @ w2.value)

    return min_ret, max_ret


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    # -------------------------
    # Load whole-dataset μ, Σ
    # -------------------------
    data = np.load("Data/mu_sigma_whole_dataset.npz")
    mu = data["mu"]
    Sigma = data["sigma"]

    # -------------------------
    # Efficient Frontier
    # -------------------------
    min_ret, max_ret = compute_return_range(mu)
    print(f"Return range = [{min_ret:.6f}, {max_ret:.6f}]")

    target_returns = np.linspace(min_ret, max_ret, 60)
    risks, rets = [], []

    for r in target_returns:
        w = solve_markowitz_L2_TE(mu, Sigma, r)
        if w is None:
            continue
        risks.append(np.sqrt(w @ Sigma @ w))
        rets.append(mu @ w)

    plt.figure(figsize=(7,5))
    plt.plot(risks, rets, "-o", markersize=3)
    plt.grid(True)
    plt.xlabel("Risk (Std Dev)")
    plt.ylabel("Expected Return")
    plt.title("Model C — L2 + TE Regularized Frontier")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Backtest
    # -------------------------
    train = pd.read_csv("Data/sp500_20yr_clean.csv",
                        index_col="Timestamp", parse_dates=True)
    test = pd.read_csv("Data/test_set.csv",
                       index_col="Timestamp", parse_dates=True)

    if "Unnamed: 0" in test.columns:
        test = test.drop(columns=["Unnamed: 0"])

    tickers = train.columns.tolist()
    common = [t for t in tickers if t in test.columns]
    test = test[common]
    idxs = [tickers.index(t) for t in common]

    muA = mu[idxs]
    SigmaA = Sigma[np.ix_(idxs, idxs)]

    target = 0.002
    w_bt = solve_markowitz_L2_TE(muA, SigmaA, target)

    p0 = test.iloc[0].values
    p1 = test.iloc[-1].values

    value0 = 1.0
    shares = w_bt * value0 / p0
    value1 = np.sum(shares * p1)
    ret = (value1 - value0)/value0

    print("\n========== Backtest (Model C) ==========")
    print(f"Actual return: {ret:.4%}")
    ann = (1+ret)**(252/len(test)) - 1
    print(f"Annualized: {ann:.4%}")
    print("========================================")

    top = np.argsort(w_bt)[-10:][::-1]
    print("\nTop 10 Holdings:")
    for i, ix in enumerate(top,1):
        print(f"{i}. {common[ix]}: {w_bt[ix]:.4%}")
