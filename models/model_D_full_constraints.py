# ============================================================
# model_D_full_constraints.py
# Full constrained model: L2 + TE + Weight Caps (SCS Solver)
# ============================================================

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GAMMA = 0.05
DELTA = 0.05
EPS_SIGMA = 1e-4
W_MAX = 0.05      # Now enforced in optimization (not only return-range)


# ------------------------------------------------------------
# SCS solver wrapper
# ------------------------------------------------------------
def solve_scs(prob):
    prob.solve(solver="SCS", verbose=False)


# ------------------------------------------------------------
# Model D: Fully Constrained Markowitz
# ------------------------------------------------------------
def solve_markowitz_full(mu, Sigma, target_return,
                         gamma=GAMMA, delta=DELTA):
    """
    Fully constrained Markowitz:
        minimize   wᵀ Σ w + γ||w||² + δ||w - w_bench||²
        s.t.       μᵀ w >= target_return
                   sum(w)=1
                   0 <= w <= W_MAX
    """
    n = len(mu)
    Sigma = 0.5*(Sigma + Sigma.T) + EPS_SIGMA*np.eye(n)

    w = cp.Variable(n)
    w_bench = np.ones(n)/n

    variance = cp.quad_form(w, Sigma)
    l2_pen   = cp.sum_squares(w)
    te_pen   = cp.sum_squares(w - w_bench)

    objective = cp.Minimize(variance + gamma*l2_pen + delta*te_pen)

    constraints = [
        cp.sum(w) == 1,
        mu @ w >= target_return,
        w >= 0,
        w <= W_MAX
    ]

    prob = cp.Problem(objective, constraints)
    solve_scs(prob)

    if w.value is None:
        return None
    return np.array(w.value)


# ------------------------------------------------------------
# Min/max return (long-only + caps)
# ------------------------------------------------------------
def compute_return_range(mu):
    n = len(mu)

    w = cp.Variable(n)
    cons = [cp.sum(w)==1, w>=0, w<=W_MAX]
    prob = cp.Problem(cp.Minimize(mu @ w), cons)
    solve_scs(prob)
    min_ret = float(mu @ w.value)

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

    npz = np.load("Data/mu_sigma_whole_dataset.npz")
    mu = npz["mu"]
    Sigma = npz["sigma"]

    min_ret, max_ret = compute_return_range(mu)
    print(f"Return range = [{min_ret:.6f}, {max_ret:.6f}]")

    ts = np.linspace(min_ret, max_ret, 60)
    risks, rets = [], []

    for r in ts:
        w = solve_markowitz_full(mu, Sigma, r)
        if w is None: 
            continue
        risks.append(np.sqrt(w @ Sigma @ w))
        rets.append(mu @ w)

    plt.figure(figsize=(7,5))
    plt.plot(risks, rets, "-o", markersize=3)
    plt.grid(True)
    plt.xlabel("Risk (Std Dev)")
    plt.ylabel("Expected Return")
    plt.title("Model D — Full Constraints Frontier")
    plt.tight_layout()
    plt.show()

    # Backtest -------------------
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
    w_bt = solve_markowitz_full(muA, SigmaA, target)

    p0 = test.iloc[0].values
    p1 = test.iloc[-1].values

    val0 = 1.0
    shares = w_bt * val0 / p0
    val1 = np.sum(shares * p1)

    actual = (val1 - val0)/val0
    ann = (1+actual)**(252/len(test)) - 1

    print("\n========== Backtest (Model D) ==========")
    print(f"Actual return: {actual:.4%}")
    print(f"Annualized: {ann:.4%}")
    print("========================================")

    top = np.argsort(w_bt)[-10:][::-1]
    print("\nTop 10 Holdings:")
    for i, ix in enumerate(top,1):
        print(f"{i}. {common[ix]}: {w_bt[ix]:.4%}")
