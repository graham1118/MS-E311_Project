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
# ------------------------------------------------------------
# Helper functions for backtesting table
# ------------------------------------------------------------
def backtest(w_bt, target, test):
    p0 = test.iloc[0].values
    p1 = test.iloc[-1].values

    initial_value = 1.0
    shares = w_bt * initial_value / p0
    final_value = np.sum(shares * p1)
    actual = (final_value - initial_value) / initial_value

    # Calculate daily returns for the portfolio
    portfolio_values = np.zeros(len(test))
    for t in range(len(test)):
        portfolio_values[t] = np.sum(shares * test.iloc[t].values)

    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Sharpe Ratio (assuming risk-free rate = 0)
    mean_daily_return = np.mean(daily_returns)
    std_daily_return = np.std(daily_returns, ddof=1)
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0

    # Sortino Ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_daily_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    # Annualized return calculation
    annualized = (1 + actual)**(252/len(test)) - 1

    return (actual, annualized, sharpe_ratio, sortino_ratio)



def get_month_name(chunk):
    """Extract month name from the middle date of a chunk."""
    mid_idx = len(chunk) // 2
    date = chunk.index[mid_idx]
    return date.strftime("%b %Y")


def find_best_regime(regime_results):
    """
    Find the regime with highest annualized return.

    Parameters:
    -----------
    regime_results : list of tuples
        Each tuple is (actual, annualized, sharpe_ratio, sortino_ratio)

    Returns:
    --------
    best_regime_idx : int
        Index of best performing regime
    """
    best_idx = 0
    best_annualized = regime_results[0][1]

    for i, result in enumerate(regime_results):
        if result[1] > best_annualized:
            best_annualized = result[1]
            best_idx = i

    return best_idx


def print_comparison_table(month_names, actual_returns, best_regimes,
                          best_regime_metrics, whole_dataset_metrics):
    """
    Print formatted comparison table.

    Parameters:
    -----------
    month_names : list of str
        Month name for each chunk
    actual_returns : list of float
        Actual return for each chunk
    best_regimes : list of int
        Best regime number for each chunk
    best_regime_metrics : list of tuples
        (annualized_return, sharpe_ratio) for best regime in each chunk
    whole_dataset_metrics : list of tuples
        (annualized_return, sharpe_ratio) for whole dataset in each chunk
    """
    num_chunks = len(month_names)

    print("\n" + "="*80)
    print("BACKTEST COMPARISON: REGIME-BASED vs WHOLE-DATASET PORTFOLIOS")
    print("="*80)

    # Header row 1: Month names
    print("\nMonth:".ljust(20), end="")
    for month in month_names:
        print(f"{month:>8}", end="")
    print()

    # Header row 2: Actual returns
    print("Actual Ret (%):".ljust(20), end="")
    for ret in actual_returns:
        print(f"{ret*100:>7.1f}%", end="")
    print()

    print("-"*80)

    # Row 3: Best regime number
    print("Best Reg #:".ljust(20), end="")
    for regime in best_regimes:
        print(f"{regime:>8}", end="")
    print()

    # Row 4: Best regime annualized return
    print("Best Reg Ann (%):".ljust(20), end="")
    for ann_ret, _ in best_regime_metrics:
        print(f"{ann_ret*100:>7.1f}%", end="")
    print()

    # Row 5: Best regime Sharpe ratio
    print("Best Reg Sharpe:".ljust(20), end="")
    for _, sharpe in best_regime_metrics:
        print(f"{sharpe:>8.1f}", end="")
    print()

    print("-"*80)

    # Row 6: Whole dataset annualized return
    print("Whole DS Ann (%):".ljust(20), end="")
    for ann_ret, _ in whole_dataset_metrics:
        print(f"{ann_ret*100:>7.1f}%", end="")
    print()

    # Row 7: Whole dataset Sharpe ratio
    print("Whole DS Sharpe:".ljust(20), end="")
    for _, sharpe in whole_dataset_metrics:
        print(f"{sharpe:>8.1f}", end="")
    print()

    print("="*80 + "\n")


# ------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------
if __name__ == "__main__":

    # -----------------------------------------
    # Load μ and Σ for whole dataset
    # -----------------------------------------
    npz_whole = np.load("../Data/mu_sigma_whole_dataset.npz")
    mu = npz_whole["mu"]
    Sigma = npz_whole["sigma"]

    npz_regimes = np.load("../Data/mu_sigma_regimes.npz")
    mu_r = npz_regimes["mu_all_regimes"]
    Sigma_r = npz_regimes["sigma_all_regimes"]

    # -----------------------------------------
    # Efficient Frontier
    # -----------------------------------------
    min_ret, max_ret = compute_return_range(mu)
    #print(f"Return range = [{min_ret:.6f}, {max_ret:.6f}]")

    target_returns = np.linspace(min_ret, max_ret, 60)
    risks, rets = [], []

    for r in target_returns:
        w = solve_markowitz_L2_TE(mu, Sigma, r)
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
    #plt.show()

    # -----------------------------------------
    # Backtesting Setup
    # -----------------------------------------
    sp500_train = pd.read_csv(
        "../Data/train_set.csv",
        index_col="Timestamp",
        parse_dates=True
    )
    test_set = pd.read_csv(
        "../Data/test_set.csv",
        index_col="Timestamp",
        parse_dates=True
    )


    if "Unnamed: 0" in test_set.columns:
        test = test_set.drop(columns=["Unnamed: 0"])


    # You can set this however you want; here we just use a fixed target
    target = 0.002  # daily return target

    chunk_size = 30
    num_chunks = len(test) // chunk_size  # Number of complete chunks

    # Initialize storage for results
    num_regimes = mu_r.shape[0]
    month_names = []
    actual_returns = []
    best_regimes = []
    best_regime_metrics = []
    whole_dataset_metrics = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = test.iloc[start_idx:end_idx]

        # Store month name and actual return
        month_names.append(get_month_name(chunk))

        # -----------------------------------------
        # Whole Dataset: Compute backtest return
        # -----------------------------------------
        w_bt_whole = solve_markowitz_L2_TE(mu, Sigma, target)

        if w_bt_whole is None:
            raise RuntimeError("Model A infeasible for backtest with this target.")
        (actual_whole, annualized_whole, sharpe_ratio_whole, sortino_ratio_whole) = backtest(w_bt_whole, target, chunk)

        # Store actual return (from whole dataset backtest)
        actual_returns.append(actual_whole)

        # Store whole dataset metrics
        whole_dataset_metrics.append((annualized_whole, sharpe_ratio_whole))

        # -----------------------------------------
        # Regime-based portfolios: Test all regimes
        # -----------------------------------------
        regime_results = []
        for regime in range(num_regimes):
            muB = mu_r[regime, :]
            SigmaB = Sigma_r[regime, :, :]
            w_bt_regime = solve_markowitz_L2_TE(muB, SigmaB, target)

            if w_bt_regime is None:
                raise RuntimeError(f"Model A infeasible for regime {regime} with this target.")
            (actual_regime, annualized_regime, sharpe_ratio_regime, sortino_ratio_regime) = backtest(w_bt_regime, target, chunk)
            regime_results.append((actual_regime, annualized_regime, sharpe_ratio_regime, sortino_ratio_regime))

        # Find best regime for this chunk
        best_regime_idx = find_best_regime(regime_results)
        best_regimes.append(best_regime_idx)

        # Store best regime metrics
        best_result = regime_results[best_regime_idx]
        best_regime_metrics.append((best_result[1], best_result[2]))  # (annualized, sharpe)

        print(f"Processed chunk {i+1}/{num_chunks}")

    # Print comparison table
    print_comparison_table(month_names, actual_returns, best_regimes,
                          best_regime_metrics, whole_dataset_metrics)