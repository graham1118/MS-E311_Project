import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LAM = 1 # higher = more penalty on variance, risk averse, 
        # lower = more risk-tolerant
GAMMA = 0.15
delta = 0.15
c = 0.5
def solve_markowitz(mu, Sigma, lambda_risk, gamma_reg, long_only=True):
    """
    Solve the mean-variance optimization:
        minimize    w^T Sigma w - λ (mu^T w)
        subject to  sum(w) = 1,  w >= 0   (if long_only=True)
    
    Parameters
    ----------
    mu : np.array of shape (n,)
        Expected returns for each asset.
    Sigma : np.array of shape (n,n)
        Covariance matrix of returns.
    lambda_risk : float
        Risk aversion parameter λ.
    long_only : bool
        If True, imposes w >= 0 (no short-selling).

    Returns
    -------
    w_opt : np.array of shape (n,)
        Optimal portfolio weights.
    """
    
    n = len(mu)  # number of assets
    
    # Define optimization variable (portfolio weights)
    w = cp.Variable(n)

    # Objective function:
    # Variance term: w^T Σ w
    # Return term: (mu^T w)
    # CVXPY minimizes, so we NEGATE the return term scaled by λ.
    objective = cp.Minimize(cp.quad_form(w, Sigma) - lambda_risk * (mu @ w) + gamma_reg * cp.sum_squares(w))

    # Constraints:
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)

    constraints.append(cp.norm(w, 2) <= c)
    # Problem declaration and execution
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)  # or cp.OSQP, cp.ECOS, cp.CVXOPT

    # Extract result
    if w.value is None:
        raise RuntimeError("Optimization failed. Check Σ positive-definiteness.")
    
    return np.array(w.value)



npzfile = np.load("Data/mu_sigma_regimes.npz")
mu_all_regimes = npzfile['mu_all_regimes']
sigma_all_regimes = npzfile['sigma_all_regimes']
K = mu_all_regimes.shape[0] #number of regimes
N = mu_all_regimes.shape[1] #number of assets (columns)

npz_whole_dataset = np.load("Data/mu_sigma_whole_dataset.npz")
mu_whole = npz_whole_dataset['mu']
sigma_whole = npz_whole_dataset['sigma']

weights_all_regimes = np.zeros((K, N))

for k in range(mu_all_regimes.shape[0]):
    weights_all_regimes[k, :] = solve_markowitz(mu_all_regimes[k, :], sigma_all_regimes[k, :, :], LAM, GAMMA)





# ------------------------------------------------------------
# Efficient Frontier Plot (Option A: Fix return, minimize variance)
# ------------------------------------------------------------

def solve_min_variance_for_return(mu, Sigma, target_return, long_only=True):
    """
    Solve:
        minimize    w^T Sigma w
        subject to  w^T mu = target_return
                    sum(w) = 1
                    w >= 0 (if long_only)
    """
    n = len(mu)
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, Sigma))

    constraints = [
        w @ mu == target_return,
        cp.sum(w) == 1
    ]
    if long_only:
        constraints.append(w >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    if w.value is None:
        raise RuntimeError("Frontier optimization failed.")

    return np.array(w.value)


# Pick a regime to plot (e.g., the first one)
#mu = mu_all_regimes[5]
#Sigma = sigma_all_regimes[5]
mu = mu_whole
Sigma = sigma_whole

# Sweep returns from minimum to maximum achievable
target_returns = np.linspace(mu.min(), mu.max(), 40)

frontier_returns = []
frontier_risks = []

for r_target in target_returns:
    w_opt = solve_min_variance_for_return(mu, Sigma, r_target)
    frontier_returns.append(r_target)
    frontier_risks.append(np.sqrt(w_opt @ Sigma @ w_opt))  # standard deviation


# Plot
plt.figure(figsize=(7, 5))
plt.plot(frontier_risks, frontier_returns, '-o', markersize=3)
plt.xlabel("Portfolio Risk (Std Dev)")
plt.ylabel("Portfolio Return")
plt.title("Efficient Frontier (Regime 0)")
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Backtesting: Apply optimal weights to test set
# ------------------------------------------------------------

# Read the training data to get column names (stock tickers)
sp500_train = pd.read_csv("Data/sp500_20yr_clean.csv", index_col="Timestamp", parse_dates=True)
stock_tickers = sp500_train.columns.tolist()

# Read test set
test_set = pd.read_csv("Data/test_set.csv", index_col="Timestamp", parse_dates=True)
# Drop 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in test_set.columns:
    test_set = test_set.drop(columns=['Unnamed: 0'])

# Align test_set columns with training data columns
# Keep only stocks that are in both datasets
common_stocks = [ticker for ticker in stock_tickers if ticker in test_set.columns]
test_set_aligned = test_set[common_stocks]

# Get indices of common stocks in the original training data
stock_indices = [stock_tickers.index(ticker) for ticker in common_stocks]

# Subset mu and Sigma to only include common stocks
mu_aligned = mu[stock_indices]
Sigma_aligned = Sigma[np.ix_(stock_indices, stock_indices)]

# Choose desired return and solve for optimal weights
target_return = 0.002  # Adjust this to your desired daily return
w_backtest = solve_min_variance_for_return(mu_aligned, Sigma_aligned, target_return)

# Get first and last day prices
first_day_prices = test_set_aligned.iloc[0, :].values  # First row (day 1)
last_day_prices = test_set_aligned.iloc[-1, :].values  # Last row (day 30)

# Calculate portfolio value change
# Assume we start with $1 invested according to weights w_backtest
# Number of shares bought on day 1: w_backtest[i] / first_day_prices[i]
# Value on day 30: sum(shares[i] * last_day_prices[i])

# Initial portfolio value (normalized to 1)
initial_value = 1.0

# Shares of each stock purchased with initial allocation
shares = w_backtest * initial_value / first_day_prices

# Final portfolio value
final_value = np.sum(shares * last_day_prices)

# Calculate actual return
actual_return = (final_value - initial_value) / initial_value

print("\n" + "="*60)
print("BACKTESTING RESULTS")
print("="*60)
print(f"Target daily return: {target_return:.6f}")
print(f"Test period: {test_set.index[0]} to {test_set.index[-1]}")
print(f"Number of days: {len(test_set)}")
print(f"Initial portfolio value: ${initial_value:.2f}")
print(f"Final portfolio value: ${final_value:.2f}")
print(f"Actual return over test period: {actual_return:.4%}")
print(f"Annualized return (assuming 252 trading days): {(1 + actual_return)**(252/len(test_set)) - 1:.4%}")
print("="*60)

# Print top 10 holdings
top_holdings_idx = np.argsort(w_backtest)[-10:][::-1]
print("\nTop 10 Holdings:")
for i, idx in enumerate(top_holdings_idx, 1):
    print(f"{i}. {common_stocks[idx]}: {w_backtest[idx]:.4%}")
print("="*60)
print(f"\nNote: Portfolio constructed using {len(common_stocks)} stocks present in both training and test sets.")