import cvxpy as cp
import numpy as np

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



npzfile = np.load("Data/mu_sigma.npz")
mu_all_regimes = npzfile['mu_all_regimes']
sigma_all_regimes = npzfile['sigma_all_regimes']
K = mu_all_regimes.shape[0] #number of regimes
N = mu_all_regimes.shape[1] #number of assets (columns)

weights_all_regimes = np.zeros((K, N))

for k in range(mu_all_regimes.shape[0]):
    weights_all_regimes[k, :] = solve_markowitz(mu_all_regimes[k, :], sigma_all_regimes[k, :, :], LAM, GAMMA)





# ------------------------------------------------------------
# Efficient Frontier Plot (Option A: Fix return, minimize variance)
# ------------------------------------------------------------
import matplotlib.pyplot as plt

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
mu = mu_all_regimes[0]
Sigma = sigma_all_regimes[0]
Sigma = (1 - delta) * Sigma + delta * np.diag(np.diag(Sigma))

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
