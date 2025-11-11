import cvxpy as cp
import numpy as np

def solve_markowitz(mu, Sigma, lambda_risk, long_only=True):
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
    objective = cp.Minimize(cp.quad_form(w, Sigma) - lambda_risk * (mu @ w))

    # Constraints:
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)

    # Problem declaration and execution
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)  # or cp.OSQP, cp.ECOS, cp.CVXOPT

    # Extract result
    if w.value is None:
        raise RuntimeError("Optimization failed. Check Σ positive-definiteness.")
    
    return np.array(w.value)


# ------------------------------------------------------------
# Example Usage (Global Model)
# ------------------------------------------------------------
# Suppose you have computed:
# mu_global = np.array([...])           # shape (n,)
# Sigma_global = np.array([...])        # shape (n, n)

# lambda_risk = 5.0  # Example risk-aversion parameter

# w_global = solve_markowitz(mu_global, Sigma_global, lambda_risk)
# print("Global Portfolio Weights:", w_global)


# ------------------------------------------------------------
# Regime-Based Usage Example
# ------------------------------------------------------------
# Suppose you detect current regime R and compute:
# mu_R = mu_regime_dict[R]
# Sigma_R = Sigma_regime_dict[R]

# w_regime = solve_markowitz(mu_R, Sigma_R, lambda_risk)
# print(f"Portfolio Weights in Regime {R}:", w_regime)
