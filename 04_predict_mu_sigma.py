import numpy as np
import pandas as pd


def ew_mu_sigma(returns, regimes, k, alpha=0.97, shrink_cov=0.2):
    """
    returns: T x N returns matrix
    regimes: length-T integer array of regime labels in {0,...,K-1}
    k: regime index we’re estimating for
    alpha: EW decay parameter (close to 1 = long memory)
    shrink_cov: shrinkage intensity for covariance
    """
    # 1. subset to regime k
    idx = np.where(regimes == k)[0]
    if len(idx) == 0:
        raise ValueError("No data in regime {}".format(k))
    Rk = returns[idx]        # shape: n_k x N
    n_k = Rk.shape[0]

    # 2. EW weights (most recent = last row)
    j = np.arange(n_k)       # 0 ... n_k-1
    w = (1 - alpha) * alpha**(n_k - 1 - j)
    w = w / w.sum()          # normalize

    # 3. weighted mean
    mu = w @ Rk              # shape: (N,)

    # 4. weighted covariance
    X = Rk - mu              # n_k x N
    Sigma_raw = (X * w[:, None]).T @ X   # N x N

    # 5. shrinkage toward diag target
    diag_target = np.diag(np.diag(Sigma_raw))
    Sigma = (1 - shrink_cov) * Sigma_raw + shrink_cov * diag_target

    return mu, Sigma



##### Read in Data #####
sp500 = pd.read_csv("Data/train_set.csv", index_col="Timestamp", parse_dates=True)



##### Read in Regimes #######
regimes = pd.read_csv("Data/regime_labels.csv", index_col="Date", parse_dates=True)
print(f"Regimes initial shape: {regimes.shape}")

# Drop first row to align with log returns (which lose the first row from .diff())
regimes = regimes.iloc[1:]
print(f"Regimes after dropping first row: {regimes.shape}")

# Use position-based alignment: take only the first len(sp500) rows
regimes = regimes.iloc[:len(sp500)]
print(f"Regimes after position-based alignment: {regimes.shape}")
print(f"SP500 shape: {sp500.shape}")



unique_regimes = regimes["Regime"].unique()
print(regimes.shape)
print(f"Unique regimes: {unique_regimes}")
K = len(unique_regimes)
N = len(sp500.columns)



#regime number is 1st index - use enumerate to avoid index errors
mu_all_regimes = np.zeros((K, N))
sigma_all_regimes = np.zeros((K, N, N))

for i, k in enumerate(unique_regimes):
    mu, sigma = ew_mu_sigma(sp500.to_numpy(), regimes["Regime"].to_numpy(), k)
    mu_all_regimes[i, :] = mu
    sigma_all_regimes[i, :, :] = sigma

np.savez("Data/mu_sigma_regimes", mu_all_regimes=mu_all_regimes, sigma_all_regimes=sigma_all_regimes)


# Calculate mu and sigma for the entire dataset (no regime separation)
# Create a dummy regime array where all observations are in regime 0
dummy_regimes = np.zeros(len(sp500))
mu_whole, sigma_whole = ew_mu_sigma(sp500.to_numpy(), dummy_regimes, 0)

np.savez("Data/mu_sigma_whole_dataset", mu=mu_whole, sigma=sigma_whole)
