import numpy as np
import pandas as pd

def log_returns(prices):
    '''
    prices: a T x N datafram with index as dates and columns as tickers
    '''
    # Ensure we operate on a pandas DataFrame so we can use shift()
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)

    # Coerce columns to numeric where possible (non-numeric -> NaN)
    prices = prices.apply(pd.to_numeric, errors='coerce')

    # Log returns: log(p_t) - log(p_{t-1}) == np.log(prices).diff()
    # This preserves the DataFrame index and columns and returns floats.
    log_rets = np.log(prices).diff()

    return log_rets

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
sp500 = pd.read_csv("Data/sp500_20yr_clean.csv", index_col="Timestamp", parse_dates=True)




sp500_logrets = log_returns(sp500).dropna()
sp500_logrets.replace([np.inf, -np.inf], 0, inplace=True)
#print(sp500_logrets.head(), sp500_logrets.shape)

##### Read in Regimes #######
regimes = pd.read_csv("Data/regime_labels.csv", index_col="Date", parse_dates=True)
regimes = regimes.iloc[1:]

# Align regimes with sp500_logrets by index (dates)
# Keep only the regimes that match the dates in sp500_logrets
regimes = regimes.loc[sp500_logrets.index]
#print(regimes.head(), regimes.shape)


unique_regimes = regimes["Regime"].unique()
K = len(unique_regimes)
N = len(sp500.columns)


#regime number is 1st index
mu_all_regimes = np.zeros((K, N))
sigma_all_regimes = np.zeros((K, N, N))

for k in unique_regimes:
    mu, sigma = ew_mu_sigma(sp500_logrets.to_numpy(), regimes["Regime"], k)
    mu_all_regimes[k, :] = mu
    sigma_all_regimes[k, :, :] = sigma

np.savez("Data/mu_sigma", mu_all_regimes=mu_all_regimes, sigma_all_regimes=sigma_all_regimes)


# Calculate mu and sigma for the entire dataset (no regime separation)
# Create a dummy regime array where all observations are in regime 0
dummy_regimes = np.zeros(len(sp500_logrets))
mu_whole, sigma_whole = ew_mu_sigma(sp500_logrets.to_numpy(), dummy_regimes, 0)

np.savez("Data/mu_sigma_whole_dataset", mu=mu_whole, sigma=sigma_whole)
