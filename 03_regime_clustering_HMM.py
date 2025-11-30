import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from matplotlib import pyplot as plt


# ============================================
# PARAMETERS
# ============================================

WINDOW_SIZE = 60        # rolling window for features
K = 5                   # number of regimes
MIN_LEN = 50            # minimum allowed contiguous regime length


# ============================================
# FEATURE ENGINEERING
# ============================================

def compute_features(prices, window):
    """
    prices : DataFrame (T × N)
        Clean close prices, no NaNs.
    window : int
        Rolling window size.

    Returns
    -------
    features_concat : DataFrame (T - window × (5*N))
        Clean rolling features. Timestamps preserved.
        Column names are like 'AAPL:mean', 'MSFT:vol', etc.
    """

    returns = prices.pct_change()
    features = []

    for ticker in prices.columns:
        r = returns[ticker]

        df_feat = pd.DataFrame({
            f"{ticker}:mean": r.rolling(window).mean(),
            f"{ticker}:vol":  r.rolling(window).std(),
            f"{ticker}:skew": r.rolling(window).skew(),
            f"{ticker}:kurt": r.rolling(window).kurt(),
            f"{ticker}:mom":  prices[ticker].pct_change(window)
        })

        features.append(df_feat)

    features_concat = pd.concat(features, axis=1)

    # Drop first  `window` rows (all NaNs)
    features_concat = features_concat.iloc[window:]

    return features_concat



# ============================================
# HMM-BASED REGIME CLUSTERING
# ============================================

def hmm_cluster(features_concat, k=K):
    """
    features_concat : DataFrame (T_clean × D)
        Output of compute_features().
    k : int
        Number of hidden regimes.

    Returns
    -------
    regimes : pd.Series (T_clean)
        Regime label for each time index.
    ranges : list of (regime_id, start_ts, end_ts)
        Contiguous regime ranges.
    """

    # 1. Standardize features (as HMM behaves better normalized)
    scaler = StandardScaler()
    X = scaler.fit_transform(features_concat)

    # 2. Fit Gaussian HMM
    hmm = GaussianHMM(
        n_components=k,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )

    hmm.fit(X)
    labels = hmm.predict(X)

    regimes = pd.Series(labels, index=features_concat.index, name="Regime")

    # 3. Extract contiguous regime segments
    ranges = []
    cur_regime = regimes.iloc[0]
    start_ts = regimes.index[0]

    for i in range(1, len(regimes)):
        if regimes.iloc[i] != cur_regime:
            end_ts = regimes.index[i - 1]
            ranges.append((cur_regime, start_ts, end_ts))
            cur_regime = regimes.iloc[i]
            start_ts = regimes.index[i]

    # final segment
    ranges.append((cur_regime, start_ts, regimes.index[-1]))

    return regimes, ranges



# ============================================
# MAIN SCRIPT
# ============================================

if __name__ == "__main__":

    # 1. Load data
    prices = pd.read_csv("sp500_20yr_clean.csv")
    dates = prices["Timestamp"]
    prices = prices.iloc[:, 2:]    # drop index + timestamp

    # Ensure date index on prices
    prices.index = dates

    # 2. Extract features
    features_concat = compute_features(prices, window=WINDOW_SIZE)

    # 3. Cluster regimes using HMM
    regimes, ranges = hmm_cluster(features_concat, k=K)

    # ============================================
    # OUTPUT SUMMARY
    # ============================================

    print("\n=== HMM Regime Segmentation Summary ===")
    for regime_id, start, end in ranges:
        count = regimes.loc[start:end].shape[0]
        print(f"Regime {regime_id}: {start} → {end}  ({count} points)")

    # Optionally save regimes
    regimes.to_csv("regimes_output.csv")
    print("\nSaved regime labels to regimes_output.csv")



    # ===========================
    # Regime Plot with Random Tickers + Vertical Regime Boundaries
    # ===========================

    M = 5   # number of tickers to sample randomly

    # Extract usable data
    all_tickers = prices.columns.to_list()
    sampled_tickers = np.random.choice(all_tickers, size=M, replace=False)

    timestamps = features_concat.index   # aligns with regimes
    price_subset = prices.loc[timestamps, sampled_tickers]

    plt.figure(figsize=(18, 9))

    # ----- Plot sampled asset prices -----
    for t in sampled_tickers:
        plt.plot(timestamps, price_subset[t], label=t, linewidth=1.0)

    plt.title(f"Sampled Asset Prices with Regime Boundaries (M={M})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="upper left")


    # ----- Plot vertical boundary lines -----
    for (regime_id, start, end) in ranges:
        # mark beginning of each regime segment (except the very first one)
        if start != ranges[0][1]:
            plt.axvline(x=start, color="red", linewidth=1.4, alpha=0.6)

    # ----- Optional: Color background by regime (uncomment if desired) -----
    # for (regime_id, start, end) in ranges:
    #     plt.axvspan(start, end, alpha=0.05 * (regime_id + 1), color="gray")

    plt.tight_layout()
    plt.show()