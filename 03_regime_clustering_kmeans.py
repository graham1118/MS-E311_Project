import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

WINDOW_SIZE = 60 #for computing features
K = 10

def compute_features(prices, window):
    """
    prices: DataFrame (T x N) of clean close prices
    window: the rolling window across which to compute features

    Returns
    -------
    features_concat : DataFrame (T - window x (5*N))
        Concatenated rolling features. The first `window` rows
        (which are all NaNs) are removed.
        Columns are named like 'AAPL:mean', 'AAPL:vol', ...
    """

    returns = prices.pct_change()
    features = []  # list of (T x 5) feature blocks per asset

    for ticker in prices.columns:
        r = returns[ticker]

        # 5D feature vector per asset, with ticker in the column names
        df_feat = pd.DataFrame({
            f"{ticker}:mean": r.rolling(window).mean(),
            f"{ticker}:vol":  r.rolling(window).std(),
            f"{ticker}:skew": r.rolling(window).skew(),
            f"{ticker}:kurt": r.rolling(window).kurt(),
            f"{ticker}:mom":  prices[ticker].pct_change(window)  # momentum
        })

        features.append(df_feat)

    # Concatenate horizontally → shape (T, 5*N)
    features_concat = pd.concat(features, axis=1)

    # Drop the first `window` rows (all NaNs from rolling + pct_change)
    features_concat = features_concat.iloc[window:]

    return features_concat

def cluster_regimes(features_concat, k, min_len=100):
    """
    features_concat : DataFrame (T_clean × D)
        Rolling-window features with time index.
    k : int
        Number of clusters (regimes).
    min_len : int
        Minimum allowed length for any contiguous regime segment.

    Returns
    -------
    regimes_smoothed : pd.Series
        Smoothed regime labels of length T_clean.
    regime_ranges : list of (regime_id, start_idx, end_idx)
        Final contiguous ranges after smoothing.
    """

    # 1. Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features_concat)

    # 2. K-means clustering
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)

    labels = np.array(labels)
    n = len(labels)

    # 3. Enforce minimum regime length
    i = 0
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        seg_len = j - i

        if seg_len < min_len:
            # Replace this short segment with the "closest" neighbor
            if i == 0:
                # Only future neighbor exists
                labels[i:j] = labels[j]
            elif j == n:
                # Only previous neighbor exists
                labels[i:j] = labels[i-1]
            else:
                # Choose the neighbor whose cluster center is closer
                prev_label = labels[i-1]
                next_label = labels[j]
                # (you can also just default to previous)
                labels[i:j] = prev_label  

        i = j

    regimes_smoothed = pd.Series(labels, index=features_concat.index, name="Regime")

    # 4. Build final contiguous regime ranges
    ranges = []
    current = regimes_smoothed.iloc[0]
    start_idx = regimes_smoothed.index[0]

    for t in range(1, len(regimes_smoothed)):
        if regimes_smoothed.iloc[t] != current:
            end_idx = regimes_smoothed.index[t-1]
            ranges.append((current, start_idx, end_idx))
            current = regimes_smoothed.iloc[t]
            start_idx = regimes_smoothed.index[t]

    # last segment
    ranges.append((current, start_idx, regimes_smoothed.index[-1]))

    return regimes_smoothed, ranges


# ===========================
prices = pd.read_csv("sp500_20yr_clean.csv")

date = prices["Timestamp"]
prices = prices.iloc[:, 2:]



features_concat = compute_features(prices, window=WINDOW_SIZE)


#2. Cluster regimes jointly using k-means
regimes, ranges = cluster_regimes(features_concat, k=3)

print("\n=== Regime Segmentation Summary ===")
for regime_id, start, end in ranges:
    count = regimes.loc[start:end].shape[0]
    print(f"Regime {regime_id}: {start} → {end}  ({count} points)")
