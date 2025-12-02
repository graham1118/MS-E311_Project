import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import sys
import pandas_ta as ta
import matplotlib.dates as mdates

# ============================================
# PARAMETERS
# ============================================

ALGORITHM = "Kmeans" #"HMM" or "Kmeans"
ZIG_THRESH = 0.08
WINDOW_SIZE = 120        # rolling window for features
K = 5                 # number of regimes
MIN_LEN = 25            # minimum allowed contiguous regime length


# ============================================
# FEATURE ENGINEERING
# ============================================

def compute_features_new(prices, window=60):
    """
    prices : DataFrame (T × N)
        Clean close prices for N assets.

    Returns
    -------
    features : DataFrame (T × 5)
        Market-level rolling features:
        1. Market log return (mean across assets)
        2. Market rolling volatility
        3. Cross-sectional average volatility across assets
        4. Cross-sectional dispersion (std across assets)
        5. Market trend slope (60-day regression slope)
    """

    # ---------- Basic returns ----------
    log_prices = np.log(prices)
    log_returns = log_prices.diff()

    # ---------- 1. Market log return (cross-sectional mean) ----------
    market_ret = log_returns.mean(axis=1)

    # ---------- 2. Market rolling volatility ----------
    market_vol = market_ret.rolling(window).std()

    # ---------- 3. Cross-sectional average volatility ----------
    # Compute each asset's rolling vol, then average across assets
    indiv_vol = log_returns.rolling(window).std()
    cross_vol = indiv_vol.mean(axis=1)

    # ---------- 4. Cross-sectional dispersion (std across assets) ----------
    if prices.shape[1] == 1:
        cross_disp = pd.Series(0.0, index=log_returns.index)
    else:
        cross_disp = log_returns.std(axis=1)

    # ---------- 5. Market trend slope (60d linear regression slope) ----------
    def rolling_slope(series, window):
        """Rolling regression slope of price vs time."""
        slopes = []
        x = np.arange(window)
        for i in range(len(series)):
            if i < window:
                slopes.append(np.nan)
                continue
            y = series.iloc[i-window:i].values
            # simple linear regression slope (no intercept)
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)
        return pd.Series(slopes, index=series.index)

    market_trend = rolling_slope(log_prices.mean(axis=1), window)

    # ---------- Combine into feature matrix ----------
    features = pd.DataFrame({
        "market_ret": market_ret,
        "market_vol": market_vol,
        "cross_vol": cross_vol,
        "cross_disp": cross_disp,
        "market_trend": market_trend
    })

    # Clean NaN rows from initial window
    features = features.iloc[window:]

    return features


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

    if isinstance(features_concat, pd.Series):
        features_concat = features_concat.to_frame()

    # 1. Standardize features
    scaler = StandardScaler()
    #X = scaler.fit_transform(features_concat)
    X = scaler.fit_transform(features_concat.values.reshape(-1, features_concat.shape[1])) #for only 1 column

    # 2. Fit Gaussian HMM
    hmm = GaussianHMM(
        n_components=k,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )

    #1.5 do PCA
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)

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

def kmeans_cluster(features_concat, k, min_len=100):
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

   # Always convert to DataFrame (works for Series or DataFrame)
    features_df = pd.DataFrame(features_concat)

    # Standardize (scaler keeps 1 or many columns correctly)
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df.values)

    # #1.5 do PCA
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)


    # plt.figure()
    # plt.plot(range(X.shape[0]), X)
    # plt.show()

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

# ============================================
# ZIGZAG SMOOTHING (PRE-CLUSTERING)
# ============================================

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

def zigzag_tv(series, percent=0.02):
    """
    TradingView-style ZigZag with linear interpolation between pivot points.
    percent = reversal threshold (0.02 = 2%)
    """
    s = series.values
    idx = series.index
    n = len(s)

    if n < 3:
        return series.copy()

    pivots = np.full(n, np.nan)
    last_pivot = 0
    last_pivot_price = s[0]
    direction = 0  # +1 uptrend, -1 downtrend

    for i in range(1, n):
        change = (s[i] - last_pivot_price) / last_pivot_price

        if direction == 0:
            if change > percent:
                direction = +1
                last_pivot = i
                last_pivot_price = s[i]
                pivots[i] = s[i]
            elif change < -percent:
                direction = -1
                last_pivot = i
                last_pivot_price = s[i]
                pivots[i] = s[i]

        elif direction == +1:
            if s[i] > last_pivot_price:   # new high continues trend
                last_pivot = i
                last_pivot_price = s[i]
            elif change < -percent:       # down reversal
                direction = -1
                last_pivot = i
                last_pivot_price = s[i]
                pivots[i] = s[i]

        elif direction == -1:
            if s[i] < last_pivot_price:   # new low continues trend
                last_pivot = i
                last_pivot_price = s[i]
            elif change > percent:        # up reversal
                direction = +1
                last_pivot = i
                last_pivot_price = s[i]
                pivots[i] = s[i]

    # ---- NOW PERFORM LINEAR INTERPOLATION BETWEEN PIVOTS ----
    zigzag = pd.Series(np.nan, index=idx)

    # get pivot indices
    pivot_idx = np.where(~np.isnan(pivots))[0]

    if len(pivot_idx) < 2:
        # not enough pivots -> no zigzag possible
        return series.copy()

    for j in range(len(pivot_idx) - 1):
        a = pivot_idx[j]
        b = pivot_idx[j + 1]

        y0 = pivots[a]
        y1 = pivots[b]

        # linear interpolation from pivot a to pivot b
        steps = b - a
        if steps <= 0:
            continue

        line = np.linspace(y0, y1, steps + 1)
        zigzag.iloc[a:b+1] = line

    # Fill start/end if needed
    zigzag.iloc[:pivot_idx[0]] = pivots[pivot_idx[0]]
    zigzag.iloc[pivot_idx[-1]:] = pivots[pivot_idx[-1]]

    return zigzag


def prepare_SPX_plot(smoothed):
    plt.figure(figsize=(12, 6))

    plt.plot(SPX.index, SPX["High"], 
         label="SPX Close", 
         alpha=0.4, 
         linewidth=1.5)

    # Plot ZigZag smoothed
    plt.plot(SPX.index, smoothed, 
            label="ZigZag (8%)", 
            linewidth=2.0)

    # Title and labels
    plt.title("SPX Close vs. ZigZag Smoothed Series (1% Threshold)")
    plt.xlabel("Date")
    plt.ylabel("Price Level")

    # Format x-axis ticks (rotate + nice date formatting)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))      # tick every 2 years
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    plt.legend()
    plt.tight_layout()
# ============================================
# MAIN SCRIPT
# ============================================

if __name__ == "__main__":

   
    SPX = pd.read_csv("SPX_raw.csv", index_col="Date", parse_dates=True) 
    smoothed = zigzag_tv(SPX["High"], percent=ZIG_THRESH)

    zig_ret = smoothed.pct_change().fillna(0)

    prepare_SPX_plot(smoothed)

    
    
   



    # 3. Extract features
    features = compute_features_new(smoothed.to_frame(), window=WINDOW_SIZE)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.interpolate(method="linear")

    # 4. Cluster regimes using HMM or KMEANS
    if ALGORITHM == "Kmeans":
        regimes, ranges = kmeans_cluster(features, k=K)

    elif ALGORITHM == "HMM":
        regimes, ranges = hmm_cluster(features, k=K)

    else:
        sys.exit(0, "incorrect algo selection")

    regimes.to_csv("regime_labels.csv", header=True)
    
    # ============================================
    # OUTPUT SUMMARY
    # ============================================

    print("\n=== HMM Regime Segmentation Summary ===")
    for regime_id, start, end in ranges:
       count = regimes.loc[start:end].shape[0]
       print(f"Regime {regime_id}: {start} → {end}  ({count} points)")


    # ===========================
    # Regime Plot with Random Tickers + Vertical Regime Boundaries
    # ===========================

    # M = 5   # number of tickers to sample randomly

    # # Extract usable data
    # all_tickers = prices.columns.to_list()
    # sampled_tickers = np.random.choice(all_tickers, size=M, replace=False)

    # timestamps = prices.index   # aligns with regimes
    # price_subset = prices.loc[timestamps, sampled_tickers]


    # ----- Plot vertical boundary lines -----
    for (regime_id, start, end) in ranges:
        # mark beginning of each regime segment (except the very first one)
        if start != ranges[0][1]:
            plt.axvline(x=start, color="red", linewidth=1.4, alpha=0.6)

    #ADD SHADING
    for (regime_id, start, end) in ranges:
        plt.axvspan(start, end,
                alpha=0.15,
                color=plt.cm.tab10(regime_id % 10))
    
    plt.tight_layout()
    plt.show()