import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

SMOOTH_FACTOR = 100

def kalman_denoise_multifeature(
    X,
    process_var=1e-4,     # process noise variance (Q)
    smoothing_amt=SMOOTH_FACTOR     # multiplies with obs_var (R)
   
):
    

    """
    Apply a simple 1D Kalman filter to each feature column independently.
    State model: x_t = x_{t-1} + w_t
    Observation: z_t = x_t + v_t

    Parameters
    ----------
    X : np.ndarray
        Shape (n_timesteps, n_features)
    process_var : float
        Process noise variance (how much the hidden state can drift)
    obs_var : float
        Observation noise variance (how noisy measurements are)

    Returns
    -------
    np.ndarray
        Smoothed data of same shape.
    """

    obs_var = process_var * smoothing_amt  # observation noise variance (R)


    X = np.asarray(X, dtype=np.float32)
    n, m = X.shape
    out = np.zeros_like(X)

    for j in range(m):
        z = X[:, j]
        x_est = np.zeros(n, dtype=np.float32)
        P = 1.0  # initial estimate uncertainty
        x_est[0] = z[0]

        for t in range(1, n):
            # Prediction
            x_pred = x_est[t-1]
            P_pred = P + process_var

            # Kalman Gain
            K = P_pred / (P_pred + obs_var)

            # Update
            x_est[t] = x_pred + K * (z[t] - x_pred)
            P = (1 - K) * P_pred

        out[:, j] = x_est

    return out


try:
    df = pd.read_csv("sp500_20yr_raw.csv")
except:
    print("sp500_20yr_close.csv has not yet been created! Run 01_gather_data.py")



## Fills in single missing NaNs using linear interpolation
df = df.interpolate(method="linear")

# 1. Identify columns whose first value is NaN
bad_cols = df.columns[df.iloc[0].isna()]
print("Columns with NaN in first row:", len(bad_cols))
df = df.drop(columns=bad_cols)



orig_df = df.copy(deep=True)
df.iloc[:,1:] = kalman_denoise_multifeature(df.iloc[:,1:])
print(df.head())

print(df.shape)
df = pd.DataFrame(df) #convert back to dataframe
df.to_csv("sp500_20yr_clean.csv")




#Plotting Results of denoising
PLOT_LENGTH = 150
columns = df.columns.to_list()
rand_ticker = np.random.randint(1, df.shape[1])            #choose a random ticker, skip the first 2 columns (index and timestamp)
start = np.random.randint(0, df.shape[0] - PLOT_LENGTH)    #choose a random period in time to plot
end = start + PLOT_LENGTH



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6))
ax1.plot(range(PLOT_LENGTH), orig_df.iloc[start:end, rand_ticker]) 
ax1.set_title(f"Raw prices for {columns[rand_ticker]} starting {df.iloc[start, 0]}")

ax2.plot(range(PLOT_LENGTH), df.iloc[start:end, rand_ticker]) 
ax2.set_title(f"Denoised prices for {columns[rand_ticker]} starting {df.iloc[start, 0]}")
plt.show()

