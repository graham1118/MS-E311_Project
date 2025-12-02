import datetime as dt
import pandas as pd
import yfinance as yf
import time
import sys

MIN_YEARS_HISTORY = 20
SAVE_PATH = f"sp500_{MIN_YEARS_HISTORY}yr_raw.csv"

end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=MIN_YEARS_HISTORY * 365)


########### SPX ONLY ###############
ticker = "^SPX"
raw_df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
    )




clean_df = raw_df.xs('^SPX', level=1, axis=1)[['High', 'Low']]

print(clean_df.head())

clean_df.to_csv("SPX_raw.csv", index=True)

sys.exit(0)
####################################

tickers = pd.read_csv("sp500_tickers.csv")["Symbol"].tolist()

accum_df = pd.DataFrame()
timestamps = None
first = True

for ticker in tickers:
    print("Downloading:", ticker)

    raw_df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False,
    )

    if raw_df.empty:
        print("  -> no data, skipping")
        continue

    # Drop the first 2 rows
    trimmed = raw_df.iloc[2:]

    if first:
        # Store timestamp column (first column)
        timestamps = trimmed.iloc[:, 0].rename("Timestamp")   # Open column? No — this is incorrect.
        # Actually extract the index (the Timestamp)
        timestamps = trimmed.index.to_series().rename("Timestamp")

        first = False

    # Extract second column only
    col = trimmed.iloc[:, 1].rename(ticker)

    # Add column to accumulator
    accum_df = pd.concat([accum_df, col], axis=1)
    time.sleep(0.2)

# Insert timestamps as the FIRST column
accum_df.insert(0, "Timestamp", timestamps.values)

print("Final shape:", accum_df.shape)
accum_df.to_csv(SAVE_PATH, index=False)
print("Saved:", SAVE_PATH)



# raw_df = pd.read_csv("sp500_clean_20yr_close.csv")
# raw_df = raw_df.rename(columns={'Ticker':'Date'})
# raw_df = raw_df.drop([0,1], axis=0)


# sliced_df = raw_df.iloc[:, 1::5]
# sliced_df.columns = sliced_df.columns.str.replace(".4", "", regex=False)
# print(sliced_df.shape)
# df = pd.concat([raw_df['Date'], sliced_df], axis=1)
# sliced_df.to_csv(CLEAN_SAVE_PATH)

# # ------------------------------
# # FILTER BY MINIMUM HISTORY
# # ------------------------------
# obs_counts = close_df.count()          # Number of non-NaN entries
# min_required = MIN_YEARS_HISTORY * 252 # Yahoo returns daily data, ~252 trading days per year 
# keep_tickers = obs_counts[obs_counts >= min_required].index

# close_df = close_df[keep_tickers]
# print(f"\nKeeping {len(keep_tickers)} tickers with ≥ {MIN_YEARS_HISTORY} years of data...\n")



# # ------------------------------
# # CLEAN NAN ROWS
# # ------------------------------
# close_df = close_df.dropna(axis=0, how='any')
# close_df.index = pd.to_datetime(close_df.index)



# # ------------------------------
# # SAVE TO CSV
# # ------------------------------
# close_df.to_csv(SAVE_PATH)
# print(f"Saved clean dataset to: {SAVE_PATH}")

# print("Final shape:", close_df.shape)
# print("Num columns", len(close_df))
# print("Columns:", list(close_df.columns))
