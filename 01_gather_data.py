import datetime as dt
import pandas as pd
import yfinance as yf
import time
import sys

MIN_YEARS_HISTORY = 20
SAVE_PATH = f"Data/sp500_{MIN_YEARS_HISTORY}yr_raw.csv"

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

clean_df.to_csv("Data/SPX_raw.csv", index=True)

sys.exit(0)
####################################

tickers = pd.read_csv("Data/sp500_tickers.csv")["Symbol"].tolist()

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



