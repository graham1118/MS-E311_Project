import pandas as pd

# Read in the three CSV files
sp500_clean = pd.read_csv("Data/sp500_20yr_clean.csv", index_col="Timestamp", parse_dates=True)
sp500_raw = pd.read_csv("Data/sp500_20yr_raw.csv", index_col="Timestamp", parse_dates=True)
spx_raw = pd.read_csv("Data/SPX_raw.csv", index_col="Date", parse_dates=True)

# Get the number of rows
N_clean = sp500_clean.shape[0]
N_raw = sp500_raw.shape[0]
N_spx = spx_raw.shape[0]

# Create test set from last 30 points of sp500_clean
test_set = sp500_clean.iloc[N_clean-30:, :]
test_set.to_csv("Data/test_set.csv")

# Shorten each file by removing last 30 points
sp500_clean_shortened = sp500_clean.iloc[:N_clean-30, :]
sp500_raw_shortened = sp500_raw.iloc[:N_raw-30, :]
spx_raw_shortened = spx_raw.iloc[:N_spx-30, :]

# Save the shortened files (overwriting originals)
sp500_clean_shortened.to_csv("Data/sp500_20yr_clean.csv")
sp500_raw_shortened.to_csv("Data/sp500_20yr_raw.csv")
spx_raw_shortened.to_csv("Data/SPX_raw.csv")

print(f"Created test_set.csv with {test_set.shape[0]} rows and {test_set.shape[1]} columns")
print(f"Shortened sp500_20yr_clean.csv from {N_clean} to {sp500_clean_shortened.shape[0]} rows")
print(f"Shortened sp500_20yr_raw.csv from {N_raw} to {sp500_raw_shortened.shape[0]} rows")
print(f"Shortened SPX_raw.csv from {N_spx} to {spx_raw_shortened.shape[0]} rows")
