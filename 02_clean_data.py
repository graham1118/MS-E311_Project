import pandas as pd

try:
    df = pd.read_csv("sp500_20yr_close.csv")
except:
    print("sp500_20yr_close.csv has not yet been created! Run 01_gather_data.py")
df = df.ffill()
print(df.shape)

df.to_csv("sp500_20yr_clean.csv")