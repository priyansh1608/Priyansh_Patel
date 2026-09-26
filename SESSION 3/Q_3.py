import pandas as pd

df = pd.read_csv("SESSION 3\ipl_data.csv")

df["team"] = df["team"].fillna("Unknown")

print("First 10 rows:")
print(df.head(10))