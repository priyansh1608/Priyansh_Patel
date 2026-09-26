import pandas as pd

df = pd.read_csv("SESSION 3\ipl_data.csv")

print("IPL Dataset:")
print(df)

print("\nMissing values in each column:")
print(df.isnull().sum())