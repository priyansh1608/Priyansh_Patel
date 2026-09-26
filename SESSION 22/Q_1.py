import pandas as pd

df = pd.read_csv("SESSION 22\zomato.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Description:")
print(df.describe(include="all"))