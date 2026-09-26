import pandas as pd

df = pd.read_csv("SESSION 4\myshow_data.csv")

ratings = df["user ratings"]

Q1 = ratings.quantile(0.25)
Q3 = ratings.quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("Q1 =", Q1)
print("Q3 =", Q3)
print("IQR =", IQR)
print("Lower Limit =", lower_limit)
print("Upper Limit =", upper_limit)

filtered_df = df[
    (df["user ratings"] >= lower_limit) &
    (df["user ratings"] <= upper_limit)
]

rows_dropped = len(df) - len(filtered_df)

print("\nOriginal number of rows:", len(df))
print("Rows dropped:", rows_dropped)
print("Remaining rows:", len(filtered_df))

print("\nData after removing outliers:")
print(filtered_df)