import pandas as pd
import numpy as np

df = pd.read_csv("SESSION 22\zomato.csv")

df = df.rename(columns={
    "rate": "rating",
    "approx_cost(for two)": "cost",
    "cuisines": "cuisine"
})

df["rating"] = pd.to_numeric(
    df["rating"].astype(str).str.replace("/5", "", regex=False),
    errors="coerce"
)

df["cost"] = pd.to_numeric(
    df["cost"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

print("Missing Values:")
print(df[["cost", "rating"]].isnull().sum())

df["rating"] = df["rating"].fillna(df["rating"].median())
df["cost"] = df["cost"].fillna(df["cost"].median())

Q1_rating = df["rating"].quantile(0.25)
Q3_rating = df["rating"].quantile(0.75)
IQR_rating = Q3_rating - Q1_rating

lower_rating = Q1_rating - 1.5 * IQR_rating
upper_rating = Q3_rating + 1.5 * IQR_rating

Q1_cost = df["cost"].quantile(0.25)
Q3_cost = df["cost"].quantile(0.75)
IQR_cost = Q3_cost - Q1_cost

lower_cost = Q1_cost - 1.5 * IQR_cost
upper_cost = Q3_cost + 1.5 * IQR_cost

print("\nRating Outliers Before Handling:")
print(((df["rating"] < lower_rating) | (df["rating"] > upper_rating)).sum())

print("\nCost Outliers Before Handling:")
print(((df["cost"] < lower_cost) | (df["cost"] > upper_cost)).sum())

df["rating"] = np.clip(df["rating"], lower_rating, upper_rating)
df["cost"] = np.clip(df["cost"], lower_cost, upper_cost)

print("\nMissing Values After Handling:")
print(df[["cost", "rating"]].isnull().sum())

print("\nOutliers Handled Successfully.")