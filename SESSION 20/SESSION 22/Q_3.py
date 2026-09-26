import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

df["rating"] = df["rating"].fillna(df["rating"].median())
df["cost"] = df["cost"].fillna(df["cost"].median())

df["cuisine"] = df["cuisine"].fillna("Unknown")
df["location"] = df["location"].fillna("Unknown")

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

encoded_data = encoder.fit_transform(
    df[["cuisine", "location"]]
)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    df[["cost", "rating"]]
)

print("Encoded Data Shape:", encoded_data.shape)
print("Scaled Data Shape:", scaled_data.shape)

print("\nFirst 5 Scaled Rows:")
print(scaled_data[:5])