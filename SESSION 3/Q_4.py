import pandas as pd

df = pd.read_csv("SESSION 3\ipl_data.csv")

df["venue"] = df["venue"].fillna("Unknown")

venue_encoded = pd.get_dummies(df["venue"], prefix="venue")

print("One-Hot Encoded Venue DataFrame:")
print(venue_encoded)