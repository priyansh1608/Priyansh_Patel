import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("SESSION 4\kart_data.csv")

features = ["price", "rating", "reviews", "discount"]

X = df[features]

print("Before Scaling:")
print("Minimum values:")
print(X.min())

print("\nMaximum values:")
print(X.max())

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(X)

scaled_df = pd.DataFrame(scaled_data, columns=features)

print("\nAfter Scaling:")
print("Minimum values:")
print(scaled_df.min())

print("\nMaximum values:")
print(scaled_df.max())

print("\nScaled Data:")
print(scaled_df)