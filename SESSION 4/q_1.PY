import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("SESSION 4\ipl_data.csv")

numeric_columns = ["runs", "wickets", "matches", "strike_rate"]

X = df[numeric_columns]

scaler = StandardScaler()

scaled_data = scaler.fit_transform(X)

scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns)

scaled_df.to_csv("ipl_scaled.csv", index=False)

print("Scaled IPL data:")
print(scaled_df)

print("\nScaled data saved as ipl_scaled.csv")