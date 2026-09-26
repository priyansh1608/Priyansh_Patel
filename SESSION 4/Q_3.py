import pandas as pd

df = pd.read_csv("SESSION 4\spotify_data.csv")

features = ["danceability", "energy", "tempo", "popularity"]

correlation = df[features].corr()

print("Correlation with popularity:")
print(correlation["popularity"])

popularity_corr = correlation["popularity"].drop("popularity")

top_2 = popularity_corr.abs().sort_values(ascending=False).head(2)

print("\nTop 2 features related to popularity:")

for feature in top_2.index:
    value = popularity_corr[feature]
    print(feature, ":", round(value, 3))