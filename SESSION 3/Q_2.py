import pandas as pd

df = pd.read_csv("SESSION 3\ipl_data.csv")

median_age = df["player_age"].median()

df["player_age"] = df["player_age"].fillna(median_age)

print("Updated player_age column:")
print(df["player_age"])