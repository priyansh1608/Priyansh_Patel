import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("SESSION 3\ipl_data.csv")

encoder = LabelEncoder()

df["player_role_encoded"] = encoder.fit_transform(df["player_role"])

print("Player Role Mapping:")

for role, number in zip(encoder.classes_, encoder.transform(encoder.classes_)):
    print(role, "=", number)

print("\nUpdated player_role column:")
print(df[["player_role", "player_role_encoded"]])