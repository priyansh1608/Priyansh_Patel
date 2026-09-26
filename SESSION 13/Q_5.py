import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = {
    "tempo": [120, 128, 140, 90, 85, 130, 125, 95, 150, 88,
              135, 100, 145, 92, 118, 132, 87, 142, 105, 127],
    "danceability": [0.8, 0.85, 0.9, 0.4, 0.35, 0.82, 0.78, 0.45, 0.92, 0.3,
                     0.88, 0.5, 0.91, 0.38, 0.75, 0.84, 0.32, 0.89, 0.55, 0.81],
    "energy": [0.9, 0.95, 0.92, 0.4, 0.3, 0.88, 0.85, 0.45, 0.97, 0.25,
               0.93, 0.5, 0.96, 0.35, 0.8, 0.9, 0.28, 0.94, 0.6, 0.87],
    "hit": [1, 1, 1, 0, 0, 1, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

df.to_csv("spotify_songs.csv", index=False)

df = pd.read_csv("spotify_songs.csv")

X = df[["tempo", "danceability", "energy"]]
y = df["hit"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Random Forest Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

print("\nFeature Importance:")

for feature, importance in zip(X.columns, model.feature_importances_):
    print(feature, ":", importance)

new_song = [[128, 0.85, 0.90]]

result = model.predict(new_song)

print("\nNew Song Prediction:")

if result[0] == 1:
    print("Hit")
else:
    print("Not Hit")

print("\nWhat I learned:")
print("Random Forest combines multiple decision trees to make a more stable prediction.")