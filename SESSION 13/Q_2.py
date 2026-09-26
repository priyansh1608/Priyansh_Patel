import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = {
    "price": [500, 700, 1200, 1500, 300, 450, 900, 2000, 250, 600,
              1300, 1700, 350, 800, 1100, 2200, 400, 750, 1400, 1900],
    "rating": [4.5, 4.2, 4.7, 4.8, 3.9, 4.0, 4.4, 4.9, 3.8, 4.1,
               4.6, 4.7, 4.0, 4.3, 4.5, 4.8, 3.7, 4.2, 4.6, 4.7],
    "brand": [
        "Samsung", "Samsung", "Apple", "Apple", "Boat",
        "Boat", "Samsung", "Apple", "Boat", "Samsung",
        "Apple", "Apple", "Boat", "Samsung", "Samsung",
        "Apple", "Boat", "Samsung", "Apple", "Apple"
    ],
    "category": [
        "Mobile", "Mobile", "Mobile", "Mobile", "Audio",
        "Audio", "Mobile", "Mobile", "Audio", "Mobile",
        "Mobile", "Mobile", "Audio", "Mobile", "Mobile",
        "Mobile", "Audio", "Mobile", "Mobile", "Mobile"
    ]
}

df = pd.DataFrame(data)

brand_encoder = LabelEncoder()

df["brand_encoded"] = brand_encoder.fit_transform(df["brand"])

X = df[["price", "rating", "brand_encoded"]]
y = df["category"]

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

print("Test Accuracy:", accuracy)

print("\nFeature Importance:")

for feature, importance in zip(X.columns, model.feature_importances_):
    print(feature, ":", importance)