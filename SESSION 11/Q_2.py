import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X = np.array([
    [120, 0.7, 0.9],
    [125, 0.8, 0.85],
    [130, 0.75, 0.92],
    [110, 0.65, 0.8],
    [115, 0.7, 0.88],
    [80, 0.4, 0.3],
    [85, 0.35, 0.25],
    [90, 0.45, 0.35],
    [75, 0.3, 0.2],
    [95, 0.5, 0.4],
    [128, 0.82, 0.91],
    [82, 0.38, 0.28]
])

y = np.array([
    "workout",
    "workout",
    "workout",
    "workout",
    "workout",
    "chill",
    "chill",
    "chill",
    "chill",
    "chill",
    "workout",
    "chill"
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

euclidean_model = KNeighborsClassifier(
    n_neighbors=3,
    metric="euclidean"
)

manhattan_model = KNeighborsClassifier(
    n_neighbors=3,
    metric="manhattan"
)

euclidean_model.fit(X_train, y_train)
manhattan_model.fit(X_train, y_train)

euclidean_prediction = euclidean_model.predict(X_test)
manhattan_prediction = manhattan_model.predict(X_test)

euclidean_accuracy = accuracy_score(y_test, euclidean_prediction)
manhattan_accuracy = accuracy_score(y_test, manhattan_prediction)

print("Euclidean Accuracy:", euclidean_accuracy)
print("Manhattan Accuracy:", manhattan_accuracy)