import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([
    [2, 90, 1],
    [5, 80, 2],
    [10, 70, 3],
    [15, 60, 4],
    [20, 50, 5],
    [25, 40, 6],
    [3, 85, 1],
    [8, 75, 2],
    [18, 55, 5],
    [30, 35, 7],
    [12, 65, 3],
    [22, 45, 6]
])

y = np.array([
    "confirmed",
    "confirmed",
    "confirmed",
    "confirmed",
    "confirmed",
    "waitlisted",
    "confirmed",
    "confirmed",
    "waitlisted",
    "waitlisted",
    "confirmed",
    "waitlisted"
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Actual:", y_test)
print("Predicted:", prediction)
print("Accuracy:", accuracy_score(y_test, prediction))

new_booking = np.array([[7, 78, 2]])
result = model.predict(new_booking)

print("New Booking Prediction:", result[0])