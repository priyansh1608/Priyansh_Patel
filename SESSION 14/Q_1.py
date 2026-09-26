import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X = np.array([
    [100, 0.80],
    [110, 0.75],
    [120, 0.85],
    [115, 0.78],
    [105, 0.82],
    [130, 0.90],
    [60, 0.20],
    [70, 0.25],
    [80, 0.30],
    [75, 0.28],
    [65, 0.22],
    [85, 0.35]
])

y = np.array([
    "Pop", "Pop", "Pop", "Pop", "Pop", "Pop",
    "Classical", "Classical", "Classical", "Classical", "Classical", "Classical"
])

model = SVC(kernel="linear")
model.fit(X, y)

plt.scatter(X[y == "Pop", 0], X[y == "Pop", 1], label="Pop")
plt.scatter(X[y == "Classical", 0], X[y == "Classical", 1], label="Classical")

w = model.coef_[0]
b = model.intercept_[0]

x_values = np.linspace(50, 140, 200)
y_values = -(w[0] * x_values + b) / w[1]

plt.plot(x_values, y_values, label="Decision Boundary")

plt.xlabel("Tempo")
plt.ylabel("Danceability")
plt.title("SVM Music Genre Classification")
plt.legend()
plt.grid()
plt.show()