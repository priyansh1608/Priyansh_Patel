import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X = np.array([
    [2.0, 2.5],
    [2.5, 3.0],
    [3.0, 2.8],
    [3.5, 3.2],
    [4.0, 3.5],
    [4.5, 3.8],
    [5.0, 4.0],
    [5.5, 4.3],
    [6.0, 4.5],
    [6.5, 4.8]
])

y = np.array([
    "Bad", "Bad", "Bad", "Bad", "Bad",
    "Good", "Good", "Good", "Good", "Good"
])

model = SVC(kernel="linear", C=1)
model.fit(X, y)

plt.scatter(
    X[y == "Bad", 0],
    X[y == "Bad", 1],
    label="Bad"
)

plt.scatter(
    X[y == "Good", 0],
    X[y == "Good", 1],
    label="Good"
)

support_vectors = model.support_vectors_

plt.scatter(
    support_vectors[:, 0],
    support_vectors[:, 1],
    s=150,
    facecolors="none",
    edgecolors="black",
    label="Support Vectors"
)

w = model.coef_[0]
b = model.intercept_[0]

x_values = np.linspace(1, 7, 200)

decision_boundary = -(w[0] * x_values + b) / w[1]
margin_positive = -(w[0] * x_values + b - 1) / w[1]
margin_negative = -(w[0] * x_values + b + 1) / w[1]

plt.plot(x_values, decision_boundary, label="Decision Boundary")
plt.plot(x_values, margin_positive, "--", label="Margin")
plt.plot(x_values, margin_negative, "--")

plt.xlabel("Average Order Rating")
plt.ylabel("Number of Reviews")
plt.title("SVM Margin and Support Vectors - Zomato")
plt.legend()
plt.grid()
plt.show()