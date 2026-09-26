import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)

X = np.linspace(-3, 3, 30).reshape(-1, 1)
y = 2 * X.ravel() ** 2 + 3 * X.ravel() + 5 + np.random.normal(0, 3, 30)

degrees = [1, 2, 5, 10]

plt.scatter(X, y, label="Actual Data")

X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    X_plot_poly = poly.transform(X_plot)

    model = LinearRegression()
    model.fit(X_poly, y)

    prediction = model.predict(X_plot_poly)

    plt.plot(X_plot, prediction, label=f"Degree {degree}")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Underfitting and Overfitting using Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()

print("Degree 1 demonstrates underfitting.")
print("Degree 2 gives a suitable fit for the data.")
print("Higher degrees can demonstrate overfitting.")
print("As model degree increases, bias generally decreases.")
print("As model degree increases, variance generally increases.")