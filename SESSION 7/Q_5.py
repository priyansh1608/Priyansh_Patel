import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)

X = np.linspace(0, 10, 20)
y = np.sin(X) + np.random.normal(0, 0.2, 20)

X = X.reshape(-1, 1)

degrees = [1, 4, 15]

plt.scatter(X, y, label="Actual Data")

X_plot = np.linspace(0, 10, 300).reshape(-1, 1)

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    X_plot_poly = poly.transform(X_plot)

    model = LinearRegression()
    model.fit(X_poly, y)

    prediction = model.predict(X_plot_poly)

    plt.plot(X_plot, prediction, label="Degree " + str(degree))

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Underfitting and Overfitting")
plt.legend()
plt.show()

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    prediction = model.predict(X_poly)
    mse = mean_squared_error(y, prediction)

    print("Degree:", degree)
    print("MSE:", mse)