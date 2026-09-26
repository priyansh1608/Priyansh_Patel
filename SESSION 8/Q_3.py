import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

posts = np.arange(1, 101)
followers = 1000 + 500 * posts + 20 * posts ** 2 + np.random.normal(0, 5000, 100)

X = posts.reshape(-1, 1)
y = followers

X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.2, random_state=42
)

degrees = range(1, 6)
training_errors = []
validation_errors = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_validation_poly = poly.transform(X_validation)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_prediction = model.predict(X_train_poly)
    validation_prediction = model.predict(X_validation_poly)

    training_errors.append(mean_squared_error(y_train, train_prediction))
    validation_errors.append(mean_squared_error(y_validation, validation_prediction))

plt.plot(degrees, training_errors, marker="o", label="Training Error")
plt.plot(degrees, validation_errors, marker="o", label="Validation Error")

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training and Validation Error")
plt.xticks(list(degrees))
plt.legend()
plt.grid(True)
plt.show()

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

print("Polynomial Regression Degree 4 trained successfully.")