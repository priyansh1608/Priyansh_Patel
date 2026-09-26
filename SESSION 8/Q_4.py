import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

np.random.seed(42)

features = np.arange(100, 1100, 50).reshape(-1, 1)
prices = 500 + 2.5 * features + 0.002 * features ** 2 + np.random.normal(0, 500, len(features))

X_train, X_test, y_train, y_test = train_test_split(
    features, prices.ravel(), test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train_poly, y_train)

linear_prediction = linear_model.predict(X_test_poly)
linear_error = mean_squared_error(y_test, linear_prediction)

ridge_model = Ridge(alpha=10)
ridge_model.fit(X_train_poly, y_train)

ridge_prediction = ridge_model.predict(X_test_poly)
ridge_error = mean_squared_error(y_test, ridge_prediction)

print("Polynomial Regression Test Error:", linear_error)
print("Ridge Regression Test Error:", ridge_error)

if ridge_error < linear_error:
    print("Ridge regression reduced the test error.")
elif ridge_error > linear_error:
    print("Ridge regression increased the test error.")
else:
    print("Both models have the same test error.")