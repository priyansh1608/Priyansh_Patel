import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

ram = np.array([2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 6, 8, 12, 16, 24, 32, 64, 4, 8, 16]).reshape(-1, 1)
price = np.array([5000, 7000, 9000, 13000, 17000, 25000, 33000, 47000, 62000, 120000, 14000, 18000, 26000, 34000, 46000, 60000, 115000, 9500, 17500, 35000])

X_train, X_test, y_train, y_test = train_test_split(
    ram,
    price,
    test_size=0.3,
    random_state=42
)

poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train_poly, y_train)

ridge_model = Ridge(alpha=10)
ridge_model.fit(X_train_poly, y_train)

linear_prediction = linear_model.predict(X_test_poly)
ridge_prediction = ridge_model.predict(X_test_poly)

linear_mse = mean_squared_error(y_test, linear_prediction)
ridge_mse = mean_squared_error(y_test, ridge_prediction)

print("Polynomial Regression Test MSE:", linear_mse)
print("Ridge Regression Test MSE:", ridge_mse)

if ridge_mse < linear_mse:
    print("Ridge Regression has lower test error.")
else:
    print("Polynomial Regression has lower test error.")