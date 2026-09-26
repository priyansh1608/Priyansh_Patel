import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)

followers = np.linspace(100, 10000, 40)
posts = 0.00000008 * followers**2 + 0.02 * followers + np.random.normal(0, 20, 40)

X_train, X_validation, y_train, y_validation = train_test_split(
    followers.reshape(-1, 1),
    posts,
    test_size=0.3,
    random_state=42
)

degrees = [1, 2, 3, 4, 5]
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
plt.legend()
plt.show()

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(followers.reshape(-1, 1))

model = LinearRegression()
model.fit(X_poly, posts)

print("Polynomial Regression Degree:", 4)
print("Training MSE:", training_errors[3])
print("Validation MSE:", validation_errors[3])