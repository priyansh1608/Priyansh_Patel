import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

reviews = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500]).reshape(-1, 1)

ratings = np.array([3.1, 3.2, 3.5, 3.7, 3.9, 4.0, 4.1, 4.2, 4.15, 4.1])

linear_model = LinearRegression()
linear_model.fit(reviews, ratings)
linear_predictions = linear_model.predict(reviews)

poly = PolynomialFeatures(degree=3)
reviews_poly = poly.fit_transform(reviews)

poly_model = LinearRegression()
poly_model.fit(reviews_poly, ratings)
poly_predictions = poly_model.predict(reviews_poly)

plt.scatter(reviews, ratings, label="Actual Ratings")
plt.plot(reviews, linear_predictions, label="Linear Regression")
plt.plot(reviews, poly_predictions, label="Polynomial Regression Degree 3")

plt.xlabel("Number of Reviews")
plt.ylabel("Restaurant Rating")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()