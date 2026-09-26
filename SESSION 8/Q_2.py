import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

reviews = np.array([50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]).reshape(-1, 1)
ratings = np.array([2.8, 3.0, 3.2, 3.5, 3.6, 3.8, 4.0, 4.2, 4.1, 4.4])

linear_model = LinearRegression()
linear_model.fit(reviews, ratings)
linear_prediction = linear_model.predict(reviews)

poly = PolynomialFeatures(degree=3)
reviews_poly = poly.fit_transform(reviews)

poly_model = LinearRegression()
poly_model.fit(reviews_poly, ratings)
poly_prediction = poly_model.predict(reviews_poly)

plt.scatter(reviews, ratings, label="Actual Ratings")
plt.plot(reviews, linear_prediction, label="Linear Regression")
plt.plot(reviews, poly_prediction, label="Polynomial Regression Degree 3")

plt.xlabel("Number of Reviews")
plt.ylabel("Restaurant Rating")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()