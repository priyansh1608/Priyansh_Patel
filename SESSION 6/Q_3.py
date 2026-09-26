import matplotlib.pyplot as plt

actual_ratings = [4.5, 3.8, 4.2, 2.5, 3.9, 4.8, 3.2, 4.0, 2.8, 4.6, 3.5, 4.3, 2.2, 3.7, 4.9, 3.0, 4.1, 2.7, 3.6, 4.4]
predicted_ratings = [4.3, 4.0, 4.0, 2.7, 3.7, 4.6, 3.4, 3.9, 3.0, 4.4, 3.7, 4.1, 2.4, 3.8, 4.7, 3.2, 4.0, 2.9, 3.5, 4.2]

residuals = [actual - predicted for actual, predicted in zip(actual_ratings, predicted_ratings)]

plt.scatter(predicted_ratings, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Movie Ratings")
plt.ylabel("Residuals")
plt.title("Residual Plot for Movie Ratings")
plt.show()

print("The residual plot shows the difference between actual and predicted ratings.")
print("Residuals randomly scattered around zero generally indicate a good model fit.")