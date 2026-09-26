from sklearn.metrics import mean_squared_error
import numpy as np

actual = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
predicted = [32, 34, 42, 44, 48, 57, 59, 67, 72, 73]

mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)