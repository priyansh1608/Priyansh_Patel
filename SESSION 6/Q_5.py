from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

actual_train = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
predicted_train_a = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
predicted_train_b = [52, 54, 61, 63, 71, 73, 79, 87, 88, 96]

actual_test = [52, 58, 63, 68, 73, 78, 83, 88, 93, 98]
predicted_test_a = [40, 70, 50, 82, 60, 95, 65, 105, 75, 115]
predicted_test_b = [53, 57, 64, 67, 72, 79, 82, 89, 92, 97]

def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mse, rmse, mae, r2

train_a = calculate_metrics(actual_train, predicted_train_a)
test_a = calculate_metrics(actual_test, predicted_test_a)

train_b = calculate_metrics(actual_train, predicted_train_b)
test_b = calculate_metrics(actual_test, predicted_test_b)

print("Model A - Training Data")
print("MSE:", train_a[0])
print("RMSE:", train_a[1])
print("MAE:", train_a[2])
print("R²:", train_a[3])

print()

print("Model A - Test Data")
print("MSE:", test_a[0])
print("RMSE:", test_a[1])
print("MAE:", test_a[2])
print("R²:", test_a[3])

print()

print("Model B - Training Data")
print("MSE:", train_b[0])
print("RMSE:", train_b[1])
print("MAE:", train_b[2])
print("R²:", train_b[3])

print()

print("Model B - Test Data")
print("MSE:", test_b[0])
print("RMSE:", test_b[1])
print("MAE:", test_b[2])
print("R²:", test_b[3])

print()

print("Model A is likely overfitting because its training errors are very low but its test errors are much higher.")