from sklearn.metrics import mean_absolute_error, r2_score

actual_prices = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750]
predicted_prices = [520, 730, 980, 1280, 1470, 1780, 2020, 2200, 2530, 2700]

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mae, r2

mae, r2 = calculate_metrics(actual_prices, predicted_prices)

print("Mean Absolute Error:", mae)
print("R² Score:", r2)