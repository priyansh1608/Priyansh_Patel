from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

instagram_hours = [[1], [2], [3], [4], [5], [3], [2]]
battery_percentage = [90, 82, 72, 60, 48, 65, 78]

model = LinearRegression()
model.fit(instagram_hours, battery_percentage)

predicted_battery = model.predict(instagram_hours)

mse = mean_squared_error(battery_percentage, predicted_battery)

print("Mean Squared Error:", mse)
print("A lower MSE indicates that the predicted values are closer to the actual values.")
