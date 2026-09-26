import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

instagram_hours = [[1], [2], [3], [4], [5], [3], [2]]
battery_percentage = [90, 82, 72, 60, 48, 65, 78]

model = LinearRegression()
model.fit(instagram_hours, battery_percentage)

predicted_battery = model.predict(instagram_hours)

plt.scatter(instagram_hours, battery_percentage)
plt.plot(instagram_hours, predicted_battery)
plt.xlabel("Instagram Usage Hours")
plt.ylabel("Battery Percentage")
plt.title("Linear Regression: Instagram Usage vs Battery")
plt.show()
