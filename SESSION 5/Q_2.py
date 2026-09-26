
from sklearn.linear_model import LinearRegression

instagram_hours = [[1], [2], [3], [4], [5], [3], [2]]
battery_percentage = [90, 82, 72, 60, 48, 65, 78]

model = LinearRegression()
model.fit(instagram_hours, battery_percentage)

print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
