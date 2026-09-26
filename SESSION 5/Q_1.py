
import matplotlib.pyplot as plt

instagram_hours = [1, 2, 3, 4, 5, 3, 2]
battery_percentage = [90, 82, 72, 60, 48, 65, 78]

plt.scatter(instagram_hours, battery_percentage)
plt.xlabel("Instagram Usage Hours")
plt.ylabel("Battery Percentage")
plt.title("Instagram Usage vs Battery Percentage")
plt.show()

