import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 200)
y = 1 / (1 + np.exp(-x))

plt.plot(x, y, label="Sigmoid Function")
plt.axhline(y=0.5, linestyle="--", label="Threshold = 0.5")
plt.axvline(x=0, linestyle="--", label="x = 0")

plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.title("Sigmoid Function")
plt.legend()
plt.grid(True)
plt.show()