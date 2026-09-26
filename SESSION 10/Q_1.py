import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

values = [-2, 0, 3]

for x in values:
    print("x =", x, "Sigmoid =", sigmoid(x))