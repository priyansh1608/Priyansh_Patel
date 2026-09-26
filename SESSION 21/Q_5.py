import numpy as np

def smape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    denominator = (np.abs(actual) + np.abs(predicted)) / 2

    return np.mean(
        np.abs(actual - predicted) / denominator
    ) * 100

actual = [25, 27, 28, 30, 29]
predicted = [24, 28, 27, 31, 30]

score = smape(actual, predicted)

print("SMAPE:", score, "%")