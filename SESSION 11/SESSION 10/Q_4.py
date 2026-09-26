def classify_probability(probability, threshold=0.5):
    if probability >= threshold:
        return "Viral"
    else:
        return "Not Viral"

probabilities = [0.2, 0.6, 0.8]

print("Threshold = 0.5")

for probability in probabilities:
    result = classify_probability(probability, 0.5)
    print(probability, "->", result)

print("\nThreshold = 0.7")

for probability in probabilities:
    result = classify_probability(probability, 0.7)
    print(probability, "->", result)