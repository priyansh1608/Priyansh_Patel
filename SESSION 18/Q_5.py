from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def evaluate_classifier(classifier, X, y, k):
    scores = cross_val_score(classifier, X, y, cv=k)

    mean_score = scores.mean()
    standard_deviation = scores.std()

    return mean_score, standard_deviation

iris = load_iris()

X = iris.data
y = iris.target

model = KNeighborsClassifier(n_neighbors=5)

mean_accuracy, std_accuracy = evaluate_classifier(
    model, X, y, 5
)

print("Mean Accuracy:", mean_accuracy)
print("Standard Deviation:", std_accuracy)