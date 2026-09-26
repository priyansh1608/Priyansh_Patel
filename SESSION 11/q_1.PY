from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

reviews = [
    "excellent product",
    "very good quality",
    "amazing product",
    "bad product",
    "poor quality",
    "worst product",
    "excellent quality",
    "good product",
    "very bad",
    "poor product"
]

labels = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "positive",
    "positive",
    "negative",
    "negative"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, labels)

test_reviews = [
    "excellent product",
    "bad quality",
    "amazing quality"
]

X_test = vectorizer.transform(test_reviews)
predictions = model.predict(X_test)

for review, prediction in zip(test_reviews, predictions):
    print("Review:", review)
    print("Prediction:", prediction)
    print()