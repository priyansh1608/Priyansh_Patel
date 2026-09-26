from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

reviews = [
    "Excellent product and very good quality",
    "Amazing product I love it",
    "Very good quality and fast delivery",
    "I am happy with this product",
    "Great product worth the money",
    "The product is excellent",
    "Good quality product",
    "Very useful and nice product",
    "Bad product and poor quality",
    "Very poor product",
    "I hate this product",
    "Worst product ever",
    "The product is damaged",
    "Very bad quality",
    "Not worth the money",
    "Poor product and slow delivery",
    "Terrible product",
    "I am unhappy with this product",
    "Waste of money",
    "Product quality is very bad"
]

labels = [
    "Positive", "Positive", "Positive", "Positive", "Positive",
    "Positive", "Positive", "Positive", "Negative", "Negative",
    "Negative", "Negative", "Negative", "Negative", "Negative",
    "Negative", "Negative", "Negative", "Negative", "Negative"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42, stratify=labels
)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")