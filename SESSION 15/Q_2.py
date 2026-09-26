from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

reviews = [
    "Excellent product and very good quality",
    "Amazing product I love it",
    "Very good product",
    "Great quality and fast delivery",
    "I am happy with this product",
    "Worth the money",
    "Good product and useful",
    "Excellent quality",
    "Bad product and poor quality",
    "Very poor product",
    "Worst product ever",
    "I hate this product",
    "Product is damaged",
    "Very bad quality",
    "Not worth the money",
    "Poor product",
    "Terrible quality",
    "Waste of money",
    "Very disappointing product",
    "I am unhappy with this product"
]

labels = [
    "Positive", "Positive", "Positive", "Positive", "Positive",
    "Positive", "Positive", "Positive", "Positive", "Negative",
    "Negative", "Negative", "Negative", "Negative", "Negative",
    "Negative", "Negative", "Negative", "Negative", "Negative"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(reviews)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42, stratify=labels
)

model = AdaBoostClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))