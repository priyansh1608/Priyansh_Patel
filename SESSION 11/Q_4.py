import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

messages = [
    "Hey how are you",
    "Mom please call me",
    "Can we meet today",
    "Happy birthday brother",
    "Please join the group meeting",
    "Everyone submit your assignment",
    "Good morning everyone",
    "Group meeting at 5 pm",
    "Congratulations you won a free prize",
    "Click this link to win money",
    "You have won a lottery",
    "Claim your free reward"
]

labels = [
    "personal",
    "personal",
    "personal",
    "personal",
    "group",
    "group",
    "group",
    "group",
    "spam",
    "spam",
    "spam",
    "spam"
]

X_train, X_test, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

prediction = model.predict(X_test_vectorized)

print("Actual Labels:")
print(y_test)

print("\nPredicted Labels:")
print(prediction)

cm = confusion_matrix(
    y_test,
    prediction,
    labels=["personal", "group", "spam"]
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["personal", "group", "spam"]
)

display.plot()
plt.title("WhatsApp Message Classification")
plt.show()