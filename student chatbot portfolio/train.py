from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

training_data = {
    "greeting": ["hi", "hello", "hey"],
    "study": ["how can I study", "study tips", "help me study"],
    "exam": ["exam tips", "how to pass exams"],
    "stress": ["i am stressed", "school stress"],
    "bye": ["bye", "goodbye"]
}

sentences = []
labels = []

for intent, phrases in training_data.items():
    for phrase in phrases:
        sentences.append(phrase)
        labels.append(intent)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

model = MultinomialNB()
model.fit(X, labels)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("AI trained successfully!")
