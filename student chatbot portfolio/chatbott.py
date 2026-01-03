import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

responses = {
    "greeting": "Hello! I'm your student assistant 😊",
    "study": "Use active recall and study in short sessions.",
    "exam": "Practice past papers and revise early.",
    "stress": "Take breaks, breathe, and don’t overwork yourself.",
    "bye": "Goodbye! You’ve got this 💪"
}

def get_response(message):
    vec = vectorizer.transform([message.lower()])
    intent = model.predict(vec)[0]
    return responses[intent]

