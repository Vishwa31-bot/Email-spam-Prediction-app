import streamlit as st
import pandas as pd
import string
import re
import random
import math
from collections import defaultdict

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("email_spam_dataset_realistic.csv")
    return df

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Naive Bayes from scratch
class SimpleNB:
    def __init__(self):
        self.word_probs = {}
        self.class_probs = {}

    def train(self, X, y):
        total_docs = len(y)
        classes = set(y)
        self.class_probs = {c: y.count(c)/total_docs for c in classes}
        
        word_counts = {c: defaultdict(int) for c in classes}
        class_counts = {c: 0 for c in classes}
        
        for doc, label in zip(X, y):
            words = doc.split()
            for word in words:
                word_counts[label][word] += 1
                class_counts[label] += 1

        self.word_probs = {
            c: {word: (count + 1) / (class_counts[c] + len(word_counts[c]))
                for word, count in word_counts[c].items()}
            for c in classes
        }

    def predict(self, doc):
        words = doc.split()
        scores = {}
        for c in self.class_probs:
            log_prob = math.log(self.class_probs[c])
            for word in words:
                prob = self.word_probs[c].get(word, 1 / (1 + len(self.word_probs[c])))
                log_prob += math.log(prob)
            scores[c] = log_prob
        return max(scores, key=scores.get)

# Main App
st.title("📧 Email Spam Detection App")

df = load_data()
st.write("### Sample Dataset", df.head())

# Preprocess the text
df['message_clean'] = df['message'].apply(clean_text)

# Split data manually
data = list(zip(df['message_clean'], df['label']))
random.shuffle(data)
split_index = int(len(data) * 0.8)
train_data = data[:split_index]
test_data = data[split_index:]

X_train = [x[0] for x in train_data]
y_train = [x[1] for x in train_data]
X_test = [x[0] for x in test_data]
y_test = [x[1] for x in test_data]

# Train model
model = SimpleNB()
model.train(X_train, y_train)

# Test Accuracy
correct = 0
for x, y in zip(X_test, y_test):
    pred = model.predict(x)
    if pred == y:
        correct += 1

accuracy = correct / len(y_test)
st.write(f"### Model Accuracy: {accuracy:.2f}")

# Prediction Input
st.write("## 📤 Test Your Email")
user_input = st.text_area("Enter your email message:")
if user_input:
    user_input_clean = clean_text(user_input)
    prediction = model.predict(user_input_clean)
    label = "🚫 Spam" if prediction == "spam" else "✅ Not Spam"
    st.write("### Prediction:", label)
