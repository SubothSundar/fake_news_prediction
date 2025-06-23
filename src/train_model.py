# src/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import nltk

nltk.download('stopwords')

# Load the data
fake_df = pd.read_csv('../data/Fake.csv')
real_df = pd.read_csv('../data/True.csv')

# Label the data: 0 = fake, 1 = real
fake_df['label'] = 0
real_df['label'] = 1

# Combine the datasets
df = pd.concat([fake_df, real_df], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Features and Labels
X = df['text']
y = df['label']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))


import json, os

metrics = {
    "accuracy": accuracy_score(y_test, y_pred)
}
with open("../metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)


# Save the model and vectorizer to project root
pickle.dump(model, open("../model.pkl", "wb"))
pickle.dump(vectorizer, open("../vectorizer.pkl", "wb"))
