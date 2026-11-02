# -*- coding: utf-8 -*-
"""Beverage.decisiontree"""

import os
from collections import Counter
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

st.title("🥤 Beverage Decision Tree & Classification App")

# Read the classes file using a relative path so it works locally and on Streamlit Cloud
file_path = os.path.join(os.path.dirname(__file__), "classes.txt")
with open(file_path, "r") as f:
  lines = [line.strip() for line in f.readlines() if line.strip()]

# Perform EDA
total_lines = len(lines)
unique_classes = set(lines)
class_counts = Counter(lines)

st.subheader("Exploratory Data Analysis")
st.write(f"**Total items:** {total_lines}")
st.write(f"**Unique classes count:** {len(unique_classes)}")

st.write("**Class Distributions:**")
for cls, count in class_counts.most_common():
  st.write(f" - {cls}: {count} ({count/total_lines*100:.2f}%)")

# Generate a pie chart of the classes
st.subheader("Class Distribution Chart")
if class_counts:
  labels, values = zip(*class_counts.most_common())

  fig, ax = plt.subplots(figsize=(8, 8))
  ax.pie(
      values,
      labels=labels,
      autopct="%1.1f%%",
      startangle=140,
      colors=plt.cm.Paired.colors,
  )
  ax.set_title("Distribution of Classes")
  ax.axis("equal")
  st.pyplot(fig)
else:
  st.write("No data available to plot.")

"""### Machine Learning with a Decision Tree"""

X = lines
y = lines

# Vectorizer and Model
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.20, random_state=42
)

st.subheader("Model Training Info")
st.write(f"Training set size: {X_train.shape[0]} samples")
st.write(f"Testing set size: {X_test.shape[0]} samples")

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
st.write(f"**Decision Tree Model Accuracy:** {accuracy * 100:.2f}%")

"""### Predict 10 Random Drinks"""

synthetic_test_data = [
    ("Pepsi", "Pepsi"),
    ("Coca Cola", "Coca Cola"),
    ("Sprite Lemon", "Sprite"),
    ("Mountain Dew", "Mountain Dew"),
    ("7UP Lemon-Lime", "7UP"),
    ("Fanta Orange", "Fanta"),
    ("Diet Pepsi", "Pepsi"),
    ("Coca Cola Classic", "Coca Cola"),
    ("Fanta", "Fanta"),
    ("Mountain Dew Ice", "Mountain Dew"),
]

test_names = [item[0] for item in synthetic_test_data]
true_labels = [item[1] for item in synthetic_test_data]

X_synthetic_vectorized = vectorizer.transform(test_names)
synthetic_preds = clf.predict(X_synthetic_vectorized)

results_df = pd.DataFrame({
    "Input Drink Name": test_names,
    "Actual Label Group": true_labels,
    "Predicted Label": synthetic_preds,
})

st.subheader("Synthetic Test Predictions (10 Drinks)")
st.dataframe(results_df)

joblib.dump(clf, "decision_tree_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
