# SENTIMENT ANALYSIS PROJECT - TASK 4 (LEVEL 1) (DATA ANALYTICS)

#Importing the necessary libraties
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from textblob import TextBlob

nltk.download('stopwords')


# LOADING THE  DATASET
df = pd.read_csv("app.csv")
print("Dataset Loaded Successfully!")
print(df.head())

# ADDING A  SAMPLE REVIEW COLUMN (Because the dataset has no text field for a better sentiment analysis)
sample_reviews = [
    "This app is amazing and very useful!",
    "Worst experience ever. Total waste.",
    "It's okay, but needs improvement.",
    "Absolutely loved it, great features.",
    "Does not work properly. Too many bugs.",
    "Good app, easy to use.",
    "Bad interface, not recommended.",
    "Excellent performance and smooth UI.",
    "Keeps crashing. Not good.",
    "A decent app overall."
]

df["Review_Text"] = np.random.choice(sample_reviews, size=len(df))

# TEXT CLEANING FUNCTION (NLP)
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special chars
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df["Clean_Text"] = df["Review_Text"].apply(clean_text)

# SENTIMENT ANALYSIS (POLARITY SCORING)
def sentiment_score(text):
    return TextBlob(text).sentiment.polarity

df["Sentiment_Score"] = df["Clean_Text"].apply(sentiment_score)

def classify(score):
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Sentiment_Score"].apply(classify)

print("\nSample sentiment output:")
print(df[["Review_Text", "Sentiment", "Sentiment_Score"]].head())

# FEATURE ENGINEERING (TF-IDF)
X = df["Clean_Text"]
y = df["Sentiment"]

tfidf = TfidfVectorizer(max_features=3000)
X_tfidf = tfidf.fit_transform(X)

# TRAIN–TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# MACHINE LEARNING MODELS
# 🔹 Model 1 — Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

print("\n🔷 NAIVE BAYES RESULTS")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# 🔹 Model 2 — Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)

print("\n🔷 LOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))

# CONFUSION MATRIX VISUALIZATION
cm = confusion_matrix(y_test, log_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Confusion Matrix.png")
plt.show()

# SENTIMENT DISTRIBUTION PLOT
plt.figure(figsize=(5,4))
df["Sentiment"].value_counts().plot(kind="bar", color=["green", "red", "orange"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment Type")
plt.ylabel("Count")
plt.savefig("Sentiment Distribution.png")
plt.show()

# WORD CLOUDS
positive_text = " ".join(df[df["Sentiment"]=="Positive"]["Review_Text"])
negative_text = " ".join(df[df["Sentiment"]=="Negative"]["Review_Text"])

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Positive WordCloud")
plt.imshow(WordCloud().generate(positive_text))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Negative WordCloud")
plt.imshow(WordCloud().generate(negative_text))
plt.axis("off")
plt.savefig("Word Cloud.png")
plt.show()

df.to_csv("sentiment_output.csv", index=False)
print("\n🎉 Sentiment Analysis Completed Successfully!")
print("File saved as: sentiment_output.csv")
