# AUTOCOMPLETE AND AUTOCORRECT DATA ANALYTICS - INTERNSHIP TASK 5 (LEVEL 2) 
 
# Import the necessary libraries
import pandas as pd
import nltk
import warnings
warnings.filterwarnings("ignore")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from textblob import TextBlob

# Download required NLP resources
nltk.download('punkt')
nltk.download('stopwords')

# LOAD DATASET (DATA COLLECTION)

df = pd.read_csv("creditcard_task5.csv")
print("Dataset Shape:", df.shape)
print(df.head())

# SYNTHETIC TEXT FOR NLP

sample_text = """
credit card transaction failed
unauthorized payment detected
fraudulent transaction blocked
payment successful
card verification failed
"""
tokens = word_tokenize(sample_text.lower())
tokens = [w for w in tokens if w.isalpha()]

print("\nTotal Tokens:", len(tokens))

# AUTOCOMPLETE IMPLEMENTATION

bigrams = list(ngrams(tokens, 2))
bigram_freq = Counter(bigrams)

def autocomplete(word, top_n=5):
    suggestions = [pair[1] for pair in bigram_freq if pair[0] == word]
    return Counter(suggestions).most_common(top_n)

print("\nAutocomplete Example:")
print(autocomplete("credit"))

# AUTOCORRECT IMPLEMENTATION

def autocorrect(word):
    return str(TextBlob(word).correct())

sample_words = ["transacton", "unauthorised", "paymnt", "verificatoin"]
corrected_words = [autocorrect(w) for w in sample_words]

print("\nAutocorrect Example:")
for w, c in zip(sample_words, corrected_words):
    print(f"{w} → {c}")

# SAFE ACCURACY CHECK

if len(sample_words) == 0:
    accuracy = 0
else:
    accuracy = sum(1 for i, j in zip(sample_words, corrected_words) if i == j) / len(sample_words)

print("\nAutocorrect Accuracy:", accuracy)

# VISUALIZATIONS

print("\nTotal Tokens:", len(tokens))

from collections import Counter
import matplotlib.pyplot as plt

# Word frequency Visualization
word_freq = Counter(tokens)
top_words = word_freq.most_common(10)
words, counts = zip(*top_words)
plt.figure(figsize=(10,5))
plt.bar(words, counts)
plt.title("Top 10 Most Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Top_10_Most_Frequent_Words.png")
plt.show()

# Bigram Frequency (Autocomplete Logic Visual)
bigram_words = [" ".join(bg) for bg in bigrams]
bigram_freq = Counter(bigram_words).most_common(10)
bg_words, bg_counts = zip(*bigram_freq)
plt.figure(figsize=(10,5))
plt.bar(bg_words, bg_counts)
plt.title("Top 10 Most Frequent Bigrams (Autocomplete Base)")
plt.xlabel("Word Pairs")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Top_10_Most_Frequent_Bigrams.png")
plt.show()

# Autocorrect Before vs After
incorrect = sample_words
corrected = corrected_words
plt.figure(figsize=(10,4))
plt.plot(incorrect, label="Incorrect Words", marker="o")
plt.plot(corrected, label="Corrected Words", marker="o")
plt.title("Autocorrect: Before vs After")
plt.xlabel("Words")
plt.ylabel("Word Representation")
plt.legend()
plt.tight_layout()
plt.savefig("Autocorrect.png")
plt.show()

# Autocorrect Accuracy Visual
plt.figure(figsize=(6,4))
plt.bar(["Correct", "Incorrect"], 
        [accuracy * 100, (1 - accuracy) * 100])
plt.title("Autocorrect Accuracy")
plt.ylabel("Percentage")
plt.ylim(0,100)
plt.tight_layout()
plt.savefig("Autocorrect_Accuracy.png")
plt.show()

#  PROJECT COMPLETION MESSAGE

print("\nThis pipeline can be extended using:")
print("- Neural Language Models")
print("- Transformer-based Autocomplete")
print("- Deep Learning Spell Correction")

print("\nPROJECT COMPLETED SUCCESSFULLY ✅")
