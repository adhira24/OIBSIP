# UNVEILING THE ANDROID APP MARKET : ANALYZING GOOGLE PLAY STORE DATA - INTERNSHIP TASK 4 (LEVEL 2) (DATA ANALYTICS)

# IMPORTING THE NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore")

# LOADING THE DATASETS
apps = pd.read_csv("app_task4.csv")
review = pd.read_csv("users_review.csv")
print("Apps Dataset Shape:\n",apps.shape)
print("Reviews Dataset Shape:\n",review.shape)

# DATA PREPARATION
# Remove the unnamed columns if present
apps = apps.loc[:, ~apps.columns.str.contains("^Unnamed")]
review = review.loc[:, ~review.columns.str.contains("^Unnamed")]

# Convert installs
def clean_installs(x):
    x = str(x).replace(",","").replace("+","")
    return int(x)
apps["Installs"] = apps["Installs"].apply(clean_installs)

# Convert price
def clean_price(x):
    x = str(x).replace("$","")
    return float(x)
apps["Price"] = apps["Price"].apply(clean_price)

# Convert Rating to numeric
apps["Rating"] = pd.to_numeric(apps["Rating"],errors='coerce')

# Merge the Datasets 
merged = pd.merge(apps, review, on="App", how="left")

# CATEGORY EXPLORATION
plt.figure(figsize=(12,6))
merged["Category"].value_counts().head(15).plot(kind="bar", color="teal")
plt.title("Top 15 Most Common App Categories")
plt.xlabel("Category")
plt.ylabel("Number of Apps")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Top_15_Most_Common_App_Categories.png")
plt.show()

# Category vs Average Rating
plt.figure(figsize=(12,6))
category_rating = merged.groupby("Category")["Rating"].mean().sort_values(ascending=False)
category_rating.head(15).plot(kind="bar", color="orange")
plt.title("Top Categories With Highest Average Ratings")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Top_Categories_with_Highest_Average_Ratings.png")
plt.show()

# METRICS ANALYSIS

df = pd.merge(apps, review, on = "App", how= "left")

# BASIC METRICS (Ratings & Reviews)

print("===== BASIC METRICS =====")
print("Total Reviews:", len(df))

if 'Rating' in df.columns:
    print("Average Rating:", df['Rating'].mean())
    print("Median Rating:", df['Rating'].median())
    print("Mode Rating:", df['Rating'].mode()[0])
    print("Rating Standard Deviation:", df['Rating'].std())


# REVIEW LENGTH METRICS
df['review_length'] = df['Reviews'].astype(str).apply(len)

print("===== REVIEW LENGTH METRICS =====")
print(df['review_length'].describe())

plt.figure(figsize=(8,5))
sns.histplot(df['review_length'], kde=True)
plt.title("Distribution of Review Length")
plt.xlabel("Length of Review (characters)")
plt.ylabel("Frequency")
plt.savefig("Distribution_of_Review_Length.png")
plt.show()

# CATEGORY-WISE METRICS 
if 'category' in df.columns:
    category_metrics = df.groupby('category')['Rating'].mean().sort_values(ascending=False)
    print("===== AVERAGE RATING PER CATEGORY =====")
    print(category_metrics)

    plt.figure(figsize=(10,5))
    category_metrics.plot(kind='bar', color='skyblue')
    plt.title("Average Rating per Category")
    plt.ylabel("Average Rating")
    plt.xlabel("Category")
    plt.xticks(rotation=45)
    plt.savefig("Average_Rating_per_Category.png")
    plt.show()

# TOP & BOTTOM RATED REVIEWS
print("===== TOP 5 REVIEWS =====")
print(df.nlargest(5, 'Rating')[['Reviews', 'Rating']])

print("===== BOTTOM 5 REVIEWS =====")
print(df.nsmallest(5, 'Rating')[['Reviews', 'Rating']])

# Rating distribution
plt.figure(figsize=(10,5))
sns.histplot(merged["Rating"], kde=True, bins=20, color="purple")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.savefig("Rating_Distribution.png")
plt.show()

# Size vs Rating
plt.figure(figsize=(10,5))
sns.scatterplot(data=merged, x="Size", y="Rating", alpha=0.5)
plt.title("App Size vs Rating")
plt.savefig("App size vs Rating.png")
plt.show()

# Installs vs Rating
plt.figure(figsize=(10,5))
sns.scatterplot(data=merged, x="Rating", y="Installs", alpha=0.5)
plt.title("Rating vs Installs")
plt.savefig("Rating vs Installs.png")
plt.show()

# Free vs Paid apps comparison
plt.figure(figsize=(8,5))
sns.boxplot(data=merged, x="Type", y="Rating", palette="viridis")
plt.title("Free vs Paid Apps - Rating Comparison")
plt.savefig("Rating_Comparison.png")
plt.show()


# SENTIMENT ANALYSIS

def get_sentiment(text):
    if pd.isnull(text):
        return "Neutral"
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

merged["Sentiment_Label"] = merged["Translated_Review"].apply(get_sentiment)

# Sentiment analysis using wordcloud 
from wordcloud import WordCloud

positive_text = " ".join(merged[merged["Sentiment_Label"]=="Positive"]["Translated_Review"].astype(str))
negative_text = " ".join(merged[merged["Sentiment_Label"]=="Negative"]["Translated_Review"].astype(str))

plt.figure(figsize=(14,6))

# Positive WordCloud
plt.subplot(1,2,1)
plt.title("Positive Reviews WordCloud")
plt.imshow(WordCloud(width=800, height=400, background_color='white').generate(positive_text))
plt.axis("off")

# Negative WordCloud
plt.subplot(1,2,2)
plt.title("Negative Reviews WordCloud")
plt.imshow(WordCloud(width=800, height=400, background_color='white').generate(negative_text))
plt.axis("off")
plt.savefig("Wordcloud.png")
plt.show()

# Sentiment distribution
plt.figure(figsize=(6,6))
merged["Sentiment_Label"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["green","red","gold"])
plt.title("User Sentiment Distribution")
plt.ylabel("")
plt.savefig("User_Sentiment_Distribution.png")
plt.show()

# Sentiment vs Rating
plt.figure(figsize=(10,5))
sns.boxplot(data=merged, x="Sentiment_Label", y="Rating", palette="pastel")
plt.title("Ratings Based on User Sentiment")
plt.savefig("Ratings_based_on_user_sentiment.png")
plt.show()

# SAVE FINAL DATASET

merged.to_csv("final_app_review_merged.csv", index=False)
print("Final cleaned merged dataset saved as final_app_review_merged.csv")
