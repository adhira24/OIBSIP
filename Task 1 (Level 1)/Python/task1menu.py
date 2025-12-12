# EXPLORATORY DATA ANALYSIS FOR DATASET 2 - INTERNSHIP TASK 1 (LEVEL 1) (DATA ANALYTICS)
#importing necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
df = pd.read_csv("menu.csv")
df.head()

#data cleaning
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
df = df.drop_duplicates()
numeric_cols = ['Calories', 'Calories from Fat', 'Total Fat', 'Saturated Fat',
                'Trans Fat', 'Cholesterol', 'Sodium', 'Carbohydrates',
                'Dietary Fiber', 'Sugars', 'Protein']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.info()

df.describe()
top_high_cal = df.nlargest(10, "Calories")
top_high_cal[['Item', 'Calories']]
cat_cal = df.groupby('Category')['Calories'].mean().sort_values(ascending=False)
print(cat_cal)  

#bar chart
plt.figure(figsize=(10,5))
sns.barplot(x = cat_cal.index, y = cat_cal.values)
plt.title("Average calories by Categories")
plt.xlabel("Category")
plt.ylabel("Calories")
plt.savefig("Average calories by Categories.png")
plt.show()

#top 10 high calorie food items
plt.figure(figsize=(10,5))
sns.barplot(data=top_high_cal, x = "Calories", y = "Item")
plt.title("Top 10 High Calorie Food Items")
plt.savefig("Top 10 High Calorie Food Items.png")
plt.show()

#correlation of nutrients
plt.figure(figsize=(12,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="Blues")
plt.title("Nutritional Correlation Heatmap")
plt.savefig("Nutritional Correlation Heatmap.png")
plt.show()

#INSIGHTS
print("\n Insights and Recommendations\n")
print("\n1️⃣ Categories with high calories:\n",cat_cal.head(3))
print("\n Recommendations:\n Promote low calorie alternatives in these categories")

high_sodium = df.nlargest(5, "Sodium")[['Item','Sodium']]
print("\n2️⃣ Items with highest Sodium:\n",high_sodium)
print("\n Recommendations:\n Highlight low-Sodium items in the menu")

high_fat = df.nlargest(5, "Total Fat")[['Item','Total Fat']]
print("\n3️⃣ High-Fat items:\n",high_fat)
print("\n Recommendations:\n Provide healthier substituitions or educate customers")

df.to_csv("cleaned_menu.csv",index=False)
