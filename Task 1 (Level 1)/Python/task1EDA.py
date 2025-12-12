#EXPLORATORY DATA ANALYSIS - INTERNSHIP TASK 1 (LEVEL 1)(DATA ANALYTICS)
# importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the dataset
df = pd.read_csv("retail_sales_dataset.csv")
df.head()
df['Date'] = pd.to_datetime(df['Date'])
print(df.isnull().sum())
df = df.drop_duplicates()

#descriptive statistics(mean,median,mode and SD)
df.describe()
print("Mean :",df['Total Amount'].mean())
print("Median :",df['Total Amount'].median())
print("Mode :",df['Product Category'].mode()[0])
print("Standard Deviation :",df['Total Amount'].std())

#Line chart for Daily sales (Time series analysis)
sales = df.groupby('Date')['Total Amount'].sum()
plt.figure(figsize=(12,5))
plt.plot(sales)
plt.title("Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.savefig("Sales trend.png")
plt.show()

#bar chart for Product sales
product = df.groupby('Product Category')['Total Amount'].sum().sort_values()
plt.figure(figsize=(10,5))
sns.barplot(x = product.index, y = product.values)
plt.title("Product Trend")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.savefig("Product Trend.png")
plt.show()

#bar chart for top customers
customers = df.groupby('Customer ID')['Total Amount'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x = customers.index, y = customers.values)
plt.title("Top 10 customers by spending")
plt.xticks(rotation = 45)
plt.savefig("Top 10 customers by spending.png")
plt.show()

#heatmap
plt.figure(figsize=(6,4))
numeric = df.select_dtypes(include=['int64','float64'])
corr = numeric.corr()
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.title("Correlated Heatmap")
plt.savefig("Correlation heatmap.png")
plt.show()

df.to_csv("cleaned_retail_sales_dataset.csv", index=False)

#INSIGHTS

print("KEY INSIGHTS & RECOMMENDATIONS")
print("------------------------------------------")

print("\n1️⃣ Best-selling category:")
print(df.groupby('Product Category')['Total Amount'].sum().idxmax())

print("\n2️⃣ Age group that spends the most:")
print(df.groupby('Age')['Total Amount'].sum().idxmax())


print("\n3️⃣ Recommendations:\n")
print("- Stock more products in the highest-selling category.")
print("- Provide offers for age groups with low spending to improve engagement.")
print("- Analyze peak days to plan staffing and inventory.")
print("- Use high correlation variables (Price, Quantity) for targeted strategies.")
