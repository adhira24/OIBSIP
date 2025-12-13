#DATA CLEANING - INTERNSHIP TASK 3 (LEVEL 1) (DATA ANALYTICS)

#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#LOAD DATA
df = pd.read_csv("task3_data.csv")
print("🔹 Data Loaded Successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe(include='all'))

# CHECK DATA INTEGRITY
print("\n🔹 Missing Values Count:")
print(df.isnull().sum())

print("\n🔹 Duplicate Rows Count:")
print(df.duplicated().sum())

# HANDLE MISSING VALUES
# Numeric columns → fill with mean
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Categorical columns → fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n✅ Missing Values Handled Successfully")

# REMOVE DUPLICATES
df_before = df.shape[0]
df = df.drop_duplicates()
df_after = df.shape[0]

print(f"\nDuplicates Removed: {df_before - df_after}")

# STANDARDIZATION (NUMERIC ONLY)
scaler = StandardScaler()
df_scaled = df.copy()

df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

print("\n✅ Standardization Completed")

# OUTLIER DETECTION USING IQR
def detect_outliers(df, columns):
    outlier_indices = []

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - (1.5 * IQR)
        upper = Q3 + (1.5 * IQR)

        outliers = df[(df[col] < lower) | (df[col] > upper)].index
        outlier_indices.extend(outliers)

        print(f"Outliers in {col}: {len(outliers)}")

    return list(set(outlier_indices))

outlier_rows = detect_outliers(df_scaled, numeric_cols)
print(f"\nTotal Outlier Rows Detected: {len(outlier_rows)}")

# Remove Outliers
df_cleaned = df_scaled.drop(outlier_rows).reset_index(drop=True)

print("\n✅ Outliers Removed Successfully")
print(f"Final Cleaned Dataset Shape: {df_cleaned.shape}")

# SAVE CLEANED DATA
df_cleaned.to_csv("task3_data_cleaned.csv", index=False)
print("\n🎉 Cleaned Data Saved as: task3_data_cleaned.csv")

# OPTIONAL VISUALIZATIONS
# Histogram for numeric columns
df[numeric_cols].hist(figsize=(12,8))
plt.suptitle("Numeric Columns Distribution")
plt.savefig("Numeric Columns Distribution.png")
plt.show()

# Heatmap for correlations
plt.figure(figsize=(10,7))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("Correlation Heatmap.png")
plt.show()
