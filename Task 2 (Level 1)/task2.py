# CUSTOMER SEGMENTATION ANALYSIS - INTERNSHIP TASK (LEVEL 1) (DATA ANALYTICS)

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.stats import mode
import warnings
warnings.filterwarnings("ignore")

print("\u2713 Libraries imported")

# DATA COLLECTION
# Load the dataset
df = pd.read_csv("customer_segmentation_analysis.csv")

print("\n\u2713 Dataset Loaded Successfully")
print("\nFirst 5 Rows:")
print(df.head())

#DATA EXPLORATION & CLEANING
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Drop rows with missing values 
df_clean = df.dropna()

print("\n\u2713 Missing values removed")
print(f"Rows before: {len(df)}, Rows after: {len(df_clean)}")

# DESCRIPTIVE STATISTICS

print("\n--- DESCRIPTIVE STATISTICS ---")
print(df_clean.describe())

# Calculate mean, median, mode, std for each numeric column
numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns

stats = pd.DataFrame({
    "Mean": df_clean[numeric_cols].mean(),
    "Median": df_clean[numeric_cols].median(),
    "Mode": df_clean[numeric_cols].mode().iloc[0],
    "Std Dev": df_clean[numeric_cols].std()
})

print("\n\u2713 Statistical Summary (Mean, Median, Mode, Std Dev):")
print(stats)

# DATA VISUALIZATION
plt.figure(figsize=(8,5))
sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.savefig("Correlation Heatmap.png")
plt.show()

for col in numeric_cols:
    plt.figure(figsize=(7,4))
    sns.histplot(df_clean[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"Distribution of {col}.png")
    plt.show()

# CUSTOMER SEGMENTATION (K-Means)
# Standardizing numeric features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean[numeric_cols])

# Elbow Method to find optimal k
inertia = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, marker='o')
plt.title("Elbow Method - Optimal K")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.savefig("Elbow Method - Optimal K.png")
plt.show()

# Choose optimal k = 4 (usually from the elbow curve)
kmeans = KMeans(n_clusters=4, random_state=42)
df_clean["Cluster"] = kmeans.fit_predict(scaled_data)

print("\n\u2713 Clustering Completed")
print(df_clean["Cluster"].value_counts())

# VISUALIZATION OF CLUSTERS
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

df_clean["PC1"] = pca_data[:, 0]
df_clean["PC2"] = pca_data[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_clean,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10"
)
plt.title("Customer Segments (PCA Visualization)")
plt.savefig("Customer Segments (PCA Visualization).png")
plt.show()


# INSIGHTS & RECOMMENDATIONS
cluster_summary = df_clean.groupby("Cluster")[numeric_cols].mean()

print("\n--- Cluster Summary (Insights) ---")
print(cluster_summary)
print("\n\u2713 Analysis Complete")

# AUTO-GENERATED INSIGHTS & RECOMMENDATIONS
print("\n==================== INSIGHTS ====================\n")

cluster_profile = df_clean.groupby("Cluster")[numeric_cols].mean()
print(cluster_profile)

for cluster in cluster_profile.index:
    print(f"\n--- Cluster {cluster} Insights ---")
    for col in numeric_cols:
        value = cluster_profile.loc[cluster, col]
        print(f"{col}: {value:.2f}")
    print("----------------------------------")

print("\n================ RECOMMENDATIONS ================\n")

recommendations = {}

for cluster in cluster_profile.index:
    rec_list = []
    cluster_data = cluster_profile.loc[cluster]

    # Example logic based on purchase behaviour
    if cluster_data.mean() > cluster_profile.mean().mean():
        rec_list.append("High-value customers → Provide loyalty rewards, premium offers, exclusive discounts.")
    else:
        rec_list.append("Low/medium-value customers → Provide personalized offers to increase engagement.")

    # If frequency exists
    if "Frequency" in numeric_cols:
        if cluster_data["Frequency"] > cluster_profile["Frequency"].mean():
            rec_list.append("Frequent shoppers → Promote membership programs.")
        else:
            rec_list.append("Low frequency → Send reminders, retargeting ads.")

    # If monetary value exists
    if "Monetary" in numeric_cols or "PurchaseAmount" in numeric_cols:
        money_col = "Monetary" if "Monetary" in numeric_cols else "PurchaseAmount"
        if cluster_data[money_col] > cluster_profile[money_col].mean():
            rec_list.append("High spenders → Upsell premium products.")
        else:
            rec_list.append("Low spenders → Provide entry-level product suggestions.")

    recommendations[cluster] = rec_list

# Print recommendations
for cluster, rec_list in recommendations.items():
    print(f"\n--- Recommendations for Cluster {cluster} ---")
    for r in rec_list:
        print(f"- {r}")

df.to_csv("cleaned_customer_segmentation_analysis.csv",index=False)
print("\u2713 Cleaned dataset loaded successfully")
