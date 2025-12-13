# WINE QUALITY PREDICTION - INTERNSHIP TASK 2 (LEVEL 2) (DATA ANALYTICS)

#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

# LOADING THE DATASET
df = pd.read_csv("WineQT.csv")
print("\n\u2713 Dataset Loaded Successfully")
print(df.head())

print("\n\u2713 Dataset Info:\n")
print(df.info())

print("\n\u2713 Summary Statistics\n")
print(df.describe())

# CHECK FOR MISSING VALUES
print("\n\u2713 Missing values:\n")
print(df.isnull().sum())

# DATA VISUALIZATION
plt.figure(figsize=(10,5))
sns.countplot(x="quality",data=df)
plt.title("Distribution of Wine Quality")
plt.savefig("Distribution of Wine Quality.png")
plt.show()

plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation heatmap of Wine Quality")
plt.savefig("Correlation heatmap of Wine Quality.png")
plt.show()

# FEATURE SELECTION 
x = df.drop("quality", axis=1)
y = df["quality"]
y = np.where(y  >= 6, 1, 0)
print("\n Converted Quality Class Distribution:")
print(pd.Series(y).value_counts())

# TRAIN TEST AND SPLIT
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

# FEATURE SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL 1 — RANDOM FOREST
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# MODEL 2 — STOCHASTIC GRADIENT DESCENT
sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)

print("\nSGD Accuracy:", accuracy_score(y_test, sgd_pred))
print(classification_report(y_test, sgd_pred))

# MODEL 3 — SUPPORT VECTOR CLASSIFIER (SVC)
svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

print("\nSVC Accuracy:", accuracy_score(y_test, svc_pred))
print(classification_report(y_test, svc_pred))

# CONFUSION MATRIX FOR BEST MODEL (Random Forest)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix – Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Confusion Matrix - Random Forest.png")
plt.show()

print("\nWorkflow Completed Successfully!")
