# FRAUD DETECTION - INTERNSHIP TASK 3 (LEVEL 2) (DATA ANALYTICS)

#Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest

import warnings
warnings.filterwarnings('ignore')

# LOADING DATASETS
df = pd.read_csv('creditcard.csv')

print("Dataset Loaded Successfully!")
print(df.head())

# BASIC DATASET INFO
print(df.info())
print(df['Class'].value_counts())

# FEATURES AND TARGET
X = df.drop('Class', axis=1)
y = df['Class']

# TRAIN - TEST - SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MACHINE LEARNING MODELS (using class weight)

# Logistic Regression
log_model = LogisticRegression(class_weight='balanced')
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

# Decision Tree
dt_model = DecisionTreeClassifier(class_weight='balanced')
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)

from sklearn.tree import plot_tree
plt.figure(figsize=(15,10))
plot_tree(dt_model, filled=True, max_depth=3, feature_names=X.columns, class_names=["Not Fraud","Fraud"])
plt.title("Decision Tree (Depth = 3)")
plt.savefig("Decision Tree.png")
plt.show()

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

from sklearn.metrics import ConfusionMatrixDisplay
plt.figure(figsize=(6,5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, cmap='Blues')
plt.title("Random Forest - Confusion Matrix")
plt.savefig("Random Forest.png")
plt.show()

importances = rf_model.feature_importances_
features = X.columns
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("Rnadom_Forest_Feature_Importance.png")
plt.show()

from sklearn.metrics import RocCurveDisplay
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
plt.figure(figsize=(6,5))
RocCurveDisplay.from_predictions(y_test, rf_probs)
plt.title("Random Forest - ROC Curve")
plt.savefig("Random Forest - ROC Curve.png")
plt.show()

# EVALUATION FUNCTION
def evaluate_model(name, y_test, y_pred):
    print(f"\nModel: {name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# MODEL EVALUATIONS
evaluate_model("Logistic Regression", y_test, y_pred_log)
evaluate_model("Decision Tree", y_test, y_pred_dt)
evaluate_model("Random Forest", y_test, y_pred_rf)

rf_probs = rf_model.predict_proba(X_test_scaled)[:,1]
print("Random Forest ROC-AUC:", roc_auc_score(y_test, rf_probs))

# ANAMOLY DETECTION (ISOLATION FOREST)
iso = IsolationForest(contamination=0.001, random_state=42)
iso_pred = iso.fit_predict(X)

iso_pred = [1 if p == -1 else 0 for p in iso_pred]

from sklearn.metrics import PrecisionRecallDisplay
plt.figure(figsize=(6,5))
PrecisionRecallDisplay.from_predictions(y_test, rf_probs)
plt.title("Precision-Recall Curve (Random Forest)")
plt.show()

from sklearn.decomposition import PCA

# PREPROCESSING THE DATA FOR THE 2D PLOT OF ISOLATION FOREST
df = df.copy()

# Remove target variable for anomaly boundary plot
X = df.drop("Class", axis=1)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2 dimensions using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply Isolation Forest
iso = IsolationForest(contamination=0.01, random_state=42)
y_pred = iso.fit_predict(X_pca)

# Convert predictions to 0=normal, 1=anomaly
anomaly = (y_pred == -1)

# Plot decision boundary
plt.figure(figsize=(10, 7))

# Create mesh grid
xx, yy = np.meshgrid(
    np.linspace(X_pca[:,0].min(), X_pca[:,0].max(), 300),
    np.linspace(X_pca[:,1].min(), X_pca[:,1].max(), 300)
)

# Predict anomaly score for grid
Z = iso.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot boundary
plt.contourf(xx, yy, Z, cmap="coolwarm", levels=50, alpha=0.7)

# Plot normal points
plt.scatter(
    X_pca[~anomaly, 0], X_pca[~anomaly, 1],
    s=5, label="Normal", color="blue"
)

# Plot anomalies
plt.scatter(
    X_pca[anomaly, 0], X_pca[anomaly, 1],
    s=10, label="Anomaly (Fraud)", color="red"
)

plt.title("Isolation Forest — Anomaly Detection Boundary (PCA 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.savefig("Isolation Forest.png")
plt.show()

print("\nIsolation Forest Results:")
print(confusion_matrix(y, iso_pred))
print(classification_report(y, iso_pred))
