#PREDICTING HOUSE PRICES WITH LINEAR REGRESSION - INTERNSHIP TASK 1 (LEVEL 2) (DATA ANALYTICS)

#Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("\n📌 Libraries Imported Successfully")

# LOADING THE  DATASET
df = pd.read_csv("housing.csv")   
print("\n📌 Dataset Loaded Successfully")
print(df.head())
print(df.info())
print(df.describe())

# DATA CLEANING
# Remove unnamed index columns if present
df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]

# Drop missing values
df = df.dropna()
print("\n📌 After Removing Missing Values:", df.shape)

# FEATURE SELECTION
# Identify numerical columns
numeric_cols = df.select_dtypes(include=['int64','float64']).columns
print("\n📌 Numerical Columns:", numeric_cols)
target = 'price'       

X = df[numeric_cols].drop(columns=[target], errors='ignore')
y = df[target]

print("\n📌 Selected Features:", X.columns)
print("📌 Target Variable:", target)

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n📌 Data Split Completed")

# MODEL TRAINING
model = LinearRegression()
model.fit(X_train, y_train)
print("\n📌 Linear Regression Model Trained Successfully")

# MODEL PREDICTION
y_pred = model.predict(X_test)

# MODEL EVALUATION
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📌 MODEL EVALUATION RESULTS")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

# VISUALIZATION
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted - Linear Regression")
plt.savefig("Actual vs Predicted - Linear Regression.png")
plt.show()

# Distribution of errors
errors = y_test - y_pred
plt.figure(figsize=(8,6))
sns.histplot(errors, kde=True)
plt.title("Distribution of Prediction Errors")
plt.xlabel("Prediction Error")
plt.savefig("Distribution of Pridiction Errors.png")
plt.show()

df.to_csv("cleaned_housing.csv",index=False)

print("\n🎉 House Price Prediction Task Completed Successfully!")
