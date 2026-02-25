import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("study_data.csv")

# New features
df["Prev_Day_Hours"] = df["Study_Hours"].shift(1)
df["Rolling_Avg"] = df["Study_Hours"].rolling(window=3).mean()

# Drop NaN rows
df = df.dropna()

print("Study Data:\n")
print(df)

# Basic stats
avg_hours = df["Study_Hours"].mean()
total_tasks = df["Tasks_Completed"].sum()
consistency = df["Study_Hours"].std()

print("\nAverage Study Hours:", avg_hours)
print("Total Tasks Done:", total_tasks)
print("Consistency Score:", consistency)

# NEW: Productivity Score
df["Productivity_Score"] = df["Tasks_Completed"] / df["Study_Hours"]

print("\nUpdated Data with Productivity Score:\n")
print(df)

# 📊 GRAPH 1 — Study Hours Trend
plt.figure()
plt.plot(df["Day"], df["Study_Hours"], marker='o')
plt.title("Study Hours Trend")
plt.xlabel("Day")
plt.ylabel("Hours")
plt.show()

# 📊 GRAPH 2 — Tasks Completed
plt.figure()
plt.bar(df["Day"], df["Tasks_Completed"])
plt.title("Tasks Completed per Day")
plt.xlabel("Day")
plt.ylabel("Tasks")
plt.show()

# 📊 GRAPH 3 — Productivity Score
plt.figure()
plt.plot(df["Day"], df["Productivity_Score"], marker='o')
plt.title("Productivity Score Trend")
plt.xlabel("Day")
plt.ylabel("Score")
plt.show()
from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data
X = df[["Study_Hours", "Prev_Day_Hours", "Rolling_Avg"]]   # input
y = df["Tasks_Completed"] # output

# Train model
model = LinearRegression()
model.fit(X, y)
# Train Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(X, y)

# Train Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X, y)
# Predictions
lr_pred = model.predict(X)
dt_pred = dt_model.predict(X)
rf_pred = rf_model.predict(X)

# Compare errors
print("\nModel Comparison:")
print("Linear Regression Error:", mean_absolute_error(y, lr_pred))
print("Decision Tree Error:", mean_absolute_error(y, dt_pred))
print("Random Forest Error:", mean_absolute_error(y, rf_pred))

# Predict for new input
new_data = pd.DataFrame([[7, 6, 5.5]],
                        columns=["Study_Hours", "Prev_Day_Hours", "Rolling_Avg"])

prediction = model.predict(new_data)
print("\nPredicted tasks for 7 study hours:", prediction[0])
import joblib

# Save model
joblib.dump(model, "productivity_model.pkl")

print("\nModel saved successfully!")
# Load model
loaded_model = joblib.load("productivity_model.pkl")

# Predict again using loaded model
new_hours = pd.DataFrame([[8]], columns=["Study_Hours"])
prediction = model.predict(new_data)
print("Advanced Prediction:", prediction[0])

print("Prediction using loaded model:", prediction[0])