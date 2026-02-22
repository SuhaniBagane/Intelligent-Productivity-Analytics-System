import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("study_data.csv")

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
X = df[["Study_Hours"]]   # input
y = df["Tasks_Completed"] # output

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict for new input
new_hours = pd.DataFrame([[7]], columns=["Study_Hours"])
prediction = model.predict(new_hours)

print("\nPredicted tasks for 7 study hours:", prediction[0])
import joblib

# Save model
joblib.dump(model, "productivity_model.pkl")

print("\nModel saved successfully!")
# Load model
loaded_model = joblib.load("productivity_model.pkl")

# Predict again using loaded model
new_hours = pd.DataFrame([[8]], columns=["Study_Hours"])
prediction = loaded_model.predict(new_hours)

print("Prediction using loaded model:", prediction[0])