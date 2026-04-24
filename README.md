# 📌 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 📂 Load Dataset
df = pd.read_csv("loan_prediction.csv")

# 👀 View Data
print(df.head())
print(df.info())

# 🧹 Handle Missing Values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

# 🔄 Convert Categorical to Numeric
le = LabelEncoder()

cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Status']
for col in cols:
    df[col] = le.fit_transform(df[col])

# 🎯 Define Features & Target
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# ✂️ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🔍 Prediction
y_pred = model.predict(X_test)

# 📊 Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 📈 Visualization: Loan Status Distribution
plt.figure()
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Approval Distribution")
plt.show()

# 📈 Visualization: Income vs Loan Amount
plt.figure()
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', data=df)
plt.title("Income vs Loan Amount")
plt.show()
