import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Reading and Cleaning Data


df = pd.read_csv('./datasets/diabetes.csv')


# Step 2: Exploratory Data Analysis (EDA)

# Perform EDA to understand your data better

plt.figure(figsize=(6, 4))
sns.countplot(x='BMI', data=df)
plt.title('Age')
plt.show()

# Step 3: Training a Logistic Regression model

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['BMI'])  # Replace 'target_column' with your target variable
y = df['Age']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional but can be beneficial for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Step 4: Testing the Logistic Regression model

# Make predictions on the testing data
logistic_predictions = logistic_model.predict(X_test)

# Calculate and display the accuracy
accuracy = accuracy_score(y_test, logistic_predictions)
print(f"Accuracy: {accuracy}")

# Generate a classification report
class_report = classification_report(y_test, logistic_predictions)
print(class_report)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, logistic_predictions)
print(conf_matrix)

# plt.figure(figsize=(12,12))
# sns.countplot(logistic_model,filled=True, feature_names= X.columns, class_names=True)

# # plot_tree(logistic_model, filled=True, feature_names= X.columns, class_names=True)
# plt.show()