import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data =pd.read_csv('./datasets/diabetes.csv')

plt.figure(figsize=(20,10))
data['BMI']. value_counts().plot(kind='bar')
plt.title('Targer Distribution', )
plt.show()

X = data.drop(columns=('Insulin'))
y =data['BloodPressure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
model_accuracy = accuracy_score(y_test, predictions)
class_report = classification_report(y_test, predictions)
print(class_report)

confusion_matrix= confusion_matrix(y_test, predictions)
print(confusion_matrix)