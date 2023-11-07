import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


dataset = pd.read_csv('./datasets/diabetes.csv')
X = dataset.drop(columns=['Insulin'])
y = dataset['BloodPressure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_prediction = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_prediction)
print("Accuracy of the model:", dt_accuracy)

plt.figure(figsize=(12,12))
plot_tree(dt_model, filled=True, feature_names= X.columns, class_names=True)
plt.show()