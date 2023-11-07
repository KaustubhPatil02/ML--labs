# import pandas as pd
# import numpy as npp
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import linear_model 
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split

# df=pd.read_csv('./datasets/housing.csv')
# df.describe()
# df = df.rename(columns={'housing_median_age':'age'})
# df.head()

# #EDA
# plt.figure(figsize=(10,10))
# sns.histplot(df['latitude'], kde=False, color='b')
# plt

# X_train, X_test, y_train, y_test= train_test_split(df[['population']], df.households, test_size=0.2)
# X_train
# X_test

# reg = linear_model.LinearRegression()
# reg.fit(df[['population']], df.households)
# reg.predict([[1000]])
# # reg.predict(X_test)

# # acc = accuracy_score(y_test, reg.predict(X_test))

# # Plot the training data
# plt.scatter(X_train, y_train, color='blue')

# # Plot the regression line
# reg_line = reg.coef_*df[['population']] + reg.intercept_
# plt.plot(df[['population']], reg_line, color='red')

# plt.xlabel('Population')
# plt.ylabel('Households')
# plt.title('Linear Regression')
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

df=pd.read_csv('./datasets/housing.csv')
df.describe()
df = df.rename(columns={'housing_median_age':'age'})
df.head()

#EDA
plt.figure(figsize=(10,10))
sns.histplot(df['latitude'], kde=False, color='b')
plt.show()

X_train, X_test, y_train, y_test= train_test_split(df[['population']], df.households, test_size=0.2)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Plot the training data
plt.scatter(X_train, y_train, color='blue')

# Plot the regression line
reg_line = reg.coef_*X_train + reg.intercept_
plt.plot(X_train, reg_line, color='red')

plt.xlabel('Population')
plt.ylabel('Households')
plt.title('Linear Regression')
plt.show()