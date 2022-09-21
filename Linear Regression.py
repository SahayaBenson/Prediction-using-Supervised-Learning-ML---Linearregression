#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#reading data from file
data = pd.read_csv("D:/Ben/MBA/GRIP/task1.csv")
print("Data imported successfully")
data.head(10)

#testing is there any null data
data.isnull==True

#plotting the data as linear
data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scores')
plt.show()

#preparing data
X=data.iloc[:, :-1].values
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=0.2, random_state=0)

#training the algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Training complete.")

#plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scores')
plt.show()

#making Prediction
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores

#comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

# Predicting Score
hours = [9.25]
own_pred = regressor.predict([hours])
print("No of Hours = {}".format([hours]))
print("Predicted Score = {}".format(own_pred[0],3))

#Evaluating the Model
from sklearn import metrics
print('Mean Absolute Error:',
      metrics.mean_absolute_error(y_test, y_pred))