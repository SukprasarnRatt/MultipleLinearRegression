import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Student_Performance.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Taking care of missing boolean data by replacing with the most frequent value of the column
col = [2]
booleanImputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
booleanImputer.fit(X[:, col])
X[:, col] = booleanImputer.transform(X[:, col])

# Taking care of missing number data by replacing with the average of the column
col = [0,1,3,4]
numberImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
numberImputer.fit(X[:, col])
X[:, col] = np.round(numberImputer.transform(X[:, col]), 2)

# Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)
Y_pred = np.round(Y_pred, 2)

# compare predicted and real student performance inexes
results = np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1)

# Saving the results to a CSV file
results_df = pd.DataFrame(results, columns=['Predicted Performance Index', 'Actual Performance Index'])
results_df.to_csv('predictions.csv', index=False)